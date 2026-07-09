"""
分類模型模組
- 優先使用 scikit-learn 的 RandomForestClassifier（若環境有安裝）
- 若 scikit-learn 不可用，自動退回本檔內建、以 NumPy 實作的簡易隨機森林
  (CART 決策樹 + bagging + 特徵隨機抽樣)，確保程式在任何環境都能執行。
"""
import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier as _SKRandomForest
    from sklearn.preprocessing import StandardScaler as _SKScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ----------------------------------------------------------------------
# Fallback: 純 NumPy 實作的簡易決策樹 + 隨機森林
# ----------------------------------------------------------------------
class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None  # 葉節點: 正類機率


def _gini(y):
    if len(y) == 0:
        return 0.0
    p = np.mean(y)
    return 1 - p ** 2 - (1 - p) ** 2


class _DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=20, n_features=None, random_state=0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.rng = np.random.RandomState(random_state)
        self.root = None

    def fit(self, X, y):
        self.root = self._build(X, y, depth=0)
        return self

    def _build(self, X, y, depth):
        node = _Node()
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            node.value = float(np.mean(y)) if len(y) else 0.5
            return node

        n_samples, n_total_features = X.shape
        k = self.n_features or max(1, int(np.sqrt(n_total_features)))
        feat_idx = self.rng.choice(n_total_features, size=min(k, n_total_features), replace=False)

        # 節點樣本數過多時，只用隨機子樣本尋找分割門檻（找到門檻後仍用「全部樣本」實際分裂），
        # 這是常見的加速手法（近似 histogram-based 分割搜尋），大幅降低大資料集的訓練時間。
        MAX_SEARCH_SAMPLES = 1500
        if n_samples > MAX_SEARCH_SAMPLES:
            search_idx = self.rng.choice(n_samples, size=MAX_SEARCH_SAMPLES, replace=False)
            X_search, y_search = X[search_idx], y[search_idx]
        else:
            X_search, y_search = X, y

        best_gain, best_feat, best_thr = -1, None, None
        parent_gini = _gini(y_search)
        n_search = len(y_search)

        for f in feat_idx:
            col = X_search[:, f]
            # 用分位數當候選門檻，加快速度
            candidates = np.unique(np.quantile(col, [0.33, 0.5, 0.67]))
            for thr in candidates:
                left_mask = col <= thr
                n_left = left_mask.sum()
                if n_left < 2 or (n_search - n_left) < 2:
                    continue
                g_left = _gini(y_search[left_mask])
                g_right = _gini(y_search[~left_mask])
                w_left = n_left / n_search
                w_right = 1 - w_left
                gain = parent_gini - (w_left * g_left + w_right * g_right)
                if gain > best_gain:
                    best_gain, best_feat, best_thr = gain, f, thr

        if best_feat is None or best_gain <= 1e-6:
            node.value = float(np.mean(y))
            return node

        node.feature = best_feat
        node.threshold = best_thr
        left_mask = X[:, best_feat] <= best_thr
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _predict_one(self, x):
        node = self.root
        while node.value is None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict_proba(self, X):
        return np.array([self._predict_one(x) for x in X])


class _NumpyRandomForest:
    """簡易隨機森林 (bagging of decision trees)，作為 sklearn 缺席時的替代方案。"""

    def __init__(self, n_estimators=50, max_depth=5, min_samples_split=20, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)
        rng = np.random.RandomState(self.random_state)
        self.trees = []
        for i in range(self.n_estimators):
            idx = rng.randint(0, n, size=n)  # bootstrap sample
            tree = _DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state + i,
            )
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        preds = np.mean([t.predict_proba(X) for t in self.trees], axis=0)
        return np.column_stack([1 - preds, preds])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class _SimpleScaler:
    def fit(self, X):
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ----------------------------------------------------------------------
# 對外統一介面
# ----------------------------------------------------------------------
class DirectionModel:
    """封裝分類模型，自動選用 sklearn 或內建 NumPy 版本。"""

    def __init__(self, n_estimators=100, max_depth=6, min_samples_split=20, random_state=42):
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
        )
        if HAS_SKLEARN:
            self.scaler = _SKScaler()
            self.clf = _SKRandomForest(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_state,
                n_jobs=-1,
            )
            self.backend = "scikit-learn RandomForestClassifier"
        else:
            self.scaler = _SimpleScaler()
            self.clf = _NumpyRandomForest(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_state,
            )
            self.backend = "內建 NumPy RandomForest (sklearn 未安裝，已自動退回替代方案)"

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(np.asarray(X, dtype=float))
        self.clf.fit(Xs, np.asarray(y))
        return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(np.asarray(X, dtype=float))
        return self.clf.predict_proba(Xs)[:, 1]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
