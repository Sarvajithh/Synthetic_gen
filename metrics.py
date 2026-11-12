import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def get_device(device=None):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def flatten_windows(X):
    """Pool observations for pointwise distribution metrics only.

    Joining overlapping windows fabricates transitions, so temporal metrics
    must operate window-by-window instead.
    """
    n_windows, window_size, n_features = X.shape
    return X.reshape(n_windows * window_size, n_features)


def _windowed_series(X):
    X = np.asarray(X)
    if X.ndim != 3 or X.shape[0] == 0 or X.shape[1] < 2:
        raise ValueError("Expected non-empty windows with shape (n_windows, time, features)")
    if not np.all(np.isfinite(X)):
        raise ValueError("Metric input contains non-finite values")
    return X


def _mean_window_acf(X, max_lag):
    X = _windowed_series(X)
    max_lag = min(int(max_lag), X.shape[1] - 1)
    series = X.transpose(2, 0, 1).reshape(-1, X.shape[1])
    return np.mean([_autocorr(row, max_lag) for row in series], axis=0)


class BaseMetric:
    name = "base"

    def compute(self, real, synthetic):
        raise NotImplementedError

    def __call__(self, real, synthetic):
        return self.compute(real, synthetic)


class _SimpleGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=1):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out)


class PredictiveScoreMetric(BaseMetric):
    name = "predictive_score"

    def __init__(self, hidden_dim=32, epochs=50, batch_size=64, lr=1e-3, eps=1e-8, device=None, seed=0):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.eps = eps
        self.device = get_device(device)
        self.seed = seed

    def _train_predictor(self, X):
        torch.manual_seed(self.seed)
        n_features = X.shape[2]
        model = _SimpleGRU(n_features, self.hidden_dim, n_features).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.L1Loss()
        inputs = torch.tensor(X[:, :-1, :], dtype=torch.float32)
        targets = torch.tensor(X[:, 1:, :], dtype=torch.float32)
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=self.batch_size, shuffle=True)
        model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return model

    def _eval_predictor(self, model, X_test):
        model.eval()
        inputs = torch.tensor(X_test[:, :-1, :], dtype=torch.float32, device=self.device)
        targets = torch.tensor(X_test[:, 1:, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pred = model(inputs)
            loss = torch.mean(torch.abs(pred - targets)).item()
        model.train()
        return loss

    def compute(self, real, synthetic, real_test=None):
        if real_test is None:
            n = real.shape[0]
            split = max(1, int(n * 0.8))
            real_train, real_test = real[:split], real[split:]
        else:
            real_train = real

        model_synth = self._train_predictor(synthetic)
        model_real = self._train_predictor(real_train)

        loss_synth = self._eval_predictor(model_synth, real_test)
        loss_real = self._eval_predictor(model_real, real_test)

        return loss_synth / (loss_real + self.eps)


class DiscriminativeScoreMetric(BaseMetric):
    name = "discriminative_score"

    def __init__(self, hidden_dim=32, epochs=50, batch_size=64, lr=1e-3, test_ratio=0.2, device=None, seed=0):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.test_ratio = test_ratio
        self.device = get_device(device)
        self.seed = seed

    def compute(self, real, synthetic):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        n = min(real.shape[0], synthetic.shape[0])
        if n < 4:
            raise ValueError("At least four real and synthetic windows are required")
        real, synthetic = real[:n], synthetic[:n]
        n_features = real.shape[2]

        X = np.concatenate([real, synthetic], axis=0)
        y = np.concatenate([np.ones(n), np.zeros(n)], axis=0)
        idx = np.random.permutation(2 * n)
        X, y = X[idx], y[idx]
        split = int(2 * n * (1 - self.test_ratio))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = _SimpleGRU(n_features, self.hidden_dim, 1).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=self.batch_size, shuffle=True)

        model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = model(xb)[:, -1, 0]
                loss = loss_fn(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test, dtype=torch.float32, device=self.device)
            logits = model(X_test_t)[:, -1, 0]
            preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()

        acc = np.mean(preds == y_test)
        return 2 * abs(acc - 0.5)


def _autocorr(x, max_lag):
    x = x - np.mean(x)
    n = len(x)
    denom = np.sum(x ** 2)
    if denom == 0:
        return np.zeros(max_lag)
    acf = np.zeros(max_lag)
    for h in range(1, max_lag + 1):
        num = np.sum(x[:n - h] * x[h:])
        acf[h - 1] = num / denom
    return acf


class RawReturnACFErrorMetric(BaseMetric):
    name = "raw_return_acf_error"

    def __init__(self, max_lag=10):
        self.max_lag = max_lag

    def compute(self, real, synthetic):
        real, synthetic = _windowed_series(real), _windowed_series(synthetic)
        if real.shape[2] != synthetic.shape[2]:
            raise ValueError("Real and synthetic feature counts differ")
        n_features = real.shape[2]
        errors = []
        for f in range(n_features):
            acf_r = _mean_window_acf(real[:, :, f:f + 1], self.max_lag)
            acf_s = _mean_window_acf(synthetic[:, :, f:f + 1], self.max_lag)
            errors.append(np.sum(np.abs(acf_r - acf_s)))
        return float(np.mean(errors))


class VolatilityClusteringErrorMetric(BaseMetric):
    name = "volatility_clustering_error"

    def __init__(self, max_lag=10, use_absolute=False):
        self.max_lag = max_lag
        self.use_absolute = use_absolute

    def compute(self, real, synthetic):
        real, synthetic = _windowed_series(real), _windowed_series(synthetic)
        n_features = real.shape[2]
        errors = []
        for f in range(n_features):
            r = np.abs(real[:, :, f:f + 1]) if self.use_absolute else real[:, :, f:f + 1] ** 2
            s = np.abs(synthetic[:, :, f:f + 1]) if self.use_absolute else synthetic[:, :, f:f + 1] ** 2
            acf_r = _mean_window_acf(r, self.max_lag)
            acf_s = _mean_window_acf(s, self.max_lag)
            errors.append(np.sum(np.abs(acf_r - acf_s)))
        return float(np.mean(errors))


def _correlation_matrix(X):
    X = X - X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    Xn = X / std
    n = Xn.shape[0]
    return (Xn.T @ Xn) / n


class CorrelationErrorMetric(BaseMetric):
    name = "correlation_error"

    def compute(self, real, synthetic):
        real_flat = flatten_windows(real)
        synth_flat = flatten_windows(synthetic)
        C_real = _correlation_matrix(real_flat)
        C_synth = _correlation_matrix(synth_flat)
        diff = C_real - C_synth
        return float(np.sqrt(np.sum(diff ** 2)))


class HillErrorMetric(BaseMetric):
    name = "hill_error"

    def __init__(self, k=50, tail="both"):
        self.k = k
        self.tail = tail

    def _hill_estimator(self, x):
        x_sorted = np.sort(x)[::-1]
        k = min(self.k, len(x_sorted) - 1)
        if k < 1:
            return 0.0
        top = x_sorted[:k + 1]
        if top[k] <= 0:
            top = top[top > 0]
            k = len(top) - 1
            if k < 1:
                return 0.0
        logs = np.log(top[:k] / top[k])
        return float(np.mean(logs))

    def _tail_values(self, x, side):
        if side == "upside":
            return x[x > 0]
        return -x[x < 0]

    def compute(self, real, synthetic):
        real_flat = flatten_windows(real)
        synth_flat = flatten_windows(synthetic)
        n_features = real_flat.shape[1]
        sides = ["upside", "downside"] if self.tail == "both" else [self.tail]
        errors = []
        for f in range(n_features):
            for side in sides:
                r_tail = self._tail_values(real_flat[:, f], side)
                s_tail = self._tail_values(synth_flat[:, f], side)
                gamma_r = self._hill_estimator(r_tail)
                gamma_s = self._hill_estimator(s_tail)
                errors.append(abs(gamma_r - gamma_s))
        return float(np.mean(errors))


def _periodogram(x):
    x = x - np.mean(x)
    n = len(x)
    fft = np.fft.rfft(x)
    psd = (np.abs(fft) ** 2) / n
    return psd


class SpectralErrorMetric(BaseMetric):
    name = "spectral_error"

    def __init__(self, eps=1e-8, use_squared=False):
        self.eps = eps
        self.use_squared = use_squared

    def compute(self, real, synthetic):
        real, synthetic = _windowed_series(real), _windowed_series(synthetic)
        if real.shape[1] != synthetic.shape[1] or real.shape[2] != synthetic.shape[2]:
            raise ValueError("Real and synthetic window dimensions differ")
        n_features = real.shape[2]
        errors = []
        for f in range(n_features):
            r = real[:, :, f] ** 2 if self.use_squared else real[:, :, f]
            s = synthetic[:, :, f] ** 2 if self.use_squared else synthetic[:, :, f]
            f_r = np.mean([_periodogram(row) for row in r], axis=0)
            f_s = np.mean([_periodogram(row) for row in s], axis=0)
            num = np.sum(np.abs(f_r - f_s))
            den = np.sum(f_r) + self.eps
            errors.append(num / den)
        return float(np.mean(errors))


class SkewnessErrorMetric(BaseMetric):
    name = "skewness_error"

    def _skew(self, x):
        mu = np.mean(x)
        sigma = np.std(x)
        if sigma == 0:
            return 0.0
        return float(np.mean((x - mu) ** 3) / sigma ** 3)

    def compute(self, real, synthetic):
        real_flat = flatten_windows(real)
        synth_flat = flatten_windows(synthetic)
        n_features = real_flat.shape[1]
        errors = [abs(self._skew(real_flat[:, f]) - self._skew(synth_flat[:, f])) for f in range(n_features)]
        return float(np.mean(errors))


class KurtosisErrorMetric(BaseMetric):
    name = "kurtosis_error"

    def __init__(self, excess=False):
        self.excess = excess

    def _kurt(self, x):
        mu = np.mean(x)
        sigma = np.std(x)
        if sigma == 0:
            return 0.0
        k = float(np.mean((x - mu) ** 4) / sigma ** 4)
        return k - 3 if self.excess else k

    def compute(self, real, synthetic):
        real_flat = flatten_windows(real)
        synth_flat = flatten_windows(synthetic)
        n_features = real_flat.shape[1]
        errors = [abs(self._kurt(real_flat[:, f]) - self._kurt(synth_flat[:, f])) for f in range(n_features)]
        return float(np.mean(errors))


class VaRErrorMetric(BaseMetric):
    name = "var_error"

    def __init__(self, quantiles=(0.95, 0.99), weights=None, portfolio=False):
        self.quantiles = quantiles
        self.weights = weights
        self.portfolio = portfolio

    def _var(self, losses, q):
        return float(np.quantile(losses, q))

    def compute(self, real, synthetic):
        real_flat = flatten_windows(real)
        synth_flat = flatten_windows(synthetic)
        n_features = real_flat.shape[1]

        if self.portfolio:
            w = self.weights if self.weights is not None else np.ones(n_features) / n_features
            r_losses = -(real_flat @ w)
            s_losses = -(synth_flat @ w)
            errors = []
            for q in self.quantiles:
                errors.append(abs(self._var(r_losses, q) - self._var(s_losses, q)))
            return float(np.mean(errors))

        errors = []
        for f in range(n_features):
            r_losses = -real_flat[:, f]
            s_losses = -synth_flat[:, f]
            for q in self.quantiles:
                errors.append(abs(self._var(r_losses, q) - self._var(s_losses, q)))
        return float(np.mean(errors))


class LeverageEffectErrorMetric(BaseMetric):
    name = "leverage_effect_error"

    def __init__(self, max_lag=10):
        self.max_lag = max_lag

    def _leverage(self, x, max_lag):
        n = len(x)
        x_c = x - np.mean(x)
        sq = x ** 2
        sq_c = sq - np.mean(sq)
        out = np.zeros(max_lag)
        std_x = np.std(x)
        std_sq = np.std(sq)
        denom = std_x * std_sq
        for h in range(1, max_lag + 1):
            if n - h <= 0 or denom == 0:
                out[h - 1] = 0.0
                continue
            num = np.mean(x_c[:n - h] * sq_c[h:])
            out[h - 1] = num / denom
        return out

    def compute(self, real, synthetic):
        real, synthetic = _windowed_series(real), _windowed_series(synthetic)
        n_features = real.shape[2]
        errors = []
        for f in range(n_features):
            max_lag = min(self.max_lag, real.shape[1] - 1, synthetic.shape[1] - 1)
            lev_r = np.mean([self._leverage(row, max_lag) for row in real[:, :, f]], axis=0)
            lev_s = np.mean([self._leverage(row, max_lag) for row in synthetic[:, :, f]], axis=0)
            errors.append(np.sum(np.abs(lev_r - lev_s)))
        return float(np.mean(errors))


METRIC_REGISTRY = {
    "predictive_score": PredictiveScoreMetric,
    "discriminative_score": DiscriminativeScoreMetric,
    "raw_return_acf_error": RawReturnACFErrorMetric,
    "volatility_clustering_error": VolatilityClusteringErrorMetric,
    "correlation_error": CorrelationErrorMetric,
    "hill_error": HillErrorMetric,
    "spectral_error": SpectralErrorMetric,
    "skewness_error": SkewnessErrorMetric,
    "kurtosis_error": KurtosisErrorMetric,
    "var_error": VaRErrorMetric,
    "leverage_effect_error": LeverageEffectErrorMetric,
}


def evaluate_all(real, synthetic, metric_kwargs=None, real_test=None):
    metric_kwargs = metric_kwargs or {}
    results = {}
    for name, cls in METRIC_REGISTRY.items():
        kwargs = metric_kwargs.get(name, {})
        metric = cls(**kwargs)
        if name == "predictive_score":
            results[name] = metric.compute(real, synthetic, real_test=real_test)
        else:
            results[name] = metric.compute(real, synthetic)
    return results
