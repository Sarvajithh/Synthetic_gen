import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import minimize
from scipy.special import gammaln
from statsmodels.tsa.arima.model import ARIMA


def get_device(device=None):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _contiguous_training_series(X, n_features):
    """Accept a raw (time, features) series or recover stride-1 windows safely."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 3:
        if X.shape[0] == 0 or X.shape[2] != n_features:
            raise ValueError("Invalid training windows")
        if X.shape[0] > 1 and not np.allclose(X[:-1, 1:, :], X[1:, :-1, :], equal_nan=False):
            raise ValueError("Statistical models require raw data or exactly overlapping stride-1 windows")
        X = np.concatenate([X[0], X[1:, -1, :]], axis=0)
    if X.ndim != 2 or X.shape[1] != n_features or X.shape[0] < 10:
        raise ValueError("Expected at least 10 contiguous observations with shape (time, features)")
    if not np.all(np.isfinite(X)):
        raise ValueError("Training series contains non-finite values")
    return X


def _validate_neural_training_windows(X, window_size, n_features):
    X = np.asarray(X)
    if X.ndim != 3 or X.shape[0] == 0 or X.shape[1:] != (window_size, n_features):
        raise ValueError(f"Expected non-empty input shaped (n, {window_size}, {n_features})")
    if not np.all(np.isfinite(X)):
        raise ValueError("Training windows contain non-finite values")
    return X


class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, X):
        self.min = X.min(axis=(0, 1), keepdims=True)
        self.max = X.max(axis=(0, 1), keepdims=True)
        return self

    def transform(self, X):
        denom = (self.max - self.min)
        denom = np.where(denom == 0, 1.0, denom)
        return (X - self.min) / denom

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        return X * (self.max - self.min) + self.min

    def state_dict(self):
        return {"min": self.min, "max": self.max}

    def load_state_dict(self, state):
        self.min = state["min"]
        self.max = state["max"]


class RNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, rnn_type="gru"):
        super().__init__()
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.act = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return self.act(out)


class RNNDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, rnn_type="gru"):
        super().__init__()
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out)


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, rnn_type="gru"):
        super().__init__()
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out)


class Supervisor(nn.Module):
    def __init__(self, hidden_dim, num_layers, rnn_type="gru"):
        super().__init__()
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(hidden_dim, hidden_dim, num_layers=max(1, num_layers - 1), batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return self.act(out)


class TimeGAN:
    def __init__(self, window_size, n_features, hidden_dim=24, num_layers=3,
                 rnn_type="gru", gamma=1.0, device=None):
        self.window_size = window_size
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.gamma = gamma
        self.device = get_device(device)
        self.scaler = MinMaxScaler()

        self.embedder = RNNEncoder(n_features, hidden_dim, num_layers, hidden_dim, rnn_type).to(self.device)
        self.recovery = RNNDecoder(hidden_dim, hidden_dim, num_layers, n_features, rnn_type).to(self.device)
        self.generator = RNNEncoder(n_features, hidden_dim, num_layers, hidden_dim, rnn_type).to(self.device)
        self.supervisor = Supervisor(hidden_dim, num_layers, rnn_type).to(self.device)
        self.discriminator = Discriminator(hidden_dim, hidden_dim, num_layers, rnn_type).to(self.device)

        self._fitted = False

    def _random_z(self, batch_size):
        return torch.rand(batch_size, self.window_size, self.n_features, device=self.device)

    def fit(self, X, epochs=1000, batch_size=128, lr=1e-3, verbose=True, log_every=100):
        X = _validate_neural_training_windows(X, self.window_size, self.n_features)
        if len(X) == 0 or batch_size < 1:
            raise ValueError("Training data and batch_size must be non-empty")
        X_scaled = self.scaler.fit_transform(X.astype(np.float32))
        tensor_X = torch.tensor(X_scaled, dtype=torch.float32)
        loader = DataLoader(TensorDataset(tensor_X), batch_size=batch_size, shuffle=True, drop_last=False)

        opt_e = torch.optim.Adam(list(self.embedder.parameters()) + list(self.recovery.parameters()), lr=lr)
        opt_g = torch.optim.Adam(list(self.generator.parameters()) + list(self.supervisor.parameters()), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        opt_s = torch.optim.Adam(list(self.generator.parameters()) + list(self.supervisor.parameters()), lr=lr)
        mse = nn.MSELoss()
        bce = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            for (x_batch,) in loader:
                x_batch = x_batch.to(self.device)
                h = self.embedder(x_batch)
                x_tilde = self.recovery(h)
                e_loss = mse(x_tilde, x_batch)
                opt_e.zero_grad()
                e_loss.backward()
                opt_e.step()

        for epoch in range(epochs):
            for (x_batch,) in loader:
                x_batch = x_batch.to(self.device)
                h = self.embedder(x_batch).detach()
                h_hat_supervise = self.supervisor(h)
                s_loss = mse(h_hat_supervise[:, :-1, :], h[:, 1:, :])
                opt_s.zero_grad()
                s_loss.backward()
                opt_s.step()

        for epoch in range(epochs):
            for (x_batch,) in loader:
                x_batch = x_batch.to(self.device)
                bsz = x_batch.size(0)
                z = self._random_z(bsz)

                h = self.embedder(x_batch)
                x_tilde = self.recovery(h)
                h_hat_supervise = self.supervisor(h)

                e_hat = self.generator(z)
                h_hat = self.supervisor(e_hat)
                x_hat = self.recovery(h_hat)

                y_fake = self.discriminator(h_hat)
                y_fake_e = self.discriminator(e_hat)
                y_real = self.discriminator(h)

                d_loss_real = bce(y_real, torch.ones_like(y_real))
                d_loss_fake = bce(y_fake, torch.zeros_like(y_fake))
                d_loss_fake_e = bce(y_fake_e, torch.zeros_like(y_fake_e))
                d_loss = d_loss_real + d_loss_fake + self.gamma * d_loss_fake_e
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

                z = self._random_z(bsz)
                e_hat = self.generator(z)
                h_hat = self.supervisor(e_hat)
                x_hat = self.recovery(h_hat)
                y_fake = self.discriminator(h_hat)
                y_fake_e = self.discriminator(e_hat)

                g_loss_u = bce(y_fake, torch.ones_like(y_fake))
                g_loss_u_e = bce(y_fake_e, torch.ones_like(y_fake_e))
                h_true = self.embedder(x_batch)
                h_hat_supervise = self.supervisor(h_true)
                g_loss_s = mse(h_hat_supervise[:, :-1, :], h_true[:, 1:, :])
                g_loss_v1 = torch.mean(torch.abs(
                    torch.sqrt(x_hat.var(dim=0) + 1e-6) - torch.sqrt(x_batch.var(dim=0) + 1e-6)))
                g_loss_v2 = torch.mean(torch.abs(x_hat.mean(dim=0) - x_batch.mean(dim=0)))
                g_loss_v = g_loss_v1 + g_loss_v2
                g_loss = g_loss_u + self.gamma * g_loss_u_e + 100 * torch.sqrt(g_loss_s + 1e-8) + 100 * g_loss_v
                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()

                h = self.embedder(x_batch)
                x_tilde = self.recovery(h)
                h_hat_supervise = self.supervisor(h)
                e_loss_t0 = mse(x_tilde, x_batch)
                g_loss_s = mse(h_hat_supervise[:, :-1, :], h[:, 1:, :])
                e_loss = 10 * torch.sqrt(e_loss_t0 + 1e-8) + 0.1 * g_loss_s
                opt_e.zero_grad()
                e_loss.backward()
                opt_e.step()

            if verbose and (epoch + 1) % log_every == 0:
                print(f"[TimeGAN] epoch {epoch + 1}/{epochs} d_loss={d_loss.item():.4f} g_loss={g_loss.item():.4f}")

        self._fitted = True
        return self

    @torch.no_grad()
    def sample(self, n_samples, batch_size=256):
        assert self._fitted, "Model must be fit before sampling"
        self.generator.eval()
        self.supervisor.eval()
        self.recovery.eval()
        outputs = []
        remaining = n_samples
        while remaining > 0:
            bsz = min(batch_size, remaining)
            z = self._random_z(bsz)
            e_hat = self.generator(z)
            h_hat = self.supervisor(e_hat)
            x_hat = self.recovery(h_hat)
            outputs.append(x_hat.cpu().numpy())
            remaining -= bsz
        self.generator.train()
        self.supervisor.train()
        self.recovery.train()
        samples = np.concatenate(outputs, axis=0)
        return self.scaler.inverse_transform(samples)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.embedder.state_dict(), os.path.join(path, "embedder.pt"))
        torch.save(self.recovery.state_dict(), os.path.join(path, "recovery.pt"))
        torch.save(self.generator.state_dict(), os.path.join(path, "generator.pt"))
        torch.save(self.supervisor.state_dict(), os.path.join(path, "supervisor.pt"))
        torch.save(self.discriminator.state_dict(), os.path.join(path, "discriminator.pt"))
        with open(os.path.join(path, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler.state_dict(), f)
        config = dict(window_size=self.window_size, n_features=self.n_features,
                      hidden_dim=self.hidden_dim, num_layers=self.num_layers,
                      rnn_type=self.rnn_type, gamma=self.gamma)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def load(cls, path, device=None):
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        model = cls(**config, device=device)
        model.embedder.load_state_dict(torch.load(os.path.join(path, "embedder.pt"), map_location=model.device))
        model.recovery.load_state_dict(torch.load(os.path.join(path, "recovery.pt"), map_location=model.device))
        model.generator.load_state_dict(torch.load(os.path.join(path, "generator.pt"), map_location=model.device))
        model.supervisor.load_state_dict(torch.load(os.path.join(path, "supervisor.pt"), map_location=model.device))
        model.discriminator.load_state_dict(torch.load(os.path.join(path, "discriminator.pt"), map_location=model.device))
        with open(os.path.join(path, "scaler.pkl"), "rb") as f:
            model.scaler.load_state_dict(pickle.load(f))
        model._fitted = True
        return model


class TimeVAEEncoder(nn.Module):
    def __init__(self, window_size, n_features, latent_dim, hidden_channels=(64, 32, 16)):
        super().__init__()
        layers = []
        in_ch = n_features
        for out_ch in hidden_channels:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU())
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, n_features, window_size)
            flat_dim = self.conv(dummy).flatten(1).shape[1]
        self.flat_dim = flat_dim
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        h = x.transpose(1, 2)
        h = self.conv(h)
        h = h.flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class TimeVAEDecoder(nn.Module):
    def __init__(self, window_size, n_features, latent_dim, hidden_channels=(16, 32, 64),
                 use_trend=True, use_seasonality=True, n_seasons=2):
        super().__init__()
        self.window_size = window_size
        self.n_features = n_features
        self.use_trend = use_trend
        self.use_seasonality = use_seasonality
        self.n_seasons = n_seasons

        self.init_len = max(1, window_size // (2 ** len(hidden_channels)))
        self.fc = nn.Linear(latent_dim, hidden_channels[0] * self.init_len)
        self.hidden_channels = hidden_channels

        deconv_layers = []
        in_ch = hidden_channels[0]
        for out_ch in hidden_channels[1:]:
            deconv_layers.append(nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            deconv_layers.append(nn.ReLU())
            in_ch = out_ch
        deconv_layers.append(nn.ConvTranspose1d(in_ch, n_features, kernel_size=4, stride=2, padding=1))
        self.deconv = nn.Sequential(*deconv_layers)

        if use_trend:
            self.trend_fc = nn.Linear(latent_dim, n_features * 3)
        if use_seasonality:
            self.season_fc = nn.Linear(latent_dim, n_features * n_seasons * 2)

    def _resid(self, z):
        h = self.fc(z)
        h = h.view(z.size(0), self.hidden_channels[0], self.init_len)
        out = self.deconv(h)
        out = out.transpose(1, 2)
        if out.size(1) != self.window_size:
            out = F.interpolate(out.transpose(1, 2), size=self.window_size, mode="linear", align_corners=False).transpose(1, 2)
        return out

    def _trend(self, z, t):
        coeffs = self.trend_fc(z).view(z.size(0), self.n_features, 3)
        t_powers = torch.stack([t ** p for p in range(3)], dim=0)
        trend = torch.einsum("bfp,pt->btf", coeffs, t_powers)
        return trend

    def _seasonality(self, z, t):
        params = self.season_fc(z).view(z.size(0), self.n_features, self.n_seasons, 2)
        out = torch.zeros(z.size(0), t.size(0), self.n_features, device=z.device)
        for s in range(self.n_seasons):
            period = self.window_size / (s + 2)
            amp = params[:, :, s, 0]
            phase = params[:, :, s, 1]
            wave = torch.sin(2 * np.pi * t.unsqueeze(0) / period + phase.unsqueeze(-1))
            out = out + wave.transpose(1, 2) * amp.unsqueeze(1)
        return out

    def forward(self, z):
        t = torch.arange(self.window_size, device=z.device, dtype=torch.float32)
        out = self._resid(z)
        if self.use_trend:
            out = out + self._trend(z, t)
        if self.use_seasonality:
            out = out + self._seasonality(z, t)
        return out


class TimeVAE:
    def __init__(self, window_size, n_features, latent_dim=8, hidden_channels=(64, 32, 16),
                 use_trend=True, use_seasonality=True, n_seasons=2, reconstruction_weight=3.0,
                 device=None):
        self.window_size = window_size
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.use_trend = use_trend
        self.use_seasonality = use_seasonality
        self.n_seasons = n_seasons
        self.reconstruction_weight = reconstruction_weight
        self.device = get_device(device)
        self.scaler = MinMaxScaler()

        self.encoder = TimeVAEEncoder(window_size, n_features, latent_dim, hidden_channels).to(self.device)
        self.decoder = TimeVAEDecoder(window_size, n_features, latent_dim, tuple(reversed(hidden_channels)),
                                       use_trend, use_seasonality, n_seasons).to(self.device)
        self._fitted = False

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def fit(self, X, epochs=500, batch_size=128, lr=1e-3, verbose=True, log_every=50):
        X = _validate_neural_training_windows(X, self.window_size, self.n_features)
        X_scaled = self.scaler.fit_transform(X.astype(np.float32))
        tensor_X = torch.tensor(X_scaled, dtype=torch.float32)
        loader = DataLoader(TensorDataset(tensor_X), batch_size=batch_size, shuffle=True, drop_last=False)
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        opt = torch.optim.Adam(params, lr=lr)

        for epoch in range(epochs):
            total_loss = 0.0
            for (x_batch,) in loader:
                x_batch = x_batch.to(self.device)
                mu, logvar = self.encoder(x_batch)
                z = self._reparameterize(mu, logvar)
                x_hat = self.decoder(z)
                recon_loss = F.mse_loss(x_hat, x_batch, reduction="mean")
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = self.reconstruction_weight * recon_loss + kl_loss
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
            if verbose and (epoch + 1) % log_every == 0:
                print(f"[TimeVAE] epoch {epoch + 1}/{epochs} loss={total_loss / len(loader):.4f}")

        self._fitted = True
        return self

    @torch.no_grad()
    def sample(self, n_samples, batch_size=256):
        assert self._fitted, "Model must be fit before sampling"
        self.decoder.eval()
        outputs = []
        remaining = n_samples
        while remaining > 0:
            bsz = min(batch_size, remaining)
            z = torch.randn(bsz, self.latent_dim, device=self.device)
            x_hat = self.decoder(z)
            outputs.append(x_hat.cpu().numpy())
            remaining -= bsz
        self.decoder.train()
        samples = np.concatenate(outputs, axis=0)
        return self.scaler.inverse_transform(samples)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(path, "encoder.pt"))
        torch.save(self.decoder.state_dict(), os.path.join(path, "decoder.pt"))
        with open(os.path.join(path, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler.state_dict(), f)
        config = dict(window_size=self.window_size, n_features=self.n_features,
                      latent_dim=self.latent_dim, hidden_channels=self.hidden_channels,
                      use_trend=self.use_trend, use_seasonality=self.use_seasonality,
                      n_seasons=self.n_seasons, reconstruction_weight=self.reconstruction_weight)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def load(cls, path, device=None):
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        config["hidden_channels"] = tuple(config["hidden_channels"])
        model = cls(**config, device=device)
        model.encoder.load_state_dict(torch.load(os.path.join(path, "encoder.pt"), map_location=model.device))
        model.decoder.load_state_dict(torch.load(os.path.join(path, "decoder.pt"), map_location=model.device))
        with open(os.path.join(path, "scaler.pkl"), "rb") as f:
            model.scaler.load_state_dict(pickle.load(f))
        model._fitted = True
        return model


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TransformerDenoiser(nn.Module):
    def __init__(self, window_size, n_features, d_model=128, n_heads=4, n_layers=4, dim_feedforward=256):
        super().__init__()
        self.window_size = window_size
        self.n_features = n_features
        self.input_proj = nn.Linear(n_features, d_model)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.pos_emb = nn.Parameter(torch.randn(1, window_size, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                     dim_feedforward=dim_feedforward,
                                                     batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.trend_proj = nn.Linear(d_model, n_features)
        self.season_proj = nn.Linear(d_model, n_features)
        self.output_proj = nn.Linear(d_model, n_features)

    def forward(self, x, t):
        h = self.input_proj(x) + self.pos_emb
        temb = self.time_mlp(t).unsqueeze(1)
        h = h + temb
        h = self.encoder(h)
        trend = self.trend_proj(h)
        season = self.season_proj(h)
        resid = self.output_proj(h)
        return trend + season + resid


class DiffusionTS:
    def __init__(self, window_size, n_features, d_model=128, n_heads=4, n_layers=4,
                 timesteps=500, beta_start=1e-4, beta_end=2e-2, device=None):
        self.window_size = window_size
        self.n_features = n_features
        self.timesteps = timesteps
        self.device = get_device(device)
        self.scaler = MinMaxScaler()

        self.model = TransformerDenoiser(window_size, n_features, d_model, n_heads, n_layers).to(self.device)

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas.to(self.device)
        self.alphas = alphas.to(self.device)
        self.alphas_cumprod = alphas_cumprod.to(self.device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(self.device)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self._fitted = False

    def _q_sample(self, x0, t, noise):
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_ac * x0 + sqrt_om * noise

    def fit(self, X, epochs=500, batch_size=128, lr=1e-4, verbose=True, log_every=50):
        X = _validate_neural_training_windows(X, self.window_size, self.n_features)
        X_scaled = self.scaler.fit_transform(X.astype(np.float32)) * 2 - 1
        tensor_X = torch.tensor(X_scaled, dtype=torch.float32)
        loader = DataLoader(TensorDataset(tensor_X), batch_size=batch_size, shuffle=True, drop_last=False)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0.0
            for (x_batch,) in loader:
                x_batch = x_batch.to(self.device)
                bsz = x_batch.size(0)
                t = torch.randint(0, self.timesteps, (bsz,), device=self.device).long()
                noise = torch.randn_like(x_batch)
                x_t = self._q_sample(x_batch, t, noise)
                x0_pred = self.model(x_t, t)
                loss = F.l1_loss(x0_pred, x_batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
            if verbose and (epoch + 1) % log_every == 0:
                print(f"[DiffusionTS] epoch {epoch + 1}/{epochs} loss={total_loss / len(loader):.4f}")

        self._fitted = True
        return self

    @torch.no_grad()
    def _p_sample(self, x_t, t_index):
        bsz = x_t.size(0)
        t = torch.full((bsz,), t_index, device=self.device, dtype=torch.long)
        x0_pred = self.model(x_t, t)
        # Do not censor tail values at the training range; calibration is measured downstream.

        alpha_t = self.alphas[t_index]
        alpha_cumprod_t = self.alphas_cumprod[t_index]
        beta_t = self.betas[t_index]

        if t_index > 0:
            alpha_cumprod_prev = self.alphas_cumprod[t_index - 1]
        else:
            alpha_cumprod_prev = torch.tensor(1.0, device=self.device)

        coef_x0 = torch.sqrt(alpha_cumprod_prev) * beta_t / (1 - alpha_cumprod_t)
        coef_xt = torch.sqrt(alpha_t) * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t)
        mean = coef_x0 * x0_pred + coef_xt * x_t

        if t_index > 0:
            posterior_var = beta_t * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t)
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(posterior_var) * noise
        return mean

    @torch.no_grad()
    def sample(self, n_samples, batch_size=64):
        assert self._fitted, "Model must be fit before sampling"
        self.model.eval()
        outputs = []
        remaining = n_samples
        while remaining > 0:
            bsz = min(batch_size, remaining)
            x_t = torch.randn(bsz, self.window_size, self.n_features, device=self.device)
            for t_index in reversed(range(self.timesteps)):
                x_t = self._p_sample(x_t, t_index)
            outputs.append(x_t.cpu().numpy())
            remaining -= bsz
        self.model.train()
        samples = np.concatenate(outputs, axis=0)
        samples = (samples + 1) / 2
        return self.scaler.inverse_transform(samples)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "denoiser.pt"))
        with open(os.path.join(path, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler.state_dict(), f)
        config = dict(window_size=self.window_size, n_features=self.n_features,
                      d_model=self.d_model, n_heads=self.n_heads, n_layers=self.n_layers,
                      timesteps=self.timesteps,
                      beta_start=self.betas[0].item(), beta_end=self.betas[-1].item())
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def load(cls, path, device=None):
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        model = cls(**config, device=device)
        model.model.load_state_dict(torch.load(os.path.join(path, "denoiser.pt"), map_location=model.device))
        with open(os.path.join(path, "scaler.pkl"), "rb") as f:
            model.scaler.load_state_dict(pickle.load(f))
        model._fitted = True
        return model


def stable_rvs(alpha, beta, sigma, mu, size, rng):
    if alpha >= 2.0 - 1e-9:
        return rng.normal(loc=mu, scale=sigma * np.sqrt(2.0), size=size)

    U = rng.uniform(-np.pi / 2.0, np.pi / 2.0, size=size)
    W = rng.exponential(1.0, size=size)

    if abs(alpha - 1.0) < 1e-8:
        term = np.pi / 2.0 + beta * U
        X1 = (2.0 / np.pi) * (term * np.tan(U) - beta * np.log((np.pi / 2.0 * W * np.cos(U)) / term))
        return sigma * X1 + (2.0 / np.pi) * beta * sigma * np.log(sigma) + mu

    zeta = beta * np.tan(np.pi * alpha / 2.0)
    B = np.arctan(zeta) / alpha
    S = (1.0 + zeta ** 2) ** (1.0 / (2.0 * alpha))

    num = np.sin(alpha * (U + B))
    den = np.cos(U) ** (1.0 / alpha)
    factor = (np.cos(U - alpha * (U + B)) / W) ** ((1.0 - alpha) / alpha)
    X1 = S * (num / den) * factor
    X0 = X1 - zeta
    return sigma * X0 + mu


def _hill_tail_index(x, k):
    x_sorted = np.sort(x)[::-1]
    k = max(1, min(k, len(x_sorted) - 1))
    top = x_sorted[:k + 1]
    top = top[top > 0]
    if len(top) < 3:
        return None
    k_eff = len(top) - 1
    logs = np.log(top[:k_eff] / top[k_eff])
    gamma = np.mean(logs)
    if gamma <= 1e-8:
        return None
    return 1.0 / gamma


def estimate_stable_params(z, tail_fraction=0.1):
    z = np.asarray(z, dtype=np.float64)
    n = len(z)
    mu_z = float(np.median(z))
    k = max(5, int(n * tail_fraction))

    upper = z[z > mu_z] - mu_z
    lower = mu_z - z[z < mu_z]

    alpha_u = _hill_tail_index(upper, k) if len(upper) > 5 else None
    alpha_l = _hill_tail_index(lower, k) if len(lower) > 5 else None

    candidates = [a for a in (alpha_u, alpha_l) if a is not None]
    if len(candidates) == 0:
        alpha = 2.0
    else:
        alpha = float(np.clip(np.mean(candidates), 0.1, 2.0))

    if alpha_u is not None and alpha_l is not None:
        mass_u = len(upper)
        mass_l = len(lower)
        beta_z = float(np.clip((mass_u - mass_l) / (mass_u + mass_l), -1.0, 1.0))
    else:
        beta_z = 0.0

    q75, q25 = np.percentile(z, [75, 25])
    iqr = q75 - q25
    if alpha >= 1.9:
        sigma_z = float(iqr / (2.0 * 0.6745 * np.sqrt(2.0)))
    else:
        sigma_z = float(max(iqr / 2.0, 1e-6))

    return dict(alpha=alpha, beta_z=beta_z, sigma_z=max(sigma_z, 1e-6), mu_z=mu_z)


class UnivariateStableGARCH:
    def __init__(self, random_state=None):
        self.rng = np.random.default_rng(random_state)
        self.mu = 0.0
        self.omega = 1e-4
        self.alpha_g = 0.05
        self.beta_g = 0.90
        self.alpha = 2.0
        self.beta_z = 0.0
        self.sigma_z = 1.0
        self.mu_z = 0.0
        self._fitted = False

    def _avgarch_sigma(self, abs_eps, omega, alpha_g, beta_g):
        n = len(abs_eps)
        sigma = np.zeros(n)
        sigma[0] = max(np.mean(abs_eps), 1e-6)
        for t in range(1, n):
            sigma[t] = omega + alpha_g * abs_eps[t - 1] + beta_g * sigma[t - 1]
        return np.maximum(sigma, 1e-8)

    def _negloglik_proxy(self, params, abs_eps):
        omega, alpha_g, beta_g = params
        if omega <= 0 or alpha_g < 0 or beta_g < 0 or alpha_g + beta_g >= 0.999:
            return 1e12
        sigma = self._avgarch_sigma(abs_eps, omega, alpha_g, beta_g)
        resid = abs_eps - sigma
        return float(np.sum(resid ** 2))

    def fit(self, returns, tail_fraction=0.1):
        returns = np.asarray(returns, dtype=np.float64).ravel()
        self.mu = float(np.median(returns))
        eps = returns - self.mu
        abs_eps = np.abs(eps)

        unconditional = max(np.mean(abs_eps), 1e-6)
        x0 = np.array([unconditional * 0.1, 0.05, 0.85])
        bounds = [(1e-8, unconditional * 5), (1e-6, 0.999), (1e-6, 0.999)]
        result = minimize(self._negloglik_proxy, x0, args=(abs_eps,), bounds=bounds, method="L-BFGS-B")

        omega, alpha_g, beta_g = result.x
        if alpha_g + beta_g >= 0.999:
            scale = 0.999 / (alpha_g + beta_g)
            alpha_g *= scale
            beta_g *= scale

        self.omega = float(omega)
        self.alpha_g = float(alpha_g)
        self.beta_g = float(beta_g)

        sigma = self._avgarch_sigma(abs_eps, self.omega, self.alpha_g, self.beta_g)
        z = eps / sigma

        stable_params = estimate_stable_params(z, tail_fraction=tail_fraction)
        self.alpha = stable_params["alpha"]
        self.beta_z = stable_params["beta_z"]
        self.sigma_z = stable_params["sigma_z"]
        self.mu_z = stable_params["mu_z"]

        self._fitted = True
        return self

    def simulate(self, n_paths, horizon, burn_in=50):
        assert self._fitted, "UnivariateStableGARCH must be fit before simulate"
        total_len = horizon + burn_in
        z = stable_rvs(self.alpha, self.beta_z, self.sigma_z, self.mu_z,
                        size=(n_paths, total_len), rng=self.rng)

        if self.alpha_g + self.beta_g < 1.0:
            sigma0 = self.omega / (1.0 - self.alpha_g - self.beta_g)
        else:
            sigma0 = self.omega

        sigma = np.zeros((n_paths, total_len))
        eps = np.zeros((n_paths, total_len))
        sigma[:, 0] = sigma0
        eps[:, 0] = sigma[:, 0] * z[:, 0]
        for t in range(1, total_len):
            sigma[:, t] = self.omega + self.alpha_g * np.abs(eps[:, t - 1]) + self.beta_g * sigma[:, t - 1]
            eps[:, t] = sigma[:, t] * z[:, t]

        returns = self.mu + eps[:, burn_in:]
        return returns

    def to_dict(self):
        return dict(mu=self.mu, omega=self.omega, alpha_g=self.alpha_g, beta_g=self.beta_g,
                    alpha=self.alpha, beta_z=self.beta_z, sigma_z=self.sigma_z, mu_z=self.mu_z)

    @classmethod
    def from_dict(cls, d, random_state=None):
        model = cls(random_state=random_state)
        model.mu = d["mu"]
        model.omega = d["omega"]
        model.alpha_g = d["alpha_g"]
        model.beta_g = d["beta_g"]
        model.alpha = d["alpha"]
        model.beta_z = d["beta_z"]
        model.sigma_z = d["sigma_z"]
        model.mu_z = d["mu_z"]
        model._fitted = True
        return model


class StableGARCH:
    def __init__(self, window_size, n_features, tail_fraction=0.1, burn_in=50,
                 random_state=None, device=None):
        self.window_size = window_size
        self.n_features = n_features
        self.tail_fraction = tail_fraction
        self.burn_in = burn_in
        self.random_state = random_state
        self.device = "cpu"
        self.asset_models = [UnivariateStableGARCH(random_state=
                              None if random_state is None else random_state + i)
                              for i in range(n_features)]
        self._fitted = False

    def fit(self, X, verbose=True):
        X = _contiguous_training_series(X, self.n_features)
        for f in range(self.n_features):
            series = X[:, f]
            self.asset_models[f].fit(series, tail_fraction=self.tail_fraction)
            if verbose:
                m = self.asset_models[f]
                print(f"[StableGARCH] asset {f}: alpha={m.alpha:.3f} beta_z={m.beta_z:.3f} "
                      f"alpha_g={m.alpha_g:.3f} beta_g={m.beta_g:.3f}")
        self._fitted = True
        return self

    def sample(self, n_samples):
        assert self._fitted, "Model must be fit before sampling"
        asset_series = []
        for f in range(self.n_features):
            paths = self.asset_models[f].simulate(n_samples, self.window_size, burn_in=self.burn_in)
            asset_series.append(paths)
        samples = np.stack(asset_series, axis=-1)
        return samples

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        config = dict(window_size=self.window_size, n_features=self.n_features,
                      tail_fraction=self.tail_fraction, burn_in=self.burn_in,
                      random_state=self.random_state)
        state = dict(config=config, assets=[m.to_dict() for m in self.asset_models])
        with open(os.path.join(path, "stable_garch.json"), "w") as f:
            json.dump(state, f)

    @classmethod
    def load(cls, path, device=None):
        with open(os.path.join(path, "stable_garch.json")) as f:
            state = json.load(f)
        config = state["config"]
        model = cls(**config, device=device)
        model.asset_models = [UnivariateStableGARCH.from_dict(d, random_state=config["random_state"])
                               for d in state["assets"]]
        model._fitted = True
        return model


def student_t_rvs(df, size, rng):
    if df <= 2.0:
        df = 2.0001
    normal = rng.standard_normal(size=size)
    chi2 = rng.chisquare(df, size=size)
    t = normal / np.sqrt(chi2 / df)
    return t / np.sqrt(df / (df - 2.0))


class UnivariateStudentTGARCH:
    def __init__(self, order=(1, 0, 1), random_state=None):
        self.order = order
        self.rng = np.random.default_rng(random_state)
        self.ar_params = np.array([])
        self.ma_params = np.array([])
        self.const = 0.0
        self.diff_order = order[1]
        self.last_values = np.array([])
        self.omega = 1e-4
        self.alpha_g = 0.05
        self.beta_g = 0.90
        self.df = 8.0
        self._fitted = False

    def _negloglik(self, params, eps):
        omega, alpha_g, beta_g, df = params
        if omega <= 0 or alpha_g < 0 or beta_g < 0 or alpha_g + beta_g >= 0.999 or df <= 2.01:
            return 1e12
        n = len(eps)
        sigma2 = np.zeros(n)
        sigma2[0] = max(np.var(eps), 1e-8)
        for t in range(1, n):
            sigma2[t] = omega + alpha_g * eps[t - 1] ** 2 + beta_g * sigma2[t - 1]
        sigma2 = np.maximum(sigma2, 1e-12)
        z2 = eps ** 2 / sigma2
        ll = (gammaln((df + 1.0) / 2.0) - gammaln(df / 2.0) - 0.5 * np.log(np.pi * (df - 2.0))
              - 0.5 * np.log(sigma2) - ((df + 1.0) / 2.0) * np.log(1.0 + z2 / (df - 2.0)))
        if not np.all(np.isfinite(ll)):
            return 1e12
        return -np.sum(ll)

    def fit(self, returns):
        returns = np.asarray(returns, dtype=np.float64).ravel()
        p, d, q = self.order
        arima_result = ARIMA(returns, order=(p, d, q), trend="c" if d == 0 else "n").fit()
        self.const = float(arima_result.params[0]) if d == 0 else 0.0
        offset = 1 if d == 0 else 0
        self.ar_params = np.array(arima_result.arparams) if p > 0 else np.array([])
        self.ma_params = np.array(arima_result.maparams) if q > 0 else np.array([])
        eps = np.asarray(arima_result.resid, dtype=np.float64)
        eps = eps[np.isfinite(eps)]
        self.last_values = returns[-max(p, 1):] if p > 0 else np.array([])
        self.last_resid = eps[-max(q, 1):] if q > 0 else np.array([])

        unconditional = max(np.var(eps), 1e-6)
        x0 = np.array([unconditional * 0.1, 0.05, 0.85, 8.0])
        bounds = [(1e-10, unconditional * 5), (1e-6, 0.999), (1e-6, 0.999), (2.1, 60.0)]
        result = minimize(self._negloglik, x0, args=(eps,), bounds=bounds, method="L-BFGS-B")

        omega, alpha_g, beta_g, df = result.x
        if alpha_g + beta_g >= 0.999:
            scale = 0.999 / (alpha_g + beta_g)
            alpha_g *= scale
            beta_g *= scale

        self.omega = float(omega)
        self.alpha_g = float(alpha_g)
        self.beta_g = float(beta_g)
        self.df = float(df)
        self._fitted = True
        return self

    def simulate(self, n_paths, horizon, burn_in=50):
        assert self._fitted, "UnivariateStudentTGARCH must be fit before simulate"
        total_len = horizon + burn_in
        z = student_t_rvs(self.df, size=(n_paths, total_len), rng=self.rng)

        if self.alpha_g + self.beta_g < 1.0:
            sigma2_0 = self.omega / (1.0 - self.alpha_g - self.beta_g)
        else:
            sigma2_0 = self.omega

        sigma2 = np.zeros((n_paths, total_len))
        eps = np.zeros((n_paths, total_len))
        sigma2[:, 0] = sigma2_0
        eps[:, 0] = np.sqrt(sigma2[:, 0]) * z[:, 0]
        for t in range(1, total_len):
            sigma2[:, t] = self.omega + self.alpha_g * eps[:, t - 1] ** 2 + self.beta_g * sigma2[:, t - 1]
            eps[:, t] = np.sqrt(sigma2[:, t]) * z[:, t]

        p = len(self.ar_params)
        q = len(self.ma_params)
        y = np.zeros((n_paths, total_len))
        y_hist = np.tile(self.last_values, (n_paths, 1)) if p > 0 else np.zeros((n_paths, max(p, 1)))
        resid_hist = np.tile(self.last_resid, (n_paths, 1)) if q > 0 else np.zeros((n_paths, max(q, 1)))

        for t in range(total_len):
            value = self.const
            for i in range(p):
                if t - i - 1 >= 0:
                    value = value + self.ar_params[i] * y[:, t - i - 1]
                else:
                    value = value + self.ar_params[i] * y_hist[:, y_hist.shape[1] - (i + 1 - t)]
            for j in range(q):
                if t - j - 1 >= 0:
                    value = value + self.ma_params[j] * eps[:, t - j - 1]
                else:
                    value = value + self.ma_params[j] * resid_hist[:, resid_hist.shape[1] - (j + 1 - t)]
            y[:, t] = value + eps[:, t]

        return y[:, burn_in:]

    def to_dict(self):
        return dict(order=list(self.order), ar_params=self.ar_params.tolist(),
                    ma_params=self.ma_params.tolist(), const=self.const,
                    last_values=self.last_values.tolist(), last_resid=self.last_resid.tolist(),
                    omega=self.omega, alpha_g=self.alpha_g, beta_g=self.beta_g, df=self.df)

    @classmethod
    def from_dict(cls, d, random_state=None):
        model = cls(order=tuple(d["order"]), random_state=random_state)
        model.ar_params = np.array(d["ar_params"])
        model.ma_params = np.array(d["ma_params"])
        model.const = d["const"]
        model.last_values = np.array(d["last_values"])
        model.last_resid = np.array(d["last_resid"])
        model.omega = d["omega"]
        model.alpha_g = d["alpha_g"]
        model.beta_g = d["beta_g"]
        model.df = d["df"]
        model._fitted = True
        return model


class StudentTARIMAGARCH:
    def __init__(self, window_size, n_features, order=(1, 0, 1), burn_in=50,
                 random_state=None, device=None):
        self.window_size = window_size
        self.n_features = n_features
        self.order = order
        self.burn_in = burn_in
        self.random_state = random_state
        self.device = "cpu"
        self.asset_models = [UnivariateStudentTGARCH(order=order, random_state=
                              None if random_state is None else random_state + i)
                              for i in range(n_features)]
        self._fitted = False

    def fit(self, X, verbose=True):
        X = _contiguous_training_series(X, self.n_features)
        for f in range(self.n_features):
            series = X[:, f]
            self.asset_models[f].fit(series)
            if verbose:
                m = self.asset_models[f]
                print(f"[StudentTARIMAGARCH] asset {f}: df={m.df:.2f} alpha_g={m.alpha_g:.3f} "
                      f"beta_g={m.beta_g:.3f}")
        self._fitted = True
        return self

    def sample(self, n_samples):
        assert self._fitted, "Model must be fit before sampling"
        asset_series = []
        for f in range(self.n_features):
            paths = self.asset_models[f].simulate(n_samples, self.window_size, burn_in=self.burn_in)
            asset_series.append(paths)
        samples = np.stack(asset_series, axis=-1)
        return samples

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        config = dict(window_size=self.window_size, n_features=self.n_features,
                      order=list(self.order), burn_in=self.burn_in,
                      random_state=self.random_state)
        state = dict(config=config, assets=[m.to_dict() for m in self.asset_models])
        with open(os.path.join(path, "student_t_arima_garch.json"), "w") as f:
            json.dump(state, f)

    @classmethod
    def load(cls, path, device=None):
        with open(os.path.join(path, "student_t_arima_garch.json")) as f:
            state = json.load(f)
        config = dict(state["config"])
        config["order"] = tuple(config["order"])
        model = cls(**config, device=device)
        model.asset_models = [UnivariateStudentTGARCH.from_dict(d, random_state=config["random_state"])
                               for d in state["assets"]]
        model._fitted = True
        return model
