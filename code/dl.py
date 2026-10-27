import copy

import numpy as np
import torch
from torch import nn

torch.set_num_threads(2)


class _MLP(nn.Module):
    def __init__(self, n_in, hidden, dropout):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, s=None):
        return self.net(x).squeeze(-1)


class _RNN(nn.Module):
    def __init__(self, n_dyn, n_static, hidden, cell, dropout):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(n_dyn, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x, s):
        out, _ = self.rnn(x)
        return self.head(torch.cat([out[:, -1, :], s], dim=1)).squeeze(-1)


def _r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


class TorchRegressor:
    def __init__(self, kind="mlp", hidden=(64, 32), dropout=0.1, lr=1e-3, weight_decay=1e-4,
                 batch_size=256, max_epochs=300, patience=20, seed=0, curves=False):
        self.kind, self.hidden, self.dropout, self.lr = kind, hidden, dropout, lr
        self.weight_decay, self.batch_size = weight_decay, batch_size
        self.max_epochs, self.patience, self.seed = max_epochs, patience, seed
        self.curves = curves

    def _build(self, n_dyn, n_static):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if self.kind == "mlp":
            return _MLP(n_dyn, self.hidden, self.dropout)
        return _RNN(n_dyn, n_static, self.hidden if isinstance(self.hidden, int) else self.hidden[0],
                    self.kind, self.dropout)

    def _scale_fit(self, X, S, y):
        ax = (0, 1) if X.ndim == 3 else 0
        self.xm, self.xs = X.mean(axis=ax), X.std(axis=ax) + 1e-8
        if S is not None:
            self.sm, self.ss = S.mean(0), S.std(0) + 1e-8
        self.ym, self.ys = y.mean(), y.std() + 1e-8

    def _tx(self, X, S):
        Xt = torch.tensor((X - self.xm) / self.xs, dtype=torch.float32)
        St = None if S is None else torch.tensor((S - self.sm) / self.ss, dtype=torch.float32)
        return Xt, St

    def _run(self, Xtr, Str, ytr, epochs, Xva=None, Sva=None, yva=None, track=True):
        model = self._build(Xtr.shape[-1], 0 if Str is None else Str.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lossf = nn.MSELoss()
        Xt, St = self._tx(Xtr, Str)
        yt = torch.tensor((ytr - self.ym) / self.ys, dtype=torch.float32)
        if Xva is not None:
            Xv, Sv = self._tx(Xva, Sva)
        g = torch.Generator().manual_seed(self.seed)
        hist, best, best_state, best_ep, wait = [], np.inf, None, 0, 0
        n = len(yt)
        for ep in range(1, epochs + 1):
            model.train()
            perm = torch.randperm(n, generator=g)
            batch_losses = []
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                out = model(Xt[idx], None if St is None else St[idx])
                loss = lossf(out, yt[idx])
                loss.backward()
                opt.step()
                batch_losses.append(loss.item() * len(idx))
            if not track:
                continue
            model.eval()
            with torch.no_grad():
                rec = {"epoch": ep, "train_loss": float(np.sum(batch_losses) / n)}
                if self.curves:
                    ptr = model(Xt, St).numpy() * self.ys + self.ym
                    rec.update(train_loss=float(np.mean(((ptr - ytr) / self.ys) ** 2)), train_r2=_r2(ytr, ptr))
                if Xva is not None:
                    pva = model(Xv, Sv).numpy() * self.ys + self.ym
                    vl = float(np.mean(((pva - yva) / self.ys) ** 2))
                    rec.update(val_loss=vl, val_r2=_r2(yva, pva))
                    if vl < best - 1e-6:
                        best, best_state, best_ep, wait = vl, copy.deepcopy(model.state_dict()), ep, 0
                    else:
                        wait += 1
            hist.append(rec)
            if Xva is not None and wait >= self.patience:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, hist, best_ep

    def fit(self, X, y, S=None, val_mask=None, refit=True):
        y = np.asarray(y, float)
        if val_mask is None:
            self._scale_fit(X, S, y)
            self.model_, self.history_, _ = self._run(X, S, y, self.max_epochs, track=False)
            self.best_epoch_ = self.max_epochs
            return self
        tr = ~val_mask
        self._scale_fit(X[tr], None if S is None else S[tr], y[tr])
        m, hist, best_ep = self._run(X[tr], None if S is None else S[tr], y[tr], self.max_epochs,
                                     X[val_mask], None if S is None else S[val_mask], y[val_mask])
        self.history_, self.best_epoch_ = hist, max(best_ep, 1)
        if refit:
            self._scale_fit(X, S, y)
            self.model_, _, _ = self._run(X, S, y, self.best_epoch_, track=False)
        else:
            self.model_ = m
        return self

    def predict(self, X, S=None):
        self.model_.eval()
        Xt, St = self._tx(X, S)
        with torch.no_grad():
            return self.model_(Xt, St).numpy() * self.ys + self.ym
