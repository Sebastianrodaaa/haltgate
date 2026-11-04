"""Typed System One decision model over the fixed option set {continue, halt, escalate}."""
import torch

OPTIONS = ("continue", "halt", "escalate")
CONTINUE, HALT, ESCALATE = range(3)


def improves(correct):
    """[M, N] bool: item is wrong at loop i but right at some later loop (halting here would be a false halt)."""
    later = correct.flip(1).cummax(1).values.flip(1)  # right at loop >= i
    later = torch.cat([later[:, 1:], torch.zeros_like(later[:, :1])], 1)  # right at loop > i
    return later & ~correct


def labels(correct):
    """Oracle labels from the offline Nmax run: continue if still improves, escalate if never right within Nmax, else halt."""
    lab = torch.full(correct.shape, HALT)
    lab[improves(correct)] = CONTINUE
    lab[~correct.any(1)] = ESCALATE
    return lab


class DM:
    """Multinomial logistic regression on standardized features. scores() is a point on the 3-simplex."""

    def fit(self, X, Y, l2=1e-3, iters=200):
        X, Y = X.reshape(-1, X.shape[-1]).double(), Y.reshape(-1)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        w = (1.0 / torch.bincount(Y, minlength=3).clamp(min=1).double())[Y]  # class-balanced
        self.W = torch.zeros(Z.shape[1], 3, dtype=torch.double, requires_grad=True)
        self.b = torch.zeros(3, dtype=torch.double, requires_grad=True)
        opt = torch.optim.LBFGS([self.W, self.b], max_iter=iters, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            nll = torch.nn.functional.cross_entropy(Z @ self.W + self.b, Y, reduction="none")
            loss = (w * nll).sum() / w.sum() + l2 * (self.W**2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        self.W, self.b = self.W.detach(), self.b.detach()
        return self

    def scores(self, X):
        Z = (X.double() - self.mu) / self.sd
        return (Z @ self.W + self.b).softmax(-1).float()


def decide(p, lam):
    """Typed decision per item: stop (halt or escalate, whichever is likelier) when P(stop) >= lam, else continue."""
    stop = 1 - p[..., CONTINUE] >= lam
    kind = torch.where(p[..., ESCALATE] > p[..., HALT], ESCALATE, HALT)
    return torch.where(stop, kind, torch.full_like(kind, CONTINUE))
