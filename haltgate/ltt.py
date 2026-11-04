"""Learn-then-Test (Angelopoulos et al. 2021) calibration of the stop threshold, binomial p-values, fixed-sequence testing."""
import math
import torch


def first_stop(stop, lam):
    """stop [M, N] scores; halt at the first loop (1-indexed) with stop >= lam, else at Nmax."""
    hit = stop >= lam
    hit[:, -1] = True
    return hit.float().argmax(1) + 1


def false_halts(halt, imp):
    """Per-item 1 if halted at a loop where the item still improves later."""
    return imp.gather(1, (halt - 1).unsqueeze(1)).squeeze(1)


def binom_p(k, n, alpha):
    """P(Binom(n, alpha) <= k): valid p-value for H0: risk > alpha."""
    lp = lambda j: math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * math.log(alpha) + (n - j) * math.log1p(-alpha)
    return min(1.0, sum(math.exp(lp(j)) for j in range(k + 1)))


def ltt(stop, imp, alpha, delta, grid=None):
    """Most aggressive lam with false-halt rate on the hard set S <= alpha w.p. >= 1-delta, or None.

    S = calib items that still improve at some loop. Fixed-sequence testing from the most conservative lam down;
    stop at the first lam whose null is not rejected (so no multiplicity correction is needed).
    """
    S = imp.any(1)
    stop, imp, n = stop[S], imp[S], int(S.sum())
    grid = torch.linspace(1, 0, 101) if grid is None else grid
    best, path = None, []
    for lam in grid.tolist():
        k = int(false_halts(first_stop(stop, lam), imp).sum())
        p = binom_p(k, n, alpha)
        path.append({"lam": lam, "false_halts": k, "n_hard": n, "p": p})
        if p > delta:
            break
        best = lam
    return best, path
