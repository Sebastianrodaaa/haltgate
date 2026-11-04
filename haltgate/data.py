"""SYNTH toy tasks: parity, multi-digit addition, copy-with-noise. All data is generated, never downloaded."""
import numpy as np
import torch

PAD, NOISE, PLUS, EQ, BAR, SLOT, PAR, ADD, CPY = 0, 11, 12, 13, 14, 15, 16, 17, 18
V, T = 19, 48
TASKS = ("parity", "addition", "copy")
D = lambda k: k + 1  # digit k -> token


def parity(rng, lo=2, hi=16, W=16):
    L = rng.integers(lo, hi + 1)  # L random bits, left-padded with zeros to fixed width W
    b = np.r_[np.zeros(W - L, int), rng.integers(0, 2, L)]
    return [PAR, *map(D, b.tolist()), EQ, SLOT], [D(int(b.sum()) % 2)]


def addition(rng, lo=1, hi=10, W=10):
    a, b = (int(rng.integers(0, 10 ** rng.integers(lo, hi + 1))) for _ in "ab")  # zero-padded to W digits
    rev = lambda v, w: [int(c) for c in str(v).zfill(w)[::-1]]  # least-significant digit first
    return [ADD, *map(D, rev(a, W)), PLUS, *map(D, rev(b, W)), EQ] + [SLOT] * (W + 1), list(map(D, rev(a + b, W + 1)))


def copy(rng, lo=3, hi=8, p=0.3, k=3):
    L = rng.integers(lo, hi + 1)
    x = rng.integers(0, 10, L).tolist()
    keep = rng.random((k, L)) > p  # k independently corrupted copies
    keep[rng.integers(0, k, L), np.arange(L)] = True  # every position survives in at least one copy
    seq = [CPY]
    for j in range(k):
        seq += [D(t) if keep[j, i] else NOISE for i, t in enumerate(x)] + [BAR]
    return seq[:-1] + [EQ] + [SLOT] * L, list(map(D, x))


GEN = {"parity": parity, "addition": addition, "copy": copy}


def make(n, seed, tasks=TASKS):
    """n items per task -> (x [M,T] long, y [M,T] long with -100 off-slot, task id [M])."""
    rng = np.random.default_rng(seed)
    xs, ys, ts = [], [], []
    for ti, t in enumerate(tasks):
        for _ in range(n):
            s, a = GEN[t](rng)
            xs.append(s), ys.append(a), ts.append(TASKS.index(t))
    return pack(xs, ys) + (torch.tensor(ts),)


def pack(xs, ys):
    x = torch.zeros(len(xs), T, dtype=torch.long)
    y = torch.full((len(xs), T), -100, dtype=torch.long)
    for i, (s, a) in enumerate(zip(xs, ys)):
        x[i, : len(s)] = torch.tensor(s)
        y[i, len(s) - len(a) : len(s)] = torch.tensor(a)
    return x, y


def batch(bs, rng):
    xs, ys = zip(*(GEN[TASKS[rng.integers(3)]](rng) for _ in range(bs)))
    return pack(xs, ys)
