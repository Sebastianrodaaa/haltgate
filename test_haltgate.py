"""python3 test_haltgate.py  (CPU, ~1 min). Uses the cached seed-0 checkpoint and a fresh 200-example parity stream."""
import json
import torch
from haltgate.data import make
from haltgate.dm import CONTINUE, DM, labels
from haltgate.features import trace
from haltgate.ltt import binom_p, false_halts, first_stop, ltt
from haltgate.run import gated, load

torch.manual_seed(0)
MODEL = load(0)
N = MODEL.nmax
X, Y, _ = make(200, 4242, tasks=("parity",))
F, C = trace(MODEL, X, Y)
XG, YG, _ = make(300, 4243)  # mixed gate-train stream
DMODEL = DM().fit(*(lambda f, c: (f, labels(c)))(*trace(MODEL, XG, YG)))
P = DMODEL.scores(F)
STOP, IMP = 1 - P[..., CONTINUE], labels(C) == CONTINUE


def test_labels_on_mock_fixture():
    fx = json.load(open("tests/mock_data/correct_toy.json"))
    assert labels(torch.tensor(fx["correct"]).bool()).tolist() == fx["labels"]


def test_scores_are_a_simplex():
    assert (P >= 0).all() and torch.allclose(P.sum(-1), torch.ones(P.shape[:-1]), atol=1e-5)


def test_ltt_returns_none_below_floor():
    n = int(IMP.any(1).sum())
    alpha = 0.5 * (1 - 0.1 ** (1 / n))  # even zero false halts cannot reach p <= delta here
    assert binom_p(0, n, alpha) > 0.1
    assert ltt(STOP, IMP, alpha, 0.1)[0] is None


def test_fixed1_underperforms_nmax_on_parity():
    assert C[:, 0].float().mean() < C[:, -1].float().mean()


def test_haltgate_saves_loops_and_keeps_bound():
    lam, _ = ltt(STOP, IMP, alpha=0.1, delta=0.1)
    assert lam is not None, "LTT found no valid threshold on this stream"
    halt = first_stop(STOP, lam)
    S = IMP.any(1)
    assert halt.float().mean() < N
    assert false_halts(halt, IMP)[S].float().mean() <= 0.1


def test_online_gate_matches_offline():
    lam, _ = ltt(STOP, IMP, alpha=0.1, delta=0.1)
    halt, _, _ = gated(MODEL, DMODEL, lam, X)
    assert torch.equal(halt, first_stop(STOP, lam))


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn()
            print("ok", k)
