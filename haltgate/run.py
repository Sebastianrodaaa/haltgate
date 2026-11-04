"""Train the looped backbone (or load the cached checkpoint), dump per-loop traces, fit the DM, LTT-calibrate, score every arm.

python3 -m haltgate.run --seeds 0 1 2            # reuse checkpoints/ if present (make repro)
python3 -m haltgate.run --seeds 0 1 2 --train    # retrain from scratch on CPU
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as Fn
from .data import TASKS, batch, make
from .dm import CONTINUE, DM, ESCALATE, HALT, OPTIONS, decide, improves, labels
from .features import SCALARS, loop_features, trace
from .ltt import false_halts, first_stop, ltt
from .model import Looped

CK, RES = "checkpoints", "results"
CFG = dict(d=128, heads=4, layers=2, nmax=8)


def train(seed, steps, bs=128, lr=1e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = Looped(**CFG)
    opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    xe, ye, te = make(200, 90_000 + seed)
    log, t0 = {"loss": [], "eval": []}, time.time()
    for s in range(1, steps + 1):
        x, y = batch(bs, rng)
        loss = sum(Fn.cross_entropy(lg.transpose(1, 2), y) for _, _, lg in m(x)) / m.nmax  # deep supervision: every loop reads out
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(), sched.step()
        if s % 50 == 0:
            log["loss"].append([s, round(loss.item(), 4)])
        if s % 500 == 0 or s == steps:
            _, c = trace(m, xe, ye)
            m.train()
            acc = [[round(c[te == k, i].float().mean().item(), 4) for k in range(3)] for i in range(m.nmax)]
            log["eval"].append({"step": s, "sec": round(time.time() - t0, 1), "acc_loop_task": acc})
            print(f"seed {seed} step {s} {time.time() - t0:.0f}s loss {loss.item():.3f} acc@1 {acc[0]} acc@N {acc[-1]}", flush=True)
    os.makedirs(CK, exist_ok=True)
    torch.save(m.state_dict(), f"{CK}/seed{seed}.pt")
    log.update(seed=seed, steps=steps, batch=bs, lr=lr, cfg=CFG, params=sum(p.numel() for p in m.parameters()), train_sec=round(time.time() - t0, 1))
    json.dump(log, open(f"{RES}/train_seed{seed}.json", "w"))


def load(seed):
    m = Looped(**CFG)
    m.load_state_dict(torch.load(f"{CK}/seed{seed}.pt"))
    return m.eval()


@torch.no_grad()
def gated(model, dm, lam, x, score="dm"):
    """Online early exit: run loops only on items the gate keeps alive. Returns halt loop [M], decision [M], preds [M, T]."""
    e, pad, h = model.embed(x), x == 0, None
    idx, M = torch.arange(len(x)), len(x)
    halt, dec, out, pp = torch.full((M,), model.nmax), torch.full((M,), HALT), torch.zeros_like(x), None
    h = torch.zeros_like(e)
    for i in range(1, model.nmax + 1):
        prev = h
        h, logits = model.step(h, e, pad)
        f, pp = loop_features(prev, h, logits, x, i, model.nmax, pp)
        d = decide(dm.scores(f), lam) if lam is not None else torch.full((len(x),), CONTINUE)
        d = torch.where(torch.tensor(i == model.nmax), torch.where(d == CONTINUE, HALT, d), d)
        stop = d != CONTINUE
        halt[idx[stop]], dec[idx[stop]], out[idx[stop]] = i, d[stop], pp[stop]
        keep = ~stop
        if not keep.any():
            break
        h, e, pad, x, pp, idx = h[keep], e[keep], pad[keep], x[keep], pp[keep], idx[keep]
    return halt, dec, out


def acc_at(correct, halt):
    return correct.gather(1, (halt - 1).unsqueeze(1)).squeeze(1).float()


def fixed_interp(correct, m):
    """Fixed-N at a fractional mean m: run floor(m) or ceil(m) loops at random in the right ratio (expected per-item accuracy)."""
    lo = int(np.floor(m)); hi = min(lo + 1, correct.shape[1]); w = m - lo
    lo = max(lo, 1)
    return (1 - w) * correct[:, lo - 1].float() + w * correct[:, hi - 1].float()


def evaluate(seed, alpha, delta, n, dump):
    model = load(seed)
    N = model.nmax
    sp = {k: make(n, base + seed) for k, base in (("gate_train", 10_000), ("calib", 20_000), ("test", 30_000))}
    tr = {k: trace(model, x, y) for k, (x, y, _) in sp.items()}
    imp = {k: improves(c) for k, (_, c) in tr.items()}
    dm = DM().fit(tr["gate_train"][0], labels(tr["gate_train"][1]))
    P = {k: dm.scores(f) for k, (f, _) in tr.items()}
    stop = {k: 1 - p[..., CONTINUE] for k, p in P.items()}
    ent = {k: -f[..., -len(SCALARS) + SCALARS.index("ent_max")] for k, (f, _) in tr.items()}  # stop score: low entropy -> stop
    C, T, task = tr["test"][1], imp["test"], sp["test"][2]
    S = T.any(1)

    def arm(name, halt, correct=None, extra=None):
        a = acc_at(C, halt) if correct is None else correct
        fh = false_halts(halt, T)[S].float()
        r = {"arm": name, "acc": a.mean().item(), "mean_loops": halt.float().mean().item(), "false_halt_rate": fh.mean().item(),
             "n_hard": int(S.sum()), "acc_task": [a[task == k].mean().item() for k in range(3)],
             "loops_hist": torch.bincount(halt, minlength=N + 1)[1:].tolist()}
        return {**r, **(extra or {})}, a

    arms, per_item = [], {}
    for k in range(1, N + 1):
        r, a = arm(f"fixed-{k}", torch.full((len(C),), k))
        arms.append(r); per_item[r["arm"]] = a
    lam, path = ltt(stop["calib"], imp["calib"], alpha, delta)
    lam_eff = 1.01 if lam is None else lam  # no valid config -> never stop early (run Nmax)
    h_gate = first_stop(stop["test"], lam_eff)
    dec = decide(P["test"], lam_eff).gather(1, (h_gate - 1).unsqueeze(1)).squeeze(1)
    dec[(h_gate == N) & (dec == CONTINUE)] = HALT
    esc = dec == ESCALATE
    cal_mean = first_stop(stop["calib"], lam_eff).float().mean().item()
    r, a = arm("haltgate", h_gate, extra={"lam": lam, "calib_mean_loops": cal_mean, "escalate_rate": esc.float().mean().item(),
                                          "escalate_precision": (~C[esc].any(1)).float().mean().item() if esc.any() else None})
    arms.append(r); per_item["haltgate"] = a
    # entropy early exit tuned on calib to match haltgate's calib mean loops
    taus = torch.quantile(ent["calib"].flatten(), torch.linspace(0, 1, 401))
    means = torch.stack([first_stop(ent["calib"], t).float().mean() for t in taus])
    tau = taus[(means - cal_mean).abs().argmin()].item()
    r, a = arm("entropy-matched", first_stop(ent["test"], tau), extra={"tau": tau})
    arms.append(r); per_item["entropy-matched"] = a
    # same LTT procedure on the raw entropy score (isolates what the DM adds)
    tau_l, path_e = ltt(ent["calib"], imp["calib"], alpha, delta, grid=taus.flip(0))
    r, a = arm("entropy-ltt", first_stop(ent["test"], 1e9 if tau_l is None else tau_l), extra={"tau": tau_l})
    arms.append(r); per_item["entropy-ltt"] = a
    # random halt with haltgate's exact loop distribution (20 shuffles, expected accuracy)
    g = torch.Generator().manual_seed(seed)
    ra = torch.stack([acc_at(C, h_gate[torch.randperm(len(h_gate), generator=g)]) for _ in range(20)]).mean(0)
    hr = h_gate[torch.randperm(len(h_gate), generator=g)]
    r, a = arm("random-matched", hr, correct=ra)
    arms.append(r); per_item["random-matched"] = a
    per_item["fixed-interp"] = fixed_interp(C, r["mean_loops"])

    # Pareto curves on test
    curve = lambda s, grid: [[first_stop(s, t).float().mean().item(), acc_at(C, first_stop(s, t)).mean().item()] for t in grid]
    pareto = {"haltgate": curve(stop["test"], torch.linspace(1, 0, 51).tolist()), "entropy": curve(ent["test"], taus[::8].tolist()),
              "fixed": [[k, per_item[f"fixed-{k}"].mean().item()] for k in range(1, N + 1)]}

    # wall clock: real online early exit vs fixed-N, batch 512
    x_test = sp["test"][0]
    def clock(fn, reps=3):
        best = 1e9
        for _ in range(reps):
            t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
        return best / len(x_test) * 1e3
    def run_fixed(k):
        with torch.no_grad():
            for s in range(0, len(x_test), 512):
                for _ in model(x_test[s:s + 512], k): pass
    def run_gate():
        return [gated(model, dm, lam, x_test[s:s + 512]) for s in range(0, len(x_test), 512)]
    online = run_gate()
    h_on = torch.cat([o[0] for o in online])
    wall = {"ms_per_item": {**{f"fixed-{k}": clock(lambda: run_fixed(k)) for k in (1, 2, 4, N)}, "haltgate": clock(run_gate)},
            "online_offline_halt_agreement": (h_on == h_gate).float().mean().item(), "threads": torch.get_num_threads()}

    res = {"seed": seed, "data": "SYNTH", "scale": "small-scale toy (CPU)", "alpha": alpha, "delta": delta, "nmax": N, "n_per_task": n,
           "cfg": CFG, "features": [f"pool[{model.emb.embedding_dim}]"] + SCALARS, "options": OPTIONS,
           "acc_loop_task": [[C[task == k, i].float().mean().item() for k in range(3)] for i in range(N)],
           "hard_frac": S.float().mean().item(), "hard_frac_task": [S[task == k].float().mean().item() for k in range(3)],
           "label_frac_gate_train": torch.bincount(labels(tr["gate_train"][1]).flatten(), minlength=3).float().div(tr["gate_train"][1].numel()).tolist(),
           "ltt_path": path, "ltt_path_entropy": path_e[:60], "arms": arms, "pareto": pareto, "wall": wall}
    if dump:  # per-item halt timeline + feature scatter for the dashboard (real test items, seed 0)
        idx = torch.randperm(len(C), generator=torch.Generator().manual_seed(0))[:240]
        res["items"] = [{"task": TASKS[task[i]], "correct": C[i].int().tolist(), "p_stop": [round(v, 4) for v in stop["test"][i].tolist()],
                         "halt": int(h_gate[i]), "decision": OPTIONS[dec[i]], "hard": bool(S[i])} for i in idx.tolist()]
        F = tr["test"][0][idx[:200]].reshape(-1, tr["test"][0].shape[-1])
        Z = (F - F.mean(0)) / (F.std(0) + 1e-6)
        U, _, _ = torch.pca_lowrank(Z, q=2, center=True)
        lab = labels(C[idx[:200]]).flatten()
        res["scatter"] = [[round(u, 4), round(v, 4), int(l), int(j % N) + 1] for j, ((u, v), l) in enumerate(zip(U.tolist(), lab.tolist()))]
    json.dump(res, open(f"{RES}/seed{seed}.json", "w"), indent=1)
    return res, {k: v for k, v in per_item.items()}


def boot(d, B=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = torch.stack([d[torch.randint(len(d), (len(d),), generator=g)].mean() for _ in range(B)])
    return [d.mean().item(), torch.quantile(s, 0.025).item(), torch.quantile(s, 0.975).item()]


def summarize(results, items, alpha):
    pool = {k: torch.cat([it[k] for it in items]) for k in items[0]}
    mean = lambda arm, key: float(np.mean([next(a[key] for a in r["arms"] if a["arm"] == arm) for r in results]))
    gate_acc, gate_loops = pool["haltgate"].mean().item(), mean("haltgate", "mean_loops")
    base = {k: pool[k].mean().item() for k in ("fixed-interp", "entropy-matched", "random-matched")}
    better = max(base, key=base.get)
    gaps = {k: boot(pool["haltgate"] - pool[k]) for k in base}
    fhr = [next(a for a in r["arms"] if a["arm"] == "haltgate")["false_halt_rate"] for r in results]
    N = results[0]["nmax"]
    cheaper = [k for k in range(1, N + 1) if k <= gate_loops and pool[f"fixed-{k}"].mean().item() >= gate_acc]
    fail = {"bound_violated": any(f > alpha for f in fhr), "ci_vs_better_includes_0": gaps[better][1] <= 0,
            "fixed_n_matches_at_leq_loops": bool(cheaper)}
    hyp = {"h1_fhr_within_alpha_all_seeds": not fail["bound_violated"], "h2a_gap_vs_fixed_ge_5pts": gaps["fixed-interp"][0] >= 0.05,
           "h2b_gap_vs_entropy_ge_3pts": gaps["entropy-matched"][0] >= 0.03}
    arms = sorted({a["arm"] for a in results[0]["arms"]}, key=[a["arm"] for a in results[0]["arms"]].index)
    table = [{"arm": a, **{k: mean(a, k) for k in ("acc", "mean_loops", "false_halt_rate")},
              "acc_sd": float(np.std([next(x["acc"] for x in r["arms"] if x["arm"] == a) for r in results]))} for a in arms]
    out = {"data": "SYNTH", "scale": "small-scale toy (CPU)", "seeds": [r["seed"] for r in results], "alpha": alpha, "delta": results[0]["delta"],
           "n_test_items_pooled": len(pool["haltgate"]), "table": table, "haltgate_fhr_per_seed": fhr, "lam_per_seed": [next(a for a in r["arms"] if a["arm"] == "haltgate")["lam"] for r in results],
           "gap_vs": gaps, "better_baseline": better, "fixed_n_at_leq_loops_matching": cheaper,
           "failure": fail, "hypothesis": hyp, "verdict": "FAIL" if any(fail.values()) or not all(hyp.values()) else "PASS"}
    json.dump(out, open(f"{RES}/summary.json", "w"), indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--n", type=int, default=1000, help="items per task per split")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--only-train", action="store_true")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    os.makedirs(RES, exist_ok=True)
    for s in a.seeds:
        if a.train or not os.path.exists(f"{CK}/seed{s}.pt"):
            train(s, a.steps)
    if a.only_train:
        return
    out = [evaluate(s, a.alpha, a.delta, a.n, dump=(s == a.seeds[0])) for s in a.seeds]
    summ = summarize([o[0] for o in out], [o[1] for o in out], a.alpha)
    for row in summ["table"]:
        print(f"{row['arm']:>16}  acc {row['acc']:.3f}  loops {row['mean_loops']:.2f}  fhr {row['false_halt_rate']:.3f}")
    print(json.dumps({k: summ[k] for k in ("gap_vs", "failure", "hypothesis", "verdict")}, indent=1))


if __name__ == "__main__":
    main()
