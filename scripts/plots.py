"""Black-and-white plots from results/*.json into docs/screenshots/, and the README results table between markers."""
import json, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = json.load(open("results/summary.json"))
R = [json.load(open(f"results/seed{s}.json")) for s in S["seeds"]]
N = R[0]["nmax"]
plt.rcParams.update({"figure.facecolor": "black", "axes.facecolor": "black", "savefig.facecolor": "black", "text.color": "#ddd",
                     "axes.labelcolor": "#bbb", "axes.edgecolor": "#444", "xtick.color": "#999", "ytick.color": "#999",
                     "font.family": "monospace", "font.size": 10, "axes.grid": True, "grid.color": "#1c1c1c", "legend.frameon": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"docs/screenshots/{name}.png", dpi=110)
    plt.close(fig)


fig, ax = plt.subplots(figsize=(8, 4.2))
for i, r in enumerate(R):
    P, kw = r["pareto"], dict(alpha=1 if i == 0 else 0.35)
    ax.plot(*zip(*P["haltgate"]), color="white", lw=1.8, label="haltgate (λ sweep)" if i == 0 else None, **kw)
    ax.plot(*zip(*P["entropy"]), color="#888", lw=1.2, ls="--", label="entropy exit (τ sweep)" if i == 0 else None, **kw)
    ax.plot(*zip(*P["fixed"]), "o", mfc="black", mec="#ddd", label="fixed-N" if i == 0 else None, **kw)
    g = next(a for a in r["arms"] if a["arm"] == "haltgate")
    ax.plot(g["mean_loops"], g["acc"], "s", color="white", ms=8, label="haltgate @ LTT λ*" if i == 0 else None)
ax.set(xlabel="mean loops used", ylabel="exact-match accuracy", title=f"accuracy vs compute · test split · {len(R)} seeds")
ax.legend(loc="lower right")
save(fig, "pareto")

fig, ax = plt.subplots(figsize=(5, 3.4))
for k, (name, st) in enumerate(zip(["parity", "addition", "copy"], ["-", "--", ":"])):
    for i, r in enumerate(R):
        ax.plot(range(1, N + 1), [a[k] for a in r["acc_loop_task"]], st, color="white", alpha=1 if i == 0 else 0.35, label=name if i == 0 else None)
ax.set(xlabel="loop", ylabel="accuracy", title="per-loop accuracy (oracle Nmax run)", ylim=(0, 1.02))
ax.legend()
save(fig, "byloop")

fig, ax = plt.subplots(figsize=(5, 3.4))
for i, r in enumerate(R):
    p = r["ltt_path"]
    ax.plot([q["lam"] for q in p], [q["p"] for q in p], color="white", alpha=1 if i == 0 else 0.4, label=f"seed {r['seed']}")
ax.axhline(R[0]["delta"], color="#888", ls="--", lw=1)
ax.text(0.98, R[0]["delta"] + 0.03, f"δ = {R[0]['delta']}", ha="right", color="#aaa")
ax.set(xlabel="stop threshold λ (tested 1 → 0)", ylabel="binomial p-value", title=f"learn then test · α = {R[0]['alpha']}", xlim=(1.0, 0))
ax.legend()
save(fig, "ltt")

fig, ax = plt.subplots(figsize=(5, 3.4))
g = [sum(x) for x in zip(*[next(a for a in r["arms"] if a["arm"] == "haltgate")["loops_hist"] for r in R])]
e = [sum(x) for x in zip(*[next(a for a in r["arms"] if a["arm"] == "entropy-matched")["loops_hist"] for r in R])]
ax.bar([i - 0.2 for i in range(1, N + 1)], g, 0.4, color="white", label="haltgate")
ax.bar([i + 0.2 for i in range(1, N + 1)], e, 0.4, color="#666", label="entropy-matched")
ax.set(xlabel="loops used", ylabel="test items (all seeds)", title="where items halt")
ax.legend()
save(fig, "hist")

rows = ["| arm | accuracy | mean loops | false-halt on S |", "|---|---:|---:|---:|"]
for t in S["table"]:
    rows.append(f"| {t['arm']} | {100 * t['acc']:.1f} ± {100 * t['acc_sd']:.1f} | {t['mean_loops']:.2f} | {100 * t['false_halt_rate']:.1f}% |")
gv = S["gap_vs"]
rows += ["", f"Gap of haltgate vs baseline at matched loops, pooled over {S['n_test_items_pooled']} test items (95% bootstrap CI, points):", ""]
rows += [f"- vs {k}: {100 * v[0]:+.1f} [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}]" for k, v in gv.items()]
rows += ["", f"Verdict: **{S['verdict']}** (failure criteria: {', '.join(k for k, v in S['failure'].items() if v) or 'none triggered'}; "
         f"hypothesis parts not met: {', '.join(k for k, v in S['hypothesis'].items() if not v) or 'none'})."]
block = "<!-- results:start -->\n" + "\n".join(rows) + "\n<!-- results:end -->"
for f in ("README.md", "docs/REPORT.md"):
    try:
        t = open(f).read()
        open(f, "w").write(re.sub(r"<!-- results:start -->.*?<!-- results:end -->", lambda _: block, t, flags=re.S))
    except FileNotFoundError:
        pass
print("\n".join(rows))
