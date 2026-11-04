## 2026-11-04 morning: haltgate (LTT-gated typed decision model for looped-transformer continue/halt)
Status: proposed to Reverse Engineer Bot; needs Sebastian's go before any Devin session. Devin model: SWE-2 only. No Cursor cloud agents. No GPU spend; CPU / free compute only.

Signals combined: looped-depth-open-weights (0.40 to 0.50, How to Loop Your Transformer amp), decision-policy-gates (hold 0.45) and dllm-decision-engines (0.70 to 0.75, REAX 1.3 / Prism open System One framing).

Repo name: haltgate
GitHub topics / hashtags: ai-frontier-research, looped-transformers, decision-models, conformal-prediction, early-exit, adaptive-compute, system-one, cpu-inference, typed-decisions, policy-gates

Idea: How to Loop Your Transformer shows that which blocks you loop, how many times, and how state passes between loops changes the size/compute/inference trade-off, but N is usually fixed at train or serve time. Spherical Fixed-Point Reasoners already do geometric adaptive halting. Policy Gates and open System One decision models (Prism, REAX framing) show typed classifiers can be fast allow/deny/route decisions with measurable error rates. haltgate inserts a typed decision model as a LOOP GATE: after each loop i in {1..Nmax}, a small System One model over the fixed option set {continue, halt, escalate} reads a cheap feature vector from the current hidden state (mean-pool, entropy of next-token logits if available, loop index, delta-norm vs prior loop) and decides whether to spend another shared-block pass. Learn-then-Test (Angelopoulos et al. 2021) picks score thresholds so that the false-halt rate on a calib set of items that still improve with more loops stays at most alpha (for example 0.05) with probability 1 - delta. escalate may jump to a deeper non-looped path or a larger N budget when configured. The looped backbone is frozen after a short toy pretrain; only the gate is calibrated. v1 trains a tiny shared-block looped transformer from scratch on CPU (parity / multi-digit addition / copy-with-noise), logs per-loop features and gold continue/halt labels from oracle improvement, fits hashing or linear features + multinomial logistic (or a tiny MLP), then LTT-calibrates.

Why new (closest prior work, checked 2026-11-04 morning on local arXiv dump, arXiv API-style title/abstract search, GitHub search API, HF papers in corpus):
- How to Loop Your Transformer (https://arxiv.org/abs/2610.40325v1): studies fixed loop design and parallel inference; does not LTT-gate continue/halt with a typed DM.
- Spherical Fixed-Point Reasoners (https://arxiv.org/abs/2610.25646v1): adaptive loop halting on the hypersphere; geometric criterion, not a typed System One DM with conformal false-halt bound.
- Truncate or Not? (https://arxiv.org/abs/2610.26030v2): truncated BPTT for training looped transformers, not inference gates.
- One Model, Two Paths (https://arxiv.org/abs/2610.33517v1): balances single-pass vs looped performance; no runtime DM gate.
- Decision Models as Policy Gates (https://arxiv.org/abs/2611.02155v1): gates tool calls, not loop iterations.
- Classic early-exit / CALM / PonderNet / Mixture-of-Depths: adaptive compute on ordinary (non-looped) stacks or token-wise depth; not a typed DM + LTT on shared-block loops.
- prefixgate / skilladmit (our prior projects): LTT gates on tool-call prefixes and skill admission, not loop depth.
GitHub search for looped transformer early-exit / adaptive-depth decision returned unrelated MoE/agent repos; no match for conformal loop-halt gate. HF papers search in corpus found no LTT + looped halt combo.

Hypothesis (falsifiable): on a held-out toy task suite where accuracy is non-decreasing in loop count for a known subset S (oracle labels from running Nmax offline), (1) haltgate keeps empirical false-halt rate on S inside the LTT alpha bound; (2) at matched mean loops used, accuracy is at least 5 points absolute above fixed-N equal to that mean, and at least 3 points above a confidence/entropy early-exit baseline tuned on the same calib split. It fails if the bound is violated, if the CI on the gap vs the better baseline includes 0, or if fixed-Nmax already matches accuracy at equal or lower mean loops.

Experiment (CPU only, this box ~8 vCPU; estimate: toy looped LM train 15 to 25 min wall for ~2 to 5M params on synthetic sequences; feature dump + logistic + LTT grid <2 min; 3 seeds full v1 30 to 60 min wall; no GPU, no ZeroGPU required):
1. Toy looped LM: 2 to 4 shared transformer blocks, loop count Nmax in {4,8,16}, vocab and seq short; tasks: parity, addition, noisy copy (scripts generate data; label SYNTH in data cards).
2. Collect per-loop features and oracle continue/halt (continue if accuracy or margin still improves by eps at a later loop).
3. Gate arms: fixed-N (N in {1,2,4,Nmax}), entropy/confidence early-exit, random halt at matched mean loops, haltgate typed DM + LTT.
4. Report accuracy vs mean loops, false-halt rate, bound violations, wall-clock, 3 seeds, all runs including negatives.
5. Optional: swap the logistic head for a frozen open decision-model-style scoring pattern if a tiny CPU DM stand-in fits; otherwise document skip. No GPU.

Key files (ponytail: fewest files): `haltgate/model.py` (tiny looped transformer), `haltgate/data.py` (SYNTH tasks), `haltgate/features.py`, `haltgate/dm.py` (typed continue/halt/escalate), `haltgate/ltt.py`, `haltgate/run.py`, `scripts/plots.py`, `scripts/demo_gif.py`, `app/server.py`, `app/index.html`, `results/` (real JSON only), `test_haltgate.py` (one file).

How to verify: `python3 test_haltgate.py` on CPU with a 200-example parity stream: scores are a valid simplex, LTT returns no configuration when alpha is below the empirical floor, fixed-N=1 underperforms Nmax on hard parity, and haltgate mean loops is strictly below Nmax while keeping false-halt under alpha on the calib-defined hard set (or document failure). `make repro` reruns from cached checkpoints; README tables regenerate from results/*.json.

UI refs (Pinterest board https://www.pinterest.com/roda0050/ui/ ; local pack /workspace/ui-refs/pinterest/; STYLE.md: dark monochrome pro dashboards, dense panels, graph/embedding viz, one restrained accent):
- pin03 https://i.pinimg.com/736x/fd/58/bb/fd58bb3831ee0e7d0ec898140e6d4c5e.jpg
- pin01 https://i.pinimg.com/736x/22/d9/2a/22d92aa1eefe62e170d8ce4cf905e90d.jpg
- pin06 https://i.pinimg.com/736x/0f/46/11/0f46119a71afbcb006290dda0b85041e.jpg
- pin04 https://i.pinimg.com/736x/0e/15/69/0e1569c60a627ab3389dc5d33aca11d0.jpg
- pin05 https://i.pinimg.com/736x/72/54/c1/7254c1bb1a1c148a89e511e43d29666e.jpg
Local images: /workspace/ui-refs/pinterest/pin01.jpg … pin06.jpg. Dashboard: loops-used histogram, accuracy vs mean loops Pareto, per-item halt timeline, embedding/scatter of hidden-state features colored by continue/halt.

Ponytail principles (paste into Devin brief):
1. YAGNI ladder: skip speculative needs; reuse; stdlib; already-installed deps; one line; only then minimum code.
2. Fewest files; deletion over addition; boring over clever; no unrequested abstractions.
3. Bug fix = root cause (one shared guard beats patches everywhere).
4. No mock data in results; MOCK fixtures only under tests/mock_data/ and labeled.
Full skill: /home/box/agent-data/workflows/ponytail/SKILL.md

License: MIT (third-party model/data terms in NOTICE.md).

README format (mandatory paste from /workspace/readme-ref/README-FORMAT.md):
# README format for every project (Sebastian, 2026-11-02)
Model: https://github.com/ghgoodreau/athanor/blob/main/README.md (copy here: athanor-README.md, banner: athanor-banner.png)
PALETTE: black and white only. Pure black background, white/grey type and art. No sepia, no color accents anywhere (banner, screenshots, diagrams, badges).

## Banner (similar to athanor's)
- 1280x400 PNG at docs/screenshots/banner.png, first line of README: ![name](docs/screenshots/banner.png)
- Centered project name in a large pixel blackletter/gothic bitmap font, lowercase.
- Under it, a 1-2 line lowercase monospace tagline.
- Left and right edges: public-domain engravings (e.g. Dore, Durer) relevant to the theme, dithered to two tones (1-bit/ordered or Floyd-Steinberg), fading into the black. Credit sources in NOTICE.md.

## Structure
1. Banner, then 2-3 plain sentences saying what it is.
2. One large hero screenshot of the real product/UI right away.
3. Short themed "## The X" sections (e.g. The Method, The Results, The Screen), each a few short plain paragraphs followed by one real screenshot or plot. Inline `code` for commands.
4. A 2x2 markdown image grid where it helps (| | | table).
5. ## Install: one sh block (clone, dry run, run), then what it touches and how to undo.
6. ## The Name: a short paragraph on the name's origin.
7. ## Inspirations: short, links INSPIRATIONS.md / related work.
8. ## License: MIT, third-party terms in NOTICE.md.
Voice: calm, literal, short sentences, a little wit, no hype, no emoji.
Keep the hashtags/topics and a single small row of black-and-white shields.io badges (style=flat-square, color black/white) under the tagline paragraph; demo GIF/video from a real run stays required.
- Banner name: haltgate. Themed sections: The Loop, The Gate, The Bound.
