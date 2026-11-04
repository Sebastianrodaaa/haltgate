PY ?= python3

repro:            ## rerun eval from cached checkpoints/, rewrite results/*.json, plots and README tables
	$(PY) -m haltgate.run --seeds 0 1 2
	$(PY) scripts/plots.py

train:            ## retrain the three looped backbones from scratch on CPU (~1 h measured on 8 vCPU, 3 procs x 2 threads)
	for s in 0 1 2; do $(PY) -m haltgate.run --seeds $$s --train --only-train --threads 2 & done; wait
	$(MAKE) repro

test:
	$(PY) test_haltgate.py

app:
	$(PY) app/server.py

assets:           ## banner + demo gif (banner needs UnifrakturMaguntia-Book.ttf, OFL, see NOTICE.md)
	$(PY) scripts/banner.py
	$(PY) scripts/demo_gif.py

.PHONY: repro train test app assets
