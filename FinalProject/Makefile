PYTHON ?= python

SCENARIOS := vertical horizontal

DATA_FILES := $(foreach s,$(SCENARIOS),data/$(s)_proportional.npz data/$(s)_mbe.npz)
FIG_FILES  := $(foreach s,$(SCENARIOS),figs/$(s)_trajectories.png figs/$(s)_time_series.png figs/$(s)_filter_consistency.png)

all: figures

data: $(DATA_FILES)

$(DATA_FILES): run.py scenario.py
	$(PYTHON) run.py

figures: $(FIG_FILES)

$(foreach s,$(SCENARIOS),figs/$(s)_trajectories.png): plot_trajectories.py $(DATA_FILES)
	$(PYTHON) plot_trajectories.py

$(foreach s,$(SCENARIOS),figs/$(s)_time_series.png): plot_time_series.py $(DATA_FILES)
	$(PYTHON) plot_time_series.py

$(foreach s,$(SCENARIOS),figs/$(s)_filter_consistency.png): plot_filter_consistency.py $(DATA_FILES)
	$(PYTHON) plot_filter_consistency.py

clean:
	rm -rf data figs

.PHONY: all data figures clean
