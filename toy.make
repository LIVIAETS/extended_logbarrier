CC = python3.7
SHELL = zsh
PP = PYTHONPATH="$(PYTHONPATH):."

# CFLAGS = -O
# DEBUG = --debug

EPC = 50
NET = ENet
B_DATA = [('img', png_transform, False), ('gt', gt_transform, True)]

# TRN = results/toy/fs
TRN = results/toy/penalty_size \
	results/toy/penalty_centroid \
	results/toy/penalty_size_centroid \
	results/toy/logbarrier_size \
	results/toy/logbarrier_centroid \
	results/toy/logbarrier_size_centroid \
	results/toy/logbarrier_size_sched \
	results/toy/logbarrier_centroid_sched \
	results/toy/logbarrier_size_centroid_sched

GRAPH = results/toy/val_dice.png results/toy/tra_dice.png \
	results/toy/val_loss.png results/toy/tra_loss.png
HIST =  results/toy/val_dice_hist.png
BOXPLOT = results/toy/val_dice_boxplot.png
PLT = $(GRAPH) $(HIST) $(BOXPLOT)

REPO = $(shell basename `git rev-parse --show-toplevel`)
DATE = $(shell date +"%y%m%d")
HASH = $(shell git rev-parse --short HEAD)
HOSTNAME = $(shell hostname)
PBASE = archives
PACK = $(PBASE)/$(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-toy.tar.gz

all: pack

plot: $(PLT)

pack: $(PACK) report
$(PACK): $(PLT) $(TRN)
	mkdir -p $(@D)
	tar cf - $^ | pigz > $@
	chmod -w $@
# tar -zc -f $@ $^  # Use if pigz is not available

data/TOY: data/TOY/train/gt data/TOY/val/gt
DTS = data/TOY
data/TOY/train/gt data/TOY/val/gt:
	rm -rf $(DTS)_tmp $(DTS)
	$(PP) $(CC) $(CFLAGS) preprocess/gen_toy.py --dest $(DTS)_tmp -n 1000 100 -r 25 -wh 256 256
	mv $(DTS)_tmp $(DTS)

data/TOY/train/gt data/TOY/val/gt:


# Trainings
results/toy/fs: OPT = --losses="[('CrossEntropy', {'idc': [0, 1]}, None, None, None, 1)]"
results/toy/fs: data/TOY/train/gt data/TOY/val/gt
results/toy/fs: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True)]"

# No labeled pixels
results/toy/penalty_size: OPT = --losses="[('NaivePenalty', {'idc': [1]}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_size', 1e-2)]"
results/toy/penalty_size: data/TOY/train/gt data/TOY/val/gt
results/toy/penalty_size: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True)]"

results/toy/penalty_centroid: OPT = --losses="[('NaivePenalty', {'idc': [1]}, 'PreciseBounds', {'margin': 20, 'mode': 'abs'}, 'soft_centroid', 1e-2)]"
results/toy/penalty_centroid: data/TOY/train/gt data/TOY/val/gt
results/toy/penalty_centroid: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True)]"

results/toy/penalty_size_centroid: OPT = --losses="[('NaivePenalty', {'idc': [1]}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_size', 1e-2), \
	('NaivePenalty', {'idc': [1]}, 'PreciseBounds', {'margin': 20, 'mode': 'abs'}, 'soft_centroid', 1e-2)]"
results/toy/penalty_size_centroid: data/TOY/train/gt data/TOY/val/gt
results/toy/penalty_size_centroid: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True), ('gt', gt_transform, True)]"

results/toy/logbarrier_size: OPT = --losses="[('LogBarrierLoss', {'idc': [1], 't': 5}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_size', 1e-2)]"
results/toy/logbarrier_size: data/TOY/train/gt data/TOY/val/gt
results/toy/logbarrier_size: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True)]"

results/toy/logbarrier_centroid: OPT = --losses="[('LogBarrierLoss', {'idc': [1], 't': 5}, 'PreciseBounds', {'margin': 20, 'mode': 'abs'}, 'soft_centroid', 1e-2)]"
results/toy/logbarrier_centroid: data/TOY/train/gt data/TOY/val/gt
results/toy/logbarrier_centroid: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True)]"

results/toy/logbarrier_size_centroid: OPT = --losses="[('LogBarrierLoss', {'idc': [1], 't': 5}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_size', 1e-2), \
	('LogBarrierLoss', {'idc': [1], 't': 5}, 'PreciseBounds', {'margin': 20, 'mode': 'abs'}, 'soft_centroid', 1e-2)]"
results/toy/logbarrier_size_centroid: data/TOY/train/gt data/TOY/val/gt
results/toy/logbarrier_size_centroid: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True), ('gt', gt_transform, True)]"

results/toy/logbarrier_size_sched: OPT = --losses="[('LogBarrierLoss', {'idc': [1], 't': 1}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_size', 1e-2)]" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': 'LogBarrierLoss', 'mu': 1.1}"
results/toy/logbarrier_size_sched: data/TOY/train/gt data/TOY/val/gt
results/toy/logbarrier_size_sched: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True)]"

results/toy/logbarrier_centroid_sched: OPT = --losses="[('LogBarrierLoss', {'idc': [1], 't': 1}, 'PreciseBounds', {'margin': 20, 'mode': 'abs'}, 'soft_centroid', 1e-2)]" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': 'LogBarrierLoss', 'mu': 1.1}"
results/toy/logbarrier_centroid_sched: data/TOY/train/gt data/TOY/val/gt
results/toy/logbarrier_centroid_sched: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True)]"

results/toy/logbarrier_size_centroid_sched: OPT = --losses="[('LogBarrierLoss', {'idc': [1], 't': 1}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_size', 1e-2), \
	('LogBarrierLoss', {'idc': [1], 't': 1}, 'PreciseBounds', {'margin': 20, 'mode': 'abs'}, 'soft_centroid', 1e-2)]" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': 'LogBarrierLoss', 'mu': 1.1}"
results/toy/logbarrier_size_centroid_sched: data/TOY/train/gt data/TOY/val/gt
results/toy/logbarrier_size_centroid_sched: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True), ('gt', gt_transform, True)]"


# Template
results/toy/%:
	rm -rf $@_tmp
	$(CC) $(CFLAGS) main.py --dataset=$(dir $(<D)) --batch_size=1 --schedule --save_train --temperature 5 \
		--n_epoch=$(EPC) --workdir=$@_tmp --csv=metrics.csv --n_class=2 --metric_axis 1 \
		--grp_regex="$(G_RGX)" --network=$(NET) $(OPT) $(DATA) $(DEBUG)
	mv $@_tmp $@


# Plotting
results/toy/val_batch_dice.png results/toy/val_dice.png results/toy/val_haussdorf.png results/toy/tra_dice.png : COLS = 1
results/toy/val_dice.png results/toy/val_batch_dice.png results/toy/val_haussdorf.png: plot.py $(TRN)
results/toy/tra_dice.png: plot.py $(TRN)

results/toy/tra_loss.png results/toy/val_loss.png: COLS = 0 1
results/toy/tra_loss.png results/toy/val_loss.png: OPT = --ylim -0.1 0.1 --no_mean
results/toy/tra_loss.png results/toy/val_loss.png: plot.py results/toy/penalty_size_centroid results/toy/logbarrier_size_centroid results/toy/logbarrier_size_centroid_sched

results/toy/val_batch_dice_hist.png results/toy/val_dice_hist.png: COLS = 1
results/toy/tra_loss_hist.png: COLS = 0
results/toy/val_dice_hist.png results/toy/tra_loss_hist.png results/toy/val_batch_dice_hist.png: hist.py $(TRN)

results/toy/val_batch_dice_boxplot.png results/toy/val_dice_boxplot.png: COLS = 1
results/toy/val_batch_dice_boxplot.png results/toy/val_dice_boxplot.png: moustache.py $(TRN)

# Nice titles:
results/toy/val_dice.png: OPT = --title "Validation dice over time" --ylabel DSC
results/toy/tra_dice.png: OPT = --title "Training dice over time" --ylabel DSC

results/toy/%.png:
	$(eval metric:=$(subst _hist,,$(@F)))
	$(eval metric:=$(subst _boxplot,,$(metric)))
	$(eval metric:=$(subst .png,.npy,$(metric)))
	$(CC) $(CFLAGS) $< --filename $(metric) --folders $(filter-out $<,$^) --columns $(COLS) \
		--savefig=$@ --headless --epc $(shell echo $$(( $(EPC) - 1 ))) $(OPT) $(DEBUG)

metrics: $(TRN)
	$(CC) $(CFLAGS) metrics.py --num_classes=4 --grp_regex="$(G_RGX)" --gt_folder data/MIDL/val/gt \
		--pred_folders $(addsuffix /best_epoch/val, $^) $(DEBUG)


# Viewing
view: $(TRN)
	viewer -n 3 --img_source data/TOY/val/img data/TOY/val/gt $(addsuffix /best_epoch/val, $^) \
		--display_names gt $(notdir $^) --no_contour

view_train: $(TRN)
	viewer -n 3 --img_source data/TOY/train/img data/TOY/train/gt $(addsuffix /best_epoch/train, $^) \
		--display_names gt $(notdir $^) --no_contour

report: $(TRN)
	$(CC) $(CFLAGS) report.py --folders $(TRN) --metrics val_dice --axises 1