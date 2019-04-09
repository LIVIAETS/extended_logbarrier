CC = python3.7
PP = PYTHONPATH="$(PYTHONPATH):."

# CFLAGS = -O
# DEBUG = --debug
EPC = 200
# EPC = 5


G_RGX = (\d+_Case\d+_\d+)_\d+
NET = ResidualUNet
B_DATA = [('img', png_transform, False), ('gt', gt_transform, True)]

TRN = results/prostate/fs results/prostate/partial \
	results/prostate/penalty_partial_size \
	results/prostate/logbarrier_partial_size_sched

GRAPH = results/prostate/val_dice.png results/prostate/tra_dice.png \
		results/prostate/val_loss.png results/prostate/tra_loss.png \
		results/prostate/val_batch_dice.png
HIST =
BOXPLOT = results/prostate/val_batch_dice_boxplot.png results/prostate/val_dice_boxplot.png
PLT = $(GRAPH) $(HIST) $(BOXPLOT)


REPO = $(shell basename `git rev-parse --show-toplevel`)
DATE = $(shell date +"%y%m%d")
HASH = $(shell git rev-parse --short HEAD)
HOSTNAME = $(shell hostname)
PBASE = archives
PACK = $(PBASE)/$(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-prostate.tar.gz

all: pack

plot: $(PLT)

pack: $(PACK) report
$(PACK): $(PLT) $(TRN)
	mkdir -p $(@D)
	# tar -zc -f $@ $^  # Use if pigz is not available
	tar cf - $^ | pigz > $@
	chmod -w $@


# Extraction and slicing
data/PROSTATE/train/gt data/PROSTATE/val/gt: data/PROSTATE
data/PROSTATE: data/promise
	rm -rf $@_tmp
	$(PP) $(CC) $(CFLAGS) preprocess/slice_promise.py --source_dir $< --dest_dir $@_tmp --n_augment=0
	mv $@_tmp $@
data/promise: data/prostate.lineage data/TrainingData_Part1.zip data/TrainingData_Part2.zip data/TrainingData_Part3.zip
	md5sum -c $<
	rm -rf $@_tmp
	unzip -q $(word 2, $^) -d $@_tmp
	unzip -q $(word 3, $^) -d $@_tmp
	unzip -q $(word 4, $^) -d $@_tmp
	mv $@_tmp $@


# Weak labels generation
weaks = data/PROSTATE/train/random data/PROSTATE/val/random
weak: $(weaks)

data/PROSTATE/train/random data/PROSTATE/val/random: OPT = --seed=0 --width=4 --r=0 --strategy=random_strat

$(weaks): data/PROSTATE
	rm -rf $@_tmp
	$(CC) $(CFLAGS) gen_weak.py --selected_class 1 --filling 1 --base_folder=$(@D) --save_subfolder=$(@F)_tmp $(OPT)
	mv $@_tmp $@


data/PROSTATE-Aug/train/gt data/PROSTATE-Aug/val/gt: data/PROSTATE-Aug
data/PROSTATE-Aug: data/PROSTATE $(weaks)
	rm -rf $@ $@_tmp
	$(CC) $(CFLAGS) augment.py --n_aug 4 --root_dir $</train --dest_dir $@_tmp/train
	$(CC) $(CFLAGS) augment.py --n_aug 0 --root_dir $</val --dest_dir $@_tmp/val  # Naming scheme for consistency
	mv $@_tmp $@


results/prostate/fs: OPT = --losses="[('CrossEntropy', {'idc': [0, 1]}, None, None, None, 1)]"
results/prostate/fs: data/PROSTATE-Aug/train/gt data/PROSTATE-Aug/val/gt
results/prostate/fs: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True)]"

results/prostate/partial: OPT = --losses="[('CrossEntropy', {'idc': [1]}, None, None, None, 1)]"
results/prostate/partial: data/PROSTATE-Aug/train/random data/PROSTATE-Aug/val/random
results/prostate/partial:  DATA = --folders="$(B_DATA)+[('random', gt_transform, True)]"

results/prostate/penalty_partial_size: OPT = --losses="[('CrossEntropy', {'idc': [1]}, None, None, None, 1), \
	('NaivePenalty', {'idc': [1]}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_size', 1e-2)]"
results/prostate/penalty_partial_size: data/PROSTATE-Aug/train/random data/PROSTATE-Aug/val/random
results/prostate/penalty_partial_size: DATA = --folders="$(B_DATA)+[('random', gt_transform, True), ('random', gt_transform, True)]"

results/prostate/logbarrier_partial_size_sched: OPT = --losses="[('CrossEntropy', {'idc': [1]}, None, None, None, 1), \
	('LogBarrierLoss', {'idc': [1], 't': 5}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_size', 1e-2)]" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': 'LogBarrierLoss', 'mu': 1.1}"
results/prostate/logbarrier_partial_size_sched: data/PROSTATE-Aug/train/random data/PROSTATE-Aug/val/random
results/prostate/logbarrier_partial_size_sched: DATA = --folders="$(B_DATA)+[('random', gt_transform, True), ('random', gt_transform, True)]"


$(TRN):
	rm -rf $@_tmp
	$(CC) $(CFLAGS) main.py --dataset=$(dir $(<D)) --batch_size=4 --group --schedule --save_train \
		--n_epoch=$(EPC) --workdir=$@_tmp --csv=metrics.csv --n_class=2 --metric_axis=1 \
		--grp_regex="$(G_RGX)" --network=$(NET) $(OPT) $(DATA) $(DEBUG)
	mv $@_tmp $@

# Plotting
results/prostate/val_batch_dice.png results/prostate/val_dice.png results/prostate/tra_dice.png: COLS = 1
results/prostate/val_dice.png results/prostate/val_batch_dice.png: plot.py $(TRN)
results/prostate/tra_dice.png: plot.py $(TRN)

results/prostate/tra_loss.png results/prostate/val_loss.png: COLS = 0 1
results/prostate/tra_loss.png results/prostate/val_loss.png: OPT = --ylim 0 10
results/prostate/tra_loss.png results/prostate/val_loss.png: plot.py results/prostate/penalty_partial_size \
																	results/prostate/logbarrier_partial_size_sched

results/prostate/val_haussdorf.png: COLS = 1
results/prostate/val_haussdorf.png: OPT = --ylim 0 7 --min
results/prostate/val_haussdorf.png: plot.py $(TRN)

results/prostate/val_batch_dice_boxplot.png results/prostate/val_dice_boxplot.png: COLS = 1
results/prostate/val_batch_dice_boxplot.png results/prostate/val_dice_boxplot.png: moustache.py $(TRN)

# Nice titles:
results/prostate/val_batch_dice.png: OPT = --title "Validation dice over time"
results/prostate/tra_dice.png: OPT = --title "Training dice over time"

$(GRAPH) $(HIST) $(BOXPLOT):
	$(eval metric:=$(subst _hist,,$(@F)))
	$(eval metric:=$(subst _boxplot,,$(metric)))
	$(eval metric:=$(subst .png,.npy,$(metric)))
	$(CC) $(CFLAGS) $< --filename $(metric) --folders $(filter-out $<,$^) --columns $(COLS) \
		--savefig=$@ --headless --epc 199 $(OPT) $(DEBUG)

# Viewing
view: $(TRN)
	viewer -n 3 --img_source data/PROSTATE/val/img data/PROSTATE/val/gt $(addsuffix /best_epoch/val, $^) --crop 10 \
		--display_names gt $(notdir $^)

view_epc: $(TRN)
	viewer -n 3 --img_source data/PROSTATE/val/img data/PROSTATE/val/gt $(addsuffix /iter$(ITER)/val, $^) --crop 10 \
		--display_names gt $(notdir $^)

report: $(TRN)
	$(CC) $(CFLAGS) report.py --folders $(TRN) --metrics val_batch_dice val_dice --axises 1