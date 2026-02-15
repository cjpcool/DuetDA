# DuetDA

Official code for **DuetDA: Decomposed and Dynamic Data Attribution with Model-State Gating for Accelerated Scientific Endeavors**.

This repo supports:
- Building MatBench CGCNN splits (count-based OOD and difficulty-based OOD)
- Meta-training a DuetDA data valuator
- Training CGCNN/SchNet/ALIGNN with **DuetDA-only** data selection

## 1) Environment

Recommended: Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy pandas scikit-learn pymatgen matbench wandb pyarrow
```

## 2) Prepare data splits

### 2.1 Count-based OOD split
```bash
python prepare_cgcnn_count_based_ood.py
```
Output (default):
- `cgcnn_data/matbench_log_kvrh_CountBasedOOD/`

### 2.2 Difficulty-based OOD split

```bash
python prepare_cgcnn_difficulty.py \
	--fit_csv fit_difficulty.csv
```
Output (default):
- `cgcnn_data/matbench_log_kvrh_difficultyOOD1/`

## 3) Meta-train DuetDA valuator

```bash
python modules/meta_train_cgcnn.py \
	--data_root cgcnn_data/matbench_log_kvrh_CountBasedOOD \
	--checkpoint-dir checkpoints/meta_train/cgcnn \
	--backbone cgcnn \
	--fold 1 \
	--num-models 3 \
	--num-outer-steps 50 \
	--truncation-steps 3 \
	--inner-steps 5 \
	--meta-lr 0.001 \
	--batch-size 256
```

Example checkpoint to use later:
- `checkpoints/meta_train/cgcnn/data_attributor_meta_step_*.pt`

## 4) Train with DuetDA curation (quick run)

`main.py` now supports **only** `--da-method duetda`.

```bash
python main.py \
	--data-root cgcnn_data/matbench_log_kvrh_difficultyOOD1 \
	--data-name cgcnn_matbench \
	--da-model-ckpt checkpoints/meta_train/cgcnn/data_attributor_meta_step_50.pt \
	--model-name schnet \
	--task-type regression \
	--da-method duetda \
	--fold 1 \
	--epochs 2 \
	--cuda \
	--optim Adam \
	--batch-size 256 \
	--print-freq 5 \
	--seed 42 \
	--selection-ratio 0.5
```

Optional W&B logging:
```bash
--use-wandb --wandb-entity <entity> --wandb-project <project> --wandb-name <run_name>
```

## 5) Key outputs

- Best checkpoint: `checkpoints/<data>_duetda_<ratio>_model_best.pth.tar`
- Last checkpoint: `checkpoints/<data>_duetda_<ratio>_last.pth.tar`
- Test predictions: `<duetda>_<ratio>_<data>_test_results.csv` (or `--test-res-path`)

## Notes

- Use split `fold` values consistent across data prep, meta-training, and final training.
- If you train on CPU, remove `--cuda`.