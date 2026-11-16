# Task 7 — CNN and HOG–SVM character recognition

The implementation audits the supplied images, compares preprocessing and hyperparameters, trains a CNN and HOG–SVM, and recognizes the Task 6 character crops. Recorded experiment tables are in [results/task7_results.md](results/task7_results.md). The generated `results/task7_results.tex` is an includable LaTeX table fragment requiring `booktabs`.

## Standalone folder

The complete `task7/` folder can be copied to another location. It contains the code, the supplied `dataset_2026/` dataset, chip image and saved model weights in `results/`. The dataset, trained checkpoints and recorded predictions are included in the repository. `segmentation.py` bundles the Task 6 functions used for chip recognition.

Create the conda environment once from inside the copied folder.

```bash
conda env create -f environment.yml
conda activate robot-vision-task7
```

For chip recognition using the included trained models, run

```bash
python recognize_chip.py
```

To reproduce the complete experiment, run

```bash
python prepare_data.py
python train.py --epochs 20
python recognize_chip.py
python summarize_results.py
```

The default dataset, chip image and output paths are resolved relative to the Python files. Commands can also be invoked by absolute path from another working directory. `--data`, `--image` and `--output` override the defaults when needed.

The models run on a CPU with four threads and seed 5411. Repeating the commands overwrites the corresponding files in `results/`. Keep both `cnn_model.pt` and `svm_model.pt` in that directory when copying the trained folder. Model weights and prediction arrays are retained locally and excluded from Git through `.gitignore`. If they are absent after a Git clone, the training command recreates them.

## Stage 1 — data preparation

From inside the standalone folder, run

```bash
python prepare_data.py
```

Class labels come from image subfolders. Image files at the dataset root, including the chip bitmap, are excluded. Paths in the manifest are relative to the dataset root.

The split reserves exactly 25% of each class for testing. The remaining 75% is the development portion. Twenty percent of development is used for internal validation, with rounding to an integer. Candidate models will train on the inner training portion. After validation selects parameters, the final model can be refitted on all development images.

Decoded grayscale pixel hashes identify identical images, including dimensions. Each duplicate group stays within one partition. The script rejects identical pixel arrays with conflicting labels. A shuffled subset-sum allocation maintains exact class counts while keeping duplicate groups together. Seed 5411 fixes the allocation.

Preprocessing maps dark glyphs on a light background to bright foreground values in [0, 1]. The output is a 32×32 float32 array. The two alternatives are direct resizing and foreground cropping followed by centered square padding and resizing. Foreground cropping uses intensity above 51 after inversion. Square padding adds a 12% margin, with a minimum of two pixels. Blank inputs produce a zero image in padding mode.

Generated outputs are in `results/`.

- `data_split.csv` records every relative path, class, pixel hash and partition.
- `data_summary.json` records counts, duplicate audit, parameters and runtime versions.
- `preprocessing_examples.png` shows two inner-training examples per class under both preprocessing modes.

Duplicate images remain within their assigned partition and contribute repeated observations there. Near-duplicate images and related font families may still span partitions. The preview uses training examples only.

## Stage 2 — CNN

`models.py` defines a network with three 3×3 convolution stages, ReLU activations, two max-pooling stages, adaptive average pooling, a 64-unit dense layer, dropout 0.25 and seven output logits. All weights start from random initialization. Adam optimizes cross-entropy with batch size 128 and weight decay 0.0001.

Three candidates compare direct resizing at learning rate 0.001, crop-and-pad at 0.001, and crop-and-pad at 0.003. Each runs for up to 20 epochs. Validation accuracy selects the configuration and its earliest best epoch. A new model is then initialized and refitted on all 5,334 development images for that fixed epoch budget.

## Stage 3 — HOG and linear SVM

`models.py` implements unsigned HOG with nine orientation bins, interpolation between orientation bins and overlapping 2×2-cell blocks. The baseline uses 8×8 cells and 324 features. The refined model uses 4×4 cells and 1,764 features. Each block uses L2 normalization, clipping at 0.2 and renormalization. Spatial interpolation between cells is omitted.

The SVM contains seven one-vs-rest linear decision functions. PyTorch optimizes a squared-hinge SVM objective with Adam, using learning rate 0.01 for up to 35 epochs. The objective is half the squared weight norm plus `C` times the mean per-image sum of squared hinge losses. Biases are unregularized. The initial comparison uses `C` values 1 and 10 under both preprocessing modes. Sequential refinement first raises `C` to 100 with 8×8 HOG cells, then retains `C=100` and uses 4×4 cells. Validation chooses the configuration and epoch, followed by a fixed-budget refit on all development images. The default training command includes these configurations.

Both optimizers use PyTorch. The non-CNN classifier consists of hand-written HOG features and linear margin scores.

## Stage 4 — evaluation and chip recognition

The held-out set contains 1,778 images. `classification_results.json` records accuracy, macro F1, per-class precision and recall, confusion matrices, parameter count and timings. `experiments.json` stores each candidate's configuration and training history. Figures include separate confusion matrices and the CNN validation curves.

`recognize_chip.py` calls the bundled Task 6 functions in `segmentation.py`. Its default grayscale crop is `(x=36, y=202, width=912, height=136)`. It uses the same 3×3 smoothing, 31×31 local mean, offset 4, 5×5 median and minimum component area 500. Both 4-connectivity and 8-connectivity are evaluated. Each connected component is rendered on a white canvas before the selected classifier preprocessing is applied.

```bash
python recognize_chip.py --image dataset_2026/Test_Image_2026.bmp --crop 36 202 912 136
```

The chip reference string is used after prediction for position scoring. The chip image also supplies feedback during SVM refinement, so its current accuracy is a development result. Check `chip_results.json` and `chip_predictions_4.png` / `chip_predictions_8.png` for the actual outputs.

Timing uses warmed-up full-test batches and excludes file input, normalization and segmentation. The separate SVM HOG-plus-model timing includes HOG extraction from normalized arrays. Training and inference measurements depend on hardware and the local software environment.

The current run gives CNN test accuracy 99.38% and refined HOG–SVM test accuracy 99.55%. Both classifiers predict `HD44780A00` under both connectivity settings. The baseline HOG–SVM test accuracy was 96.51% with chip prediction `H844780A00`.

## Sequential refinement records

`results/baseline/` retains the original run, including its weights, predictions and figures. `results/iterations/c100/` records the first trial, which improved validation accuracy while retaining the chip error. `results/iterations/c100_hog4/` records the successful second trial. The default `results/svm_model.pt` now contains the selected refined model.

Read `results/refinement_comparison.md` for the recorded progression. The dataset split is identical across rounds. The successful model has 12,355 learned parameters and was refitted for eight epochs. The finer HOG descriptor requires more computation.

To reproduce individual rounds from inside the standalone folder, choose fresh output names.

```bash
python refine_svm.py --name repeat_c100 --C 100 --cell-size 8 \
  --reason "Increase C after inspecting the baseline D versus 8 error"
python refine_svm.py --name repeat_hog4 --C 100 --cell-size 4 \
  --reason "Refine spatial HOG after C100 retained the chip error"
```

The refinement command records validation history and chip inference in its own iteration directory. The selected second trial has a separate held-out evaluation in `results/iterations/c100_hog4/test_results.json`. The default `train.py` command reproduces the complete configuration comparison and final test evaluation.
