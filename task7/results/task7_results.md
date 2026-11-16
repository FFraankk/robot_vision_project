# Task 7 experimental results

The grouped dataset split uses 7112 images, including 5334 development images and 1778 held-out test images.

## Validation search

| Method | Preprocessing | HOG cell | Learning rate | C | Best epoch | Validation accuracy |
|---|---|---:|---:|---:|---:|---:|
| CNN | resize | — | 0.001 | — | 18 | 97.74% |
| CNN | pad | — | 0.001 | — | 19 | 98.21% |
| CNN | pad | — | 0.003 | — | 19 | 98.87% |
| SVM | resize | 8 | 0.01 | 1.0 | 12 | 93.98% |
| SVM | resize | 8 | 0.01 | 10.0 | 35 | 96.05% |
| SVM | pad | 8 | 0.01 | 1.0 | 14 | 94.64% |
| SVM | pad | 8 | 0.01 | 10.0 | 5 | 96.52% |
| SVM | pad | 8 | 0.01 | 100.0 | 7 | 97.93% |
| SVM | pad | 4 | 0.01 | 100.0 | 8 | 99.53% |

## Held-out evaluation after full-development refit

| Method | Test accuracy | Macro F1 | Parameters | Refit seconds | Model-only ms/image |
|---|---:|---:|---:|---:|---:|
| CNN | 99.38% | 0.99382 | 89351 | 22.94 | 0.10678 |
| SVM | 99.55% | 0.99550 | 12355 | 0.23 | 0.00123 |

The model-only times are warmed-up full-test batch throughput on a four-thread CPU. File decoding, normalization and segmentation are excluded. HOG extraction is excluded from SVM model-only timing. A separate HOG-plus-model measurement is 0.55322 ms/image.

## Chip-image transfer

The reference line is `HD44780A00`.

| Connectivity | Method | Predicted line | Position accuracy |
|---|---|---|---:|
| 4 | CNN | `HD44780A00` | 100.00% |
| 4 | SVM | `HD44780A00` | 100.00% |
| 8 | CNN | `HD44780A00` | 100.00% |
| 8 | SVM | `HD44780A00` | 100.00% |

## Comparison

On this held-out split, CNN accuracy is 99.38% and HOG–SVM accuracy is 99.55%. The CNN learns local features jointly with the classifier. HOG fixes the gradient representation before the linear classifier is fitted. Learned features provide a plausible explanation for the observed differences. The SVM refinement additionally investigates finer spatial gradient bins.

The SVM has fewer learned parameters and a fast linear scoring step. Its HOG computation is required on every input, so model-only timing should be read alongside the HOG-plus-model timing. CNN inference includes its learned feature extraction. Refit time excludes the earlier parameter search and feature preparation.

The validation table isolates preprocessing by holding learning rate or C fixed. The padded CNN learning-rate comparison and each SVM C comparison show parameter sensitivity on the same split. The conclusions are limited to this search range and random seed.

## Interpretation limits

Model selection uses internal validation. The selected configuration is refitted on all development images for the validation-selected epoch budget. The chip image supplies feedback during sequential SVM refinement, so its position accuracy describes a development example.

The SVM is a linear one-vs-rest classifier trained with a squared-hinge objective and Adam. Its HOG features are hand-written. Both methods use PyTorch tensor operations for optimization.

The results use one grouped split and one random seed. Duplicate groups remain repeated within their assigned partition, and related fonts may occur across partitions. One chip image provides a small transfer example. The clean character-image test accuracy and chip position accuracy measure different conditions.

The crop, local threshold settings and area cutoff for chip segmentation follow Task 6. The classifier checkpoints are selected using validation accuracy on the supplied character dataset.

## Sequential SVM refinement

The original run and subsequent trials are recorded in [refinement_comparison.md](refinement_comparison.md). Baseline artifacts are retained in `baseline/` and each new trial is stored in `iterations/`.
