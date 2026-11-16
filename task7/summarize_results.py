"""Create readable Task 7 tables from recorded experimental results."""
import json
from pathlib import Path


output = Path(__file__).with_name("results")
data = json.loads((output / "data_summary.json").read_text())
experiments = json.loads((output / "experiments.json").read_text())
results = json.loads((output / "classification_results.json").read_text())
chip = json.loads((output / "chip_results.json").read_text())
lines = ["# Task 7 experimental results", "",
         f"The grouped dataset split uses {data['total_images']} images, including "
         f"{data['development_images']} development images and {data['split_counts']['test']} held-out test images.",
         "", "## Validation search", "",
         "| Method | Preprocessing | HOG cell | Learning rate | C | Best epoch | Validation accuracy |",
         "|---|---|---:|---:|---:|---:|---:|"]
for record in experiments:
    config = record["config"]
    cell_size = config.get("hog_cell_size", 8) if record["method"] == "svm" else "—"
    lines.append(f"| {record['method'].upper()} | {config['mode']} | {cell_size} | {config['lr']} | "
                 f"{config.get('C', '—')} | {record['best_epoch']} | {record['validation_accuracy']:.2%} |")
lines += ["", "## Held-out evaluation after full-development refit", "",
          "| Method | Test accuracy | Macro F1 | Parameters | Refit seconds | Model-only ms/image |",
          "|---|---:|---:|---:|---:|---:|"]
for method, result in results.items():
    lines.append(f"| {method.upper()} | {result['test_accuracy']:.2%} | {result['macro_f1']:.5f} | "
                 f"{result['parameters']} | {result['refit_seconds']:.2f} | {result['model_ms_per_image']:.5f} |")
lines += ["", "The model-only times are warmed-up full-test batch throughput on a four-thread CPU. "
          "File decoding, normalization and segmentation are excluded. HOG extraction is excluded from SVM model-only timing. "
          f"A separate HOG-plus-model measurement is {results['svm']['hog_and_model_ms_per_image']:.5f} ms/image.",
          "", "## Chip-image transfer", "", "The reference line is `HD44780A00`.", "",
          "| Connectivity | Method | Predicted line | Position accuracy |",
          "|---|---|---|---:|"]
for connectivity, result in chip["connectivity"].items():
    for method, prediction in result["models"].items():
        accuracy = prediction["position_accuracy"]
        score = f"{accuracy:.2%}" if accuracy is not None else "Different segment count"
        lines.append(f"| {connectivity} | {method.upper()} | `{prediction['prediction']}` | {score} |")
lines += ["", "## Comparison", "",
          f"On this held-out split, CNN accuracy is {results['cnn']['test_accuracy']:.2%} "
          f"and HOG–SVM accuracy is {results['svm']['test_accuracy']:.2%}. "
          "The CNN learns local features jointly with the classifier. HOG fixes the gradient representation before "
          "the linear classifier is fitted. Learned features provide a plausible explanation for the observed "
          "differences. The SVM refinement additionally investigates finer spatial gradient bins.", "",
          "The SVM has fewer learned parameters and a fast linear scoring step. Its HOG computation is required "
          "on every input, so model-only timing should be read alongside the HOG-plus-model timing. "
          "CNN inference includes its learned feature extraction. Refit time excludes the earlier parameter "
          "search and feature preparation.", "",
          "The validation table isolates preprocessing by holding learning rate or C fixed. The padded CNN "
          "learning-rate comparison and each SVM C comparison show parameter sensitivity on the same split. "
          "The conclusions are limited to this search range and random seed.", "",
          "## Interpretation limits", "",
          "Model selection uses internal validation. The selected configuration is refitted on all development images "
          "for the validation-selected epoch budget. The chip image supplies feedback during sequential "
          "SVM refinement, so its position accuracy describes a development example.", "",
          "The SVM is a linear one-vs-rest classifier trained with a squared-hinge objective and Adam. "
          "Its HOG features are hand-written. Both methods use PyTorch tensor operations for optimization.", "",
          "The results use one grouped split and one random seed. Duplicate groups remain repeated within their "
          "assigned partition, and related fonts may occur across partitions. One chip image provides a small "
          "transfer example. The clean character-image test accuracy and chip position accuracy measure different conditions.", "",
          "The crop, local threshold settings and area cutoff for chip segmentation follow Task 6. "
          "The classifier checkpoints are selected using validation accuracy on the supplied character dataset.", ""]

baseline_path = output / "baseline" / "classification_results.json"
if baseline_path.exists():
    baseline = json.loads(baseline_path.read_text())["svm"]
    baseline_chip = json.loads((output / "baseline" / "chip_results.json").read_text())
    baseline_text = baseline_chip["connectivity"]["8"]["models"]["svm"]["prediction"]
    comparison = ["# Sequential SVM refinement", "",
                  "Each round was chosen after observing the preceding result. The dataset split and seed remain fixed.", "",
                  "| Round | HOG cell | C | Validation accuracy | Test accuracy | Chip prediction |",
                  "|---|---:|---:|---:|---:|---|",
                  f"| Baseline | 8 | 10 | {baseline['validation_accuracy']:.2%} | "
                  f"{baseline['test_accuracy']:.2%} | `{baseline_text}` |"]
    for name in ("c100", "c100_hog4"):
        record = json.loads((output / "iterations" / name / "iteration.json").read_text())
        config = record["config"]
        test_score = f"{record['test_accuracy']:.2%}" if record["test_evaluated"] else "Not evaluated"
        text = record["chip_predictions"]["8"]["prediction"]
        comparison.append(f"| {name} | {config.get('hog_cell_size', 8)} | {config['C']:g} | "
                          f"{record['validation_accuracy']:.2%} | {test_score} | `{text}` |")
    comparison += ["", "The first round increased C from 10 to 100. Validation accuracy improved, "
                   "while the second chip character retained the prediction 8. This motivated a feature change.", "",
                   "The second round held C at 100 and changed HOG cells from 8x8 to 4x4. "
                   "The descriptor grew from 324 to 1764 entries. This resolved the second character "
                   "and improved validation accuracy. The validation-selected checkpoint was refitted "
                   "for eight epochs on the full development partition, then evaluated on the reserved test set.", "",
                   "The chip image participates in this refinement feedback. Its correct transcription is a development "
                   "result. Each classifier predicts all positions using its learned scores.", "",
                   "The finer descriptor increases learned parameters from 2275 to 12355. "
                   f"HOG-plus-model timing changed from {baseline['hog_and_model_ms_per_image']:.4f} to "
                   f"{results['svm']['hog_and_model_ms_per_image']:.4f} ms per image on this CPU run.", "",
                   "Original weights, predictions, metrics and plots are retained in baseline/. "
                   "Each iteration has its own weights, training history and chip outputs under iterations/.", ""]
    (output / "refinement_comparison.md").write_text("\n".join(comparison))
    lines += ["## Sequential SVM refinement", "",
              "The original run and subsequent trials are recorded in [refinement_comparison.md](refinement_comparison.md). "
              "Baseline artifacts are retained in `baseline/` and each new trial is stored in `iterations/`.", ""]
(output / "task7_results.md").write_text("\n".join(lines))

# A table fragment can be included in a report that loads booktabs.
latex = [r"\section{Task 7 experimental results}",
         f"The supplied dataset contains {data['total_images']} character images. "
         f"The development portion has {data['development_images']} images and the held-out test has "
         f"{data['split_counts']['test']} images. Exact duplicate groups remain within one partition.",
         r"\subsection{Held-out classification}",
         r"\begin{center}\begin{tabular}{lrr}\toprule",
         r"Method & Accuracy & Macro F1 \\\midrule"]
for method, result in results.items():
    latex.append(f"{method.upper()} & {100 * result['test_accuracy']:.2f}\\% & {result['macro_f1']:.5f} " + r"\\")
latex += [r"\bottomrule\end{tabular}\end{center}",
          r"\subsection{Chip-image inference}",
          r"The reference string is \texttt{HD44780A00}.",
          r"\begin{center}\begin{tabular}{lllr}\toprule",
          r"Connectivity & Method & Prediction & Position accuracy \\\midrule"]
for connectivity, result in chip["connectivity"].items():
    for method, prediction in result["models"].items():
        accuracy = prediction["position_accuracy"]
        score = f"{100 * accuracy:.2f}\\%" if accuracy is not None else "Unaligned"
        latex.append(f"{connectivity} & {method.upper()} & " + r"\texttt{" +
                     prediction["prediction"] + "} & " + score + r" \\")
latex += [r"\bottomrule\end{tabular}\end{center}",
          "Model configurations and epoch budgets were chosen using internal validation, followed by "
          "a fixed-budget refit on all development images. These measurements use one grouped split and one seed. "
          "The SVM uses hand-written HOG and a linear squared-hinge objective optimized with Adam. "
          "Both optimizers use PyTorch. The chip image supplies feedback during SVM refinement, "
          "and its transcription score describes this development example.",
          "The experiment tables are generated from the saved JSON records. "
          "The group should develop its report discussion from its own assessment of these results."]
(output / "task7_results.tex").write_text("\n".join(latex) + "\n")
print(output / "task7_results.md")
