"""Run one explicitly chosen SVM refinement and retain its evidence."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from models import hog_features
from recognize_chip import recognize
from train import fit, read_data, save_json


def run_refinement(data_root, results, name, regularization, cell_size, reason):
    output = results / "iterations" / name
    if (output / "iteration.json").exists():
        raise FileExistsError("Iteration already recorded. Choose a new --name.")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    images, labels, indices, classes = read_data(data_root, results)
    features = torch.from_numpy(hog_features(images["pad"], cell_size=cell_size))
    config = {"mode": "pad", "lr": 0.01, "C": regularization, "epochs": 35,
              "hog_cell_size": cell_size}
    _, selection = fit("svm", features, labels, indices["train"], indices["validation"],
                       config, len(classes), 5411)
    final_config = {**config, "epochs": selection["best_epoch"]}
    development = np.concatenate([indices["train"], indices["validation"]])
    model, refit = fit("svm", features, labels, development, None,
                       final_config, len(classes), 5411, refit=True)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": classes,
                "config": final_config, "method": "svm"}, output / "svm_model.pt")
    shutil.copy2(results / "baseline" / "cnn_model.pt", output / "cnn_model.pt")
    recognize(data_root / "Test_Image_2026.bmp", output, (36, 202, 912, 136))
    chip = json.loads((output / "chip_results.json").read_text())
    record = {"name": name, "reason": reason, "config": final_config,
              "validation_accuracy": selection["validation_accuracy"],
              "history": selection["history"], "search_fit_seconds": selection["fit_seconds"],
              "refit_seconds": refit["fit_seconds"], "test_evaluated": False,
              "chip_predictions": {connectivity: result["models"]["svm"]
                                   for connectivity, result in chip["connectivity"].items()},
              "selection_note": "The chip image participates in refinement feedback. Its score is a development result."}
    save_json(output / "iteration.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("dataset_2026"))
    parser.add_argument("--results", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--C", type=float, default=100)
    parser.add_argument("--cell-size", type=int, choices=(4, 8), default=8)
    parser.add_argument("--reason", required=True)
    arguments = parser.parse_args()
    if arguments.C <= 0:
        parser.error("C must be positive")
    run_refinement(arguments.data, arguments.results, arguments.name, arguments.C,
                   arguments.cell_size, arguments.reason)
