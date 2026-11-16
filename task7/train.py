"""Train and compare CNN and HOG-SVM using the fixed data_split.csv."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from models import CharacterCNN, LinearSVM, hog_features, squared_hinge_loss
from prepare_data import preprocess_character


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def read_data(root, output):
    with (output / "data_split.csv").open() as stream:
        records = list(csv.DictReader(stream))
    classes = sorted({record["label"] for record in records})
    labels = torch.tensor([classes.index(record["label"]) for record in records])
    indices = {split: np.array([index for index, record in enumerate(records)
                               if record["split"] == split])
               for split in ("train", "validation", "test")}
    images = {mode: [] for mode in ("resize", "pad")}
    for record in records:
        with Image.open(root / record["path"]) as source:
            gray = source.convert("L")
            content = f"L:{gray.width}x{gray.height}:".encode() + gray.tobytes()
            if hashlib.sha256(content).hexdigest() != record["pixel_sha256"]:
                raise ValueError(f"Image changed after splitting: {record['path']}")
            for mode in images:
                images[mode].append(preprocess_character(gray, mode))
    return {mode: np.asarray(array) for mode, array in images.items()}, labels, indices, classes


def predict(model, features):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in features.split(256):
            predictions.extend(model(batch).argmax(1).tolist())
    return np.asarray(predictions)


def fit(kind, features, labels, train_indices, validation_indices, config,
        class_count, seed, refit=False):
    torch.manual_seed(seed)
    model = CharacterCNN(class_count) if kind == "cnn" else LinearSVM(class_count, features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                 weight_decay=1e-4 if kind == "cnn" else 0)
    train_features, train_labels = features[train_indices], labels[train_indices]
    best_accuracy = -1
    best_epoch = config["epochs"]
    best_state = None
    history = []
    start = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        loss_sum = 0
        for batch in torch.randperm(len(train_indices)).split(128):
            optimizer.zero_grad()
            scores = model(train_features[batch])
            if kind == "cnn":
                loss = torch.nn.functional.cross_entropy(scores, train_labels[batch])
            else:
                loss = squared_hinge_loss(scores, train_labels[batch], model, config["C"])
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
        entry = {"epoch": epoch, "loss": loss_sum / len(train_indices)}
        if not refit:
            prediction = predict(model, features[validation_indices])
            accuracy = float(np.mean(prediction == labels[validation_indices].numpy()))
            entry["validation_accuracy"] = accuracy
            if accuracy > best_accuracy:
                best_accuracy, best_epoch = accuracy, epoch
                best_state = copy.deepcopy(model.state_dict())
        history.append(entry)
        if epoch == 1 or epoch % 5 == 0:
            print(kind, config, entry, flush=True)
    if not refit:
        model.load_state_dict(best_state)
    return model, {"config": config, "best_epoch": best_epoch,
                   "validation_accuracy": best_accuracy if not refit else None,
                   "fit_seconds": time.perf_counter() - start, "history": history}


def evaluate(truth, prediction, classes, path):
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    np.add.at(matrix, (truth, prediction), 1)
    precision = np.diag(matrix) / np.maximum(matrix.sum(0), 1)
    recall = np.diag(matrix) / np.maximum(matrix.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(classes)), classes)
    axis.set_yticks(range(len(classes)), classes)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    for row in range(len(classes)):
        for column in range(len(classes)):
            value = matrix[row, column]
            axis.text(column, row, str(value), ha="center", va="center",
                      color="white" if value > matrix.max() / 2 else "black")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return {"test_accuracy": float(np.mean(truth == prediction)), "macro_f1": float(f1.mean()),
            "confusion_matrix": matrix.tolist(),
            "per_class": {label: {"precision": float(p), "recall": float(r), "f1": float(f)}
                          for label, p, r, f in zip(classes, precision, recall, f1)}}


def train(root, output, epochs, seed):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    images, labels, indices, classes = read_data(root, output)
    development = np.concatenate([indices["train"], indices["validation"]])
    experiments, results = [], {}
    for kind in ("cnn", "svm"):
        configurations = ([{"mode": mode, "lr": lr, "epochs": epochs}
                           for mode, lr in [("resize", .001), ("pad", .001), ("pad", .003)]]
                          if kind == "cnn" else
                          [{"mode": mode, "lr": .01, "C": c, "epochs": 35}
                           for mode in ("resize", "pad") for c in (1., 10.)] +
                          [{"mode": "pad", "lr": .01, "C": 100., "epochs": 35,
                            "hog_cell_size": cell_size} for cell_size in (8, 4)])
        feature_start = time.perf_counter()
        arrays = {}
        for config in configurations:
            key = (config["mode"], config.get("hog_cell_size", 8))
            if key not in arrays:
                array = images[config["mode"]]
                features = array[:, None] if kind == "cnn" else hog_features(array, cell_size=key[1])
                arrays[key] = torch.from_numpy(features)
        feature_seconds = time.perf_counter() - feature_start
        candidates = []
        for config in configurations:
            key = (config["mode"], config.get("hog_cell_size", 8))
            model, record = fit(kind, arrays[key], labels, indices["train"],
                                indices["validation"], config, len(classes), seed)
            record["method"] = kind
            candidates.append(record)
            experiments.append(record)
            save_json(output / "experiments.json", experiments)
        selected = max(candidates, key=lambda item: item["validation_accuracy"])
        config = {**selected["config"], "epochs": selected["best_epoch"]}
        features = arrays[(config["mode"], config.get("hog_cell_size", 8))]
        model, refit = fit(kind, features, labels, development, None,
                           config, len(classes), seed, refit=True)
        test_features = features[indices["test"]]
        prediction = predict(model, test_features)
        truth = labels[indices["test"]].numpy()
        result = evaluate(truth, prediction, classes, output / f"{kind}_confusion.png")
        times = []
        for repeat in range(5):
            start = time.perf_counter()
            predict(model, test_features)
            times.append(time.perf_counter() - start)
        model_path = output / f"{kind}_model.pt"
        torch.save({"state_dict": model.state_dict(), "classes": classes, "config": config,
                    "method": kind}, model_path)
        result.update(config=config, validation_accuracy=selected["validation_accuracy"],
                      refit_seconds=refit["fit_seconds"], feature_extraction_seconds=feature_seconds,
                      parameters=sum(parameter.numel() for parameter in model.parameters()),
                      search_fit_seconds=sum(record["fit_seconds"] for record in candidates),
                      model_ms_per_image=float(np.median(times) * 1000 / len(truth)),
                      model_bytes=model_path.stat().st_size)
        if kind == "svm":
            start = time.perf_counter()
            predict(model, torch.from_numpy(hog_features(
                images[config["mode"]][indices["test"]], cell_size=config.get("hog_cell_size", 8))))
            result["hog_and_model_ms_per_image"] = (time.perf_counter() - start) * 1000 / len(truth)
        results[kind] = result
        np.savez(output / f"{kind}_predictions.npz", truth=truth, prediction=prediction, indices=indices["test"])
        save_json(output / "classification_results.json", results)
    figure, axis = plt.subplots(figsize=(8, 4))
    for record in experiments:
        if record["method"] == "cnn":
            axis.plot([entry["epoch"] for entry in record["history"]],
                      [entry["validation_accuracy"] for entry in record["history"]],
                      label=f"{record['config']['mode']}, lr={record['config']['lr']}")
    axis.set(xlabel="Epoch", ylabel="Validation accuracy")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "cnn_learning_curves.png", dpi=160)
    plt.close(figure)
    save_json(output / "training_parameters.json", {"seed": seed, "maximum_cnn_epochs": epochs,
              "device": "cpu", "threads": 4, "batch_size": 128, "torch": torch.__version__,
              "classes": classes, "test_images": len(indices["test"]),
              "final_training_images": len(development)})
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("dataset_2026"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=5411)
    arguments = parser.parse_args()
    if arguments.epochs < 1:
        parser.error("epochs must be positive")
    train(arguments.data, arguments.output, arguments.epochs, arguments.seed)
