"""Apply Task 7 models to the Task 6 connected-component glyphs."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from models import CharacterCNN, LinearSVM, hog_features
from prepare_data import preprocess_character
from train import predict

from segmentation import connected_components, prepare_character_binary


def recognize(image_path, output, crop):
    torch.set_num_threads(4)
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)
    x, y, width, height = crop
    if min(x, y) < 0 or min(width, height) <= 0 or x + width > image.shape[1] or y + height > image.shape[0]:
        raise ValueError("Crop must be inside the image")
    gray_crop = image[y:y + height, x:x + width]
    _, _, binary = prepare_character_binary(gray_crop, 3, 31, 4, 5)
    checkpoints = {kind: torch.load(output / f"{kind}_model.pt", map_location="cpu", weights_only=True)
                   for kind in ("cnn", "svm")}
    result = {"image": image_path.name, "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
              "crop_xywh": list(crop), "reference": "HD44780A00", "connectivity": {}}
    for connectivity in (4, 8):
        labels, components = connected_components(binary, connectivity, min_area=500)
        if not components:
            raise ValueError("No character components found")
        glyphs = []
        for component in components:
            left, top = component["x"], component["y"]
            right = left + component["width"]
            bottom = top + component["height"]
            mask = labels[top:bottom, left:right] == component["label"]
            # Dataset preprocessing accepts dark foreground on a white canvas.
            glyphs.append(Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)))
        mode_result = {"character_count": len(glyphs), "boxes": components, "models": {}}
        for kind, checkpoint in checkpoints.items():
            classes = checkpoint["classes"]
            mode = checkpoint["config"]["mode"]
            normalized = np.stack([preprocess_character(glyph, mode) for glyph in glyphs])
            features = normalized[:, None] if kind == "cnn" else hog_features(
                normalized, cell_size=checkpoint["config"].get("hog_cell_size", 8))
            model = CharacterCNN(len(classes)) if kind == "cnn" else LinearSVM(len(classes), features.shape[1])
            model.load_state_dict(checkpoint["state_dict"])
            prediction = predict(model, torch.from_numpy(features))
            text = "".join(classes[index] for index in prediction)
            reference = result["reference"]
            mode_result["models"][kind] = {
                "prediction": text, "preprocessing": mode,
                "position_accuracy": sum(a == b for a, b in zip(text, reference)) / len(reference)
                if len(text) == len(reference) else None,
            }
        result["connectivity"][str(connectivity)] = mode_result
        figure, axes = plt.subplots(2, (len(glyphs) + 1) // 2, figsize=(12, 5), squeeze=False)
        for axis in axes.flat:
            axis.axis("off")
        for index, (axis, glyph) in enumerate(zip(axes.flat, glyphs)):
            axis.imshow(glyph, cmap="gray", vmin=0, vmax=255)
            cnn_text = mode_result["models"]["cnn"]["prediction"]
            svm_text = mode_result["models"]["svm"]["prediction"]
            axis.set_title(f"{index + 1}: CNN {cnn_text[index]}, SVM {svm_text[index]}")
        figure.tight_layout()
        figure.savefig(output / f"chip_predictions_{connectivity}.png", dpi=160)
        plt.close(figure)
    (output / "chip_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path(__file__).with_name("dataset_2026") / "Test_Image_2026.bmp")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--crop", type=int, nargs=4, default=(36, 202, 912, 136),
                        metavar=("X", "Y", "WIDTH", "HEIGHT"))
    arguments = parser.parse_args()
    recognize(arguments.image, arguments.output, arguments.crop)
