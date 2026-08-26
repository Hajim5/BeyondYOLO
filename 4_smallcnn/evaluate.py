"""
evaluate.py

Evaluate a trained SmallCNN checkpoint.

Features:
    - Load saved .pth checkpoint
    - Recreate model automatically
    - Load test dataset
    - Accuracy
    - Balanced accuracy
    - Precision
    - Recall
    - F1 score
    - Per-class metrics
    - Confusion matrix
    - Inference speed measurement
    - Save results to JSON
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from torch.utils.data import DataLoader

from torchvision import (
    datasets,
    transforms,
)

from model import SmallCNN


# ============================================================
# DEVICE
# ============================================================

def get_device(
    requested_device="auto"
):

    if requested_device == "auto":

        return torch.device(

            "cuda"

            if torch.cuda.is_available()

            else "cpu"
        )

    return torch.device(
        requested_device
    )


# ============================================================
# CREATE TRANSFORM
# ============================================================

def create_transform(
    image_size,
    input_channels,
    normalization
):

    mean = normalization["mean"]

    std = normalization["std"]

    transform_steps = []

    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    if input_channels == 1:

        transform_steps.append(

            transforms.Grayscale(
                num_output_channels=1
            )
        )

    # --------------------------------------------------------
    # RESIZE
    # --------------------------------------------------------

    transform_steps.append(

        transforms.Resize(
            (
                image_size,
                image_size
            )
        )
    )

    # --------------------------------------------------------
    # TENSOR
    # --------------------------------------------------------

    transform_steps.append(

        transforms.ToTensor()
    )

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    transform_steps.append(

        transforms.Normalize(

            mean=mean,

            std=std
        )
    )

    return transforms.Compose(
        transform_steps
    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(
    checkpoint_path,
    device
):

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(

        checkpoint_path,

        map_location=device
    )

    # --------------------------------------------------------
    # VALIDATE CHECKPOINT
    # --------------------------------------------------------

    required_keys = [

        "model_state_dict",

        "model_config",

        "image_config",

        "class_names"
    ]

    missing_keys = [

        key

        for key

        in required_keys

        if key not in checkpoint
    ]

    if missing_keys:

        raise ValueError(

            "Checkpoint is missing "
            "required information:\n"

            f"{missing_keys}"
        )

    return checkpoint


# ============================================================
# CREATE MODEL
# ============================================================

def create_model_from_checkpoint(
    checkpoint,
    device
):

    model_config = (

        checkpoint[
            "model_config"
        ]
    )

    model = SmallCNN(

        num_classes=

            model_config[
                "num_classes"
            ],

        input_channels=

            model_config[
                "input_channels"
            ],

        dropout=

            model_config[
                "dropout"
            ]
    )

    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(
        device
    )

    model.eval()

    return model


# ============================================================
# LOAD TEST DATASET
# ============================================================

def load_test_dataset(
    dataset_dir,
    transform,
    expected_class_names
):

    dataset_dir = Path(
        dataset_dir
    )

    test_dir = (

        dataset_dir

        / "test"
    )

    if not test_dir.exists():

        raise FileNotFoundError(

            "Test dataset not found:\n"

            f"{test_dir}"
        )

    dataset = datasets.ImageFolder(

        test_dir,

        transform=transform
    )

    # --------------------------------------------------------
    # VERIFY CLASS ORDER
    # --------------------------------------------------------

    if (

        dataset.classes

        !=

        expected_class_names
    ):

        raise ValueError(

            "Dataset classes do not match "
            "the checkpoint.\n\n"

            f"Checkpoint classes:\n"
            f"{expected_class_names}\n\n"

            f"Dataset classes:\n"
            f"{dataset.classes}"
        )

    return dataset


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    loader,
    device,
    class_names
):

    y_true = []

    y_pred = []

    total_images = 0

    total_inference_time = 0.0

    model.eval()

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            # ------------------------------------------------
            # GPU SYNCHRONIZATION
            # ------------------------------------------------

            if device.type == "cuda":

                torch.cuda.synchronize()

            start_time = time.perf_counter()

            logits = model(
                images
            )

            if device.type == "cuda":

                torch.cuda.synchronize()

            end_time = time.perf_counter()

            inference_time = (

                end_time

                -

                start_time
            )

            predictions = torch.argmax(

                logits,

                dim=1
            )

            total_inference_time += (

                inference_time
            )

            total_images += (

                images.size(0)
            )

            y_true.extend(

                labels.cpu()
                .numpy()
                .tolist()
            )

            y_pred.extend(

                predictions.cpu()
                .numpy()
                .tolist()
            )

    # --------------------------------------------------------
    # OVERALL METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(

        y_true,

        y_pred
    )

    balanced_accuracy = (

        balanced_accuracy_score(

            y_true,

            y_pred
        )
    )

    precision, recall, f1, _ = (

        precision_recall_fscore_support(

            y_true,

            y_pred,

            average="weighted",

            zero_division=0
        )
    )

    # --------------------------------------------------------
    # PER-CLASS METRICS
    # --------------------------------------------------------

    (
        class_precision,

        class_recall,

        class_f1,

        class_support

    ) = precision_recall_fscore_support(

        y_true,

        y_pred,

        labels=list(
            range(
                len(class_names)
            )
        ),

        zero_division=0
    )

    per_class_metrics = {}

    for index, class_name in enumerate(
        class_names
    ):

        per_class_metrics[
            class_name
        ] = {

            "precision":

                float(
                    class_precision[
                        index
                    ]
                ),

            "recall":

                float(
                    class_recall[
                        index
                    ]
                ),

            "f1":

                float(
                    class_f1[
                        index
                    ]
                ),

            "support":

                int(
                    class_support[
                        index
                    ]
                )
        }

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    matrix = confusion_matrix(

        y_true,

        y_pred,

        labels=list(
            range(
                len(class_names)
            )
        )
    )

    # --------------------------------------------------------
    # INFERENCE SPEED
    # --------------------------------------------------------

    if total_inference_time > 0:

        images_per_second = (

            total_images

            /

            total_inference_time
        )

        milliseconds_per_image = (

            (
                total_inference_time

                /

                total_images
            )

            * 1000
        )

    else:

        images_per_second = 0.0

        milliseconds_per_image = 0.0

    return {

        "total_images":

            total_images,

        "accuracy":

            float(
                accuracy
            ),

        "balanced_accuracy":

            float(
                balanced_accuracy
            ),

        "precision":

            float(
                precision
            ),

        "recall":

            float(
                recall
            ),

        "f1":

            float(
                f1
            ),

        "per_class_metrics":

            per_class_metrics,

        "confusion_matrix":

            matrix.tolist(),

        "inference_speed": {

            "total_inference_seconds":

                total_inference_time,

            "images_per_second":

                images_per_second,

            "milliseconds_per_image":

                milliseconds_per_image
        }
    }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    results,
    class_names
):

    print()

    print("=" * 70)

    print(
        "EVALUATION RESULTS"
    )

    print("=" * 70)

    print()

    print(
        f"Total images: "
        f"{results['total_images']}"
    )

    print()

    print(
        f"Accuracy: "
        f"{results['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{results['balanced_accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{results['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{results['recall']:.4f}"
    )

    print(
        f"F1 Score: "
        f"{results['f1']:.4f}"
    )

    # --------------------------------------------------------
    # PER CLASS
    # --------------------------------------------------------

    print()

    print("-" * 70)

    print(
        "PER-CLASS METRICS"
    )

    print("-" * 70)

    for class_name in class_names:

        metrics = (

            results[
                "per_class_metrics"
            ][
                class_name
            ]
        )

        print()

        print(
            f"{class_name}"
        )

        print(
            f"  Precision: "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"  Recall: "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"  F1: "
            f"{metrics['f1']:.4f}"
        )

        print(
            f"  Support: "
            f"{metrics['support']}"
        )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    print()

    print("-" * 70)

    print(
        "CONFUSION MATRIX"
    )

    print("-" * 70)

    print()

    print(
        "True → rows"
    )

    print(
        "Predicted → columns"
    )

    print()

    print(
        "Classes:"
    )

    for index, class_name in enumerate(
        class_names
    ):

        print(
            f"  {index}: "
            f"{class_name}"
        )

    print()

    for row in results[
        "confusion_matrix"
    ]:

        print(
            row
        )

    # --------------------------------------------------------
    # INFERENCE SPEED
    # --------------------------------------------------------

    print()

    print("-" * 70)

    print(
        "INFERENCE SPEED"
    )

    print("-" * 70)

    print()

    speed = results[
        "inference_speed"
    ]

    print(
        f"Images per second: "
        f"{speed['images_per_second']:.2f}"
    )

    print(
        f"Milliseconds per image: "
        f"{speed['milliseconds_per_image']:.4f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results,
    output_path
):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(

        parents=True,

        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(

            results,

            file,

            indent=4
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=(

            "Evaluate a trained "
            "SmallCNN checkpoint."
        )
    )

    parser.add_argument(

        "--model",

        required=True,

        type=str,

        help=(

            "Path to the "
            ".pth checkpoint."
        )
    )

    parser.add_argument(

        "--dataset",

        required=True,

        type=str,

        help=(

            "Dataset root containing "
            "the test folder."
        )
    )

    parser.add_argument(

        "--batch-size",

        type=int,

        default=32,

        help=(

            "Evaluation batch size."
        )
    )

    parser.add_argument(

        "--num-workers",

        type=int,

        default=0,

        help=(

            "Number of DataLoader "
            "workers."
        )
    )

    parser.add_argument(

        "--device",

        type=str,

        default="auto",

        choices=[

            "auto",

            "cpu",

            "cuda"
        ],

        help=(

            "Device to use."
        )
    )

    parser.add_argument(

        "--output",

        type=str,

        default="evaluation_results.json",

        help=(

            "Path for evaluation "
            "results."
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = get_device(

        args.device
    )

    print()

    print("=" * 70)

    print(
        "SMALLCNN EVALUATION"
    )

    print("=" * 70)

    print()

    print(
        f"Device: "
        f"{device}"
    )

    print(
        f"Model: "
        f"{args.model}"
    )

    print(
        f"Dataset: "
        f"{args.dataset}"
    )

    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    print()

    print(
        "Loading checkpoint..."
    )

    checkpoint = load_checkpoint(

        args.model,

        device
    )

    class_names = (

        checkpoint[
            "class_names"
        ]
    )

    model_config = (

        checkpoint[
            "model_config"
        ]
    )

    image_config = (

        checkpoint[
            "image_config"
        ]
    )

    print()

    print(
        "Classes:"
    )

    for index, class_name in enumerate(
        class_names
    ):

        print(
            f"  {index}: "
            f"{class_name}"
        )

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    print()

    print(
        "Creating model..."
    )

    model = (

        create_model_from_checkpoint(

            checkpoint,

            device
        )
    )

    # --------------------------------------------------------
    # TRANSFORM
    # --------------------------------------------------------

    transform = create_transform(

        image_size=

            image_config[
                "image_size"
            ],

        input_channels=

            model_config[
                "input_channels"
            ],

        normalization=

            image_config[
                "normalization"
            ]
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print()

    print(
        "Loading test dataset..."
    )

    test_dataset = (

        load_test_dataset(

            args.dataset,

            transform,

            class_names
        )
    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=
            args.batch_size,

        shuffle=False,

        num_workers=
            args.num_workers,

        pin_memory=
            torch.cuda.is_available()
    )

    print(
        f"Test images: "
        f"{len(test_dataset)}"
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    print()

    print(
        "Running evaluation..."
    )

    results = evaluate_model(

        model,

        test_loader,

        device,

        class_names
    )

    # --------------------------------------------------------
    # ADD METADATA
    # --------------------------------------------------------

    results["model_path"] = (

        str(
            Path(
                args.model
            ).resolve()
        )
    )

    results["dataset_path"] = (

        str(
            Path(
                args.dataset
            ).resolve()
        )
    )

    results["class_names"] = (

        class_names
    )

    results["model_config"] = (

        model_config
    )

    results["image_config"] = (

        image_config
    )

    results["device"] = (

        str(device)
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print_results(

        results,

        class_names
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_results(

        results,

        args.output
    )

    print()

    print("=" * 70)

    print(
        "EVALUATION COMPLETE"
    )

    print("=" * 70)

    print()

    print(
        f"Results saved to:\n"
        f"{args.output}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
