"""
train.py

General training pipeline for SmallCNN.

Features:
    - Train / validation / test datasets
    - Roboflow-exported ImageFolder datasets
    - Grayscale or RGB input
    - Configurable image size
    - Configurable number of classes
    - Automatic CPU / CUDA selection
    - Adam optimizer
    - CrossEntropyLoss
    - Best validation checkpoint saving
    - Training history CSV
    - Final test evaluation
    - Accuracy
    - Balanced accuracy
    - Precision
    - Recall
    - F1 score
    - Confusion matrix
    - Inference speed measurement
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from model import SmallCNN


# ============================================================
# DEFAULT SETTINGS
# ============================================================

DEFAULT_CONFIG = {
    "dataset_dir": "dataset",

    "output_dir": "outputs",

    "image_size": 64,

    "input_channels": 1,

    "dropout": 0.20,

    "batch_size": 32,

    "epochs": 30,

    "learning_rate": 0.001,

    "weight_decay": 0.0,

    "num_workers": 0,

    "seed": 42,

    "device": "auto",
}


# ============================================================
# LOAD CONFIGURATION
# ============================================================

def load_config(
    config_path
):

    config = DEFAULT_CONFIG.copy()

    if config_path is None:

        return config

    config_path = Path(
        config_path
    )

    if not config_path.exists():

        raise FileNotFoundError(
            f"Config file not found:\n"
            f"{config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as file:

        user_config = json.load(
            file
        )

    if not isinstance(
        user_config,
        dict
    ):

        raise ValueError(
            "Config file must contain "
            "a JSON object."
        )

    config.update(
        user_config
    )

    return config


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(
    seed
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# DEVICE
# ============================================================

def get_device(
    requested_device
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
# IMAGE TRANSFORMS
# ============================================================

def create_transforms(
    image_size,
    input_channels
):

    if input_channels == 1:

        transform = transforms.Compose([

            transforms.Grayscale(
                num_output_channels=1
            ),

            transforms.Resize(
                (
                    image_size,
                    image_size
                )
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            ),
        ])

    elif input_channels == 3:

        transform = transforms.Compose([

            transforms.Resize(
                (
                    image_size,
                    image_size
                )
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],

                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            ),
        ])

    else:

        raise ValueError(

            "input_channels must be "
            "1 or 3."
        )

    return transform


# ============================================================
# LOAD DATASETS
# ============================================================

def load_datasets(
    dataset_dir,
    transform
):

    dataset_dir = Path(
        dataset_dir
    )

    train_dir = (
        dataset_dir
        / "train"
    )

    valid_dir = (
        dataset_dir
        / "valid"
    )

    test_dir = (
        dataset_dir
        / "test"
    )

    for directory in [
        train_dir,
        valid_dir,
        test_dir
    ]:

        if not directory.exists():

            raise FileNotFoundError(

                "Dataset directory not found:\n"
                f"{directory}"
            )

    train_dataset = datasets.ImageFolder(

        train_dir,

        transform=transform
    )

    valid_dataset = datasets.ImageFolder(

        valid_dir,

        transform=transform
    )

    test_dataset = datasets.ImageFolder(

        test_dir,

        transform=transform
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # All dataset splits must use the same class ordering.
    # --------------------------------------------------------

    if (
        train_dataset.classes
        !=
        valid_dataset.classes
    ):

        raise ValueError(
            "Train and validation "
            "classes do not match."
        )

    if (
        train_dataset.classes
        !=
        test_dataset.classes
    ):

        raise ValueError(
            "Train and test "
            "classes do not match."
        )

    return (

        train_dataset,

        valid_dataset,

        test_dataset
    )


# ============================================================
# CREATE DATALOADERS
# ============================================================

def create_dataloaders(
    train_dataset,
    valid_dataset,
    test_dataset,
    batch_size,
    num_workers
):

    pin_memory = (
        torch.cuda.is_available()
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=num_workers,

        pin_memory=pin_memory
    )

    valid_loader = DataLoader(

        valid_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=pin_memory
    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=pin_memory
    )

    return (

        train_loader,

        valid_loader,

        test_loader
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):

    model.train()

    total_loss = 0.0

    total_samples = 0

    correct_predictions = 0

    for images, labels in loader:

        images = images.to(
            device
        )

        labels = labels.to(
            device
        )

        optimizer.zero_grad()

        logits = model(
            images
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        batch_size = (
            images.size(0)
        )

        total_loss += (
            loss.item()
            *
            batch_size
        )

        total_samples += (
            batch_size
        )

        predictions = torch.argmax(

            logits,

            dim=1
        )

        correct_predictions += (

            predictions
            ==
            labels

        ).sum().item()

    average_loss = (

        total_loss
        /
        total_samples

        if total_samples > 0

        else 0.0
    )

    accuracy = (

        correct_predictions
        /
        total_samples

        if total_samples > 0

        else 0.0
    )

    return {

        "loss":
            average_loss,

        "accuracy":
            accuracy
    }


# ============================================================
# VALIDATE / EVALUATE
# ============================================================

def evaluate_model(
    model,
    loader,
    criterion,
    device,
    class_names
):

    model.eval()

    total_loss = 0.0

    total_samples = 0

    y_true = []

    y_pred = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            logits = model(
                images
            )

            loss = criterion(
                logits,
                labels
            )

            predictions = torch.argmax(

                logits,

                dim=1
            )

            batch_size = (
                images.size(0)
            )

            total_loss += (

                loss.item()
                *
                batch_size
            )

            total_samples += (
                batch_size
            )

            y_true.extend(

                labels.cpu().numpy().tolist()
            )

            y_pred.extend(

                predictions.cpu()
                .numpy()
                .tolist()
            )

    average_loss = (

        total_loss
        /
        total_samples

        if total_samples > 0

        else 0.0
    )

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

    matrix = confusion_matrix(

        y_true,

        y_pred,

        labels=list(
            range(
                len(class_names)
            )
        )
    )

    return {

        "loss":
            average_loss,

        "accuracy":
            float(accuracy),

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

        "confusion_matrix":
            matrix.tolist(),

        "y_true":
            y_true,

        "y_pred":
            y_pred
    }


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    output_path,
    model,
    config,
    class_names,
    epoch,
    metrics
):

    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "model_config": {

            "num_classes":
                len(
                    class_names
                ),

            "input_channels":
                config[
                    "input_channels"
                ],

            "dropout":
                config[
                    "dropout"
                ],
        },

        "image_config": {

            "image_size":
                config[
                    "image_size"
                ],

            "normalization": {

                "mean":

                    [0.5]

                    if config[
                        "input_channels"
                    ] == 1

                    else [
                        0.485,
                        0.456,
                        0.406
                    ],

                "std":

                    [0.5]

                    if config[
                        "input_channels"
                    ] == 1

                    else [
                        0.229,
                        0.224,
                        0.225
                    ],
            }
        },

        "class_names":
            class_names,

        "epoch":
            epoch,

        "metrics":
            metrics,
    }

    torch.save(

        checkpoint,

        output_path
    )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

def save_history(
    history,
    output_path
):

    if not history:

        return

    fieldnames = list(
        history[0].keys()
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(

            file,

            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            history
        )


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    data,
    output_path
):

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(

            data,

            file,

            indent=4
        )


# ============================================================
# MEASURE INFERENCE SPEED
# ============================================================

def measure_inference_speed(
    model,
    loader,
    device,
    warmup_batches=5,
    test_batches=20
):

    model.eval()

    batches = []

    for index, batch in enumerate(
        loader
    ):

        batches.append(
            batch
        )

        if (
            index + 1
            >=
            warmup_batches + test_batches
        ):

            break

    if not batches:

        return {}

    # --------------------------------------------------------
    # WARMUP
    # --------------------------------------------------------

    with torch.no_grad():

        for images, _ in batches[
            :warmup_batches
        ]:

            images = images.to(
                device
            )

            _ = model(
                images
            )

    if device.type == "cuda":

        torch.cuda.synchronize()

    # --------------------------------------------------------
    # MEASURE
    # --------------------------------------------------------

    measured_batches = batches[
        warmup_batches:
    ]

    if not measured_batches:

        return {}

    total_images = 0

    start_time = time.perf_counter()

    with torch.no_grad():

        for images, _ in measured_batches:

            images = images.to(
                device
            )

            _ = model(
                images
            )

            total_images += (
                images.size(0)
            )

    if device.type == "cuda":

        torch.cuda.synchronize()

    elapsed = (

        time.perf_counter()
        -
        start_time
    )

    if elapsed <= 0:

        return {}

    images_per_second = (

        total_images
        /
        elapsed
    )

    milliseconds_per_image = (

        1000.0
        /
        images_per_second
    )

    return {

        "images_tested":
            total_images,

        "elapsed_seconds":
            elapsed,

        "images_per_second":
            images_per_second,

        "milliseconds_per_image":
            milliseconds_per_image
    }


# ============================================================
# PRINT DATASET SUMMARY
# ============================================================

def print_dataset_summary(
    train_dataset,
    valid_dataset,
    test_dataset
):

    print()

    print("=" * 70)

    print(
        "DATASET SUMMARY"
    )

    print("=" * 70)

    print()

    print(
        "Classes:"
    )

    for index, class_name in enumerate(
        train_dataset.classes
    ):

        print(
            f"  {index}: "
            f"{class_name}"
        )

    print()

    print(
        f"Train samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(valid_dataset)}"
    )

    print(
        f"Test samples: "
        f"{len(test_dataset)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Train a general "
            "SmallCNN classifier."
        )
    )

    parser.add_argument(

        "--config",

        type=str,

        default=None,

        help=(
            "Path to JSON "
            "configuration file."
        )
    )

    parser.add_argument(

        "--dataset",

        type=str,

        default=None,

        help=(
            "Override dataset directory."
        )
    )

    parser.add_argument(

        "--output",

        type=str,

        default=None,

        help=(
            "Override output directory."
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------

    config = load_config(
        args.config
    )

    if args.dataset is not None:

        config[
            "dataset_dir"
        ] = args.dataset

    if args.output is not None:

        config[
            "output_dir"
        ] = args.output

    set_seed(
        config["seed"]
    )

    output_dir = Path(

        config[
            "output_dir"
        ]
    )

    output_dir.mkdir(

        parents=True,

        exist_ok=True
    )

    device = get_device(

        config[
            "device"
        ]
    )

    print()

    print("=" * 70)

    print(
        "SMALLCNN TRAINING"
    )

    print("=" * 70)

    print()

    print(
        f"Device: {device}"
    )

    print(
        f"Dataset: "
        f"{config['dataset_dir']}"
    )

    print(
        f"Output: "
        f"{output_dir}"
    )

    # --------------------------------------------------------
    # TRANSFORM
    # --------------------------------------------------------

    transform = create_transforms(

        image_size=
        config[
            "image_size"
        ],

        input_channels=
        config[
            "input_channels"
        ]
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    (
        train_dataset,

        valid_dataset,

        test_dataset

    ) = load_datasets(

        config[
            "dataset_dir"
        ],

        transform
    )

    print_dataset_summary(

        train_dataset,

        valid_dataset,

        test_dataset
    )

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    (
        train_loader,

        valid_loader,

        test_loader

    ) = create_dataloaders(

        train_dataset,

        valid_dataset,

        test_dataset,

        batch_size=
        config[
            "batch_size"
        ],

        num_workers=
        config[
            "num_workers"
        ]
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    class_names = (
        train_dataset.classes
    )

    model = SmallCNN(

        num_classes=
        len(
            class_names
        ),

        input_channels=
        config[
            "input_channels"
        ],

        dropout=
        config[
            "dropout"
        ]

    ).to(
        device
    )

    total_parameters = sum(

        parameter.numel()

        for parameter

        in model.parameters()

        if parameter.requires_grad
    )

    print()

    print(
        f"Model parameters: "
        f"{total_parameters:,}"
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = (
        nn.CrossEntropyLoss()
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=
        config[
            "learning_rate"
        ],

        weight_decay=
        config[
            "weight_decay"
        ]
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    best_validation_accuracy = (
        -1.0
    )

    best_checkpoint_path = (

        output_dir
        /
        "best_model.pth"
    )

    history = []

    print()

    print("=" * 70)

    print(
        "TRAINING"
    )

    print("=" * 70)

    for epoch in range(

        1,

        config["epochs"]
        +
        1
    ):

        epoch_start = (
            time.perf_counter()
        )

        train_metrics = (
            train_one_epoch(

                model,

                train_loader,

                optimizer,

                criterion,

                device
            )
        )

        validation_metrics = (
            evaluate_model(

                model,

                valid_loader,

                criterion,

                device,

                class_names
            )
        )

        epoch_time = (

            time.perf_counter()
            -
            epoch_start
        )

        row = {

            "epoch":
                epoch,

            "train_loss":
                train_metrics[
                    "loss"
                ],

            "train_accuracy":
                train_metrics[
                    "accuracy"
                ],

            "validation_loss":
                validation_metrics[
                    "loss"
                ],

            "validation_accuracy":
                validation_metrics[
                    "accuracy"
                ],

            "validation_balanced_accuracy":
                validation_metrics[
                    "balanced_accuracy"
                ],

            "validation_f1":
                validation_metrics[
                    "f1"
                ],

            "epoch_seconds":
                epoch_time
        }

        history.append(
            row
        )

        print()

        print(
            f"Epoch "
            f"{epoch}/"
            f"{config['epochs']}"
        )

        print(
            f"  Train Loss: "
            f"{train_metrics['loss']:.6f}"
        )

        print(
            f"  Train Accuracy: "
            f"{train_metrics['accuracy']:.4f}"
        )

        print(
            f"  Validation Loss: "
            f"{validation_metrics['loss']:.6f}"
        )

        print(
            f"  Validation Accuracy: "
            f"{validation_metrics['accuracy']:.4f}"
        )

        print(
            f"  Validation Balanced Accuracy: "
            f"{validation_metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"  Validation F1: "
            f"{validation_metrics['f1']:.4f}"
        )

        print(
            f"  Epoch Time: "
            f"{epoch_time:.2f} sec"
        )

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if (

            validation_metrics[
                "accuracy"
            ]

            >

            best_validation_accuracy
        ):

            best_validation_accuracy = (

                validation_metrics[
                    "accuracy"
                ]
            )

            save_checkpoint(

                best_checkpoint_path,

                model,

                config,

                class_names,

                epoch,

                validation_metrics
            )

            print(
                "  Best model saved."
            )

    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    save_history(

        history,

        output_dir
        /
        "training_history.csv"
    )

    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "LOADING BEST MODEL"
    )

    print("=" * 70)

    checkpoint = torch.load(

        best_checkpoint_path,

        map_location=device
    )

    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    # --------------------------------------------------------
    # TEST EVALUATION
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "TEST EVALUATION"
    )

    print("=" * 70)

    test_metrics = evaluate_model(

        model,

        test_loader,

        criterion,

        device,

        class_names
    )

    print()

    print(
        f"Test Loss: "
        f"{test_metrics['loss']:.6f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Test Balanced Accuracy: "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Test Precision: "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Test Recall: "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"Test F1: "
        f"{test_metrics['f1']:.4f}"
    )

    print()

    print(
        "Confusion Matrix:"
    )

    for row in test_metrics[
        "confusion_matrix"
    ]:

        print(
            row
        )

    # --------------------------------------------------------
    # INFERENCE SPEED
    # --------------------------------------------------------

    speed_metrics = (

        measure_inference_speed(

            model,

            test_loader,

            device
        )
    )

    print()

    print("=" * 70)

    print(
        "INFERENCE SPEED"
    )

    print("=" * 70)

    if speed_metrics:

        print()

        print(
            f"Images per second: "
            f"{speed_metrics['images_per_second']:.2f}"
        )

        print(
            f"Milliseconds per image: "
            f"{speed_metrics['milliseconds_per_image']:.4f}"
        )

    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    final_results = {

        "device":
            str(device),

        "dataset_dir":
            str(
                config[
                    "dataset_dir"
                ]
            ),

        "class_names":
            class_names,

        "num_classes":
            len(
                class_names
            ),

        "model_parameters":
            total_parameters,

        "best_validation_accuracy":
            best_validation_accuracy,

        "test_metrics": {

            key:
                value

            for key, value

            in test_metrics.items()

            if key
            not in [
                "y_true",
                "y_pred"
            ]
        },

        "inference_speed":
            speed_metrics,
    }

    save_json(

        final_results,

        output_dir
        /
        "results.json"
    )

    print()

    print("=" * 70)

    print(
        "TRAINING COMPLETE"
    )

    print("=" * 70)

    print()

    print(
        f"Best model:\n"
        f"{best_checkpoint_path}"
    )

    print()

    print(
        f"Results:\n"
        f"{output_dir / 'results.json'}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
