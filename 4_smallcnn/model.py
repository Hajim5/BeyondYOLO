"""
model.py

General lightweight CNN for ROI-based image classification.

Designed for:

    - Small image inputs
    - ROI-based classification
    - Low computational cost
    - CPU or GPU inference
    - Industrial inspection
    - Edge deployment experiments

Default architecture:

    Input
      ↓
    Conv Block 1
    16 channels
      ↓
    MaxPool
      ↓
    Conv Block 2
    32 channels
      ↓
    MaxPool
      ↓
    Conv Block 3
    64 channels
      ↓
    Global Average Pooling
      ↓
    Dropout
      ↓
    Fully Connected Output
"""

import torch
import torch.nn as nn


class SmallCNN(nn.Module):

    def __init__(
        self,
        num_classes,
        input_channels=1,
        dropout=0.20
    ):

        super().__init__()

        self.features = nn.Sequential(

            # ------------------------------------------------
            # BLOCK 1
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=input_channels,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                kernel_size=2
            ),

            # ------------------------------------------------
            # BLOCK 2
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                kernel_size=2
            ),

            # ------------------------------------------------
            # BLOCK 3
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(
                inplace=True
            ),
        )

        # ----------------------------------------------------
        # GLOBAL AVERAGE POOLING
        # ----------------------------------------------------

        self.global_pool = (
            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(
                p=dropout
            ),

            nn.Linear(
                64,
                num_classes
            ),
        )


    def forward(
        self,
        x
    ):

        x = self.features(
            x
        )

        x = self.global_pool(
            x
        )

        x = self.classifier(
            x
        )

        return x


# ============================================================
# MODEL INFORMATION
# ============================================================

def count_parameters(
    model
):

    return sum(

        parameter.numel()

        for parameter

        in model.parameters()

        if parameter.requires_grad
    )


def get_model_info(
    model
):

    return {

        "model_name":
            model.__class__.__name__,

        "trainable_parameters":
            count_parameters(
                model
            ),
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    model = SmallCNN(

        num_classes=3,

        input_channels=1,

        dropout=0.20
    )

    print()
    print("=" * 60)
    print("SMALLCNN MODEL TEST")
    print("=" * 60)

    print()

    print(model)

    print()

    info = get_model_info(
        model
    )

    print(
        f"Trainable parameters: "
        f"{info['trainable_parameters']:,}"
    )

    # --------------------------------------------------------
    # TEST INPUT
    # --------------------------------------------------------

    test_input = torch.randn(

        1,

        1,

        64,

        64
    )

    output = model(
        test_input
    )

    print()

    print(
        f"Input shape: "
        f"{tuple(test_input.shape)}"
    )

    print(
        f"Output shape: "
        f"{tuple(output.shape)}"
    )
