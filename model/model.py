import torch.nn as nn
from torchvision import models


def build_model(num_classes=3):

    # =========================
    # LOAD PRETRAINED DENSENET
    # =========================

    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

    # =========================
    # FREEZE BACKBONE
    # =========================

    for param in model.features.parameters():
        param.requires_grad = False


    # =========================
    # REPLACE CLASSIFIER
    # =========================

    num_features = model.classifier.in_features

    model.classifier = nn.Sequential(

        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),

        nn.Linear(512, num_classes)

    )

    return model