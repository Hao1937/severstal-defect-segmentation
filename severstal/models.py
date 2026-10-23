"""Architectures are rebuilt for training; historical TorchScript stays separate."""

import segmentation_models_pytorch as smp
import torchvision.models as vision


def make_model(config, pretrained=True):
    use_imagenet = pretrained and config.get("encoder_weights") == "imagenet"
    if config["task"] == "classification":
        model = vision.resnet34(weights=vision.ResNet34_Weights.IMAGENET1K_V1 if use_imagenet else None)
        import torch
        model.fc = torch.nn.Linear(model.fc.in_features, 4)
        return model
    # torchvision hosts the original ResNet34 weights directly and avoids a Hub dependency.
    resnet = config["encoder"] == "resnet34"
    model = getattr(smp, config["architecture"])(
        encoder_name=config["encoder"], encoder_weights="imagenet" if use_imagenet and not resnet else None,
        in_channels=3, classes=4, activation=None,
    )
    if use_imagenet and resnet:
        state = vision.ResNet34_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
        model.encoder.load_state_dict(state)
    return model
