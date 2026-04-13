import torch
from torchvision import models
from pytorch_grad_cam import GradCAM

device = torch.device('cpu')
model = models.densenet121(weights=None)
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(model.classifier.in_features, 22)
)
# We don't need to load the weights just to test GradCAM layer names.

target_layers = [model.features[-1]]
try:
    cam = GradCAM(model=model, target_layers=target_layers)
    print("Success with model.features[-1]!")
except Exception as e:
    print("Error with model.features[-1]:", e)

target_layers = [model.features.denseblock4.denselayer16.conv2]
try:
    cam = GradCAM(model=model, target_layers=target_layers)
    print("Success with model.features.denseblock4.denselayer16.conv2!")
except Exception as e:
    print("Error with convolution:", e)
