import torch.nn as nn
import torchvision.models as models

class VGGFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = models.vgg19(pretrained=True).features
        self.layers = {
            "0": "conv1",
            "5": "conv2",
            "10": "conv3",
            "19": "conv4",
            "28": "conv5",
        }

        for p in self.vgg.parameters():
            p.requires_grad = False

    def forward(self, x):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
        return features
