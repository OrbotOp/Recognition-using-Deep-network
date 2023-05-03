"""""
author: HRITHIK KANOJE, RISHABH SINGH
Class: CS5330 Pattern Recog & Computer Vision
Prof: Bruce Maxwell
Project 5: Recognition using Deep Networks
"""""

import cv2
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt

import sys

"""
The model inherited the architecture of the pre-trained VGG-16 and extracted its layers using self.features. 
This enables the ability to easily return to a specific layer within the model, which can be useful for 
customizing the model's behavior for specific tasks.
"""
class VGG16_model(torch.nn.Module):
    def __init__(self, layer_idx):
        super(VGG16_model, self).__init__()
        features = list(models.vgg16(pretrained=True).features)[:23]
        self.features = nn.ModuleList(features).eval()
        self.layer_idx = layer_idx
        # print(features)

    # override the forward method
    def forward(self, x):
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == self.layer_idx:
                return x

def layer1_eff(network, img):
    """
    Plot the first 64 filters from  VGG-16 network

    """
    network.eval()
    with torch.no_grad():
        output = network(img)
        img_list = [output[0, i] for i in range(64)]

    fig, axs = plt.subplots(8, 8, figsize=(28, 28), subplot_kw={'xticks': [], 'yticks': []})
    i = 0
    while i < 64:
        row = i // 8
        col = i % 8
        axs[row, col].imshow(img_list[i], cmap='gray')
        i += 1
    plt.show()


def main(argv):
    """
    Build the VGG-16 model and input a 3-channel image to obtain the first convolutional layer's output.
    """
    model = VGG16_model(0)

    img = cv2.imread('./bike.jpg')  #import image
    img = cv2.resize(img, (224, 224)) #resize image

    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    layer1_eff(model, img_tensor)


if __name__ == "__main__":
    main(sys.argv)
