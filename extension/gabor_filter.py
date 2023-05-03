"""""
author: HRITHIK KANOJE, RISHABH SINGH
Class: CS5330 Pattern Recog & Computer Vision
Prof: Bruce Maxwell
Project 5: Recognition using Deep Networks
"""""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from task_1.training_model import MyNetwork
import torch.nn.functional as F
import torch.nn as nn
import sys


# Network with first convolutional layer replaced with filter bank of Gabor.
class gabor_convolve_ntwrk(MyNetwork):

    def __init__(self, gabor_kernels, check_first_layer=False):
        super().__init__()
        self.gabor_kernels = gabor_kernels
        self.check_first_layer = check_first_layer

    # override the forward method
    def forward(self, x):
        gabor_conv = nn.Conv2d(1, 10, kernel_size=(5, 5))
        gabor_conv.weight = nn.Parameter(torch.Tensor(self.gabor_kernels))

        # if checking the result of first convolutional layer
        if self.check_first_layer:
            return gabor_conv(x)
        x = F.relu(F.max_pool2d(gabor_conv(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def create_kernel():
    """
   Gabor filter bank has replaced the network's first convolutional layer, enhancing its feature extraction capabilities.
    """
    angles = np.linspace(0, 180, 10)
    gabor_kernels = [cv2.getGaborKernel((5, 5), 10, np.degrees(angle), 0.05, 0.05, 0, cv2.CV_32F) for angle in angles]
    gabor_kernels = np.stack(gabor_kernels, axis=0)
    gabor_kernels = gabor_kernels[:, None, :, :]
    return gabor_kernels


def dataset_accuracy(network):
    """
    Evaluate the network's accuracy by inputting the test dataset into the model.

    """
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in network.test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(network.test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(network.test_loader.dataset),
        100. * correct / len(network.test_loader.dataset)))


def layer1_eff(weights, network, images):
    """
    Visualize the first Gabor convolutional layer filters and their output on the initial test dataset
    image using the modified network with a Gabor filter bank.
    """
    network.eval()
    with torch.no_grad():
        output = network(images[0:1])
        img_list = []
        i = 0
        while i < 10:
            img_list.append(weights[i, 0])
            img_list.append(output[0, i])
            i += 1

    fig, axs = plt.subplots(5, 4, figsize=(9, 6), subplot_kw={'xticks': [], 'yticks': []})
    j = 0
    while j < 20:
        row = j // 4
        col = j % 4
        axs[row, col].imshow(img_list[j], cmap='gray')
        j += 1
    plt.show()



def main(argv):
    """
   To enhance the feature extraction capabilities of the original MyNetwork, create a new network that
   inherits from MyNetwork and replaces the initial convolutional layer with a new layer containing Gabor kernels.
   After training the network on the dataset, evaluate its performance by checking the prediction and output of the
   first test dataset example. Additionally, calculate the accuracy of this network on the entire test dataset to
   determine its effectiveness.
    """
    gabor_kernels = create_kernel()

    network = gabor_convolve_ntwrk(gabor_kernels)
    network.load_state_dict(torch.load('../task_1/results/model.pth'))
    network.eval()

    images, labels = next(iter(network.test_loader))
    image = images[0:1]

    # Make the prediction on the first image in the test dataset
    output = network(image)
    prediction = output.data.max(1, keepdim=True)[1].item()
    print(f'Prediction for the first test image: {prediction}')

    dataset_accuracy(network)

    first_layer_network = gabor_convolve_ntwrk(gabor_kernels, True)
    first_layer_network.load_state_dict(torch.load('../task_1/results/model.pth'))
    layer1_eff(gabor_kernels, first_layer_network, images)


if __name__ == "__main__":
    main(sys.argv)
