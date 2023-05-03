"""""
author: HRITHIK KANOJE, RISHABH SINGH
Class: CS5330 Pattern Recog & Computer Vision
Prof: Bruce Maxwell
Project 5: Recognition using Deep Networks
"""""
import sys
import torch
from task_1.training_model import MyNetwork
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
import seaborn as sns
sns.set_theme()

class sub_model(MyNetwork):

    def __init__(self, layer_num):
        super().__init__()
        self.layer_num = layer_num

    # This method overrides the forward method, where the layer_num argument specifies whether one or two layers from the original network should be included.
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # Apply ReLU activation function on the output of max pooling layer after conv1
        if self.layer_num == 2:
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))  # Apply ReLU activation function on the output of max pooling layer after conv2
        return x


def analysis_first_layer(network):
    """
    Visualize the ten filters of a trained network using pyplot.
    """
    network.eval()
    with torch.no_grad():
        weights = network.conv1.weight
        fig, axs = plt.subplots(3, 4, figsize=(9, 6), subplot_kw={'xticks': [], 'yticks': []})
        i = 0
        while i < 10:
            row = i // 4
            col = i % 4
            axs[row, col].imshow(weights[i, 0].numpy())
            axs[row, col].set_title(f'Filter {i}')
            i += 1

        axs[2, 2].set_visible(False)
        axs[2, 3].set_visible(False)
        plt.show()


def filter_eff(network, img):
    """
    Visualize the 10 filters applied to the first training example image using the OpenCV filter2D function.
    Plot each filter's result and the corresponding filter together.
    """
    network.eval()
    with torch.no_grad():
        img_list = []
        weights = network.conv1.weight
        i = 0
        while i < 10:
            img_list.append(weights[i, 0].numpy())
            filtered_img = cv2.filter2D(img.numpy()[0], -1, weights[i, 0].numpy())
            img_list.append(filtered_img)
            i += 1

    fig, axs = plt.subplots(5, 4, figsize=(9, 6), subplot_kw={'xticks': [], 'yticks': []})
    j = 0
    while j < 20:
        row = j // 4
        col = j % 4
        axs[row, col].imshow(img_list[j], cmap='gray')
        j += 1
    plt.show()


def truncated_model(network, images):
    """
    Apply the first 10 filters of the truncated model to the first training example image and plot the outputs.
    """
    network.eval()
    with torch.no_grad():
        output = network(images[0:1])
        fig, axs = plt.subplots(5, 2, figsize=(9, 6), subplot_kw={'xticks': [], 'yticks': []})
        i = 0
        while i < 10:
            row = i // 2
            col = i % 2
            axs[row, col].imshow(output[0, i].numpy(), cmap='gray')
            i += 1
        plt.show()


def main(argv):
    """
    Load the network, plot the first layer filters, and their effects on the example image.
    Create two truncated models with one and two layers, and plot their results on the example image
    """
    network = MyNetwork()
    network.load_state_dict(torch.load('../task_1/results/model.pth'))

    samples = next(iter(network.train_loader))
    images, labels = samples
    img = images[0]

    analysis_first_layer(network)

    filter_eff(network, img)

    single_layer_model = sub_model(1)
    single_layer_model.load_state_dict(torch.load('../task_1/results/model.pth'))
    truncated_model(single_layer_model, images)

    double_layer_model = sub_model(2)
    double_layer_model.load_state_dict(torch.load('../task_1/results/model.pth'))
    truncated_model(double_layer_model, images)


if __name__ == "__main__":
    main(sys.argv)
