"""""
author: HRITHIK KANOJE, RISHABH SINGH
Class: CS5330 Pattern Recog & Computer Vision
Prof: Bruce Maxwell
Project 5: Recognition using Deep Networks
"""""
import os
import sys
import torch
from task_1.training_model import MyNetwork
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
import numpy as np
import pandas as pd


class truncatedmodel(MyNetwork):

    def __init__(self):
        super().__init__()

    # override the forward method
    def forward(self, x):
        # Apply first convolutional layer and max pooling, then apply ReLU activation
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        # Apply second convolutional layer, dropout, max pooling, then apply ReLU activation
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        # Reshape tensor and apply fully connected layer with ReLU activation
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = F.relu(x)

        return x


def make_dataset(directory):
    """
    Load the greek symbols dataset from the given directory and create a PyTorch dataset object.
    The dataset includes the intensity values and the labels for each symbol.
    """
    fig, axes = plt.subplots(nrows=9, ncols=3, figsize=(8, 10))
    plt.subplots_adjust(hspace=0.5)
    axes = axes.flatten()

    # save the intensity values
    data = []

    # save the corresponding category(alpha = 0, beta = 1, gamma = 2)
    categories = []
    symbol_dict = {}
    i = 0
    for file_name in os.listdir(directory):

        #the corresponding category
        symbol = file_name.split('_')[0]
        if symbol not in symbol_dict:
            symbol_dict[symbol] = len(symbol_dict)
        categories.append(symbol_dict[symbol])

        #porcess the image
        img = cv2.imread(os.path.join(directory, file_name))
        img = cv2.resize(img, (28, 28))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.bitwise_not(img)

        axes[i].imshow(img, cmap='gray', interpolation='none')
        axes[i].set_xticks([])
        axes[i].set_yticks([])

        data.append(np.array(img).flatten())
        i += 1
    plt.show()

    # save the intensity values into data.csv
    data_header = [str(i) for i in range(len(data[0]))]
    data = pd.DataFrame(data)
    data.to_csv('./data.csv', header=data_header, index=False)

    # save the categories into category.csv
    categories = pd.DataFrame(categories)
    categories.to_csv('./category.csv', header=['category'], index=False)


def get_embeddings(network):
    """
    Load greek dataset, get embeddings and categories with trained network.
    """
    with torch.no_grad():
        # Load data and categories
        data = pd.read_csv('./data.csv').values.reshape((-1, 1, 28, 28))
        categories = pd.read_csv('./category.csv').values.flatten()

        # Compute embeddings
        embedding = network(torch.Tensor(data)).numpy()

        return embedding, categories

def sum_sq_dist(a, b):
    """
    the sum squared distance between two embedding arrays
    """
    return np.sum(np.square(a - b))


def compute_ssd(categories, embedding):
    """
    Calculate the sum squared distance of the first greek digit for each category from all the greek symbols,
    and sort these three distance lists to analyze the pattern.
    """
    category_first_idx = [np.where(categories == i)[0][0] for i in range(3)]
    labels = {0: 'alpha', 1: 'beta', 2: 'gamma'}
    dist_label_list = [[(sum_sq_dist(embedding[category_first_idx[i]], embedding[j]), labels[categories[j]]) for j in
                        range(embedding.shape[0])] for i in range(3)]
    # sorting
    for dist_label in dist_label_list:
        dist_label.sort(key=lambda v: v[0])
        print(dist_label)



def test_greek(network, embedding, categories):
    """
    Preprocess and feed the test images from the "test_greek" folder into the network to obtain embeddings.
    Calculate the distances between each test image's embedding and the embeddings of the entire training dataset.
    Choose the training image with the smallest distance as the prediction.

    """
    labels = {0: 'alpha', 1: 'beta', 2: 'gamma'}
    test_data = []
    file_path = './test_greek/'

    # load test images and preprocess
    for file_name in os.listdir(file_path):
        img = cv2.imread(file_path + file_name, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img = cv2.bitwise_not(img)
        test_data.append(img)

    # convert test data to tensor
    test_data = torch.Tensor(np.array(test_data).reshape(-1, 1, 28, 28))

    with torch.no_grad():
        # get embeddings for test data
        output = network(test_data)
        test_embeddings = output.numpy()

        print("Test dataset result:")
        # iterate over test images
        for i, img in enumerate(test_data):
            plt.subplot(3, 2, i + 1)
            plt.tight_layout()
            plt.imshow(test_data[i].squeeze(), cmap='gray', interpolation='none')

            # calculate distances between test embedding and training embeddings
            distances = [(sum_sq_dist(embedding[j], test_embeddings[i]), labels[categories[j]]) for j in
                         range(len(categories))]
            distances.sort(key=lambda v: v[0])
            print(distances)

            # set title as predicted label
            plt.title("Prediction: {}".format(distances[0][1]))
            plt.xticks([])
            plt.yticks([])
        plt.show()


def main(argv):
    """
    Create greek symbols dataset in csv format, use truncated model to get embeddings,
    calculate SSD for each category's first example, and apply this method to a new test dataset to get the prediction.

    """
    make_dataset('./greek')

    truncated_model = truncatedmodel()
    truncated_model.load_state_dict(torch.load('../task_1/results/model.pth'))
    truncated_model.eval()

    samples = next(iter(truncated_model.test_loader))
    images, labels = samples
    output = truncated_model(images[0:1])
    #output of the truncated model is 50
    assert output.shape[1] == 50

    embedding, categories = get_embeddings(truncated_model)

    compute_ssd(categories, embedding)

    test_greek(truncated_model, embedding, categories)


if __name__ == "__main__":
    main(sys.argv)
