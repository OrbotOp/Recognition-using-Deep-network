"""""
author: HRITHIK KANOJE, RISHABH SINGH
Class: CS5330 Pattern Recog & Computer Vision
Prof: Bruce Maxwell
Project 5: Recognition using Deep Networks
"""""

import cv2
import torch
import matplotlib.pyplot as plt
import sys
import training_model
import numpy as np
import seaborn as sns
sns.set_theme()


def sample_test(network):
    """
    Create predictions for the first 10 test dataset images, displaying output values and index of max value.
    Plot the first 9 digits in a 3x3 grid with their predicted values.

    """
    # Retrieve the first 10 images and their corresponding labels from the test dataset,
    #  then use the trained model to make predictions on these images.
    samples = next(iter(network.test_loader))
    images, labels = samples
    ground_truth_labels = labels[:10].numpy()
    output = network(images[:10])

    # Create a plot that displays the first 9 digits along with their predicted labels.
    # Print the required results, including the output values and the index of the maximum output value,
    # for all 10 digits.
    for i in range(10):
        if i != 9:
            plt.subplot(3, 3, i + 1)
            plt.tight_layout()
            plt.imshow(images[i][0], cmap='gray', interpolation='none')
            plt.title("Prediction: {}".format(
                output.data.max(1, keepdim=True)[1][i].item()))
            plt.xticks([])
            plt.yticks([])
        output_values = ["{:.2f}".format(v) for v in output[i].tolist()]
        print(
            f"image {i + 1}, output values: {output_values}, max value index: {torch.argmax(output[i]).item()}, ground truth label: {ground_truth_labels[i]}")
    plt.show()


def new_input_test(network):
    """
    Preprocess the digits from the test_digits directory, input them into the network for prediction,
    and visualize the results.

    """
    file_path = './test_digits/'
    test_img = []
    for i in range(10):
        # load as grayscale
        img = cv2.imread(f'{file_path}{i}.png', cv2.IMREAD_GRAYSCALE)
        # resize to 28x28 square
        img = cv2.resize(img, (28, 28))
        # invert the intensity
        img = cv2.bitwise_not(img)
        test_img.append(img)
    test_data = np.array(test_img)
    #transform into tensor as input
    test_data = np.expand_dims(test_data, 1)
    test_data = torch.Tensor(test_data)
    output = network(test_data)

    for i in range(10):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        plt.imshow(test_img[i], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
            output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def main(argv):
    """
    Load the network and apply it to the test set to evaluate its performance on new inputs.

    """
    # load the model
    network = training_model.MyNetwork()
    network.load_state_dict(torch.load('./results/model.pth'))
    # enter eval mode
    network.eval()

    sample_test(network)
    new_input_test(network)


if __name__ == "__main__":
    main(sys.argv)
