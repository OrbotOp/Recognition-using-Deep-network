"""""
author: HRITHIK KANOJE, RISHABH SINGH
Class: CS5330 Pattern Recog & Computer Vision
Prof: Bruce Maxwell
Project 5: Recognition using Deep Networks
"""""

import sys
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchviz import make_dot
import seaborn as sns
import os
sns.set_theme()

"""""
# The network for the MNIST dataset classification. It contains:
# Convolution layer applies 10 convolutional filter with a size of 5x5 to the input data
# Max pooling layer applies max pooling with a window size of 2x2 to the output of the previous convolution layer,
 and then applies a ReLU activation function to the result
# Convolution layer applies 20 convolutional filter with a size of 5x5 to the output of the previous max pooling layer
# Dropout layer randomly sets half of the input units to zero during each training iteration, which helps to prevent overfitting
# Max pooling layer applies max pooling with size of 2x2 to the output of the previous layer,
 and then applies a ReLU activation function to the result
# Flattening operation followed by a fully connected linear layer with 50 nodes and a ReLU function on the output
# Fully connected linear layer with 10 nodes and log_softmax function applied to output.
"""""
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        # initialize the configurations for this network
        self.n_epochs = 5
        self.batch_size_train = 64
        self.batch_size_test = 1000
        self.learning_rate = 0.01
        self.momentum = 0.5
        self.log_interval = 10

        self.random_seed = 42
        torch.backends.cudnn.enabled = False
        torch.manual_seed(self.random_seed)
        # load the training dataset
        self.train_dataset = torchvision.datasets.MNIST(root='../data', train=True, download=True,
                                                        transform=torchvision.transforms.Compose([
                                                            torchvision.transforms.ToTensor(),
                                                            torchvision.transforms.Normalize(
                                                                (0.1307,), (0.3081,))
                                                        ]))
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size_train)

        # load the test dataset
        self.test_dataset = torchvision.datasets.MNIST('../data', train=False, download=True,
                                                       transform=torchvision.transforms.Compose([
                                                           torchvision.transforms.ToTensor(),
                                                           torchvision.transforms.Normalize(
                                                               (0.1307,), (0.3081,))
                                                       ]))
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size_test)

        # define the layers
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(10, 20, kernel_size=(5, 5))
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """
        This function performs a forward pass through the network on the input data 'x', returning a set of 10 confidence scores
        that represent the network's predictions for each of the 10 possible class labels.
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def show_examples(network):
    """
    This function plots the first 6 digits in the test dataset with their ground truth labels, using the neural network 'network' to generate predictions.

    """
    examples = enumerate(network.test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    # plotting results for 6 test digits
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def train_network(network, epoch, train_losses, train_counter):
    """
    This function trains the neural network 'network' for one epoch on the training dataset, updating its parameters
    and storing the loss values in 'train_losses'. It takes in the current epoch, training counter, and network as inputs.

    """
    optimizer = optim.SGD(network.parameters(), lr=network.learning_rate,
                          momentum=network.momentum)

    network.train()
    # check if results folder exists if not then it creates one to save model
    if not os.path.exists('results'):
        os.makedirs('results')

    for batch_idx, (data, target) in enumerate(network.train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % network.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(network.train_loader.dataset),
                       100. * batch_idx / len(network.train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(network.train_loader.dataset)))
            torch.save(network.state_dict(), 'results/model.pth') #saving model
            torch.save(optimizer.state_dict(), 'results/optimizer.pth') #saving optimizer


def test_network(network, test_losses):
    """
    This function evaluates the performance of the neural network 'network' on the test dataset, computing the loss values
    and storing them in 'test_losses'. It takes in the network as input and can be used to assess the model's accuracy and
    generalization ability.

    """
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in network.test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(network.test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(network.test_loader.dataset),
        100. * correct / len(network.test_loader.dataset)))


def main(argv):
    """
    This function creates a neural network, visualizes test dataset images, trains the network, and plots the training and
    validation loss curves.
    """
    network = MyNetwork()
    show_examples(network)

    # plot the diagram
    x = torch.randn(1000, 1, 28, 28)
    dot = make_dot(network(x), params=dict(network.named_parameters()))

    dot.render('diagram', format='png') #save the diagram in task_1 folder

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(network.train_loader.dataset) for i in range(network.n_epochs + 1)]

    test_network(network, test_losses)
    for epoch in range(1, network.n_epochs + 1):
        train_network(network, epoch, train_losses, train_counter)
        test_network(network, test_losses)

    # plotting results
    plt.plot(train_counter, train_losses, color='green')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples')
    plt.ylabel('negative log likelihood loss')
    plt.show()


if __name__ == "__main__":
    main(sys.argv)
