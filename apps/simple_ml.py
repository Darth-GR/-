"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl

import needle.nn as nn
from apps.models import *
import time
device = ndl.cuda() if ndl.cuda().enabled() else ndl.cpu()

def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    with gzip.open(image_filesname, "rb") as img_f:
        magic, num_images, rows, cols = struct.unpack(">IIII", img_f.read(16))
        if magic != 2051:
            raise ValueError(f"invalid MNIST image magic number: {magic}")
        X = np.frombuffer(img_f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
    with gzip.open(label_filename, "rb") as label_f:
        magic, num_labels = struct.unpack(">II", label_f.read(8))
        if magic != 2049:
            raise ValueError(f"invalid MNIST label magic number: {magic}")
        y = np.frombuffer(label_f.read(), dtype=np.uint8)
    if num_images != num_labels:
        raise ValueError("MNIST image and label counts differ")
    return X.astype(np.float32) / 255.0, y.astype(np.int8)


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    batch_size = Z.shape[0]
    return (ndl.ops.logsumexp(Z, axes=(1,)) - (Z * y_one_hot).sum(axes=(1,))).sum() / batch_size


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """
    n = X.shape[0]
    for start in range(0, n, batch):
        end = min(start + batch, n)
        X_batch = ndl.Tensor(X[start:end], device=W1.device, dtype="float32")
        y_batch = y[start:end]
        y_one_hot = np.zeros((end - start, W2.shape[1]), dtype=np.float32)
        y_one_hot[np.arange(end - start), y_batch] = 1
        y_batch_one_hot = ndl.Tensor(y_one_hot, device=W1.device, dtype="float32")
        logits = ndl.ops.relu(X_batch @ W1) @ W2
        loss = softmax_loss(logits, y_batch_one_hot)
        loss.backward()
        W1.data = W1.data - lr * W1.grad
        W2.data = W2.data - lr * W2.grad
        W1.grad = None
        W2.grad = None
    return W1, W2

### CIFAR-10 training ###
def epoch_general_cifar10(dataloader, model, epoch, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    training = opt is not None
    model.train() if training else model.eval()
    total_correct = 0
    total_loss = 0.0
    total = 0
    for X, y in dataloader:
        X = ndl.Tensor(X, device=device, dtype="float32")
        y = ndl.Tensor(y, device=device, dtype="int32", requires_grad=False)
        logits = model(X)
        loss = loss_fn(logits, y)
        batch_size = X.shape[0]
        total_loss += float(loss.numpy()) * batch_size
        total_correct += int((logits.numpy().argmax(axis=1) == y.numpy()).sum())
        total += batch_size
        if training:
            opt.reset_grad()
            loss.backward()
            opt.step()
    return total_correct / total, total_loss / total


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss()):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    avg_acc, avg_loss = 0.0, 0.0
    for epoch in range(n_epochs):
        avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, epoch, loss_fn, opt)
    return avg_acc, avg_loss


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss()):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    return epoch_general_cifar10(dataloader, model, 0, loss_fn, opt=None)

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
