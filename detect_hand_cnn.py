from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def count_similiarity(y_pred, y):

    '''
    :param y_pred: predicted values
    :param y: real values
    :return: similiarity value
    '''

    differences = torch.abs(y_pred - y)

    squared_diff_sum = torch.sum(differences)

    total_elements = y_pred.numel()
    result = squared_diff_sum / total_elements

    return result


class Create_net(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=8, output_dim=4):

        '''

        Function which creates a CNN with specified input dimensions and output dimensions.

        :param input_dim: number of input channels (int)
        :param output_dim: number of output classes (int)

        '''

        super().__init__()
        self.seq_nn = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(hidden_dim, 2 * hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(120 * 160 * 2 * hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, output_dim)
        )

    def forward(self, X):
        """

        :param X:
        :return:
        """
        return self.seq_nn(X)


def train(model_nn, train_loader: torch, val_loader: torch, epochs=100, lr=0.01, optimizer=torch.optim.SGD,
          choosen_loss_function=nn.L1Loss, count_similiarity=count_similiarity):

    '''

    :param model_nn: pytorch model
    :param train_loader: training data as torch DataLoader
    :param val_loader: validation data as torch DataLoader
    :param epochs: number of epochs (loops)
    :param lr: learning rate (speed of learning)
    :param optimizer: type of optimizer (SGD, Adam)
    :param choosen_loss_function: type of loss function (nn.CrossEntropyLoss)
    :param count_similiarity: prepared function to count good and bad similiarity
    :return:
    '''

    # creating metrics
    # it returns two dictionaries
    # each for train and val data

    train_metrics = {}
    train_metrics['acc'] = []
    train_metrics['loss'] = []

    val_metrics = {}
    val_metrics['acc'] = []
    val_metrics['loss'] = []

    loss_function = choosen_loss_function()
    optimizer = optimizer(params=model_nn.parameters(), lr=lr)

    best_score = 0
    best_model = None

    for epoch in tqdm(range(epochs), desc="Training"):
        # tqdm na epoch
        numb = 0
        acc_epoch = 0
        loss_epoch = 0

        # dropout
        model_nn.train()
        for X, y in train_loader:
            optimizer.zero_grad()

            X = X.unsqueeze(1)

            output = model_nn(X)

            output = output.squeeze()

            loss_value = loss_function(output, y)

            numb_acc = count_similiarity(output, y)

            loss_value.backward()
            optimizer.step()

            acc_epoch += numb_acc
            numb += len(y.float())
            loss_epoch += loss_value.item()

        train_metrics['acc'].append(acc_epoch / numb)
        train_metrics['loss'].append(loss_epoch / numb)

        acc_epoch = 0
        loss_epoch = 0

        numb2 = 0
        model_nn.eval()
        with torch.no_grad():
            for X, y in val_loader:

                X = X.unsqueeze(1)

                output = model_nn(X)

                output = output.squeeze()

                loss_value = loss_function(output.squeeze(), y.float())

                acc = count_similiarity(output, y)

                numb2 += len(y.float())
                acc_epoch += acc
                loss_epoch += loss_value.item()

        val_metrics['acc'].append(acc_epoch / numb2)
        val_metrics['loss'].append(loss_epoch / numb2)

        tqdm.write(f"Epoch {epoch + 1}/{epochs} - "
                   f"Train Accuracy: {train_metrics['acc'][epoch]:.4f}, "
                   f"Train Loss: {train_metrics['loss'][epoch]:.4f},"
                   f"Val Accuracy: {val_metrics['acc'][epoch]:.4f},"
                   f"Val Loss: {val_metrics['loss'][epoch]:.4f}")

        if acc_epoch / numb2 > best_score:
            best_score = acc_epoch / numb2
            best_model = model_nn

    return train_metrics, val_metrics, best_model


def show_comparision(train_metrics, val_metrics, train_metrics2, val_metrics2):

    train_acc_model1 = train_metrics['acc']
    train_loss_model1 = train_metrics['loss']
    val_acc_model1 = val_metrics['acc']
    val_loss_model1 = val_metrics['loss']

    train_acc_model2 = train_metrics2['acc']
    train_loss_model2 = train_metrics2['loss']
    val_acc_model2 = val_metrics2['acc']
    val_loss_model2 = val_metrics2['loss']

    plt.style.use('ggplot')

    plt.figure(figsize=(10, 5))
    plt.plot(train_acc_model1, label='Model 1 Train')
    plt.plot(train_acc_model2, label='Model 2 Train')
    plt.xlabel('Epoch')
    plt.ylabel('Train Accuracy')
    plt.title('Train Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_model1, label='Model 1 Train')
    plt.plot(train_loss_model2, label='Model 2 Train')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title('Train Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(val_acc_model1, label='Model 1 Validation')
    plt.plot(val_acc_model2, label='Model 2 Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(val_loss_model1, label='Model 1 Validation')
    plt.plot(val_loss_model2, label='Model 2 Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

def predict_result(model, X):

    with torch.no_grad():

        output = model(X)

    return output.squeeze().cpu().detach().numpy()