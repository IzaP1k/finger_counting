
"""
That module is to create model, train and show its results
"""

# Packages

from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# Functions

def search_prob(tensor):

    '''
    It get a list of elements between 0 and 1, it chooses the highest propability as a predicted int output.
    :param tensor
    :return: predicted letter
    '''

    max_indices = []
    for row in tensor:
        max_index = torch.argmax(row).item()
        max_indices.append(max_index)
    return max_indices


def count_similiarity(y_pred, y):

    '''
    It counts similarity between predicted value and true value.
    :param y_pred: predicted values
    :param y: real values
    :return: similarity value float
    '''

    count = 0
    for num1, num2 in zip(y_pred, y):

        if num1 == num2:
            count += 1

    return count

class Simple_CNN(nn.Module):

    def __init__(self, input_dim=1, hidden_dim=8, output_dim=6):

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
            nn.Linear(25 * 25 * 2 * hidden_dim, 2 * hidden_dim),
            nn.ReLU(),

            nn.Linear(2 * hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, X):
        return self.seq_nn(X)


class Advanced_CNN(nn.Module):
    """
    Function which creates a CNN with specified input dimensions and output dimensions.
    input = 1
    output = 6
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16 X 50 X 50

            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 X 25 X 25

            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(5, 5),  # 64 X 5 X 5

            nn.Flatten(),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(64 * 5 * 5, 6),
            nn.Sigmoid())

    def forward(self, xb):
        return self.network(xb)

def train(model_nn, train_loader: torch, val_loader: torch, epochs=100, lr=0.01, optimizer=torch.optim.AdamW,
          choosen_loss_function=nn.CrossEntropyLoss, count_similiarity=count_similiarity, model_name='new_model'):

    '''

    :param model_nn: pytorch model
    :param train_loader: training data as torch DataLoader
    :param val_loader: validation data as torch DataLoader
    :param epochs: number of epochs (loops)
    :param lr: learning rate (speed of learning)
    :param optimizer: type of optimizer (SGD, Adam)
    :param choosen_loss_function: type of loss function (nn.CrossEntropyLoss)
    :param count_similiarity: prepared function to count good and bad similiarity
    :return: train_metrics, val_metrics, best_model
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


            y_pred = search_prob(output)
            y_true = search_prob(y)

            numb_acc = count_similiarity(y_pred, y_true)

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

                y_pred = search_prob(output)
                y_true = search_prob(y)

                acc = count_similiarity(y_pred, y_true)

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
            torch.save(model_nn.state_dict(), f"{model_name}.pt")


    best_model = model_nn.load_state_dict(torch.load(f"{model_name}.pt"))

    return train_metrics, val_metrics, best_model


def show_comparision(train_metrics, val_metrics, train_metrics2, val_metrics2, train_metrics3, val_metrics3):

    """
    It shows plots that compares results of train and validation datasets during training loop.
    :param train_metrics:
    :param val_metrics:
    :param train_metrics2:
    :param val_metrics2:
    :param train_metrics3:
    :param val_metrics3:
    :return: plots
    """

    train_acc_model1 = train_metrics['acc']
    train_loss_model1 = train_metrics['loss']
    val_acc_model1 = val_metrics['acc']
    val_loss_model1 = val_metrics['loss']

    train_acc_model2 = train_metrics2['acc']
    train_loss_model2 = train_metrics2['loss']
    val_acc_model2 = val_metrics2['acc']
    val_loss_model2 = val_metrics2['loss']

    train_acc_model3 = train_metrics3['acc']
    train_loss_model3 = train_metrics3['loss']
    val_acc_model3 = val_metrics3['acc']
    val_loss_model3 = val_metrics3['loss']

    plt.style.use('ggplot')

    plt.figure(figsize=(10, 5))
    plt.plot(train_acc_model1, label='Gray scale model Train')
    plt.plot(train_acc_model2, label='HOG model Train')
    plt.plot(train_acc_model3, label='LoG model Train')
    plt.xlabel('Epoch')
    plt.ylabel('Train Accuracy')
    plt.title('Train Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_model1, label='Gray scale model Train')
    plt.plot(train_loss_model2, label='HOG model Train')
    plt.plot(train_loss_model3, label='LoG model Train')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title('Train Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(val_acc_model1, label='Gray scale model Validation')
    plt.plot(val_acc_model2, label='HOG model Validation')
    plt.plot(val_acc_model3, label='LoG model Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(val_loss_model1, label='Gray scale model Validation')
    plt.plot(val_loss_model2, label='HOG model Validation')
    plt.plot(val_loss_model3, label='LoG model Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

def predict_result(model, X):

    """
    It predicts results on trained model
    :param model: pytorch class nn.model
    :param X: pytroch torch tensor
    :return: list of lists
    """

    with torch.no_grad():

        output = model(X)
        output = search_prob(output)

    return output