import argparse
import configparser
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from model_construction.common import MachineMaintenanceDataset, plot_results
from model_construction.model import Classifier

torch.manual_seed(0)


def train(model, criterion, optimizer, loader, epochs):

    """ Train a classifier """

    model.train()
    for e in range(epochs):
        for samples, labels in loader:
            optimizer.zero_grad()
            output = model(samples)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

    return model

def evaluate(model, loader, example_inputs):

    """ Evaluate a classifier """

    model.eval()
    loss = 0.0
    count = 0.0
    with torch.no_grad():
        example_outputs = model(example_inputs)
        n_classes = example_outputs.shape[1]
        cmatrix = torch.zeros(n_classes, n_classes, dtype=torch.int)
        for samples, labels in loader:
            output = model(samples)
            eval_loss = criterion(output, labels)
            loss += eval_loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            for y_true, y_pred in zip(labels, pred):
                cmatrix[y_true][y_pred] += 1
            count += len(labels)

    return loss / count, cmatrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate a model')
    parser.add_argument('config', type=str, help='Configuration file')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    config_epochs = config.getint('CONFIGURATION', 'epochs')
    config_batch = config.getint('CONFIGURATION', 'batch')
    config_lr = config.getfloat('CONFIGURATION', 'lr')
    config_shuffle = config.getboolean('CONFIGURATION', 'shuffle')

    train_file = config.get('CONFIGURATION', 'train_file')
    test_file = config.get('CONFIGURATION', 'test_file')

    train_dataset = MachineMaintenanceDataset(train_file)
    test_dataset = MachineMaintenanceDataset(test_file)

    model = Classifier()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=config_lr)

    train_losses = []
    test_losses = []
    confusion_matrices = []

    for _ in range(config_epochs):

        train_loader = DataLoader(train_dataset, batch_size=config_batch, shuffle=config_shuffle)
        test_loader = DataLoader(test_dataset, batch_size=config_batch, shuffle=config_shuffle)

        train(model, criterion, optimizer, train_loader, 1)
        loss, _ = evaluate(model, train_loader, torch.zeros(1, 10))
        train_losses.append(loss)
        loss, cmatrix = evaluate(model, test_loader, torch.zeros(1, 10))
        test_losses.append(loss)
        confusion_matrices.append(cmatrix)

    #plot_results([m.numpy() for m in confusion_matrices], train_losses, test_losses)
    print(train_losses, test_losses)