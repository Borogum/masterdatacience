import argparse
import configparser
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

N_CLASSES = 3

plt.style.use('seaborn')

# Reproducibility
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class MachineMaintenanceDataset(Dataset):

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file, sep=',', decimal='.')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.df.iloc[idx, :-1].astype('float')
        y = self.df.iloc[idx, -1].astype('int')
        return torch.FloatTensor(x), torch.tensor(y, dtype=torch.long)


class Classifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, N_CLASSES)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


def cm2pred(x):

    y_true = []
    y_pred = []

    for i in range(len(x)):
        for j in range(len(x)):
            for k in range(int(x[i][j])):
                y_true.append(i)
                y_pred.append(j)

    return y_true, y_pred


def train_model(train, test, epochs=30, lr=0.005):
    model = Classifier()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, eval_losses, confusion_matrices = [], [], []

    for e in range(epochs):

        running_loss = 0
        running_eval_loss = 0
        n_total_train = 0
        c_matrix = np.zeros((N_CLASSES, N_CLASSES))

        for samples, labels in train:
            optimizer.zero_grad()
            log_ps = model(samples)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_total_train += len(labels)
        else:
            with torch.no_grad():
                model.eval()
                for samples, labels in test:
                    log_ps = model(samples)
                    eval_loss = criterion(log_ps, labels)
                    running_eval_loss += eval_loss.item()
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    c_matrix += confusion_matrix(labels.view(*top_class.shape).numpy(), top_class.numpy(),
                                                labels=range(N_CLASSES))
                model.train()

            train_losses.append(running_loss / n_total_train)
            eval_losses.append(running_eval_loss / n_total_train)
            confusion_matrices.append(c_matrix)

            print("Epoch: {}/{}.. ".format(e + 1, epochs),
                  "Training Loss: {:.4f}.. ".format(running_loss / n_total_train),
                  "Evaluation Loss: {:.4f}.. ".format(running_eval_loss / n_total_train),
                  "Evaluation Accuracy: {:.4f}..".format(c_matrix.trace() / c_matrix.sum()))

    return confusion_matrices, train_losses, eval_losses


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

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=config_shuffle)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=config_shuffle)

    conf_matrices, train_losses, eval_losses = train_model(train_loader, test_loader, epochs=config_epochs, lr=config_lr)
    reports = [classification_report(*cm2pred(cm), labels=range(N_CLASSES), output_dict=True) for cm in conf_matrices]

    fig, ax = plt.subplots(2, 2)
    fig.tight_layout(pad=2.5)
    fig.suptitle('MODEL STATS')

    # Plot loss
    ax[0, 0].set_title('Loss')
    ax[0, 0].set_xlabel('Epochs')
    ax[0, 0].set_ylabel('Loss')
    ax[0, 0].set_xticks(np.arange(start=0, stop=config_epochs, step=2))
    ax[0, 0].plot(train_losses, label='Training loss')
    ax[0, 0].plot(eval_losses, label='Validation loss')
    ax[0, 0].legend(frameon=False)

    # Plot confusion matrix
    labels = [str(i) for i in range(N_CLASSES)]
    df_cm = pd.DataFrame(conf_matrices[-1].astype(int), index=labels, columns=labels)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", linewidths=.5, cmap='Blues', cbar=False, ax=ax[1, 0])
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='center', fontsize=10)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
    ax[1, 0].set_ylabel('True label')
    ax[1, 0].set_xlabel('Predicted label')

    # Plot support
    ax[0, 1].bar(range(N_CLASSES), [reports[-1][l]['support'] for l in labels])
    ax[0, 1].set_xticks(range(N_CLASSES))
    ax[0, 1].set_xticklabels(labels)
    ax[0, 1].set_xlabel('Class')
    ax[0, 1].set_ylabel('Count')
    ax[0, 1].set_title('Support')

    # Plot scores
    precisions = [r['macro avg']['precision'] for r in reports]
    recalls = [r['macro avg']['recall'] for r in reports]
    f1_scores = [r['macro avg']['f1-score'] for r in reports]
    ax[1, 1].set_title('Scores')
    ax[1, 1].set_xlabel('Epochs')
    ax[1, 1].plot(precisions, label='Precision')
    ax[1, 1].plot(recalls, label='Recall')
    ax[1, 1].plot(f1_scores, label='F1')
    ax[1, 1].legend(frameon=False)



    # Print summary
    print(classification_report(*cm2pred(conf_matrices[-1]), labels=range(N_CLASSES), output_dict=False))
    plt.show()


