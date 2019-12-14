import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from syft.workers.websocket_server import WebsocketServerWorker
from syft.workers.websocket_client import WebsocketClientWorker
from sklearn.metrics import classification_report

torch.manual_seed(0)

plt.style.use('seaborn')


def cm2pred(x):
    """ Helper method that creates y_true, y_pred based on confusion matrix"""

    y_true = []
    y_pred = []

    for i in range(len(x)):
        for j in range(len(x)):
            for k in range(int(x[i][j])):
                y_true.append(i)
                y_pred.append(j)
    return y_true, y_pred


def hist2cm(x):
    """ Helper method that transforms a histogram into a confusion matrix"""
    pass


def plot_results(conf_matrices, train_losses, eval_losses):
    n_classes = conf_matrices[0].shape[0]
    reports = [classification_report(*cm2pred(cm), labels=range(n_classes), output_dict=True, zero_division=0) for cm in
               conf_matrices]  # Zero division equals 0 to avoid warnings when class not found

    fig, ax = plt.subplots(2, 2)
    fig.tight_layout(pad=2.5)
    fig.suptitle('MODEL STATS')

    # Plot loss
    ax[0, 0].set_title('Loss')
    ax[0, 0].set_xlabel('Epochs')
    ax[0, 0].set_ylabel('Loss')
    ax[0, 0].set_xticks(np.arange(start=0, stop=len(train_losses), step=2))
    ax[0, 0].plot(train_losses, label='Training loss')
    ax[0, 0].plot(eval_losses, label='Validation loss')
    ax[0, 0].legend(frameon=False)

    # Plot confusion matrix
    labels = [str(i) for i in range(n_classes)]
    df_cm = pd.DataFrame(conf_matrices[-1], index=labels, columns=labels).astype(int)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", linewidths=.5, cmap='Blues', cbar=False, ax=ax[1, 0])
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='center', fontsize=10)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
    ax[1, 0].set_ylabel('True label')
    ax[1, 0].set_xlabel('Predicted label')

    # Plot support
    ax[0, 1].bar(range(n_classes), [reports[-1][l]['support'] for l in labels])
    ax[0, 1].set_xticks(range(n_classes))
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
    print(classification_report(*cm2pred(conf_matrices[-1]), labels=range(n_classes), output_dict=False))
    plt.show()


class MachineMaintenanceDataset(Dataset):

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file, sep=',', decimal='.')
        self.data = torch.from_numpy(self.df[self.df.columns[:-1]].values).float()
        self.targets = torch.from_numpy(self.df[self.df.columns[-1]].values).long()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.df.iloc[idx, :-1].astype('float')
        y = self.df.iloc[idx, -1].astype('int')
        return torch.FloatTensor(x), torch.tensor(y, dtype=torch.long)


class CustomWebsocketServerWorker(WebsocketServerWorker):

    def evaluate(
            self,
            dataset_key: str,
            return_histograms: bool = False,
            nr_bins: int = -1,
            return_loss: bool = True,
            return_raw_accuracy: bool = True,
            device: str = "cpu",
            return_confusion_matrix: bool = False,
            example_inputs=None
    ):
        """Evaluates a model on the local dataset as specified in the local TrainConfig object.
        Args:
            dataset_key: Identifier of the local dataset that shall be used for training.
            return_histograms: If True, calculate the histograms of predicted classes.
            nr_bins: Used together with calculate_histograms. Provide the number of classes/bins.
            return_loss: If True, loss is calculated additionally.
            return_raw_accuracy: If True, return nr_correct_predictions and nr_predictions
        Returns:
            Dictionary containing depending on the provided flags:
                * loss: avg loss on data set, None if not calculated.
                * nr_correct_predictions: number of correct predictions.
                * nr_predictions: total number of predictions.
                * histogram_predictions: histogram of predictions.
                * histogram_target: histogram of target values in the dataset.
        """
        self._check_train_config()

        if dataset_key not in self.datasets:
            raise ValueError("Dataset {} unknown.".format(dataset_key))

        eval_result = dict()
        model = self.get_obj(self.train_config._model_id).obj
        loss_fn = self.get_obj(self.train_config._loss_fn_id).obj

        model.eval()
        device = "cuda" if device == "cuda" else "cpu"
        data_loader = self._create_data_loader(dataset_key=dataset_key, shuffle=False)
        test_loss = 0.0
        correct = 0
        if return_confusion_matrix:
            example_outputs = model(example_inputs)
            n_classes = example_outputs.shape[1]
            cmatrix = torch.zeros(n_classes, n_classes)

        if return_histograms:
            hist_target = np.zeros(nr_bins)
            hist_pred = np.zeros(nr_bins)

        with torch.no_grad():
            for data, target in data_loader:

                data, target = data.to(device), target.to(device)
                output = model(data)
                if return_loss:
                    test_loss += loss_fn(output, target).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                if return_histograms:
                    hist, _ = np.histogram(target, bins=nr_bins, range=(0, nr_bins))
                    hist_target += hist
                    hist, _ = np.histogram(pred, bins=nr_bins, range=(0, nr_bins))
                    hist_pred += hist
                if return_raw_accuracy:
                    correct += pred.eq(target.view_as(pred)).sum().item()
                if return_confusion_matrix:
                    for y_true, y_pred in zip(target, pred):
                        cmatrix[y_true][y_pred] += 1

        if return_loss:
            test_loss /= len(data_loader.dataset)
            eval_result["loss"] = test_loss
        if return_raw_accuracy:
            eval_result["nr_correct_predictions"] = correct
            eval_result["nr_predictions"] = len(data_loader.dataset)
        if return_histograms:
            eval_result["histogram_predictions"] = hist_pred
            eval_result["histogram_target"] = hist_target
        if return_confusion_matrix:
            eval_result['confusion_matrix'] = cmatrix

        return eval_result


class CustomWebsocketClientWorker(WebsocketClientWorker):

    def evaluate(
            self,
            dataset_key: str,
            return_histograms: bool = False,
            nr_bins: int = -1,
            return_loss=True,
            return_raw_accuracy: bool = True,
            return_confusion_matrix: bool = False,
            example_inputs=None,

    ):
        """Call the evaluate() method on the remote worker (WebsocketServerWorker instance).
        Args:
            dataset_key: Identifier of the local dataset that shall be used for training.
            return_histograms: If True, calculate the histograms of predicted classes.
            nr_bins: Used together with calculate_histograms. Provide the number of classes/bins.
            return_loss: If True, loss is calculated additionally.
            return_raw_accuracy: If True, return nr_correct_predictions and nr_predictions
        Returns:
            Dictionary containing depending on the provided flags:
                * loss: avg loss on data set, None if not calculated.
                * nr_correct_predictions: number of correct predictions.
                * nr_predictions: total number of predictions.
                * histogram_predictions: histogram of predictions.
                * histogram_target: histogram of target values in the dataset.
        """

        return self._send_msg_and_deserialize(
            "evaluate",
            dataset_key=dataset_key,
            return_histograms=return_histograms,
            nr_bins=nr_bins,
            return_loss=return_loss,
            return_raw_accuracy=return_raw_accuracy,
            return_confusion_matrix=return_confusion_matrix,
            example_inputs=example_inputs,
        )
