import torch
import numpy as np
from syft.workers.websocket_server import WebsocketServerWorker
from syft.workers.websocket_client import WebsocketClientWorker

""" Change original classes to get custom stats """


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
            return_raw_accuracy: If True, return nr_correct_predictions and nr_predictions.
            device: "cpu" or "cuda"
            return_confusion_matrix: If True, return confusion matrix.
            example_inputs: Example of input of the model, required if return_confusion_matrix is True.

        Returns:
            Dictionary containing depending on the provided flags:
                * loss: avg loss on data set, None if not calculated.
                * nr_correct_predictions: number of correct predictions.
                * nr_predictions: total number of predictions.
                * histogram_predictions: histogram of predictions.
                * histogram_target: histogram of target values in the dataset.
                * confusion matrix: confusion matrix
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
            return_confusion_matrix: If True, return confusion matrix.
            example_inputs: Example of input of the model, required if return_confusion_matrix is True.
        Returns:
            Dictionary containing depending on the provided flags:
                * loss: avg loss on data set, None if not calculated.
                * nr_correct_predictions: number of correct predictions.
                * nr_predictions: total number of predictions.
                * histogram_predictions: histogram of predictions.
                * histogram_target: histogram of target values in the dataset.
                * confusion matrix: confusion matrix
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
