import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
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


def show_results(conf_matrices, train_losses, eval_losses):
    n_classes = conf_matrices[0].shape[0]
    reports = [classification_report(*cm2pred(cm), labels=range(n_classes), output_dict=True, zero_division=0) for cm in
               conf_matrices]  # Zero division equals 0 to avoid warnings when a class not found (but doesn't work :D)

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
    print(
        classification_report(*cm2pred(conf_matrices[-1]), labels=range(n_classes), output_dict=False, zero_division=0))
    plt.show()
