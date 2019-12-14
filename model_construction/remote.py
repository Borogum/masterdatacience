import time
import signal
import argparse
import subprocess
import configparser
import torch
import torch.nn.functional as F
import syft as sy
from model_construction.common import CustomWebsocketClientWorker, plot_results
from model_construction.model import Classifier

torch.manual_seed(0)

# Loss function
@torch.jit.script
def loss_fn(pred, target):
    return F.nll_loss(input=pred, target=target)


def train_model():
    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train model remotely')
    parser.add_argument('config', type=str, help='Configuration file')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    # Train configuration
    config_epochs = config.getint('TRAIN', 'epochs')
    config_batch = config.getint('TRAIN', 'batch')
    config_lr = config.getfloat('TRAIN', 'lr')
    config_shuffle = config.getboolean('TRAIN', 'shuffle')
    config_train = config.get('TRAIN', 'train_file')
    config_test = config.get('TRAIN', 'test_file')

    # Server configuration
    config_id = config.get('SERVER', 'id')
    config_host = config.get('SERVER', 'host')
    config_port = config.get('SERVER', 'port')

    config_verbose = config.getboolean('SERVER', 'verbose')

    # Start server
    command = ['python',
               'server.py',
               config_id,
               config_host,
               config_port,
               config_train,
               config_test,
               ]

    if config_verbose:
        command.append('--verbose')

    print('Starting server ... ', end='')
    p = subprocess.Popen(command)

    # Wait for server up
    time.sleep(2)

    print('Done!')

    # Train and validate

    hook = sy.TorchHook(torch)
    kwargs_websocket = {'hook': hook, 'id': config_id, 'host': config_host, 'port': config_port,
                        'verbose': config_verbose}
    worker = CustomWebsocketClientWorker(**kwargs_websocket)
    traced_model = torch.jit.trace(Classifier(),  torch.rand(1, 10))

    train_losses = []
    test_losses = []
    confusion_matrices = []

    for epoch in range(config_epochs):
        # Train
        traced_model.train()

        train_config = sy.TrainConfig(
            model=traced_model,
            loss_fn=loss_fn,
            batch_size=config_batch,
            shuffle=config_shuffle,
            max_nr_batches=-1,
            epochs=1,
            optimizer="Adam",
            optimizer_args={"lr": config_lr},
        )

        worker.clear_objects_remote()
        train_config.send(worker)
        loss = worker.fit(dataset_key="train")
        traced_model = train_config.model_ptr.get().obj

        print(traced_model.fc1.weight.data,'<')

        # Evaluate

        traced_model.eval()

        #Train set
        train_config = sy.TrainConfig(
            batch_size=config_batch,
            model=traced_model,
            loss_fn=loss_fn,
            optimizer_args=None,
        )
        train_config.send(worker)
        result = worker.evaluate(
            dataset_key='train',
            return_histograms=False,
            return_loss=True,
            return_raw_accuracy=False,
            return_confusion_matrix=False,
            example_inputs=None,
        )
        train_losses.append(result['loss'])

        # Validate set
        train_config = sy.TrainConfig(
            batch_size=config_batch,
            model=traced_model,
            loss_fn=loss_fn,
            optimizer_args=None,
        )
        train_config.send(worker)
        result = worker.evaluate(
            dataset_key='test',
            return_histograms=False,
            return_loss=True,
            return_raw_accuracy=False,
            return_confusion_matrix=True,
            example_inputs=torch.rand(1, 10),
        )
        test_losses.append(result['loss'])
        confusion_matrices.append(result['confusion_matrix'])

        print(traced_model.fc1.weight.data, '<')

    #plot_results([m.numpy() for m in confusion_matrices], train_losses, test_losses)

    print(train_losses, test_losses)

    # Stop server
    p.send_signal(signal.CTRL_BREAK_EVENT)

    # Waiting for server to end
    while p.poll() is None:
        time.sleep(.5)
