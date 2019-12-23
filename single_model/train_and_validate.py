import time
import argparse
import subprocess
import configparser
import torch
from torch.jit import trace
import syft as sy
from single_model.utils import show_results
from single_model.workers import CustomWebsocketClientWorker
from single_model.model import Classifier, loss_fn

torch.manual_seed(0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and validate the model')
    parser.add_argument('config', type=str, help='Configuration file')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    # Train configuration
    config_epochs = config.getint('CONFIGURATION', 'epochs')
    config_batch = config.getint('CONFIGURATION', 'batch')
    config_optimizer = config.get('CONFIGURATION', 'optimizer')
    config_lr = config.getfloat('CONFIGURATION', 'lr')
    config_shuffle = config.getboolean('CONFIGURATION', 'shuffle')
    config_train = config.get('CONFIGURATION', 'train')
    config_test = config.get('CONFIGURATION', 'test')

    # Worker configuration
    config_id = config.get('CONFIGURATION', 'worker_id')
    config_host = config.get('CONFIGURATION', 'host')
    config_port = config.get('CONFIGURATION', 'port')
    config_verbose = config.getboolean('CONFIGURATION', 'verbose')

    # Start worker
    command = ['python',
               'start_worker.py',
               config_id,
               config_host,
               config_port,
               config_train,
               config_test,
               ]

    if config_verbose:
        command.append('--verbose')

    print('Starting worker ... ', end='')
    p = subprocess.Popen(command, stdout=subprocess.DEVNULL)
    time.sleep(2)  # Wait for worker
    print('Done! (PID: %s)' % p.pid)

    print('Training and validating ...')
    hook = sy.TorchHook(torch)
    kwargs_websocket = {'hook': hook, 'id': config_id, 'host': config_host, 'port': config_port,
                        'verbose': config_verbose}
    worker = CustomWebsocketClientWorker(**kwargs_websocket)
    worker.clear_objects_remote()
    traced_model = trace(Classifier(),  torch.rand(1, 10))

    train_losses = []
    test_losses = []
    confusion_matrices = []

    for epoch in range(config_epochs):

        traced_model.train()

        train_config = sy.TrainConfig(
            model=traced_model,
            loss_fn=loss_fn,
            batch_size=config_batch,
            shuffle=config_shuffle,
            max_nr_batches=-1,
            epochs=1,
            optimizer=config_optimizer,
            optimizer_args={'lr': config_lr},
        )

        worker.clear_objects_remote()
        train_config.send(worker)
        loss = worker.fit(dataset_key='train')
        traced_model = train_config.model_ptr.get().obj

        # Evaluation
        traced_model.eval()

        # Evaluate on train set
        train_config = sy.TrainConfig(
            batch_size=config_batch,
            model=traced_model,
            loss_fn=loss_fn,
            optimizer_args={},
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

        # Evaluate in test set
        train_config = sy.TrainConfig(
            batch_size=config_batch,
            model=traced_model,
            loss_fn=loss_fn,
            optimizer_args={},
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

        print('Epoch %d. Train loss: %.4f. Test loss: %.4f' % (epoch+1, train_losses[-1], test_losses[-1]))

    print('Done!')

    # Show stats
    show_results([m.numpy() for m in confusion_matrices], train_losses, test_losses)

    # Kill to worker subprocess
    p.kill()

    # Waiting for worker to stop
    while p.poll() is None:
        time.sleep(1)
