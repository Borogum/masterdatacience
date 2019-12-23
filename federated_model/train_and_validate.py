import asyncio
import argparse
import configparser
import torch
from torch.jit import trace
import syft as sy
from syft.frameworks.torch.federated import utils
from single_model.model import Classifier, loss_fn
from single_model.workers import CustomWebsocketClientWorker
from single_model.utils import show_results

torch.manual_seed(0)


async def fit_model_on_worker(worker, traced_model, optimizer, batch_size, epochs, lr, dataset_key, shuffle):
    """ Fit model on a worker """

    print('Training on "%s" ...' % (worker.id,))
    # Create and send train config
    train_config = sy.TrainConfig(
        model=traced_model,
        loss_fn=loss_fn,
        batch_size=batch_size,
        shuffle=shuffle,
        epochs=epochs,
        optimizer=optimizer,
        optimizer_args={'lr': lr},
    )
    train_config.send(worker)
    await worker.async_fit(dataset_key=dataset_key, return_ids=[0])
    model = train_config.model_ptr.get().obj
    return worker.id, model


def evaluate_model_on_worker(worker, dataset_key, model, batch_size):
    """ Evaluate a model on worker"""

    # Create and send train config
    train_config = sy.TrainConfig(
        batch_size=batch_size, model=model, loss_fn=loss_fn, optimizer_args={}, epochs=1
    )
    train_config.send(worker)
    result = worker.evaluate(
        dataset_key=dataset_key,
        return_histograms=False,
        return_loss=True,
        return_raw_accuracy=False,
        return_confusion_matrix=True,
        example_inputs=torch.rand(1, 10),
    )
    return result['loss'], result['confusion_matrix']


async def main():
    """ Main """

    hook = sy.TorchHook(torch)
    parser = argparse.ArgumentParser(description='Train and validate a Federated model')
    parser.add_argument('config', type=str, help='Configuration file')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    # Train configuration

    config_rounds = config.getint('TRAIN', 'rounds')
    config_epochs = config.getint('TRAIN', 'epochs')
    config_batch = config.getint('TRAIN', 'batch')
    config_optimizer = config.get('TRAIN', 'optimizer')
    config_lr = config.getfloat('TRAIN', 'lr')
    config_shuffle = config.getboolean('TRAIN', 'shuffle')

    clients = {}
    clients_results = {}

    for section in config.sections():
        if section.startswith('WORKER'):
            kwargs_websocket = {'hook': hook, 'id': config.get(section, 'id'), 'host': config.get(section, 'host'),
                                'port': config.getint(section, 'port'),
                                'verbose': config.getboolean(section, 'verbose')}
            federation_participant = config.getboolean(section, 'federation_participant')
            client = CustomWebsocketClientWorker(**kwargs_websocket)
            client.federation_participant = federation_participant
            client.clear_objects_remote()
            clients[kwargs_websocket['id']] = client
            clients_results[kwargs_websocket['id']] = []

    model = Classifier()
    traced_model = trace(model, torch.zeros([1, 10], dtype=torch.float))

    for curr_round in range(config_rounds):

        print('Round %s/%s ¡Ding Ding!:' % (curr_round + 1, config_rounds))

        results = await asyncio.gather(
            *[
                fit_model_on_worker(
                    worker=clients[client],
                    traced_model=traced_model,
                    optimizer=config_optimizer,
                    batch_size=config_batch,
                    epochs=config_epochs,
                    lr=config_lr,
                    dataset_key='test',
                    shuffle=config_shuffle
                )
                for client in clients if clients[client].federation_participant
            ]
        )

        print('Training done!')

        print('Federating model ... ', end='')
        models = {}
        for worker_id, worker_model in results:
            if worker_model is not None:
                models[worker_id] = worker_model
        traced_model = utils.federated_avg(models)
        print('Done!')

        for client in clients:
            # Evaluate train
            train_loss, train_confusion_matrix = evaluate_model_on_worker(
                worker=clients[client],
                dataset_key='train',
                model=traced_model,
                batch_size=config_batch,
            )
            # Evaluate test
            test_loss, test_confusion_matrix = evaluate_model_on_worker(
                worker=clients[client],
                dataset_key='test',
                model=traced_model,
                batch_size=config_batch,
            )

            clients_results[client].append((train_loss, test_loss, test_confusion_matrix))
            print('"%s" => Train loss: %.4f. Test loss: %.4f' % (client, train_loss, test_loss))

    print('Confusion matrices:')

    for client in clients_results:
        print('Model "%s" stats:' % client)
        train_losses = [cr[0] for cr in clients_results[client]]
        test_losses = [cr[1] for cr in clients_results[client]]
        conf_matrices = [cr[2] for cr in clients_results[client]]
        show_results(conf_matrices, train_losses, test_losses, label=client, loss_xlabel='Round')


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
