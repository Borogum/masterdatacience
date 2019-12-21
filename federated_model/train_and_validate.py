import asyncio
import argparse
import configparser
import torch
import syft as sy
from syft.frameworks.torch.federated import utils
from single_model.model import Classifier, loss_fn
from single_model.workers import CustomWebsocketClientWorker
from single_model.utils import show_results

torch.manual_seed(0)


async def fit_model_on_worker(worker, traced_model, batch_size, epochs, max_nr_batches, lr):

    print('-> Training on "%s" ...' % (worker.id,))
    train_config = sy.TrainConfig(
        model=traced_model,
        loss_fn=loss_fn,
        batch_size=batch_size,
        shuffle=True,
        max_nr_batches=max_nr_batches,
        epochs=epochs,
        optimizer="Adam",
        optimizer_args={"lr": lr},
    )
    train_config.send(worker)
    await worker.async_fit(dataset_key='train', return_ids=[0])
    model = train_config.model_ptr.get().obj
    return worker.id, model


def evaluate_model_on_worker( worker, dataset_key, model, batch_size):

    model.eval()
    # Create and send train config
    train_config = sy.TrainConfig(
        batch_size=batch_size, model=model, loss_fn=loss_fn, optimizer_args=None, epochs=1
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

    hook = sy.TorchHook(torch)
    parser = argparse.ArgumentParser(description='Run a federated_model model training')
    parser.add_argument('workers', type=str, help='Configuration file')
    parser.add_argument('rounds', type=int, help='Rounds')
    parser.add_argument('batch_size', type=int, help='Train batch size')
    parser.add_argument('federate_after_n_epochs', type=int, help='Federate after n_epochs')
    parser.add_argument('lr', type=float, help='Learning rate')

    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.workers)

    clients = {}
    clients_results = {}

    for section in config.sections():
        kwargs_websocket = {'hook': hook, }
        kwargs_websocket['id'] = config_id = config.get(section, 'id')
        kwargs_websocket['host'] = config.get(section, 'host')
        kwargs_websocket['port'] = config.getint(section, 'port')
        kwargs_websocket['verbose'] = config.getboolean(section, 'verbose')
        federation_participant = config.getboolean(section, 'federation_participant')
        client = CustomWebsocketClientWorker(**kwargs_websocket)
        client.federation_participant = federation_participant
        client.clear_objects_remote()
        clients[kwargs_websocket['id']] = client
        clients_results[kwargs_websocket['id']] = []

    model = Classifier()
    traced_model = torch.jit.trace(model, torch.zeros([1, 10], dtype=torch.float))

    for curr_round in range(1, args.rounds + 1):

        print('Round %s/%s ¡Ding Ding!:' % (curr_round, args.rounds))

        results = await asyncio.gather(
            *[
                fit_model_on_worker(
                    worker=clients[client],
                    traced_model=traced_model,
                    batch_size=args.batch_size,
                    epochs=args.federate_after_n_epochs,
                    max_nr_batches=-1,
                    lr=args.lr,
                )
                for client in clients if clients[client].federation_participant
            ]
        )
        print('All done!')

        print('Federating ... ', end='')
        models = {}
        # Federate models
        for worker_id, worker_model in results:
            if worker_model is not None:
                models[worker_id] = worker_model
        traced_model = utils.federated_avg(models)
        print('Federated!')

        print('Stats:')
        for client in clients:

            train_loss, train_confusion_matrix = evaluate_model_on_worker(
                worker=clients[client],
                dataset_key='train',
                model=traced_model,
                batch_size=args.batch_size,
            )

            test_loss, test_confusion_matrix = evaluate_model_on_worker(
                worker=clients[client],
                dataset_key='test',
                model=traced_model,
                batch_size=args.batch_size,
            )

            clients_results[client].append((train_loss, test_loss,  test_confusion_matrix))
            print('* Round %d. "%s" stats ==> Train loss: %.4f. Test loss: %.4f' % (curr_round, client, train_loss,
                                                                                  test_loss))

    print('Confusion matrices:')

    for client in clients_results:
        print(client)
        train_losses = [cr[0] for cr in clients_results[client]]
        test_losses = [cr[1] for cr in clients_results[client]]
        conf_matrices= [cr[2] for cr in clients_results[client]]
        show_results(conf_matrices, train_losses, test_losses, label=client, loss_xlabel='Round')


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
