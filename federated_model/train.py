import asyncio
import argparse
import configparser
import torch
import syft as sy
from syft.frameworks.torch.federated import utils
from single_model.model import Classifier, loss_fn
from single_model.workers import CustomWebsocketClientWorker

torch.manual_seed(0)


async def fit_model_on_worker(worker, traced_model, batch_size, epochs, curr_round, max_nr_batches, lr):
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
    loss = await worker.async_fit(dataset_key='train', return_ids=[0])
    model = train_config.model_ptr.get().obj
    return worker.id, model, loss


def evaluate_model_on_worker(model_identifier, worker, dataset_key, model, batch_size):

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
        return_raw_accuracy=True,
        return_confusion_matrix=True,
        example_inputs=torch.rand(1, 10),
    )

    print(result['loss'])

    # return result['lost'], result['confusion_matrix']

    # test_loss = result["loss"]
    # correct = result["nr_correct_predictions"]
    # len_dataset = result["nr_predictions"]
    # hist_pred = result["histogram_predictions"]
    # hist_target = result["histogram_target"]
    # TODO imprimir aqui los resultados


async def main():
    hook = sy.TorchHook(torch)
    parser = argparse.ArgumentParser(description='Run a federated_model model training')
    parser.add_argument('config', type=str, help='Configuration file')
    parser.add_argument('rounds', type=int, help='Rounds')
    parser.add_argument('batch_size', type=int, help='Train batch size')
    parser.add_argument('federate_after_n_epochs', type=int, help='Federate after n_epochs')
    parser.add_argument('lr', type=float, help='Learning rate')


    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    clients = {}
    for section in config.sections():
        kwargs_websocket = {'hook': hook, }
        kwargs_websocket['id'] = config_id = config.get(section, 'id')
        kwargs_websocket['host'] = config.get(section, 'host')
        kwargs_websocket['port'] = config.getint(section, 'port')
        kwargs_websocket['verbose'] = config.getboolean(section, 'verbose')
        client = CustomWebsocketClientWorker(**kwargs_websocket)
        client.clear_objects_remote()
        clients[kwargs_websocket['id']] = client

    model = Classifier()

    traced_model = torch.jit.trace(model, torch.zeros([1, 10], dtype=torch.float))
    learning_rate = args.lr

    for curr_round in range(1, args.rounds + 1):

        #logger.info("Training round %s/%s", curr_round, args.training_rounds)

        results = await asyncio.gather(
            *[
                fit_model_on_worker(
                    worker=clients[client],
                    traced_model=traced_model,
                    batch_size=args.batch_size,
                    epochs=args.federate_after_n_epochs,
                    curr_round=curr_round,
                    max_nr_batches=-1,
                    lr=learning_rate,
                )
                for client in clients
            ]
        )

        models = {}
        loss_values = {}

        test_models = curr_round % 10 == 1 or curr_round == args.rounds

        # Federate models (note that this will also change the model in models[0]
        for worker_id, worker_model, worker_loss in results:
            if worker_model is not None:
                models[worker_id] = worker_model
                loss_values[worker_id] = worker_loss

        traced_model = utils.federated_avg(models)

        if test_models:

            for client in clients:
                evaluate_model_on_worker(
                    model_identifier='Federated model',
                    worker=clients[client],
                    dataset_key='test',
                    model=traced_model,
                    batch_size=128,
                )


        # decay learning rate
        learning_rate = max(0.98 * learning_rate, args.lr * 0.01)


if __name__ == "__main__":
    # Logging setup
    # FORMAT = "%(asctime)s | %(message)s"
    # logging.basicConfig(format=FORMAT)
    # logger = logging.getLogger("run_websocket_client")
    # logger.setLevel(level=logging.DEBUG)

    # Websockets setup
    # websockets_logger = logging.getLogger("websockets")
    # websockets_logger.setLevel(logging.INFO)
    # websockets_logger.addHandler(logging.StreamHandler())

    # Run main
    asyncio.get_event_loop().run_until_complete(main())
