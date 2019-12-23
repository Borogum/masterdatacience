import argparse
import torch
import syft as sy
from single_model.datasets import MachineDataset
from single_model.workers import CustomWebsocketServerWorker

torch.manual_seed(0)

""" Starts a server worker """


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and evaluate a model')
    parser.add_argument('id', type=str, help='Server id')
    parser.add_argument('host', type=str, help='Server ip')
    parser.add_argument('port', type=int, help='Server port')
    parser.add_argument('train', type=str, help='Train data')
    parser.add_argument('test', type=str, help='Test data')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    hook = sy.TorchHook(torch)
    kwargs_websocket = {'id': args.id, 'hook': hook, 'host': args.host, 'port': args.port, 'verbose': args.verbose}
    train_dataset = MachineDataset(args.train)
    test_dataset = MachineDataset(args.test)
    server = CustomWebsocketServerWorker(**kwargs_websocket)
    train_dataset = sy.BaseDataset(data=train_dataset.data, targets=train_dataset.targets)
    server.add_dataset(train_dataset, key='train')
    test_dataset = sy.BaseDataset(data=test_dataset.data, targets=test_dataset.targets)
    server.add_dataset(test_dataset, key='test')
    server.start()
