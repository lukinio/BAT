"""
How to run
python run.py --dataset BACE --exp_name EXPERIMENT_NAME ...
"""

import os
import sys
import argparse
import logging
from logging.handlers import RotatingFileHandler

import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.utils import load_pre_trained_weights
from models.transformer import make_model

from utils.train_and_test import train_and_valid, test
from utils.plots import plot_history
from utils.utils import EarlyStopping

from utils.data_loaders import get_bbbp, get_bace


def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=str, default="0")
    parser.add_argument('--dataset', type=str, default='BACE', help="BACE(default)|BBBP")
    parser.add_argument('--exp_name', type=str, default="test")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, default='ADAM', help="ADAM(default)|SGD")
    parser.add_argument('--scaffold_id', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--grad_clip', type=float, default=0.0)
    parser.add_argument('--patience', type=int, default=50, help="EarlyStopping patience")

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

    exp_path = f"../experiments/{args.dataset}/{args.exp_name}/"
    os.makedirs(exp_path, exist_ok=True)
    # LOGGER
    logfile = exp_path + args.exp_name + ".log"
    should_roll_over = os.path.isfile(logfile)
    handler = RotatingFileHandler(logfile, mode='w', backupCount=0)
    if should_roll_over:
        handler.doRollover()
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            handler,
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('my_logger')

    # DATASET
    if args.dataset == "BACE":
        d_atom, data_loaders = get_bace('../data/bace_docked/bace_docked.csv', args.batch_size)
    elif args.dataset == "BBBP":
        d_atom, data_loaders = get_bbbp('../data/bbbp', 64, scaffolds=[args.scaffold_id])
    else:
        raise ValueError('No such dataset')

    train_loader, valid_loader, test_loader = data_loaders

    model_params = {
        'd_atom': d_atom,
        'd_model': 1024,
        'N': 8,
        'h': 16,
        'N_dense': 1,
        'lambda_attention': 0.33,
        'lambda_distance': 0.33,
        'leaky_relu_slope': 0.1,
        'dense_output_nonlinearity': 'relu',
        'distance_matrix_kernel': 'exp',
        'dropout': 0.0,
        'aggregation_type': 'mean'
    }

    model = make_model(**model_params)
    load_pre_trained_weights(model, '../pretrained/pretrained_weights.pt')

    if args.optimizer == "ADAM":
        optimizer = Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "SGD":
        optimizer = SGD(model.parameters(), lr=args.lr)
    else:
        raise ValueError("Lack of implementation for this optimizer")

    loss_fn = BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', min_lr=1e-8, patience=10, verbose=True)
    early_stop = EarlyStopping("max", patience=args.patience)

    history = train_and_valid(train_loader, valid_loader, model, loss_fn, optimizer,
                              exp_path=exp_path, exp_name=args.exp_name,
                              scheduler=scheduler, epochs=args.epochs, early_stop=early_stop, grad_clip=args.grad_clip)
    plot_history(history, exp_path=exp_path)

    # TEST
    logger.info("BEST")
    test(test_loader, model, loss_fn, model_weights=exp_path+args.exp_name+"_best.pt")
    logger.info("LAST")
    test(test_loader, model, loss_fn, model_weights=exp_path+args.exp_name+"_last.pt")








