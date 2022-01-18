"""
How to run
python run.py --dataset BACE --exp_name EXPERIMENT_NAME ...
"""

import os
import sys
import argparse
import logging
from logging.handlers import RotatingFileHandler

from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.utils import load_pre_trained_weights
from models.transformer import make_model

from utils.train_and_test import train_and_valid, test
from utils.plots import plot_history
from utils.utils import EarlyStopping, update_dict, mean_over_run

from utils.data_loaders import get_bbbp, get_bace_from_smiles, get_bace_from_mol


def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=str, default="0")
    parser.add_argument('--dataset', type=str, default='BACE', help="BACE(default)|BBBP")
    parser.add_argument('--mol_files', default=False, action='store_true', help="read dataset from mol files")
    parser.add_argument('--save_model', default=False, action='store_true', help="delete models")
    parser.add_argument('--exp_name', type=str, default="test")
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, default='ADAM', help="ADAM(default)|SGD")
    parser.add_argument('--scaffold_id', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--grad_clip', type=float, default=0.0, help="if grad_clip equal 0 (zero) gradient clipping "
                                                                     "is disabled otherwise gradient is clipping to "
                                                                     "given value")
    parser.add_argument('--patience', type=int, default=50, help="EarlyStopping patience")

    # Model parameters
    parser.add_argument('--d_model', type=int, default=1024)
    parser.add_argument('--N', type=int, default=8)
    parser.add_argument('--h', type=int, default=16)
    parser.add_argument('--N_dense', type=int, default=1)
    parser.add_argument('--lambda_attention', type=float, default=0.33)
    parser.add_argument('--lambda_distance', type=float, default=0.33)
    parser.add_argument('--leaky_relu_slope', type=float, default=0.1)
    parser.add_argument('--dense_output_nonlinearity', type=str, default='relu', help="relu(default)|tanh|none")
    parser.add_argument('--distance_matrix_kernel', type=str, default='exp', help="exp(default)|softmax")
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--aggregation_type', type=str, default='mean', help="mean(default)|sum|dummy_node")

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

    store_metrics = {
        "BEST": {"accuracy": [], "precision": [], "recall": [], "F1": [], "AUC": []},
        "LAST": {"accuracy": [], "precision": [], "recall": [], "F1": [], "AUC": []}
    }
    for r in range(1, args.repeat+1):

        exp_path = f"../experiments/{args.dataset}/{args.exp_name}/repeat_{str(r)}/"
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
            handlers=[logging.StreamHandler()]
        )
        logger = logging.getLogger('my_logger')
        logger.addHandler(handler)

        # DATASET
        if args.dataset == "BACE":
            if args.mol_files:
                d_atom, data_loaders = get_bace_from_mol('../data/bace_docked/', args.batch_size)
            else:
                d_atom, data_loaders = get_bace_from_smiles('../data/bace_docked/bace_docked.csv', args.batch_size)
        elif args.dataset == "BBBP":
            d_atom, data_loaders = get_bbbp('../data/bbbp', args.batch_size, scaffolds=[args.scaffold_id])
        else:
            raise ValueError('No such dataset')

        train_loader, valid_loader, test_loader = data_loaders

        model_params = {
            'd_atom': d_atom,
            'd_model': args.d_model,
            'N': args.N,
            'h': args.h,
            'N_dense': args.N_dense,
            'lambda_attention': args.lambda_attention,
            'lambda_distance': args.lambda_distance,
            'leaky_relu_slope': args.leaky_relu_slope,
            'dense_output_nonlinearity': args.dense_output_nonlinearity,
            'distance_matrix_kernel': args.distance_matrix_kernel,
            'dropout': args.dropout,
            'aggregation_type': args.aggregation_type
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
        scheduler = ReduceLROnPlateau(optimizer, mode='max', min_lr=1e-7, patience=10, verbose=True)
        early_stop = EarlyStopping(logger, "max", patience=args.patience)

        history = train_and_valid(logger, train_loader, valid_loader, model, loss_fn, optimizer,
                                  exp_path=exp_path, exp_name=args.exp_name, scheduler=scheduler,
                                  epochs=args.epochs, early_stop=early_stop, grad_clip=args.grad_clip)
        plot_history(history, exp_path=exp_path)

        # TEST
        for s in ("BEST", "LAST"):
            logger.info("\n"+s)
            model_weights = exp_path + args.exp_name + f"_{s.lower()}.pt"
            metrics = test(test_loader, model, loss_fn, model_weights=model_weights)
            logger.info(f"TEST - {metrics.show}")
            update_dict(metrics, store_metrics[s])
            mean_over_run(logger, store_metrics[s])
            if not args.save_model:
                os.remove(model_weights)
                logger.info(f"DELETE: {model_weights}")

        logger.removeHandler(handler)
    logging.shutdown()

