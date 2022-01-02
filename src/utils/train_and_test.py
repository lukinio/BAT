import logging

import torch
import torch.nn as nn

from .metrics import MetricsMeter
from .utils import EarlyStopping


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger('my_logger')


def _epoch(loader, model, loss_fn, optimizer=None, grad_clip: float = 0.):
    """Standard training/evaluation epoch over the dataset"""
    if optimizer is not None:
        phase_name, phase = "train", torch.enable_grad()
        model.train()
    else:
        phase_name, phase = "test", torch.no_grad()
        model.eval()

    metrics = MetricsMeter()
    with phase:
        for k, (adjacency_matrix, node_features, distance_matrix, y) in enumerate(loader, 1):
            adjacency_matrix, y = adjacency_matrix.to(device), y.to(device)
            node_features, distance_matrix = node_features.to(device), distance_matrix.to(device)
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
            outputs = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None)
            loss = loss_fn(outputs, y)
            if phase_name == "train":
                optimizer.zero_grad()
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            # Update metrics
            metrics.update(torch.sigmoid(outputs).round().detach(), y.detach(), loss.item())

    return metrics


def train_and_valid(train_loader, valid_loader, model, loss_fn, optimizer, exp_path, exp_name, scheduler=None,
                    epochs=30, early_stop: EarlyStopping = None, grad_clip=1, save_model=True):
    logs = {
        'loss': {"train": [], "valid": []},
        'accuracy': {"train": [], "valid": []},
        'AUC': {"train": [], "valid": []}
    }
    model = model.to(device)

    for e in range(1, epochs + 1):
        train_metrics = _epoch(train_loader, model, loss_fn, optimizer, grad_clip=grad_clip)
        valid_metrics = _epoch(valid_loader, model, loss_fn, optimizer=None)

        tmp = valid_metrics.auc

        if scheduler is not None:
            scheduler.step(tmp)

        out = "Epoch: {} {} {}"
        logger.info(out.format(e, "TRAIN", train_metrics.show))
        logger.info(out.format(e, "VALID", valid_metrics.show) + "\n")

        # Store logs
        logs['loss']["train"].append(train_metrics.loss)
        logs['loss']["valid"].append(valid_metrics.loss)
        logs['accuracy']["train"].append(train_metrics.accuracy)
        logs['accuracy']["valid"].append(valid_metrics.accuracy)
        logs['AUC']["train"].append(train_metrics.auc)
        logs['AUC']["valid"].append(valid_metrics.auc)

        # Early stopping
        if early_stop is not None:
            if early_stop.has_improved(tmp) and save_model:
                torch.save(model.state_dict(), exp_path + exp_name + "_best.pt")
            elif early_stop.should_stop:
                break

    if save_model:
        logger.info("Save model from last epoch")
        torch.save(model.state_dict(), exp_path + exp_name + "_last.pt")

    return logs


def test(test_loader, model, loss_fn, model_weights=None):
    model = model.to(device)
    if model_weights is not None:
        model.load_state_dict(torch.load(model_weights, map_location=device))
    test_metrics = _epoch(test_loader, model, loss_fn, optimizer=None)
    logger.info(f"TEST - {test_metrics.show}")
