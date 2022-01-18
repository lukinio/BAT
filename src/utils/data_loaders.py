import os
import numpy as np
import pandas as pd
import pickle
import logging

from .featurization import load_data_from_df, construct_loader, load_data_from_smiles, load_data_from_mol_file


logger = logging.getLogger('my_logger')


def get_bbbp(path, batch_size, scaffolds):
    splits = []
    data_x, data_y = load_data_from_df(f'{path}/bbbp.csv', one_hot_formal_charge=True)
    data_x, data_y = np.array(data_x, dtype=object), np.array(data_y, dtype=object)
    for i in scaffolds:
        split = np.load(f'{path}/split-scaffold-{i}.npy', allow_pickle=True)
        data_loaders = []
        for j in range(3):
            x, y = data_x[split[j]], data_y[split[j]]
            data_loaders.append(construct_loader(x.tolist(), y.tolist(), batch_size))
        splits.append(data_loaders)

    d_atom = data_x[0][0].shape[1]
    return d_atom, splits[0]


def get_bace_from_smiles(dataset_path, batch_size, add_dummy_node=True, one_hot_formal_charge=True, use_data_saving=True):
    data = pd.read_csv(dataset_path)
    data_loaders = []

    for name in ("train", "valid", "test"):
        feat_stamp = f'_{name}{"_dn" if add_dummy_node else ""}{"_ohfc" if one_hot_formal_charge else ""}'
        feature_path = dataset_path.replace('.csv', f'{feat_stamp}.p')

        if use_data_saving and os.path.exists(feature_path):
            logging.info(f"Loading features stored at '{feature_path}'")
            x_all, y_all = pickle.load(open(feature_path, "rb"))
            d_atom = x_all[0][0].shape[1]
            data_loaders.append(construct_loader(x_all, y_all, batch_size))
        else:
            tmp = data[data["scaffold_split"] == name]
            tmp_x, tmp_y = load_data_from_smiles(tmp["smiles"], tmp["class"], add_dummy_node=add_dummy_node,
                                                 one_hot_formal_charge=one_hot_formal_charge)
            d_atom = tmp_x[0][0].shape[1]

            data_loaders.append(construct_loader(tmp_x, tmp_y, batch_size))

            if use_data_saving and not os.path.exists(feature_path):
                logging.info(f"Saving features at '{feature_path}'")
                pickle.dump((tmp_x, tmp_y), open(feature_path, "wb"))

    return d_atom, data_loaders


def get_bace_from_mol(dataset_path, batch_size, add_dummy_node=True, one_hot_formal_charge=True, use_data_saving=True):
    csv_path = dataset_path + "bace_docked.csv"
    poses_path = dataset_path + "bace_poses/"
    data = pd.read_csv(csv_path)
    data_loaders = []

    for name in ("train", "valid", "test"):
        feat_stamp = f'_{name}{"_dn_mol" if add_dummy_node else ""}{"_ohfc_mol" if one_hot_formal_charge else ""}'
        feature_path = csv_path.replace('.csv', f'{feat_stamp}.p')

        if use_data_saving and os.path.exists(feature_path):
            logging.info(f"Loading features stored at '{feature_path}'")
            x_all, y_all = pickle.load(open(feature_path, "rb"))
            d_atom = x_all[0][0].shape[1]
            data_loaders.append(construct_loader(x_all, y_all, batch_size))
        else:
            tmp = data[data["scaffold_split"] == name]
            tmp_x, tmp_y = load_data_from_mol_file(poses_path, tmp["id"], tmp["class"], add_dummy_node=add_dummy_node,
                                                   one_hot_formal_charge=one_hot_formal_charge)
            d_atom = tmp_x[0][0].shape[1]
            data_loaders.append(construct_loader(tmp_x, tmp_y, batch_size))

            if use_data_saving and not os.path.exists(feature_path):
                logging.info(f"Saving features at '{feature_path}'")
                pickle.dump((tmp_x, tmp_y), open(feature_path, "wb"))

    return d_atom, data_loaders
