import torch
from torch.utils.data import DataLoader, TensorDataset
import pdb


def torch_dataloaders(args, dsets, X, Y):
    """A wrapper function.

    Modifies the datasets to torch.cuda if possible and creates dataloaders.
    """

    X_torch, Y_torch = create_torch_data(args, dsets, X, Y)
    loaders = create_dataloaders(args, X_torch, Y_torch)

    return loaders


def create_dataloaders(args, X_torch, Y_torch):
    """A function to create dataloaders for PyTorch.

    """

    loaders = {}
    # You only need train and valid dataloaders
    dsets = ('train', 'valid')
    for d in dsets:
        loaders[d] = DataLoader(TensorDataset(X_torch[d], Y_torch[d]),
                                args.batch_size, shuffle=True, drop_last=True)

    return loaders


def create_torch_data(args, dsets, X, Y):
    """A function to create dataset for PyTorch.

    """

    X_torch, Y_torch = {}, {}
    for d in dsets:
        X_torch[d] = torch.from_numpy(X[d]).float()
        Y_torch[d] = torch.from_numpy(Y[d]).float()

    return X_torch, Y_torch
