import os
import matplotlib.pyplot as plt
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
from pathlib import Path
import pickle
from sklearn.metrics import r2_score
import adabound
# My own functions
from models.linear_decoders import LinearRegression, RidgeRegression
from models.neighbors_decoders import kNNregression
from models.stochastic_decoders import BayesRidge
from models.kalman_decoders import KalmanFilter, UnscentedKalmanFilter
from models.boost_decoders import XGBoost, LightGBM, CatBoost
from models.deep_decoders import StackedRNN, TCN
# ===== Need this to run the code on cluster
import matplotlib
matplotlib.use('agg')
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
# ===== until here


# Keep this in the main file
def define_model(args):
    """A function that takes care of defining a model.

    Arguments:
        args:   argparse arguments. It should have args.decode_type.

    Returns:
        model:  the model class.

    """

    # Define the model and train
    # Separate between the models that require 2d input vs 3d input
    if args.decode_type in args.input_2d_decoders:
        if args.decode_type == "LR":
            model = LinearRegression(args)
        elif args.decode_type == "RR":
            model = RidgeRegression(args)
        elif args.decode_type == "KNN":
            model = kNNregression(args)
        elif args.decode_type == "BRR":
            model = BayesRidge(args)
        elif args.decode_type == "KF":
            model = KalmanFilter(args)
        elif args.decode_type == "UKF":
            model = UnscentedKalmanFilter(args)
        elif args.decode_type == "XGB":
            model = XGBoost(args)
        elif args.decode_type == "LGB":
            model = LightGBM(args)
        elif args.decode_type == "CB":
            model = CatBoost(args)
    # Mostly deep learning models
    elif args.decode_type in args.input_3d_decoders:
        # If GPU is available, making it cuda and using multiple GPUs
        if args.decode_type in args.rnn_decoders:
            model = StackedRNN(args)
        elif args.decode_type in args.cnn_decoders:
            model = TCN(args).cuda()
        # For GPU use
        model = model.cuda() if args.cuda else model

    return model


# ===== Optuna related functions =====
def optuna_optimizer(trial, model, args):
    """A function for optimizing optimizers using Optuna.

    Arguments:
        trial:  Optuna related. Details: https://optuna.readthedocs.io/en/latest/
        model:  a model class
        args:   argparse arguments.

    Returns:
        optimizer:  optimizer of choice. Optuna is define-by-run and it optimizes
                    for the optimizer with parameters if you define like below.

    """

    optimizer_names = ['adam', 'momentum', 'adabound']
    optimizer_name = trial.suggest_categorical('optim_type', optimizer_names)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-8, 1e-3)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)

    if optimizer_name == optimizer_names[0]:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay,
                                     amsgrad=True)
    elif optimizer_name == optimizer_names[1]:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=lr,
                                    momentum=0.9,
                                    weight_decay=weight_decay)
    elif optimizer_name == optimizer_names[2]:
        optimizer = adabound.AdaBound(model.parameters(),
                                      lr=lr,
                                      weight_decay=weight_decay)
    else:
        raise("The optimizer type not defined. Double check the configuration file.")

    return optimizer


def optuna_model_params(trial, args):
    """A function for optimizing various model parameters.

    Arguments:
        trial:  Optuna related. Details: https://optuna.readthedocs.io/en/latest/
        args:   argparse arguments.

    Returns:
        args:   argparse arguments.
        model:  a model class with hyperparameter that's going to be tuned using Optuna.

    """

    # RNN hyperparameters
    if args.decode_type in args.rnn_decoders:
        # Define what to tune regarding the model
        args.rnn_num_hidden = int(
            trial.suggest_discrete_uniform("rnn_num_hidden", 8, 64, 8))
        args.rnn_num_stack_layers = trial.suggest_int(
            "rnn_num_stack_layers", 1, 4)
        args.init_std = trial.suggest_loguniform("init_std", 0.001, 0.01)
    # CNN (TCN) hyperparameters
    elif args.decode_type in args.cnn_decoders:
        args.tcn_num_hidden = int(
            trial.suggest_discrete_uniform("tcn_num_hidden", 8, 64, 8))
        args.tcn_num_layers = trial.suggest_int("tcn_num_layers", 1, 4)
        args.tcn_kernel_size = trial.suggest_int("tcn_kernel_size", 5, 8)

    # Define the model with above hyperparameters
    model = define_model(args)
    # Use CUDA if GPU available
    model = model.cuda() if args.cuda else model

    return args, model


def objective(trial, args, loaders, save_model_path):
    """Objective function to be minimized.

    Arguments:
        trial:              Optuna related. Details: https://optuna.readthedocs.io/en/latest/
        args:               argparse arguments.
        loaders:            a data loader class from PyTorch.
        save_model_path:    where you want to save the model

    Returns:
        -r2_score:          Returning a negative R2 score to be minimized.
                            Optuna needs some values that needs to be minimized.
                            You can choose whatever metrics you want.

    """

    # Using optuna to tune some hyperparameters for model and optimizer
    args, model = optuna_model_params(trial, args)
    optimizer = optuna_optimizer(trial, model, args)

    # Choosing some other hyperparameters using optuna
    args.num_epochs = trial.suggest_int("num_epochs", 5, 30)
    if args.decode_type in args.rnn_decoders:
        args.optim_clip = trial.suggest_discrete_uniform(
            "optim_clip", 0.25, 1.5, 0.25)

    # Specify the type of loss criterion
    if args.loss_type == "MSE":
        criterion = nn.MSELoss()
    elif args.loss_type == "Huber":
        criterion = nn.SmoothL1Loss()

    # Iterate through epochs for training
    for e in range(args.num_epochs):
        print(f"===== Num epochs: {e} =====")
        # Initialize hidden
        if args.decode_type in args.rnn_decoders:
            hidden = model.init_hidden(bsize=args.batch_size)
        else:
            hidden = None
        for _, (data_t, data_v) in enumerate(zip(loaders['train'], loaders['valid'])):
            # Run train
            _ = train(args, model, data_t[0].cuda(),
                      data_t[1].cuda(), criterion, optimizer, [], hidden)
            # Run validation
            loss_v, pred = valid(args, model, data_v[0].cuda(),
                      data_v[1].cuda(), criterion, [], hidden)
            # Calculate other measurements for reporting:
            # R2 score
            r2_val = r2_score(data_v[1], pred.detach().cpu().numpy())
            # for reporting loss for pruning purpose
            trial.report(loss_v.item(), e)
            # if terminating in the middle because of the performance
            if trial.should_prune(e):
                raise optuna.structs.TrialPruned()

    # Save the model with trialID
    save_model = str(Path(save_model_path / str(trial.trial_id)))+".pth.tar"
    torch.save(model.state_dict(), save_model)

    return loss_v.item() # -r2_val  loss_v.item()


def objective_2d(trial, args, X_2d, Y, save_model_path):
    """Objective function for 2D input models. Return the error to be minimized.

    Similar to the above function, this is for 2D input models (Non Deep Learning).

    """

    # Optuna optimization
    if args.decode_type == "RR":
        args.rr_alpha = trial.suggest_uniform("rr_alpha", 0.1, 5.0)
    elif args.decode_type == "KF":
        args.kalman_lambda_F = trial.suggest_uniform(
            "kalman_lambda_F", 0.1, 5.0)
        args.kalman_lambda_B = trial.suggest_uniform(
            "kalman_lambda_B", 0.1, 5.0)
    elif args.decode_type == "UKF":
        args.kalman_lambda_F = trial.suggest_uniform(
            "kalman_lambda_F", 0.1, 5.0)
        args.kalman_lambda_B = trial.suggest_uniform(
            "kalman_lambda_B", 0.1, 5.0)
        args.ukf_kappa = trial.suggest_uniform("ukf_kappa", 0, 3)
    elif args.decode_type == "CB":
        args.cb_lr = trial.suggest_loguniform("cb_lr", 1e-4, 1e-1)
        args.cb_depth = trial.suggest_int("cb_depth", 4, 10)
        args.cb_l2 = trial.suggest_int("cb_l2", 1, 6)
    elif args.decode_type == "XGB":
        args.xgb_eta = trial.suggest_loguniform("xgb_eta", 1e-4, 1e-1)
        args.xgb_max_depth = trial.suggest_int("xgb_max_depth", 4, 10)
        args.xgb_min_child_weight = trial.suggest_int(
            "xgb_min_child_weight", 1, 6)
    elif args.decode_type == "LGB":
        args.lgb_learning_rate = trial.suggest_loguniform(
            "lgb_learning_rate", 1e-4, 1e-1)
        args.lgb_max_depth = trial.suggest_int("lgb_max_depth", 4, 10)
        args.lgb_num_leaves = trial.suggest_int(
            "lgb_num_leaves", 2, 2 ^ args.lgb_max_depth)
        args.lgb_min_data_in_leaf = trial.suggest_int(
            "lgb_min_data_in_leaf", 1, 6)

    # Define the model with the parameters
    model = define_model(args)

    # Train the model
    model.train(X_2d['train'], Y['train'])
    # Now start to validate
    if (args.decode_type == "KF") or (args.decode_type == "UKF"):
        prediction = model.process(X_2d['valid'], Y['valid'][0, :].T)
    else:
        prediction = model.predict(X_2d['valid'])

    # Save the model
    save_model = str(Path(save_model_path / str(trial.trial_id)))+".sav"
    with open(save_model, 'wb') as file_name:
        pickle.dump(model, file_name)

    # Calculate the validation r2 score
    r2_val = r2_score(Y['valid'], prediction)

    return -r2_val


def objective_fix(trial, args, loaders, save_model_path):
    """Objective function to be minimized.

    Fixing num_layers and num_hidden only optimizing other hyperparameters.

    """

    args.init_std = trial.suggest_loguniform("init_std", 0.001, 0.01)
    model = define_model(args)
    optimizer = optuna_optimizer(trial, model, args)

    args.num_epochs = trial.suggest_int("num_epochs", 5, 30)
    if args.decode_type in args.rnn_decoders:
        args.optim_clip = trial.suggest_discrete_uniform(
            "optim_clip", 0.25, 1.5, 0.25)

    if args.loss_type == "MSE":
        criterion = nn.MSELoss()
    elif args.loss_type == "Huber":
        criterion = nn.SmoothL1Loss()

    for e in range(args.num_epochs):
        print(f"===== Num epochs: {e} =====")
        # Initialize hidden
        if args.decode_type in args.rnn_decoders:
            hidden = model.init_hidden(bsize=args.batch_size)
        else:
            hidden = None
        for _, (data_t, data_v) in enumerate(zip(loaders['train'], loaders['valid'])):
            # Run train
            _ = train(args, model, data_t[0].cuda(), data_t[1].cuda(), criterion, optimizer, [], hidden)
            # Run validation
            loss_v, pred = valid(args, model, data_v[0].cuda(), data_v[1].cuda(), criterion, [], hidden)
            # Calculate other measurements for reporting:
            # R2 score
            r2_val = r2_score(data_v[1], pred.detach().cpu().numpy())
            # for reporting loss for pruning purpose
            trial.report(loss_v.item(), e)
            # if terminating in the middle because of the performance
            if trial.should_prune(e):
                raise optuna.structs.TrialPruned()

    # Save the model with trialID
    save_model = str(Path(save_model_path / str(trial.trial_id)))+".pth.tar"
    torch.save(model.state_dict(), save_model)

    return -r2_val  # loss_v.item()


# ===== Normal training related functions =====
def get_optimizer(args, model):
    """A function to define optimizer

    """

    if args.optim_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.optim_lr,
                               amsgrad=True, weight_decay=args.optim_weight_decay)
    elif args.optim_type == "adamax":
        optimizer = optim.Adamax(model.parameters(), lr=args.optim_lr)
    elif args.optim_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.optim_lr)
    elif args.optim_type == "momentum":
        optimizer = optim.SGD(model.parameters(),
                              lr=args.optim_lr, momentum=0.9)
    else:
        raise ValueError(
            "The optimizer name you specified, does not exist. Double check.")

    return optimizer


def train_wrapper(args, model, path_results, trialname, loaders):
    """A function wrapper for training, validation, plotting.

    """
    # Define the optimizer
    optimizer = get_optimizer(args, model)
    # Define the loss type
    if args.loss_type == "MSE":
        criterion = nn.MSELoss()
    elif args.loss_type == "Huber":
        criterion = nn.SmoothL1Loss()
    # later you could delete
    loss_list_tr = []
    loss_list_va = []
    for e in range(args.num_epochs):
        print(f"===== Num epochs: {e} =====")
        # Initialize hidden (only in RNN decoders)
        if args.decode_type in args.rnn_decoders:
            hidden = model.init_hidden(bsize=args.batch_size)
        else:
            hidden = None
        for i, (data_t, data_v) in enumerate(zip(loaders['train'], loaders['valid'])):
            # Run train
            loss_t = train(args, model, data_t[0].cuda(), data_t[1].cuda(),
                           criterion, optimizer, loss_list_tr, hidden)
            # Run validation
            loss_v, _ = valid(args, model, data_v[0].cuda(), data_v[1].cuda(),
                              criterion, loss_list_va, hidden)
            # for log on print
            if i % (args.batch_size*2) == 0:
                print(
                    f"Training loss: {loss_t.item():.3f}, Validation loss: {loss_v.item():.3f}")

    # visualize the loss
    N = 30
    plot_fig(N, loss_list_tr, loss_list_va, path_results, trialname)


def train(args, model, inputs, target, criterion, optimizer, loss_track, hidden=None):
    """A function to train the model.

    Supposed to be called every iteration on batch.

    """

    # Make sure it's in training mode
    model.train()
    model.zero_grad()
    # Clear previous gradients
    optimizer.zero_grad()
    # Make prediction
    if args.decode_type in args.rnn_decoders:
        out, _ = model(inputs, hidden)
    else:
        out = model(inputs)
    # Calculate the loss
    loss = criterion(out, target)
    # Compute gradients of all variables wrt loss
    loss.backward()
    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    nn.utils.clip_grad_norm_(model.parameters(), args.optim_clip)
    # performs updates using calculated gradients
    optimizer.step()
    # tracking the loss
    loss_track.append(loss.item())

    return loss


def valid(args, model, inputs, target, criterion, loss_track, hidden=None):
    """A function to validate the model.

    Supposed to be called every iteration on batch processing.
    """

    # Validating
    model.eval()
    # Predict
    if args.decode_type in args.rnn_decoders:
        out, _ = model(inputs, hidden)
    else:
        out = model(inputs)
    # Calculate the loss
    loss = criterion(out, target)
    loss_track.append(loss.item())

    return loss, out


def test(model, testX, rnn_flag=False):
    """A function to test the tuned model.
    """

    # Change the model to evaluation mode
    model.eval()
    with torch.no_grad():
        if rnn_flag:
            hidden = model.init_hidden(bsize=testX.shape[0])
        # somehow you need this .float() for CUDADoubleTensor vs CUDAFloatTensor RunTimeError
        testX_ = torch.from_numpy(testX).float().cuda()
        if rnn_flag:
            tmp, _ = model(testX_, hidden)
        else:
            tmp = model(testX_)
        # Extracting the last hidden
        # Since I'm not doing batch first, the 2nd part is hidden where I need the last hidden output
        prediction = tmp.cpu()

    return prediction


def plot_fig(N, loss_list_tr, loss_list_va, path_results, trialname):
    """For visualizing the loss.

    Instead of using tensorboardX, I like this one better.
    Don't need this when doing hyper-parameter optimization.

    """

    # running average
    N = 30
    avg_loss_tr = np.convolve(np.array(loss_list_tr),
                              np.ones((N,))/N, mode='valid')
    avg_loss_va = np.convolve(np.array(loss_list_va),
                              np.ones((N,))/N, mode='valid')

    fig = plt.figure()
    plt.plot(avg_loss_tr)
    plt.plot(avg_loss_va)
    plt.legend(['train', 'validation'])
    plt.savefig(str(path_results)+"/loss_"+trialname +
                ".png", format="png", dpi=300)
    plt.close()
