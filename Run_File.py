import pandas as pd
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.model_selection import train_test_split
import math as m
import argparse
from joblib import Parallel, delayed
from torch.func import jacrev, vmap, functional_call
import networkx as nx
from collections import deque
import seaborn as sns
from pathlib import Path
import h5py
import ast
from Plotter import Plotter
from Trial_Recorders_Model import MultiLayerNet, Trial
from DataProcessing import DataHandler






X_train_sorted = torch.linspace(-1,1,16).view(-1,1).type(torch.DoubleTensor)
Y_train_stretch = torch.multiply(X_train_sorted, 4*m.pi)
Y_linear_term = torch.multiply(X_train_sorted,4*m.pi)
# Y_train_sorted = torch.add(torch.sin(Y_train_stretch),Y_linear_term).squeeze()+5

Y_train_stretch_conjecture = torch.multiply(X_train_sorted, (4/3)*m.pi)

Y_train_sorted = torch.sin(Y_train_stretch_conjecture.squeeze())+1

slope_term = torch.multiply(X_train_sorted,4*m.pi).squeeze()
sin_term = torch.sin(Y_train_stretch_conjecture.squeeze())
Y_train_slope = torch.add(sin_term, slope_term)+1
#See if it learns the shift

X_numpy = X_train_sorted.numpy()
X_sq = X_numpy**2
print(f'<X^2> = {np.mean(X_sq)}')


X_eval = torch.linspace(-1, 1, 200).view(-1, 1).type(torch.DoubleTensor)


# Y_eval = torch.sin(torch.multiply(X_eval, 4*m.pi).squeeze())+1 +torch.multiply(X_eval,4*m.pi).squeeze()
Y_eval = torch.sin(torch.multiply(X_eval,(4/3)*m.pi))+1



def make_snapshot_epochs(total_epochs, lr, num_early, num_mid, num_end, early_cutoff, mid_cutoff):
        """
    Builds a set of snapshot epochs concentrated early (where dynamics are fast)
    and geometrically spread across the rest of training.
    early_time / mid_time are in TRAINING TIME (i.e., epoch*lr), not epochs.
    """
        early = np.linspace(0,early_cutoff, num_early) /lr
        mid = np.linspace(early_cutoff,mid_cutoff, num_mid) /lr
        late = np.linspace(mid_cutoff/lr,total_epochs-1, num_end)
        epochs = np.unique(np.concatenate([early, mid, late]).astype(int))
        return epochs[epochs < total_epochs]

def make_regions(bounds, steps):
    """
    bounds: list of boundary points, e.g. [0, 1, 20, 1000]
    steps: list of window widths for each segment between consecutive bounds,
            e.g. [0.1, 5.0, 98.0]  (one width per gap)
    Returns non-overlapping (start, end) tuples covering [bounds[0], bounds[-1]].
    """
    assert len(steps) == len(bounds) - 1
    windows = []
    for (t0, t1), w in zip(zip(bounds[:-1], bounds[1:]), steps):
        windows.append([(t, min(t + w, t1)) for t in np.arange(t0, t1, w)])
    return np.concatenate(windows, axis=0)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--OutputDir', type = str, help = 'Determines the file output of saved plots, data, etc', 
                        default =str(Path.home()/"Downloads"/"SummerWork"))
    parser.add_argument('--InputSize', type = int, help='Defines the size of the input node, almost always 1', default = 1)
    parser.add_argument('--OutputSize', type = int, help='Defines the size of the output node, almost always 1', default = 1)
    parser.add_argument('--Width', type = int, help='Defines how wide we want the hidden layers to be, i.e., how many nodes is ' \
    'the initial data mapped onto when going from initial --> HiddenLayer1', default = 10)
    parser.add_argument('--Depth', type = int, help='Defines how many hidden layers we want', default = 3)
    parser.add_argument('--lr', type = float, help='Determines the learning rate for the model', default = 0.01)
    parser.add_argument('--Epochs',type=int, help='Determines the number of training epochs', default = 10000)
    parser.add_argument('--STD',type=float, help='Determines the standard deviation (width) of the normal distribution for the hidden layers weights', default = 0.2)
    parser.add_argument('--EnsembleNum', type = int, help= ' Determines the number of models to create for the purposes of ensemble averages', default= 20)
    parser.add_argument('--Performances', type = int, help='Determines the number of printouts of model performance desired', default=4)
    parser.add_argument('--Bootstraps', type= int, help='Determines the number of bootstraps to calculate for error propagation', default=100)
    parser.add_argument('--AlignmentInterval', type = int, help='Determines how frequently to calculate the NTK alignment', default=10)
    parser.add_argument('--SaveFig', action='store_true', help='If set, saves figures to args.Filename')
    parser.add_argument('--ShowFig', action='store_true', help='If set, shows figures')
    parser.add_argument('--Filename', type = str, help='Determines the file to save data to', default= 'Unsorted')
    parser.add_argument('--Linear', type = bool, help = 'Determines whether to run as a linear model', default = False)
    parser.add_argument('--EvalAmount', type = int, help = 'Determines the number of eigenvalues/eigenvectors of interest', default = 6)
    parser.add_argument('--onnx_filename', type = str, help='Determines the filename for the onnx data', default= 'onnx_unsorted')
    parser.add_argument('--PhiTarget', type = float, help='Determines the target value for when the model is outside the linear regime', default=0.2)
    parser.add_argument('--PlateauEnd', type= int, help='When plateau is considered to be ended, data is recorded after this point', default=2000)
    parser.add_argument('--LossScans', type = int, help= 'Determines the number of loss points to include in the plateau region', default=10)
    parser.add_argument('--Activation', type = str, help='Determines the type of activation used', default= 'tanh')
    parser.add_argument('--Edgesize', type = int, help= 'Determines the edgesize of the model snapshot', default=1.0)
    parser.add_argument('--Nodesize', type = int, help= 'Determines the nodesize of the model snapshot', default=300.0)   
    args = parser.parse_args()


    output_directory = Path(args.OutputDir)/args.Filename
    output_directory.mkdir(parents = True, exist_ok = True)#Creates the folder if it does not already exist



    regions_list = make_regions(
        [0, 1, 10,200],
        [0.4, 3.0, 20]
    ).tolist()

    performances_array = make_snapshot_epochs(
        total_epochs=args.Epochs, lr=args.lr,
        num_early=5, num_mid=5, num_end=7,
        early_cutoff=1.0, mid_cutoff=10.0
    )




    model = MultiLayerNet(args.InputSize, args.Depth, args.Width,args.OutputSize,args.STD,args.Activation)
    model_fn = lambda: MultiLayerNet(args.InputSize, args.Depth, args.Width, args.OutputSize, args.STD, args.Activation).double()
    Trial_run = Trial(model_fn, args.InputSize, args.OutputSize,args.Width,args.Depth,args.lr,args.Epochs,args.STD,args.EnsembleNum,
                     args.AlignmentInterval,X_train_sorted,Y_train_sorted,X_eval,Y_eval,args.EvalAmount,performances_array,output_directory)
    Trial_run.save_initial_params(seed=1)
    Trial_run.train_ensemble()


    read_file = Path(output_directory)/"run_data.h5"
    param_file = Path(output_directory)/"simulation_parameters.csv"
    with h5py.File(name = read_file, mode = 'r') as f:
        saved_loss = np.array(f["loss/loss_dataset"])
        last_saved_performance = np.array(f["performance/performance_dataset"][:,-1,:])
        f.close()

    params_df = pd.read_csv(param_file)
    epochs = params_df["epochs"][0]

    epoch_axis = np.arange(epochs)
    
    lr = params_df["lr"]
    ensemble = params_df["ensemble"][0]
    test_true = params_df["test_truth"].apply(ast.literal_eval)[0]
    test_input = params_df["test_input"].apply(ast.literal_eval)[0]
    train_x = params_df["train_input"].apply(ast.literal_eval)[0]
    train_y = params_df["train_truth"].apply(ast.literal_eval)[0]
    #Plot loss test

    loss_avg = saved_loss.mean(axis = 0)
    loss_uncert = saved_loss.std(axis =0)/np.sqrt(ensemble)
    fig, ax = plt.subplots(figsize = (10,6))
    ax.plot(epoch_axis , loss_avg)
    for i in range(saved_loss.shape[0]):
         ax.plot(epoch_axis , saved_loss[i,:], linestyle = 'dashed', alpha =0.5)
    ax.fill_between(epoch_axis , loss_avg+loss_uncert, loss_avg-loss_uncert, alpha = 0.3)
    ax.set_xlabel("Epochs") 
    ax.set_ylabel("Loss")
    ax.set_title("Replica and Ensemble loss vs Epochs")
    plt.savefig(f"{Path(output_directory)}/Loss_test.png")
    plt.close()

    #Plotting last performance
    avg_perf = last_saved_performance.mean(axis = 0)
    perf_uncert = last_saved_performance.std(axis =0)/np.sqrt(ensemble)
    fig, ax = plt.subplots(figsize = (10,6))
    ax.plot(test_input, test_true, label = "True", linestyle = '-')
    ax.plot(test_input,avg_perf, label = "Model Output")
    ax.fill_between(test_input,avg_perf+perf_uncert,avg_perf-perf_uncert, alpha = 0.3)
    ax.scatter(train_x,train_y, label = "training points")
    plt.savefig(f"{Path(output_directory)}/LastPerformance.png")







    
    
