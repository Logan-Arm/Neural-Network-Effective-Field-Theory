import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math as m
from joblib import Parallel, delayed
from torch.func import jacrev, vmap, functional_call
import networkx as nx
from collections import deque
import logging

logger = logging.getLogger(__name__)

logging.basicConfig(level = logging.INFO)

def NTK_calc(model, X):
    params = {k: v.detach() for k,v in model.named_parameters()}
    #model.named_parameters() returns all the learnable tensors within the model, i.e., each layers weight and bias tensor
    #model.named... returns a tuple of shape (name, tensor), k is then attached to the name, and v gets attached to the actual tensor
    #v.detach ensures that the v (tensor) we store is not changed by any autograds, in essence we are decoupling it from the learning procedure

    def fnet_single(params, x):
        """
    Above lets us work in a functionally "pure" space, see june 12th notes in word document for more details
    Input model describes the architecture and forward pass procedure of our network, however, parameter space values are overwritten with the values provided
    Input params, gives the functional call all the information known about the parameter tensors of each layer of the network
    Functional call requires the data be provided in a tuple form, hence the x.unsqueeze(0), which just adds a batch dimension 
    .squeeze(0) removes this instant added dimensionality so that the data can be passed as a scalar    
        """
        return functional_call(model, params, (x.unsqueeze(0),)).squeeze(0)
    J = vmap(jacrev(fnet_single),(None,0))(params,X)
   
    """Vmap vectorises/parallelises the function/calculation being called, in this case jacrev of fnet_single
    jacrev calls the jacobian with respect to the inputs first argument, which in the case of fnet_single is params (our weights and biases)
    The (None,0) tells vmap which dimensionality to parallelise along, as seen by the function inputs i.e., the (params,X)
    None describes parallelisation of the first input of jacrev(fnet_single), i.e., the parameters
    Since we want the parameters to stay the same (as is the goal of calculating the jacobian) we set this to none
    We do however want to vectorise the X calculations, so we say we vectorise along the 0th dimension of the X inputs, i.e., row by row
    (params,X) give us the inputs to feed into the fnet_single

    The output of this, for a model with N data points is the following
    layer_0_weight = [N,10,1]
    layer_0_bias = [N,10]
    layer_1_weight = [N,10,10]
    layer_1_bias = [N,10]
    """

    #We now need to flatten these arrays
    J_flat = torch.cat([j.flatten(1) for j in J.values()], dim=1)
    """
    J.values() simply grabs one of the rows described previously (end of previous docstring)

    j.flatten(1) collapses all dimensionality beyond 1, --> everything to 2 dimensionals
    layer_0_weight --> [N,10] as 10X1 =10
    layer_1_weight --> [N,100] as 10X10 =100

    concatanating joins all the different gradients along the parameter space, i.e., for layer 1,2, etc

    
    """
    return J_flat @ J_flat.T #Final matrix multiplication to provide the NTK
    #Explicitly provides grad_theta_f (xi) dot grad_theta_f (xj), which is the definition of the NTK 

"""Creating a function to remove all hooks from the model"""
def remove_hooks(hooks):
    #Added the len(hooks) !=0 so this function can be called when no hooks are attached (i.e., for initial NTK)
    if len(hooks) !=0:
        for hook in hooks:
            hook.remove()
        hooks.clear()
"""Now defining a loss function"""

def criterion(y_pred, y_true):
    """We use 1/2 mean squared loss as the loss function in most NTK theory papers, the 1/2 is important to simplify arithmetic further down the line,
     although ultimately as this is "experimental", it is not strictly required. """
    #mean squared loss is simply (predicted-actual)^2
    return 0.5* torch.mean((y_pred-y_true)**2)





class MultiLayerNet(nn.Module): #nn.module is the base class for all neural networks in pytorch
    """The outline of the MLP which will be passed to the trial class.  The MLP is designed with input and output layers of size specified by the function
    dependencies, with (num_layers) hidden layers of width (width).  The std passed is the standard deviation of the initial weights and biases (when combined with
    nn.init.normal_).  One can pass a string to the activation function, in this case either 'tanh' or 'linear', which corresponds to the activation function desired."""

    def __init__(self, input_size, num_layers,width, output_size, std, activation):
        super().__init__()#This runs the __init__ from the parent class, i.e., nn.module which is necessary to initialize correctly



        Activations = {
            'tanh':torch.tanh,
            'linear': lambda x:x,
        }
        #Recall lambda is pythons version of an unnamed function, so we are just saying if you call linear, use a function lambda, which takes in x, and outputs x

        if callable(activation):
            self.activation = activation
            """If the inputted activation function is itself a callable function, simply use that"""
        else:
            self.activation = Activations[activation]
            """Otherwise, find the corresponding entry in the Activations list above"""

        """Since we have a variable number of hidden layers, it is best to create a list in itialisation"""


        self.hidden_layers = nn.ModuleList()
        #It should be noted I include the initial layer in this 'hidden layer' list, this is not strictly clear, and is something to be aware of

        #Create the initial layer
        self.hidden_layers.append(nn.Linear(input_size,width))

        for i in range(num_layers-1): 
            self.hidden_layers.append(nn.Linear(width,width))

        self.output_layer = nn.Linear(width, output_size)
        """What above does
        Maps initial values onto the first hidden layer Init--> Hidden
        Creates all other hidden layers
        Maps hidden to output hidden --> Output
        """
        self.layer_widths = []
        for layer in self.hidden_layers:
            nn.init.normal_(layer.weight, mean = 0, std = std)
            nn.init.normal_(layer.bias,mean = 0, std= std)
            self.layer_widths.append(layer.out_features)#Appending widths of the layer
        """IMPORTANT Note regarding layer modification  the _ at the end of each nn.init.shape creates an IN PLACE change, so we are actually editing the layers"""
    

        ###Output layer is not included in self.hidden_layers so need to handle that one externally
        nn.init.normal_(self.output_layer.weight,mean = 0, std = std)
        nn.init.normal_(self.output_layer.bias, mean = 0, std= std)
        self.layer_widths.append(self.output_layer.out_features) #appending the last layer to the widths array

    def forward(self,x):
        """passes input through the hidden layer applying specified activation.  Note, it is important here that the activation is itself outside of the layer,
        certain pytorch functions lets you combine the two; which while most of the time is rather convenient, here would be very bad.  Since we are oftentimes 
        looking at the pre-activations themselves (much easier to track than activated values), separating the layer identity from what the forward pass allows
        us to leverge pytorch hooks later on, to make tracking pre-activations much easier."""
        #
        
        for layer in self.hidden_layers:
            x= self.activation(layer(x))
        y_pred = self.output_layer(x)
        """What above line is doing
        
        Calling the layer as a function and passing x through it

        X is fed into hidden layer
        Hidden layer configuration applies the corresponding weights and biases ... (X * weights)+ bias
        Tanh activation is applied
        We are then using a tanh activation function to modify this final data
        """

        return y_pred
    

class LossRecorder():
    """In loop helper class for the trial class, meant to record the loss values and the epoch of recording"""
    def __init__(self,record_start = 0, loss_points = None, convergence_criteria = 0.01):
        self.record_start = record_start
        self.loss_points = loss_points

        self.convergence_criteria = convergence_criteria
        
        self.inst_loss = [] #Track the loss for one specific replica
        self.recorded_epochs = []
        self.ensemble_loss = []
        self.convergence_tracker = []

    def loss_track_in_loop(self,epoch,loss,j):
        if (epoch >= self.record_start) or (self.loss_points is not None and epoch in self.loss_points):
            self.inst_loss.append(loss)
        if j ==0:
            self.recorded_epochs.append(epoch)

    def loss_end_of_loop(self):
        """Append the replica's loss to the set, and then clear the instant loss to be used further later"""
        self.ensemble_loss.append(self.inst_loss)

        #convergence check
        converged = any(loss <= self.convergence_criteria for loss in self.inst_loss)
        self.convergence_tracker.append(converged)
        
        self.inst_loss = []
    def to_numpy(self):
        self.ensemble_loss = np.array(self.ensemble_loss)
    
class PlateauRecorder():
    """In loop tracker of the loss shape, identifies when the replica's begin their plateau phase.  A key note is that this records raw epochs, not training time,
    any comparison with inverse eigenvalues etc must therefore be modified by the learning rate."""
    def __init__(self, start_thresh = 1e-3, end_thresh = 0.005, window = 0.5, lr = 0.01, iter_thresh = 3):
        """
        Observes the shape of the slope of the loss in a specified range (loss window), if this slope is above or below set values, then the replica is 
        considered to be in various stages of the plateau.
        
        Plateau start and end thresh are the threshold values that the slope must meet/exceed/be under (depending on what it is we are 
        looking for at given moment), to determine the plateau status.  The window is the window of data to be looking at in order to observe the plateau,
        it is to be given in training times, and was set to 0.5 during experiment. 
         
        There is no need to record specific epochs where data was collected, since we are only observing data already collected in the loss recorder

        Independent analysis of the per replica slopes confirm that at plateau end the magnitude of the slope windows is non-universal, and subject to significant 
        variation on the per replica basis (see plotter.plot_slopes()).  As such, thresholds for plateau start and end cannot be the same value, as it would lead to 
        decrease in fidelity amongst one of the values.

        
        Current start and end thresholds have been found experimentally for replicas initialised with a std of weight and bias distribution of 0.2

        """

        self.plat_start_thresh = start_thresh
        self.plat_end_thresh = end_thresh
        self.iter_thresh =iter_thresh
        # self.window = window

        self.plateau_bounds = [] #will append this with lists of shape [plateau_epoch_start, plateau_epoch_end]

        #loss values are recorded very epoch, so there is no need to consider any alignment int etc
        self.max_len = max(2,int(round(window/lr)))

        self.update_lr = lambda : None #making a dummy function similar to how the hooks work in the other recorders

        
        self.loss_deck = deque(maxlen=self.max_len) #Create a deque of size of window in epochs to store the slopes at given points
        self.epoch_deck = deque(maxlen=self.max_len)

        dummy_list = np.arange(self.max_len) *lr
        self.x_centered = dummy_list - np.mean(dummy_list)
        self.denom = np.sum(self.x_centered**2)

        self.epochOI = None

        self.plat_start = None
        self.plat_end = None

        self.plateau_iter = 0

    def track_in_loop(self,epoch,loss):
        self.loss_deck.append(loss)
        self.epoch_deck.append(epoch)
        if len(self.loss_deck) == self.max_len: #If the list has been filled to the appropriate window size, can begin the analysis of shape
            self.plat_analysis()
        else:
            pass

    def plat_analysis(self):
        """Convert the losses to logarithms, then finds lsr line of best fit.  The slope of this is the compared against the set thresholds to determine 
        what stage of activity the model is in"""
        # log_loss = np.log(self.loss_deck)
        log_loss = np.log(np.clip(self.loss_deck,1e-12,None)) #Need to clip to avoid any numerical issues
        slope  =np.sum(self.x_centered*log_loss)/self.denom

        if self.plat_start is None: #check if we are now in plateau
            if np.abs(slope) < self.plat_start_thresh:
                if self.epochOI is None:
                    self.epochOI = self.epoch_deck[0] #set the epoch of interest to be the first epoch value in the current slope window
                    self.plateau_iter+=1
                else:
                    self.plateau_iter+=1

                if self.plateau_iter==self.iter_thresh:
                    self.plat_start = self.epochOI #append the epochOI
                    self.epochOI = None #Reset epochOI
                    self.plateau_iter = 0 #Reset plateau iter

##############################################################################################
                    #This has been commented out since we are no longer looking to increase lr in plateau region
                    # self.update_lr() #Call the update lr function


##############################################################################################                    
            else:#reset the iterator and the epochOI
                self.plateau_iter = 0
                self.epochOI = None           

        if (self.plat_start is not None) & (self.plat_end is None):
            if np.abs(slope)>= self.plat_end_thresh:
                if self.epochOI is None:
                    self.epochOI = self.epoch_deck[0]
                    self.plateau_iter+=1
                else:
                    self.plateau_iter+=1

                if self.plateau_iter==self.iter_thresh:
                    self.plat_end = self.epochOI #append the epochOI
                    self.epochOI = None #Reset epochOI
            else:#reset the iterator and the epochOI
                self.plateau_iter = 0
                self.epochOI = None           

    def end_of_loop(self):
        if self.plat_start is None:
            self.plat_start = np.nan

        if self.plat_end is None:
            self.plat_end = np.nan

        zipped =[self.plat_start,self.plat_end]
        self.plateau_bounds.append(zipped)

        #Reset the recorded data
        self.plat_start = None
        self.plat_end = None
        self.epochOI = None

        self.plateau_iter = 0

        #Reset the decks
        self.epoch_deck.clear()
        self.loss_deck.clear()

    def convert_to_numpy(self):
        self.plateau_bounds = np.array(self.plateau_bounds)
        #Dim is [replica, 2], second dimensionality is list of [plat_start, plat_end]

    def bind_update_lr_funct(self, update_lr_fn = None):
        """Function to bind the update lr function which will be passed from the trial class into the PlateauRecorder"""
        self.update_lr = update_lr_fn

class NTKRecorder():
    """All calculated vectors should be based off the ensemble average NTK, not necessarily the ensemble average of replica's data"""

    def __init__(self, alignmentint, e_amount = 6, record_start = 0):
        self.alignmentint = alignmentint
        self.e_amount = e_amount
        self.record_start = record_start

        self.recorded_epochs = []

        

        self.add_hooks_fn = lambda:None

        self.remove_hooks_fn =lambda:None
        
        self.inst_NTK=[]
        self.inst_output = []

        self.initial_NTK = []

        self.initial_evals = []

        self.ensemble_output = []
        self.ensemble_NTK=[]
        



    def bind_hook_fn(self,add_hooks_fn =None, remove_hooks_fn = None):
        self.add_hooks_fn = add_hooks_fn
        self.remove_hooks_fn = remove_hooks_fn

    def NTK_track_in_loop(self, model, train_data, target_train_data, epoch, j, norm_const): #NOTE TO SELF, THERE IS REDUNDANCY IN CODE AS A RESULT OF IMPLEMENTING RESIDUALS CALCULATIONS   
     
        if epoch ==0:
            self.remove_hooks_fn()
            NTK_matrix = NTK_calc(model,train_data).numpy()           
            self.initial_NTK.append(NTK_matrix)

            """Adding collection of initial eigenvalues"""
            evals = list(reversed(np.linalg.eigvalsh(NTK_matrix))) #eigvalsh returns eigenvalues in ascending order, reverse so we get it in descending
            assert evals[0]> evals[1], "eval 2 is greater than eval 1"
            self.initial_evals.append(evals)
            """End of initial eigenvalue collection"""

            self.add_hooks_fn()


        if (epoch >= self.record_start) and (epoch%self.alignmentint ==0):
            #Need to remove hooks
            self.remove_hooks_fn()
            if j==0:
                self.recorded_epochs.append(epoch)
            NTK_matrix = NTK_calc(model,train_data).numpy()
            self.inst_NTK.append(NTK_matrix)
            self.capture_output(model,train_data)        
            self.add_hooks_fn()


    def capture_output(self, model, train_data):
        model.eval()
        with torch.no_grad():

            ###############################################
            predictions = model(train_data).squeeze()#Think the squeeze is necessary here, most torch tensors add the extra dimensionality for no reason
            ############################################### 
        self.inst_output.append(predictions)
    

    def NTK_end_of_loop(self):
        self.ensemble_output.append(self.inst_output)
        self.ensemble_NTK.append(self.inst_NTK)
        
        self.inst_output = []
        self.inst_NTK = []

    def NTK_conversion(self):
        """Calculates the rotation of the eigenvectors as well as converts everything to a numpy array"""

        self.initial_NTK = np.array(self.initial_NTK)

        self.ensemble_NTK = np.array(self.ensemble_NTK)
        #[replica, epoch, NTK]

        self.ensemble_output = np.array(self.ensemble_output)

        self.initial_evals = np.array(self.initial_evals)
                  
class ParameterRecorder():
    """Class meant to view and store all the parameters of neurons in a replica
    Note, shares the same alignmentint as the NTK recorder, thus can just use NTK recorded epochs for any further calculations/plots
    Model.hidden_layer_list stores the following layers : input layer, 1st, 2nd and 3rd hidden layers, does not store hidden layers
    
    The data collected by this helper function was originally meant to be used to identify subnetworks within the larger MLP on a per-replica basis.
        """
    def __init__(self, numlayer, alignmentint):
        self.ensemble_weights = {i:[] for i in range(numlayer-1) }
        #Similar to the phi array, dictionary which will store the weights per layer
        self.ensemble_biases = {i:[] for i in range(numlayer-1) }


        self.replica_weights = {i:[] for i in range(numlayer-1) }
        self.replica_bias = {i:[] for i in range(numlayer-1) }

        self.inst_weights = []
        self.inst_biases = []


        self.alignmentint = alignmentint
        self.numlayer = numlayer-1 
        #Will be used in for loops later to make sure we have layer information
        #Note: the output layer is currently (July 30th) NOT in the model.hidden_layer list, as such it will be treated separately (also why numlayer-1)

    def track_in_loop(self,epoch,model):
        """Updates the inst lists with values at corresponding epoch using torch built in functions layer.weights, layer.biases
        .weights returns a [n_l,n_(l-1)] dimensional array, where n_(l-1) is the number of neurons in the previous layer, and n_l is number of neurons in layer of interest
        for instance, in our case if we took layer2.weights, we would get a [10,10] array since 10 neurons in layer 1, and 10 neurons in layer 2.  
        
        Example: layer2.weights[i,j], j is the current layer's weight matrix applied to previous layer's neuron m's activated outputs, and i corresponds 
        to a specific neuron in the current layer.  So layer2.weights[i,j] is layer2's ith neuron's weight applied to the activated output of layer1's j neuron

        The input and output neuron are special cases.  The input size (what gets fed into the initial layer) is determined during construction of the model, and is 
        defined by the second dimension of the training dataset.  I.e., in our case, since we have 16 points (scalars), the second dimension is 1, and thus we have 1
        input size --> layer1.weights ~ [1,1]  
        
        Similarly the output neuron would be of shape [1,width(layer_(L-1))]

        Alternatively, since biases are a flat additive term, the output is simply the width of the layer 
        
        
        """
        if epoch%self.alignmentint ==0:
            for i in range(self.numlayer):
                #Starting at 1 since input layer is not of interest
                layer_of_int = model.hidden_layers[i]
                logger.debug("layer of interest has %d input features and %d output features",
                             layer_of_int.in_features, layer_of_int.out_features)
                #Want above to print [1,10],[10,10],[10,10]
                self.replica_weights[i].append(layer_of_int.weight.clone().detach())
                """Pytorch .weight tensors have grad information baked into them, this presents problems when converting to a numpy array
                To get around this, detach the weight tensor, this copies the shape and data, without keeping the connection to the grad

                Clone is necessary here, this is what gives us the raw values of the weights, truthfully I am not sure what .detach() fully does.  I know it
                disconnects from the autograd, but why that is insufficient (and still needs the clone) I do not know.
                """
                self.replica_bias[i].append(layer_of_int.bias.clone().detach())

    def end_of_loop(self):
        for i in range(len(self.ensemble_weights)):
            self.ensemble_weights[i].append(self.replica_weights[i])
            self.ensemble_biases[i].append(self.replica_bias[i])

        #Reset the inst dictionaries
        self.replica_weights = {i:[] for i in range(self.numlayer) }
        self.replica_bias = {i:[] for i in range(self.numlayer) }

    def convert_to_numpy(self):
        """Dimensionality of the ensemble weights array is as following:
        [layer][replica][epoch][weight tensor (see comment under the track in loop function)]"""
        for i in range(len(self.ensemble_weights)):
            self.ensemble_weights[i] = np.array(self.ensemble_weights[i])
            self.ensemble_biases[i] =np.array(self.ensemble_biases[i])
            #Biases I expect to be of shape [layer][replica][epoch][width]

class PerformanceRecorder():
    """Helper function which records the model's output (performance) at various times (epochs also recorded) """
    def __init__(self,X_test, performance_times,epochs,performance_count = 10):

        if performance_times is None:
            """If performance times are not specified, take them on a log basis
            Broken up into two zones to hopefully capture the details more explicitly, as it is the learning does not fit any one equation type
            """
            cutoff =int(epochs*(2/3))

            snapshots_initial = np.unique(np.geomspace(1, cutoff, int(performance_count/2)).astype(int))
            snapshots_final = np.unique(np.geomspace(cutoff, epochs-1, int(performance_count/2)).astype(int))

            self.snapshot_array = np.unique(np.concat(snapshots_initial,snapshots_final))



        else:
            self.snapshot_array = performance_times

        logger.debug("snapshot array is %s ",self.snapshot_array)

        self.recorded_epochs = []
        self.performances = []
        self.inst_performs = []

        self.X_test = X_test

        self.add_hooks_fn = lambda:None
        self.remove_hooks_fn =lambda:None
        

    def bind_hook_fn(self,add_hooks_fn =None, remove_hooks_fn = None):
        """Same ideas the NTK version"""
        self.add_hooks_fn = add_hooks_fn
        self.remove_hooks_fn = remove_hooks_fn

    def capture_performance(self,epoch,model,j):
        
        if epoch in self.snapshot_array:        
            if j==0:
                self.recorded_epochs.append(epoch)
            self.remove_hooks_fn()
            model.eval()
            with torch.no_grad():
                Y_test_pred = model(self.X_test).squeeze()
                logger.debug("The shape of Y_test_pred is %s ",Y_test_pred.shape)
            self.inst_performs.append(Y_test_pred)
            logger.debug("Performance appended") #could be argued this is an info debug statement, but it would occur very frequently during training
            self.add_hooks_fn()
            model.train()
    
    def after_loop(self):
        self.performances.append(self.inst_performs)

        self.inst_performs = []

    def convert_numpy(self):
        self.performances = np.array(self.performances)

class Trial():
    def __init__(self, model_fn, input,output,width, depth,lr,epochs,STD,ensemblenum,
                 alignmentint,x_train,y_train,x_test,y_test, eval_amount,
                 performance_times, lr_factor =2):
        
        self.model_fn = model_fn
        """This is what we will use to instantiate the model at each relevant time"""
        self.input = input
        self.output = output
        self.width = width
        self.depth = depth
        self.lr = lr
        self.epochs = epochs
        self.std = STD
        self.ensemble = ensemblenum
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.numlayer = self.depth+1
        self.lr_factor =lr_factor

        self.ensemble_derivs = {i:[] for i in range(self.numlayer)}

        self.phi = {i:[] for i in range(self.numlayer)}
        
        """Instantiating the lists which store relavent times"""
        self.recorded_epochs_activation = []
        self.recorded_epochs_activation_roc = []

        self.y_vector = self.y_train.numpy()
        
        self.norm_const = np.dot(self.y_vector.T, self.y_vector)
        
        """Since its total time it is never impacted regardless of where we focus our information"""
        self.train_time_total = np.arange(self.epochs) *self.lr #To be used for any plot which requires the total training time
        """An array to hold the times where all layer's phi> target phi"""
        self.phi_target_time_layer = []

        """An array to hold the time where neurons in a specific layer are > target phi
        Note we have excluded the final layer, as it only has one layer, so only one value
        """
        self.phi_target_time_neuron = np.zeros((self.numlayer-1, self.width)) 


        self.ntk_recorder = NTKRecorder(alignmentint,e_amount=eval_amount)

        self.loss_recorder = LossRecorder()

        self.performance_recorder = PerformanceRecorder(self.x_test, performance_times,epochs)

        self.parameter_recorder = ParameterRecorder(self.numlayer,alignmentint)

        self.plateau_recorder = PlateauRecorder(lr = self.lr)


    def trial_update_lr(self,optimiser):
        new_lr = self.lr * self.lr_factor
        optimiser.param_groups[0]['lr'] = new_lr
        logger.info("Learning rate has been updated")
        logger.debug("New learning rate is %f ",optimiser.param_groups[0]['lr'])

    
    def training_loop(self,j):
        """Order of training should be 
        NTK calculations
        Actual training/loss calculation
        Loss recorder
        Performance Recorder
        
        """

        """Creates the model, this needs to be done in loop, as opposed to simply passing a model, such that we actually run our experiment over the different 
        initialisation shapes"""
        self.model = self.model_fn()

        activation_history = {i: [] for i in range(self.numlayer)}        
        """Creates a dictionary of lists corresponding to each layer, where we will store the activation history to calculate finite differences, and batch means"""

        def make_hook(layer_id):
            """This NEEDS to be a nested function to safely pass the layer_id value to the hook function, which only ever takes the module, input and output as function inputs"""
            def hook(module, input, output):
                activation_history[layer_id].append(output.detach())     
                """Appends the output of the layer (before tanh activation) to the corresponding list in the activation history
                .detach() is needed as the output is technically a tensor (which the dictionary will not accept/work well with)
                """           
            return hook

        def add_hooks():
            """Function which actually adds/appends the hooks to the hidden layers of the network"""
            for i, layer in enumerate(self.model.hidden_layers):
                self.hooks.append(layer.register_forward_hook(make_hook(i)))
            self.hooks.append(self.model.output_layer.register_forward_hook(make_hook(len(self.model.hidden_layers))))
        
        self.hooks =[]
        add_hooks()

        self.ntk_recorder.bind_hook_fn(add_hooks_fn= lambda: add_hooks(), remove_hooks_fn=lambda : remove_hooks(self.hooks))
        #This binds the hooks once per loop, allowing the NTK tracker to exist outside the training loop

        self.performance_recorder.bind_hook_fn(add_hooks_fn=lambda:add_hooks(), remove_hooks_fn= lambda: remove_hooks(self.hooks))
        #Same idea as above

        optimiser  = torch.optim.SGD(self.model.parameters(), lr = self.lr)
        """Note on gradient descent... since we are using the full batch of data, this is actually a regular gradient descent not stoichastic 
        i.e., no random sampling"""

##############################################################################################
        """This is the section responsible for the updating of the learning rate, if ever wants to be turned off comment out the binding function"""

        # self.plateau_recorder.bind_update_lr_funct(update_lr_fn= lambda :self.trial_update_lr(optimiser))
        # print("printing param groups")
        # print(optimiser.param_groups)

        # print(f"The length of param_groups is {len(optimiser.param_groups)}")
        # print(type(optimiser.param_groups[0]))
        # print(optimiser.param_groups[0].keys())
        # for groups in optimiser.param_groups:
        #     print(f"The learning rate is {groups['lr']}")
##############################################################################################



        """Training the model"""
        for epoch in range(self.epochs):
            self.ntk_recorder.NTK_track_in_loop(self.model,self.x_train,self.y_train.numpy(),epoch,j,self.norm_const)

            """Start of training loop"""
            self.parameter_recorder.track_in_loop(epoch,self.model)

            if j == 0:
                self.recorded_epochs_activation.append(epoch)
            self.model.train()
            y_pred = self.model(self.x_train)
            loss = criterion(y_pred.squeeze(),self.y_train)
            self.loss_recorder.loss_track_in_loop(epoch,loss.item(),j)
            self.plateau_recorder.track_in_loop(epoch,loss.item())
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            self.performance_recorder.capture_performance(epoch,self.model,j)

            """End of training loop"""

        self.ntk_recorder.NTK_end_of_loop()
        self.loss_recorder.loss_end_of_loop()
        self.performance_recorder.after_loop()
        self.parameter_recorder.end_of_loop()
        self.plateau_recorder.end_of_loop()

        self.activation_history_func(activation_history)

    def activation_history_func(self, activation_history):
        """Function which handles appending and reorganising the activation history function
        Stacks each activation history array on top of one another, then appends both the batch mean and time derivative to appropriate Trial class arrays
        """

        for i in range(self.numlayer):
            """Calculating the pre-activation derivative from the hooks' data"""
            self.stacked = torch.stack(activation_history[i]).numpy() #Converts it to a numpy array
            #above has the following dimensionality: [epochs, batch (or data point), neuron]
            #axis 0 = epochs
            #axis 1 = batch/data
            #axis 2 = neuron

            logger.debug("Layer %d stacked has shape %s",i, self.stacked.shape)
            logger.debug("Layer %d's phi values have been stacked",i)

            self.stacked_batch = np.mean(np.abs(self.stacked),axis = 1)
            logger.debug("Dimensionality of the stacked_batch is %s ",self.stacked_batch.shape)
            self.phi[i].append(self.stacked_batch)
            #Dimensionality of the phi array is [layer][replica][epoch,neuron]

            time_deriv = np.diff(self.stacked, 1, 0) #Takes first derivative along the epoch axis
            logger.debug('time_deriv shape: %s',time_deriv.shape) #Sanity check
            self.ensemble_derivs[i].append(time_deriv)
            #appends the layer information to the corresponding layer list in the ensemble_derivs dictionary
            #New dimensionality would be the following:
            #[layer] [ensemble_num, epoch-1 (since taken derivative), batch, neuron]

    def train_ensemble(self):
        """Full standard training loop, meant to make running a trial easier"""
        for j in range(self.ensemble):
            self.training_loop(j)
            logger.info("Done with replica %d", j)

        self.ntk_recorder.NTK_conversion()
        self.loss_recorder.to_numpy()
        self.performance_recorder.convert_numpy()
        self.parameter_recorder.convert_to_numpy()
        self.plateau_recorder.convert_to_numpy()
        logger.info("Dimensionality of the performances array is %s ",self.performance_recorder.performances.shape)
        # self.phi = np.array(self.phi)
        for i in range(self.numlayer):
            self.phi[i] = np.array(self.phi[i]) #just converting it to dictionary of numpy arrays
