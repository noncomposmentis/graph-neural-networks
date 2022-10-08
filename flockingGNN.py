# 2020/01/01~
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu
# Kate Tolstaya, eig@seas.upenn.edu

# Learn decentralized controllers for flocking. There is a team of robots that
# start flying at random velocities and we want them to coordinate so that they
# can fly together while avoiding collisions. We learn a decentralized 
# controller by using imitation learning.

# In this simulation, the number of agents is fixed for training, but can be
# set to a different number for testing.

# Outputs:
# - Text file with all the hyperparameters selected for the run and the 
#   corresponding results (hyperparameters.txt)
# - Pickle file with the random seeds of both torch and numpy for accurate
#   reproduction of results (randomSeedUsed.pkl)
# - The parameters of the trained models, for both the Best and the Last
#   instance of each model (savedModels/)
# - The figures of loss and evaluation through the training iterations for
#   each model (figs/ and trainVars/)
# - Videos for some of the trajectories in the dataset, following the optimal
#   centralized controller (datasetTrajectories/)
# - Videos for some of the learned trajectories following the controles 
#   learned by each model (learnedTrajectories/)

#%%##################################################################
#                                                                   #
#                    IMPORTING                                      #
#                                                                   #
#####################################################################

#\\\ Standard libraries:
import os
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']
import matplotlib.pyplot as plt
import pickle
import datetime
from copy import deepcopy

import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim

#\\\ Own libraries:
import alegnn.utils.dataTools as dataTools
import alegnn.utils.graphML as gml
import alegnn.modules.architecturesTime as architTime
import alegnn.modules.model as model
import alegnn.modules.training as training
import alegnn.modules.evaluation as evaluation

#\\\ Separate functions:
from alegnn.utils.miscTools import writeVarValues
from alegnn.utils.miscTools import saveSeed

# Start measuring time
start_run_time = datetime.datetime.now()

#%%##################################################################
#                                                                   #
#                    SETTING PARAMETERS                             #
#                                                                   #
#####################################################################

this_filename = 'flockingGNN' # This is the general name of all related files

n_agents = 50 # Number of agents at training time

save_dir_root = 'experiments' # In this case, relative location
save_dir = os.path.join(save_dir_root, this_filename) # Dir where to save all
    # the results from each run

#\\\ Create .txt to store the values of the setting parameters for easier
# reference when running multiple experiments
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# Append date and time of the run to the directory, to avoid several runs of
# overwritting each other.
save_dir = save_dir + '-%03d-' % n_agents + today
# Create directory
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Create the file where all the (hyper)parameters and results will be saved.
vars_file = os.path.join(save_dir,'hyperparameters.txt')
with open(vars_file, 'w+') as file:
    file.write('%s\n\n' % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

#\\\ Save seeds for reproducibility
#   PyTorch seeds
torch_state = torch.get_rng_state()
torch_seed = torch.initial_seed()
#   Numpy seeds
numpy_state = np.random.RandomState().get_state()
#   Collect all random states
random_states = []
random_states.append({})
random_states[0]['module'] = 'numpy'
random_states[0]['state'] = numpy_state
random_states.append({})
random_states[1]['module'] = 'torch'
random_states[1]['state'] = torch_state
random_states[1]['seed'] = torch_seed
#   This list and dictionary follows the format to then be loaded, if needed,
#   by calling the loadSeed function in Utils.miscTools
saveSeed(random_states, save_dir)

########
# DATA #
########

use_gpu = True # If true, and GPU is available, use it.

n_agents_max = n_agents # Maximum number of agents to test the solution
n_sim_points = 1 # Number of simulations between n_agents and n_agents_max
    # At test time, the architectures trained on n_agents will be tested on a
    # varying number of agents, starting at n_agents all the way to n_agents_max;
    # the number of simulations for different number of agents is given by
    # n_sim_points, i.e. if n_agents = 50, n_agents_max = 100 and n_sim_points = 3,
    # then the architectures are trained on 50, 75 and 100 agents.
comm_radius = 2. # Communication radius
repel_dist = 1. # Minimum distance before activating repelling potential
n_train = 400 # Number of training samples
n_valid = 20 # Number of valid samples
n_test = 20 # Number of testing samples
duration = 2. # Duration of the trajectory
sampling_time = 0.01 # Sampling time
init_geometry = 'circular' # Geometry of initial positions
init_vel_value = 3. # Initial velocities are samples from an interval
    # [-init_vel_value, init_vel_value]
init_min_dist = 0.1 # No two agents are located at a distance less than this
accel_max = 10. # This is the maximum value of acceleration allowed

n_realizations = 10 # Number of data realizations
    # How many times we repeat the experiment

#\\\ Save values:
writeVarValues(vars_file,
               {'n_agents': n_agents,
                'n_agents_max': n_agents_max,
                'n_sim_points': n_sim_points,
                'comm_radius': comm_radius,
                'repel_dist': repel_dist,
                'n_train': n_train,
                'n_valid': n_valid,
                'n_test': n_test,
                'duration': duration,
                'sampling_time': sampling_time,
                'init_geometry': init_geometry,
                'init_vel_value': init_vel_value,
                'init_min_dist': init_min_dist,
                'accel_max': accel_max,
                'n_realizations': n_realizations,
                'use_gpu': use_gpu})

############
# TRAINING #
############

#\\\ Individual model training options
optim_alg = 'ADAM' # Options: 'SGD', 'ADAM', 'RMSprop'
learning_rate = 0.0005 # In all options
beta1 = 0.9 # beta1 if 'ADAM', alpha if 'RMSprop'
beta2 = 0.999 # ADAM option only

#\\\ Loss function choice
loss_function = nn.MSELoss

#\\\ Training algorithm
trainer = training.TrainerFlocking

#\\\ Evaluation algorithm
evaluator = evaluation.evaluateFlocking

#\\\ Overall training options
prob_expert = 0.993 # Probability of choosing the expert in DAGger
#DAGgerType = 'fixedBatch' # 'replaceTimeBatch', 'randomEpoch'
n_epochs = 30 # Number of epochs
batch_size = 20 # Batch size
do_learning_rate_decay = False # Learning rate decay
learning_rate_decay_rate = 0.9 # Rate
learning_rate_decay_period = 1 # How many epochs after which update the lr
validation_interval = 5 # How many training steps to do the validation

#\\\ Save values
writeVarValues(vars_file,
               {'optimizationAlgorithm': optim_alg,
                'learning_rate': learning_rate,
                'beta1': beta1,
                'beta2': beta2,
                'loss_function': loss_function,
                'trainer': trainer,
                'evaluator': evaluator,
                'prob_expert': prob_expert,
                'n_epochs': n_epochs,
                'batch_size': batch_size,
                'do_learning_rate_decay': do_learning_rate_decay,
                'learning_rate_decay_rate': learning_rate_decay_rate,
                'learning_rate_decay_period': learning_rate_decay_period,
                'validation_interval': validation_interval})

#################
# ARCHITECTURES #
#################

# In this section, we determine the (hyper)parameters of models that we are
# going to train. This only sets the parameters. The architectures need to be
# created later below. Do not forget to add the name of the architecture
# to model_list.

# If the hyperparameter dictionary is called 'hParams' + name, then it can be
# picked up immediately later on, and there's no need to recode anything after
# the section 'Setup' (except for setting the number of nodes in the 'N'
# variable after it has been coded).

# The name of the keys in the hyperparameter dictionary have to be the same
# as the names of the variables in the architecture call, because they will
# be called by unpacking the dictionary.

#nFeatures = 32 # Number of features in all architectures
#nFilterTaps = 4 # Number of filter taps in all architectures
# [[The hyperparameters are for each architecture, and they were chosen 
#   following the results of the hyperparameter search]]
nonlinearity_hidden = torch.tanh
nonlinearity_output = torch.tanh
nonlinearity = nn.Tanh # Chosen nonlinearity for nonlinear architectures

# Select desired architectures
do_local_flt = True # Local filter (no nonlinearity)
do_local_gnn = True # Local GNN (include nonlinearity)
do_dl_agg_gnn = True
do_graph_rnn = True

model_list = []

#\\\\\\\\\\\\\\\\\\
#\\\ FIR FILTER \\\
#\\\\\\\\\\\\\\\\\\

if do_local_flt:

    #\\\ Basic parameters for the Local Filter architecture

    h_params_local_flt = {} # Hyperparameters (hParams) for the Local Filter

    h_params_local_flt['name'] = 'LocalFlt'
    # Chosen architecture
    h_params_local_flt['archit'] = architTime.LocalGNN_DB
    h_params_local_flt['device'] = 'cuda:0' \
                                    if (use_gpu and torch.cuda.is_available()) \
                                    else 'cpu'

    # Graph convolutional parameters
    h_params_local_flt['dimNodeSignals'] = [6, 32] # Features per layer
    h_params_local_flt['nFilterTaps'] = [4] # Number of filter taps
    h_params_local_flt['bias'] = True # Decide whether to include a bias term
    # Nonlinearity
    h_params_local_flt['nonlinearity'] = gml.NoActivation # Selected nonlinearity
        # is affected by the summary
    # Readout layer: local linear combination of features
    h_params_local_flt['dimReadout'] = [2] # Dimension of the fully connected
        # layers after the FIR filter layers (map); this fully connected layer
        # is applied only at each node, without any further exchanges nor
        # considering all nodes at once, making the architecture entirely
        # local.
    # Graph structure
    h_params_local_flt['dimEdgeFeatures'] = 1 # Scalar edge weights

    #\\\ Save Values:
    writeVarValues(vars_file, h_params_local_flt)
    model_list += [h_params_local_flt['name']]

#\\\\\\\\\\\\\\\\\
#\\\ LOCAL GNN \\\
#\\\\\\\\\\\\\\\\\

if do_local_gnn:

    #\\\ Basic parameters for the Local GNN architecture

    h_params_local_gnn = {} # Hyperparameters (hParams) for the Local GNN (LclGNN)

    h_params_local_gnn['name'] = 'LocalGNN'
    # Chosen architecture
    h_params_local_gnn['archit'] = architTime.LocalGNN_DB
    h_params_local_gnn['device'] = 'cuda:0' \
                                    if (use_gpu and torch.cuda.is_available()) \
                                    else 'cpu'

    # Graph convolutional parameters
    h_params_local_gnn['dimNodeSignals'] = [6, 64] # Features per layer
    h_params_local_gnn['nFilterTaps'] = [3] # Number of filter taps
    h_params_local_gnn['bias'] = True # Decide whether to include a bias term
    # Nonlinearity
    h_params_local_gnn['nonlinearity'] = nonlinearity # Selected nonlinearity
        # is affected by the summary
    # Readout layer: local linear combination of features
    h_params_local_gnn['dimReadout'] = [2] # Dimension of the fully connected
        # layers after the GCN layers (map); this fully connected layer
        # is applied only at each node, without any further exchanges nor
        # considering all nodes at once, making the architecture entirely
        # local.
    # Graph structure
    h_params_local_gnn['dimEdgeFeatures'] = 1 # Scalar edge weights

    #\\\ Save Values:
    writeVarValues(vars_file, h_params_local_gnn)
    model_list += [h_params_local_gnn['name']]

#\\\\\\\\\\\\\\\\\\\\\\\
#\\\ AGGREGATION GNN \\\
#\\\\\\\\\\\\\\\\\\\\\\\

if do_dl_agg_gnn:

    #\\\ Basic parameters for the Aggregation GNN architecture

    h_params_dagnn1_ly = {} # Hyperparameters (hParams) for the Local GNN (LclGNN)

    h_params_dagnn1_ly['name'] = 'DAGNN1Ly'
    # Chosen architecture
    h_params_dagnn1_ly['archit'] = architTime.AggregationGNN_DB
    h_params_dagnn1_ly['device'] = 'cuda:0' \
                                    if (use_gpu and torch.cuda.is_available()) \
                                    else 'cpu'

    # Graph convolutional parameters
    h_params_dagnn1_ly['dimFeatures'] = [6] # Features per layer
    h_params_dagnn1_ly['nFilterTaps'] = [] # Number of filter taps
    h_params_dagnn1_ly['bias'] = True # Decide whether to include a bias term
    # Nonlinearity
    h_params_dagnn1_ly['nonlinearity'] = nonlinearity # Selected nonlinearity
        # is affected by the summary
    h_params_dagnn1_ly['poolingFunction'] = gml.NoPool
    h_params_dagnn1_ly['poolingSize'] = []
    # Readout layer: local linear combination of features
    h_params_dagnn1_ly['dimReadout'] = [64, 2] # Dimension of the fully connected
        # layers after the GCN layers (map); this fully connected layer
        # is applied only at each node, without any further exchanges nor
        # considering all nodes at once, making the architecture entirely
        # local.
    # Graph structure
    h_params_dagnn1_ly['dimEdgeFeatures'] = 1 # Scalar edge weights
    h_params_dagnn1_ly['nExchanges'] = 2 - 1

    #\\\ Save Values:
    writeVarValues(vars_file, h_params_dagnn1_ly)
    model_list += [h_params_dagnn1_ly['name']]

#\\\\\\\\\\\\\\\\\
#\\\ GRAPH RNN \\\
#\\\\\\\\\\\\\\\\\

if do_graph_rnn:

    #\\\ Basic parameters for the Graph RNN architecture

    h_params_graph_rnn = {} # Hyperparameters (hParams) for the Local GNN (LclGNN)

    h_params_graph_rnn['name'] = 'GraphRNN'
    # Chosen architecture
    h_params_graph_rnn['archit'] = architTime.GraphRecurrentNN_DB
    h_params_graph_rnn['device'] = 'cuda:0' \
                                    if (use_gpu and torch.cuda.is_available()) \
                                    else 'cpu'

    # Graph convolutional parameters
    h_params_graph_rnn['dimInputSignals'] = 6 # Features per layer
    h_params_graph_rnn['dimOutputSignals'] = 64
    h_params_graph_rnn['dimHiddenSignals'] = 64
    h_params_graph_rnn['nFilterTaps'] = [3] * 2 # Number of filter taps
    h_params_graph_rnn['bias'] = True # Decide whether to include a bias term
    # Nonlinearity
    h_params_graph_rnn['nonlinearity_hidden'] = nonlinearity_hidden
    h_params_graph_rnn['nonlinearity_output'] = nonlinearity_output
    h_params_graph_rnn['nonlinearityReadout'] = nonlinearity
    # Readout layer: local linear combination of features
    h_params_graph_rnn['dimReadout'] = [2] # Dimension of the fully connected
        # layers after the GCN layers (map); this fully connected layer
        # is applied only at each node, without any further exchanges nor
        # considering all nodes at once, making the architecture entirely
        # local.
    # Graph structure
    h_params_graph_rnn['dimEdgeFeatures'] = 1 # Scalar edge weights

    #\\\ Save Values:
    writeVarValues(vars_file, h_params_graph_rnn)
    model_list += [h_params_graph_rnn['name']]

###########
# LOGGING #
###########

# Options:
do_print = True # Decide whether to print stuff while running
do_logging = False # Log into tensorboard
do_save_vars = True # Save (pickle) useful variables
do_figs = True # Plot some figures (this only works if do_save_vars is True)
# Parameters:
print_interval = 1 # After how many training steps, print the partial results
#   0 means to never print partial results while training
x_axis_multiplier_train = 10 # How many training steps in between those shown in
    # the plot, i.e., one training step every x_axis_multiplier_train is shown.
x_axis_multiplier_valid = 2 # How many validation steps in between those shown,
    # same as above.
fig_size = 5 # Overall size of the figure that contains the plot
line_width = 2 # Width of the plot lines
marker_shape = 'o' # Shape of the markers
marker_size = 3 # Size of the markers
video_speed = 0.5 # Slow down by half to show transitions
n_videos = 3 # Number of videos to save

#\\\ Save values:
writeVarValues(vars_file,
               {'do_print': do_print,
                'do_logging': do_logging,
                'do_save_vars': do_save_vars,
                'do_figs': do_figs,
                'save_dir': save_dir,
                'print_interval': print_interval,
                'fig_size': fig_size,
                'line_width': line_width,
                'marker_shape': marker_shape,
                'marker_size': marker_size,
                'video_speed': video_speed,
                'n_videos': n_videos})

#%%##################################################################
#                                                                   #
#                    SETUP                                          #
#                                                                   #
#####################################################################

#\\\ If CUDA is selected, empty cache:
if use_gpu and torch.cuda.is_available():
    torch.cuda.empty_cache()

#\\\ Notify of processing units
if do_print:
    print("Selected devices:")
    for this_model in model_list:
        h_params_dict = eval('hParams' + this_model)
        print("\t%s: %s" % (this_model, h_params_dict['device']))

#\\\ Logging options
if do_logging:
    # If logging is on, load the tensorboard visualizer and initialize it
    from alegnn.utils.visualTools import Visualizer
    logs_tb = os.path.join(save_dir, 'logs_tb')
    logger = Visualizer(logs_tb, name='visualResults')

#\\\ Number of agents at test time
n_agents_test = np.linspace(n_agents, n_agents_max, num = n_sim_points,dtype = np.int)
n_agents_test = np.unique(n_agents_test).tolist()
n_sim_points = len(n_agents_test)
writeVarValues(vars_file, {'n_agents_test': n_agents_test}) # Save list

#\\\ Save variables during evaluation.
# We will save all the evaluations obtained for each of the trained models.
# The first list is one for each value of n_agents that we want to simulate 
# (i.e. these are test results, so if we test for different number of agents,
# we need to save the results for each of them). Each element in the list will
# be a dictionary (i.e. for each testing case, we have a dictionary).
# It basically is a dictionary, containing a list. The key of the
# dictionary determines the model, then the first list index determines
# which split realization. Then, this will be converted to numpy to compute
# mean and standard deviation (across the split dimension).
# We're saving the cost of the full trajectory, as well as the cost at the end
# instant.
cost_best_full = [None] * n_sim_points
cost_best_end = [None] * n_sim_points
cost_last_full = [None] * n_sim_points
cost_last_end = [None] * n_sim_points
cost_opt_full = [None] * n_sim_points
cost_opt_end = [None] * n_sim_points
for n in range(n_sim_points):
    cost_best_full[n] = {} # Accuracy for the best model (full trajectory)
    cost_best_end[n] = {} # Accuracy for the best model (end time)
    cost_last_full[n] = {} # Accuracy for the last model
    cost_last_end[n] = {} # Accuracy for the last model
    for this_model in model_list: # Create an element for each split realization,
        cost_best_full[n][this_model] = [None] * n_realizations
        cost_best_end[n][this_model] = [None] * n_realizations
        cost_last_full[n][this_model] = [None] * n_realizations
        cost_last_end[n][this_model] = [None] * n_realizations
    cost_opt_full[n] = [None] * n_realizations # Accuracy for optimal controller
    cost_opt_end[n] = [None] * n_realizations # Accuracy for optimal controller

if do_figs:
    #\\\ SAVE SPACE:
    # Create the variables to save all the realizations. This is, again, a
    # dictionary, where each key represents a model, and each model is a list
    # for each data split.
    # Each data split, in this case, is not a scalar, but a vector of
    # length the number of training steps (or of validation steps)
    loss_train = {}
    eval_valid = {}
    # Initialize the splits dimension
    for this_model in model_list:
        loss_train[this_model] = [None] * n_realizations
        eval_valid[this_model] = [None] * n_realizations


####################
# TRAINING OPTIONS #
####################

# Training phase. It has a lot of options that are input through a
# dictionary of arguments.
# The value of these options was decided above with the rest of the parameters.
# This just creates a dictionary necessary to pass to the train function.

training_options = {}

if do_logging:
    training_options['logger'] = logger
if do_save_vars:
    training_options['save_dir'] = save_dir
if do_print:
    training_options['print_interval'] = print_interval
if do_learning_rate_decay:
    training_options['learning_rate_decay_rate'] = learning_rate_decay_rate
    training_options['learning_rate_decay_period'] = learning_rate_decay_period
training_options['validation_interval'] = validation_interval

# And in case each model has specific training options (aka 'DAGger'), then
# we create a separate dictionary per model.

training_opts_per_model= {}

# Create relevant dirs: we need directories to save the videos of the dataset
# that involve the optimal centralized controllers, and we also need videos
# for the learned trajectory of each model. Note that all of these depend on
# each realization, so we will be saving videos for each realization.
# Here, we create all those directories.
dataset_trajectory_dir = os.path.join(save_dir,'datasetTrajectories')
if not os.path.exists(dataset_trajectory_dir):
    os.makedirs(dataset_trajectory_dir)

dataset_train_trajectory_dir = os.path.join(dataset_trajectory_dir,'train')
if not os.path.exists(dataset_train_trajectory_dir):
    os.makedirs(dataset_train_trajectory_dir)

dataset_test_trajectory_dir = os.path.join(dataset_trajectory_dir,'test')
if not os.path.exists(dataset_test_trajectory_dir):
    os.makedirs(dataset_test_trajectory_dir)

dataset_test_agent_trajectory_dir = [None] * n_sim_points
for n in range(n_sim_points):
    dataset_test_agent_trajectory_dir[n] = os.path.join(dataset_test_trajectory_dir,
                                                    '%03d' % n_agents_test[n])

if n_realizations > 1:
    dataset_train_trajectory_dir_orig = dataset_train_trajectory_dir
    dataset_test_agent_trajectory_dir_orig = dataset_test_agent_trajectory_dir.copy()

#%%##################################################################
#                                                                   #
#                    DATA SPLIT REALIZATION                         #
#                                                                   #
#####################################################################

# Start generating a new data realization for each number of total realizations

for realization in range(n_realizations):

    # On top of the rest of the training options, we pass the identification
    # of this specific data split realization.

    if n_realizations > 1:
        training_options['realizationNo'] = realization

        # Create new directories (specific for this realization)
        dataset_train_trajectory_dir = os.path.join(dataset_train_trajectory_dir_orig,
                                                 '%03d' % realization)
        if not os.path.exists(dataset_train_trajectory_dir):
            os.makedirs(dataset_train_trajectory_dir)

        for n in range(n_sim_points):
            dataset_test_agent_trajectory_dir[n] = os.path.join(
                                          dataset_test_agent_trajectory_dir_orig[n],
                                          '%03d' % realization)
            if not os.path.exists(dataset_test_agent_trajectory_dir[n]):
                os.makedirs(dataset_test_agent_trajectory_dir[n])

    if do_print:
        print("", flush = True)

    #%%##################################################################
    #                                                                   #
    #                    DATA HANDLING                                  #
    #                                                                   #
    #####################################################################

    ############
    # DATASETS #
    ############

    if do_print:
        print("Generating data", end = '')
        if n_realizations > 1:
            print(" for realization %d" % realization, end = '')
        print("...", flush = True)

    #   Generate the dataset
    data = dataTools.Flocking(
                # Structure
                n_agents,
                comm_radius,
                repel_dist,
                # Samples
                n_train,
                n_valid,
                1, # We do not care about testing, we will re-generate the
                   # dataset for testing
                # Time
                duration,
                sampling_time,
                # Initial conditions
                init_geometry = init_geometry,
                init_vel_value = init_vel_value,
                init_min_dist = init_min_dist,
                accel_max = accel_max)

    ###########
    # PREVIEW #
    ###########

    if do_print:
        print("Preview data", end = '')
        if n_realizations > 1:
            print(" for realization %d" % realization, end = '')
        print("...", flush = True)

    # Generate the videos
    data.saveVideo(dataset_train_trajectory_dir, # Where to save them
                    data.pos['train'], # Which positions to plot
                    n_videos, # Number of videos to create
                    commGraph = data.commGraph['train'], # Graph to plot
                    vel = data.vel['train'], # Velocity arrows to plot
                    videoSpeed = video_speed) # Change speed of animation

    #%%##################################################################
    #                                                                   #
    #                    MODELS INITIALIZATION                          #
    #                                                                   #
    #####################################################################

    # This is the dictionary where we store the models (in a model.Model
    # class).
    models_gnn = {}

    # If a new model is to be created, it should be called for here.

    if do_print:
        print("Model initialization...", flush = True)

    for this_model in model_list:

        # Get the corresponding parameter dictionary
        h_params_dict = deepcopy(eval('hParams' + this_model))
        # and training options
        training_opts_per_model[this_model] = deepcopy(training_options)

        # Now, this dictionary has all the hyperparameters that we need to pass
        # to the architecture, but it also has the 'name' and 'archit' that
        # we do not need to pass them. So we are going to get them out of
        # the dictionary
        this_name = h_params_dict.pop('name')
        call_archit = h_params_dict.pop('archit')
        this_device = h_params_dict.pop('device')
        # If there's a specific DAGger type, pop it out now
        if 'DAGgerType' in h_params_dict.keys() \
                                        and 'prob_expert' in h_params_dict.keys():
            training_opts_per_model[this_model]['prob_expert'] = \
                                                  h_params_dict.pop('prob_expert')
            training_opts_per_model[this_model]['DAGgerType'] = \
                                                  h_params_dict.pop('DAGgerType')

        # If more than one graph or data realization is going to be carried out,
        # we are going to store all of thos models separately, so that any of
        # them can be brought back and studied in detail.
        if n_realizations > 1:
            this_name += 'G%02d' % realization

        if do_print:
            print("\tInitializing %s..." % this_name,
                  end = ' ',flush = True)

        ##############
        # PARAMETERS #
        ##############

        #\\\ Optimizer options
        #   (If different from the default ones, change here.)
        this_optim_alg = optim_alg
        this_learning_rate = learning_rate
        this_beta1 = beta1
        this_beta2 = beta2

        ################
        # ARCHITECTURE #
        ################

        this_archit = call_archit(**h_params_dict)
        this_archit.to(this_device)

        #############
        # OPTIMIZER #
        #############

        if this_optim_alg == 'ADAM':
            this_optim = optim.Adam(this_archit.parameters(),
                                   lr = learning_rate,
                                   betas = (beta1, beta2))
        elif this_optim_alg == 'SGD':
            this_optim = optim.SGD(this_archit.parameters(),
                                  lr = learning_rate)
        elif this_optim_alg == 'RMSprop':
            this_optim = optim.RMSprop(this_archit.parameters(),
                                      lr = learning_rate, alpha = beta1)

        ########
        # LOSS #
        ########

        this_loss_function = loss_function()

        ###########
        # TRAINER #
        ###########

        this_trainer = trainer

        #############
        # EVALUATOR #
        #############

        this_evaluator = evaluator

        #########
        # MODEL #
        #########

        model_created = model.Model(this_archit,
                                   this_loss_function,
                                   this_optim,
                                   this_trainer,
                                   this_evaluator,
                                   this_device,
                                   this_name,
                                   save_dir)

        models_gnn[this_name] = model_created

        writeVarValues(vars_file,
                       {'name': this_name,
                        'thisOptimizationAlgorithm': this_optim_alg,
                        'this_trainer': this_trainer,
                        'this_evaluator': this_evaluator,
                        'this_learning_rate': this_learning_rate,
                        'this_beta1': this_beta1,
                        'this_beta2': this_beta2})

        if do_print:
            print("OK")

    #%%##################################################################
    #                                                                   #
    #                    TRAINING                                       #
    #                                                                   #
    #####################################################################


    ############
    # TRAINING #
    ############

    print("")

    for this_model in models_gnn.keys():

        if do_print:
            print("Training model %s..." % this_model)

        for m in model_list:
            if m in this_model:
                model_name = m

        this_train_vars = models_gnn[this_model].train(data,
                                                   n_epochs,
                                                   batch_size,
                                                   **training_opts_per_model[m])

        if do_figs:
        # Find which model to save the results (when having multiple
        # realizations)
            for m in model_list:
                if m in this_model:
                    loss_train[m][realization] = this_train_vars['loss_train']
                    eval_valid[m][realization] = this_train_vars['eval_valid']
    # And we also need to save 'nBatch' but is the same for all models, so
    if do_figs:
        n_batches = this_train_vars['n_batches']

    #%%##################################################################
    #                                                                   #
    #                    EVALUATION                                     #
    #                                                                   #
    #####################################################################

    # Now that the model has been trained, we evaluate them on the test
    # samples.

    # We have two versions of each model to evaluate: the one obtained
    # at the best result of the validation step, and the last trained model.

    for n in range(n_sim_points):

        if do_print:
            print("")
            print("[%3d Agents] Generating test set" % n_agents_test[n],
                  end = '')
            if n_realizations > 1:
                print(" for realization %d" % realization, end = '')
            print("...", flush = True)

        #   Load the data, which will give a specific split
        data_test = dataTools.Flocking(
                        # Structure
                        n_agents_test[n],
                        comm_radius,
                        repel_dist,
                        # Samples
                        1, # We don't care about training
                        1, # nor validation
                        n_test,
                        # Time
                        duration,
                        sampling_time,
                        # Initial conditions
                        init_geometry = init_geometry,
                        init_vel_value = init_vel_value,
                        init_min_dist = init_min_dist,
                        accel_max = accel_max)

        ###########
        # OPTIMAL #
        ###########

        #\\\ PREVIEW
        #\\\\\\\\\\\

        # Save videos for the optimal trajectories of the test set (before it
        # was for the otpimal trajectories of the training set)

        pos_test = data_test.getData('pos', 'test')
        vel_test = data_test.getData('vel', 'test')
        comm_graph_test = data_test.getData('commGraph', 'test')

        if do_print:
            print("[%3d Agents] Preview data"  % n_agents_test[n], end = '')
            if n_realizations > 1:
                print(" for realization %d" % realization, end = '')
            print("...", flush = True)

        data_test.saveVideo(dataset_test_agent_trajectory_dir[n],
                           pos_test,
                           n_videos,
                           commGraph = comm_graph_test,
                           vel = vel_test,
                           videoSpeed = video_speed)

        #\\\ EVAL
        #\\\\\\\\

        # Get the cost for the optimal trajectories

        # Full trajectory
        cost_opt_full[n][realization] = data_test.evaluate(vel = vel_test)

        # Last time instant
        cost_opt_end[n][realization] = data_test.evaluate(vel = vel_test[:,-1:,:,:])

        writeVarValues(vars_file,
                   {'cost_opt_full%03dR%02d' % (n_agents_test[n],realization):
                                                     cost_opt_full[n][realization],
                    'cost_opt_end%04dR%02d' % (n_agents_test[n],realization):
                                                     cost_opt_end[n][realization]})

        del pos_test, vel_test, comm_graph_test

        ##########
        # MODELS #
        ##########

        for this_model in models_gnn.keys():

            if do_print:
                print("[%3d Agents] Evaluating model %s" % \
                                         (n_agents_test[n], this_model), end = '')
                if n_realizations > 1:
                    print(" for realization %d" % realization, end = '')
                print("...", flush = True)

            add_kw = {}
            add_kw['n_videos'] = n_videos
            add_kw['graphNo'] = n_agents_test[n]
            if n_realizations > 1:
                add_kw['realizationNo'] = realization

            this_eval_vars = models_gnn[this_model].evaluate(data_test, **add_kw)

            this_cost_best_full = this_eval_vars['cost_best_full']
            this_cost_best_end = this_eval_vars['cost_best_end']
            this_cost_last_full = this_eval_vars['cost_last_full']
            this_cost_last_end = this_eval_vars['cost_last_end']

            # Save values
            writeVarValues(vars_file,
                   {'cost_best_full%s%03dR%02d' % \
                                       (this_model, n_agents_test[n], realization):
                                                                this_cost_best_full,
                    'cost_best_end%s%04dR%02d' % \
                                       (this_model, n_agents_test[n], realization):
                                                                 this_cost_best_end,
                    'cost_last_full%s%03dR%02d' % \
                                       (this_model, n_agents_test[n], realization):
                                                                this_cost_last_full,
                    'cost_last_end%s%04dR%02d' % \
                                       (this_model, n_agents_test[n], realization):
                                                                this_cost_last_end})

            # Find which model to save the results (when having multiple
            # realizations)
            for m in model_list:
                if m in this_model:
                    cost_best_full[n][m][realization] = this_cost_best_full
                    cost_best_end[n][m][realization] = this_cost_best_end
                    cost_last_full[n][m][realization] = this_cost_last_full
                    cost_last_end[n][m][realization] = this_cost_last_end


############################
# FINAL EVALUATION RESULTS #
############################

mean_cost_best_full = [None] * n_sim_points # Mean across data splits
mean_cost_best_end = [None] * n_sim_points # Mean across data splits
mean_cost_last_full = [None] * n_sim_points # Mean across data splits
mean_cost_last_end = [None] * n_sim_points # Mean across data splits
std_dev_cost_best_full = [None] * n_sim_points # Standard deviation across data splits
std_dev_cost_best_end = [None] * n_sim_points # Standard deviation across data splits
std_dev_cost_last_full = [None] * n_sim_points # Standard deviation across data splits
std_dev_cost_last_end = [None] * n_sim_points # Standard deviation across data splits
mean_cost_opt_full = [None] * n_sim_points
std_dev_cost_opt_full = [None] * n_sim_points
mean_cost_opt_end = [None] * n_sim_points
std_dev_cost_opt_end = [None] * n_sim_points

for n in range(n_sim_points):

    # Now that we have computed the accuracy of all runs, we can obtain a final
    # result (mean and standard deviation)

    mean_cost_best_full[n] = {} # Mean across data splits
    mean_cost_best_end[n] = {} # Mean across data splits
    mean_cost_last_full[n] = {} # Mean across data splits
    mean_cost_last_end[n] = {} # Mean across data splits
    std_dev_cost_best_full[n] = {} # Standard deviation across data splits
    std_dev_cost_best_end[n] = {} # Standard deviation across data splits
    std_dev_cost_last_full[n] = {} # Standard deviation across data splits
    std_dev_cost_last_end[n] = {} # Standard deviation across data splits

    if do_print:
        print("\n[%3d Agents] Final evaluations (%02d data splits)" % \
                                               (n_agents_test[n], n_realizations))

    cost_opt_full[n] = np.array(cost_opt_full[n])
    mean_cost_opt_full[n] = np.mean(cost_opt_full[n])
    std_dev_cost_opt_full[n] = np.std(cost_opt_full[n])
    cost_opt_end[n] = np.array(cost_opt_end[n])
    mean_cost_opt_end[n] = np.mean(cost_opt_end[n])
    std_dev_cost_opt_end[n] = np.std(cost_opt_end[n])

    if do_print:
        print("\t%8s: %8.4f (+-%6.4f) [Optm/Full]" % (
                'Optimal',
                mean_cost_opt_full[n],
                std_dev_cost_opt_full[n]))
        print("\t%9s %8.4f (+-%6.4f) [Optm/End ]" % (
                '',
                mean_cost_opt_end[n],
                std_dev_cost_opt_end[n]))

    # Save values
    writeVarValues(vars_file,
               {'mean_cost_opt_full%03d' % n_agents_test[n]: mean_cost_opt_full[n],
                'std_dev_cost_opt_full%03d' % n_agents_test[n]: std_dev_cost_opt_full[n],
                'mean_cost_opt_end%04d' % n_agents_test[n]: mean_cost_opt_end[n],
                'std_dev_cost_opt_end%04d' % n_agents_test[n]: std_dev_cost_opt_end[n]})

    for this_model in model_list:
        # Convert the lists into a nDataSplits vector
        cost_best_full[n][this_model] = np.array(cost_best_full[n][this_model])
        cost_best_end[n][this_model] = np.array(cost_best_end[n][this_model])
        cost_last_full[n][this_model] = np.array(cost_last_full[n][this_model])
        cost_last_end[n][this_model] = np.array(cost_last_end[n][this_model])

        # And now compute the statistics (across graphs)
        mean_cost_best_full[n][this_model] = np.mean(cost_best_full[n][this_model])
        mean_cost_best_end[n][this_model] = np.mean(cost_best_end[n][this_model])
        mean_cost_last_full[n][this_model] = np.mean(cost_last_full[n][this_model])
        mean_cost_last_end[n][this_model] = np.mean(cost_last_end[n][this_model])
        std_dev_cost_best_full[n][this_model] = np.std(cost_best_full[n][this_model])
        std_dev_cost_best_end[n][this_model] = np.std(cost_best_end[n][this_model])
        std_dev_cost_last_full[n][this_model] = np.std(cost_last_full[n][this_model])
        std_dev_cost_last_end[n][this_model] = np.std(cost_last_end[n][this_model])

        # And print it:
        if do_print:
            print(
              "\t%s: %8.4f (+-%6.4f) [Best/Full] %8.4f (+-%6.4f) [Last/Full]"%(
                    this_model,
                    mean_cost_best_full[n][this_model],
                    std_dev_cost_best_full[n][this_model],
                    mean_cost_last_full[n][this_model],
                    std_dev_cost_last_full[n][this_model]))
            print(
              "\t%9s %8.4f (+-%6.4f) [Best/End ] %8.4f (+-%6.4f) [Last/End ]"%(
                    '',
                    mean_cost_best_end[n][this_model],
                    std_dev_cost_best_end[n][this_model],
                    mean_cost_last_end[n][this_model],
                    std_dev_cost_last_end[n][this_model]))

        # Save values
        writeVarValues(vars_file,
                   {'meanAccBestFull%s%03d' % (this_model, n_agents_test[n]):
                                               mean_cost_best_full[n][this_model],
                    'stdDevAccBestFull%s%03d' % (this_model, n_agents_test[n]):
                                               std_dev_cost_best_full[n][this_model],
                    'meanAccBestEnd%s%04d' % (this_model, n_agents_test[n]):
                                               mean_cost_best_end[n][this_model],
                    'stdDevAccBestEnd%s%04d' % (this_model, n_agents_test[n]):
                                               std_dev_cost_best_end[n][this_model],
                    'meanAccLastFull%s%03d' % (this_model, n_agents_test[n]):
                                               mean_cost_last_full[n][this_model],
                    'stdDevAccLastFull%s%03d' % (this_model, n_agents_test[n]):
                                               std_dev_cost_last_full[n][this_model],
                    'meanAccLastEnd%s%04d' % (this_model, n_agents_test[n]):
                                               mean_cost_last_end[n][this_model],
                    'stdDevAccLastEnd%s%04d' % (this_model, n_agents_test[n]):
                                               std_dev_cost_last_end[n][this_model]})

    # Save the printed info into the .txt file as well
    with open(vars_file, 'a+') as file:
        file.write("\n[%3d Agents] Final evaluations (%02d data splits)" % \
                                               (n_agents_test[n], n_realizations))
        file.write("\t%8s: %8.4f (+-%6.4f) [Optm/Full]" % (
                   'Optimal',
                   mean_cost_opt_full[n],
                   std_dev_cost_opt_full[n]))
        file.write("\t%9s %8.4f (+-%6.4f) [Optm/End ]" % (
                   '',
                   mean_cost_opt_end[n],
                   std_dev_cost_opt_end[n]))
        for this_model in model_list:
            file.write(
              "\t%s: %8.4f (+-%6.4f) [Best/Full] %8.4f (+-%6.4f) [Last/Full]"%(
                    this_model,
                    mean_cost_best_full[n][this_model],
                    std_dev_cost_best_full[n][this_model],
                    mean_cost_last_full[n][this_model],
                    std_dev_cost_last_full[n][this_model]))
            file.write(
              "\t%9s %8.4f (+-%6.4f) [Best/End ] %8.4f (+-%6.4f) [Last/End ]"%(
                    '',
                    mean_cost_best_end[n][this_model],
                    std_dev_cost_best_end[n][this_model],
                    mean_cost_last_end[n][this_model],
                    std_dev_cost_last_end[n][this_model]))
        file.write('\n')

#%%##################################################################
#                                                                   #
#                    PLOT                                           #
#                                                                   #
#####################################################################

# Finally, we might want to plot several quantities of interest

if do_figs:

    ###################
    # DATA PROCESSING #
    ###################

    #\\\ FIGURES DIRECTORY:
    save_dir_figs = os.path.join(save_dir,'figs')
    # If it doesn't exist, create it.
    if not os.path.exists(save_dir_figs):
        os.makedirs(save_dir_figs)

    #\\\ COMPUTE STATISTICS:
    # The first thing to do is to transform those into a matrix with all the
    # realizations, so create the variables to save that.
    mean_loss_train = {}
    mean_eval_valid = {}
    std_dev_loss_train = {}
    std_dev_eval_valid = {}
    # Initialize the variables
    for this_model in model_list:
        # Transform into np.array
        loss_train[this_model] = np.array(loss_train[this_model])
        eval_valid[this_model] = np.array(eval_valid[this_model])
        # Each of one of these variables should be of shape
        # nDataSplits x numberOfTrainingSteps
        # And compute the statistics
        mean_loss_train[this_model] = np.mean(loss_train[this_model], axis = 0)
        mean_eval_valid[this_model] = np.mean(eval_valid[this_model], axis = 0)
        std_dev_loss_train[this_model] = np.std(loss_train[this_model], axis = 0)
        std_dev_eval_valid[this_model] = np.std(eval_valid[this_model], axis = 0)

    ####################
    # SAVE FIGURE DATA #
    ####################

    # And finally, we can plot. But before, let's save the variables mean and
    # stdDev so, if we don't like the plot, we can re-open them, and re-plot
    # them, a piacere.
    #   Pickle, first:
    vars_pickle = {}
    vars_pickle['n_epochs'] = n_epochs
    vars_pickle['n_batches'] = n_batches
    vars_pickle['mean_loss_train'] = mean_loss_train
    vars_pickle['std_dev_loss_train'] = std_dev_loss_train
    vars_pickle['mean_eval_valid'] = mean_eval_valid
    vars_pickle['std_dev_eval_valid'] = std_dev_eval_valid
    with open(os.path.join(save_dir_figs,'figVars.pkl'), 'wb') as figVarsFile:
        pickle.dump(vars_pickle, figVarsFile)

    ########
    # PLOT #
    ########

    # Compute the x-axis
    x_train = np.arange(0, n_epochs * n_batches, x_axis_multiplier_train)
    x_valid = np.arange(0, n_epochs * n_batches, \
                          validation_interval*x_axis_multiplier_valid)

    # If we do not want to plot all the elements (to avoid overcrowded plots)
    # we need to recompute the x axis and take those elements corresponding
    # to the training steps we want to plot
    if x_axis_multiplier_train > 1:
        # Actual selected samples
        select_samples_train = x_train
        # Go and fetch tem
        for this_model in model_list:
            mean_loss_train[this_model] = mean_loss_train[this_model]\
                                                    [select_samples_train]
            std_dev_loss_train[this_model] = std_dev_loss_train[this_model]\
                                                        [select_samples_train]
    # And same for the validation, if necessary.
    if x_axis_multiplier_valid > 1:
        select_samples_valid = np.arange(0, len(mean_eval_valid[this_model]), \
                                       x_axis_multiplier_valid)
        for this_model in model_list:
            mean_eval_valid[this_model] = mean_eval_valid[this_model]\
                                                    [select_samples_valid]
            std_dev_eval_valid[this_model] = std_dev_eval_valid[this_model]\
                                                        [select_samples_valid]

    #\\\ LOSS (Training and validation) for EACH MODEL
    for key in mean_loss_train.keys():
        loss_fig = plt.figure(figsize=(1.61*fig_size, 1*fig_size))
        plt.errorbar(x_train, mean_loss_train[key], yerr = std_dev_loss_train[key],
                     color = '#01256E', linewidth = line_width,
                     marker = marker_shape, markersize = marker_size)
        plt.ylabel(r'Loss')
        plt.xlabel(r'Training steps')
        plt.legend([r'Training'])
        plt.title(r'%s' % key)
        loss_fig.savefig(os.path.join(save_dir_figs,'loss%s.pdf' % key),
                        bbox_inches = 'tight')
        plt.close(fig = loss_fig)

    #\\\ Cost (Training and validation) for EACH MODEL
    for key in mean_eval_valid.keys():
        acc_fig = plt.figure(figsize=(1.61*fig_size, 1*fig_size))
        plt.errorbar(x_valid, mean_eval_valid[key], yerr = std_dev_eval_valid[key],
                     color = '#01256E', linewidth = line_width,
                     marker = marker_shape, markersize = marker_size)
        plt.ylabel(r'Cost')
        plt.xlabel(r'Training steps')
        plt.legend([r'Training', r'Validation'])
        plt.title(r'%s' % key)
        acc_fig.savefig(os.path.join(save_dir_figs,'eval%s.pdf' % key),
                        bbox_inches = 'tight')
        plt.close(fig = acc_fig)

    # LOSS (training) for ALL MODELS
    all_loss_train = plt.figure(figsize=(1.61*fig_size, 1*fig_size))
    for key in mean_loss_train.keys():
        plt.errorbar(x_train, mean_loss_train[key], yerr = std_dev_loss_train[key],
                     linewidth = line_width,
                     marker = marker_shape, markersize = marker_size)
    plt.ylabel(r'Loss')
    plt.xlabel(r'Training steps')
    plt.legend(list(mean_loss_train.keys()))
    all_loss_train.savefig(os.path.join(save_dir_figs,'all_loss_train.pdf'),
                    bbox_inches = 'tight')
    plt.close(fig = all_loss_train)

    # Cost (validation) for ALL MODELS
    all_eval_valid = plt.figure(figsize=(1.61*fig_size, 1*fig_size))
    for key in mean_eval_valid.keys():
        plt.errorbar(x_valid, mean_eval_valid[key], yerr = std_dev_eval_valid[key],
                     linewidth = line_width,
                     marker = marker_shape, markersize = marker_size)
    plt.ylabel(r'Cost')
    plt.xlabel(r'Training steps')
    plt.legend(list(mean_eval_valid.keys()))
    all_eval_valid.savefig(os.path.join(save_dir_figs,'all_eval_valid.pdf'),
                    bbox_inches = 'tight')
    plt.close(fig = all_eval_valid)

# Finish measuring time
end_run_time = datetime.datetime.now()

total_run_time = abs(end_run_time - start_run_time)
total_run_time_h = int(divmod(total_run_time.total_seconds(), 3600)[0])
total_run_time_m, total_run_time_s = \
               divmod(total_run_time.total_seconds() - total_run_time_h * 3600., 60)
total_run_time_m = int(total_run_time_m)

if do_print:
    print(" ")
    print("Simulation started: %s" %start_run_time.strftime("%Y/%m/%d %H:%M:%S"))
    print("Simulation ended:   %s" % end_run_time.strftime("%Y/%m/%d %H:%M:%S"))
    print("Total time: %dh %dm %.2fs" % (total_run_time_h,
                                         total_run_time_m,
                                         total_run_time_s))

# And save this info into the .txt file as well
with open(vars_file, 'a+') as file:
    file.write("Simulation started: %s\n" %
                                     start_run_time.strftime("%Y/%m/%d %H:%M:%S"))
    file.write("Simulation ended:   %s\n" %
                                       end_run_time.strftime("%Y/%m/%d %H:%M:%S"))
    file.write("Total time: %dh %dm %.2fs" % (total_run_time_h,
                                              total_run_time_m,
                                              total_run_time_s))
