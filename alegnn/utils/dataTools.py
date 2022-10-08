# 2021/03/04~
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu
# Kate Tolstaya, eig@seas.upenn.edu
"""
dataTools.py Data management module

Functions:
    
normalize_data: normalize data along a specified axis
change_data_type: change data type of data

Classes (datasets):

FacebookEgo (class): loads the Facebook adjacency matrix of EgoNets
SourceLocalization (class): creates the datasets for a source localization 
    problem
Authorship (class): loads and splits the dataset for the authorship attribution
    problem
MovieLens (class): Loads and handles handles the MovieLens-100k dataset

Flocking (class): creates trajectories for the problem of flocking

TwentyNews (class): handles the 20NEWS dataset

Epidemics (class): loads the edge list of the friendship network of the high 
    school in Marseille and generates the epidemic spread data based on the SIR 
    model
"""

import os

import matplotlib

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

import numpy as np
import torch

import alegnn.utils.graph_tools as graph

zero_tolerance = 1e-9  # Values below this number are considered zero.


def normalize_data(x, ax):
    """
    normalize_data(x, ax): normalize data x (subtract mean and divide by standard 
    deviation) along the specified axis ax
    """

    this_shape = x.shape  # get the shape
    assert ax < len(this_shape)  # check that the axis that we want to normalize
    # is there
    data_type = type(x)  # get data type so that we don't have to convert

    if 'numpy' in repr(data_type):

        # Compute the statistics
        x_mean = np.mean(x, axis=ax)
        x_dev = np.std(x, axis=ax)
        # Add back the dimension we just took out
        x_mean = np.expand_dims(x_mean, ax)
        x_dev = np.expand_dims(x_dev, ax)

    elif 'torch' in repr(data_type):

        # Compute the statistics
        x_mean = torch.mean(x, dim=ax)
        x_dev = torch.std(x, dim=ax)
        # Add back the dimension we just took out
        x_mean = x_mean.unsqueeze(ax)
        x_dev = x_dev.unsqueeze(ax)

    # Subtract mean and divide by standard deviation
    x = (x - x_mean) / x_dev

    return x


def change_data_type(x, data_type):
    """
    change_data_type(x, data_type): change the data_type of variable x into data_type
    """

    # So this is the thing: To change data type it depends on both, what dtype
    # the variable already is, and what dtype we want to make it.
    # Torch changes type by .type(), but numpy by .astype()
    # If we have already a torch defined, and we apply a torch.tensor() to it,
    # then there will be warnings because of gradient accounting.

    # All of these facts make changing types considerably cumbersome. So we
    # create a function that just changes type and handles all this issues
    # inside.

    # If we can't recognize the type, we just make everything numpy.

    # Check if the variable has an argument called 'dtype' so that we can now
    # what type of data type the variable is
    if 'dtype' in dir(x):
        var_type = x.dtype

    # So, let's start assuming we want to convert to numpy
    if 'numpy' in repr(data_type):
        # Then, the variable con be torch, in which case we move it to cpu, to
        # numpy, and convert it to the right type.
        if 'torch' in repr(var_type):
            x = x.cpu().numpy().astype(data_type)
        # Or it could be numpy, in which case we just use .astype
        elif 'numpy' in repr(type(x)):
            x = x.astype(data_type)
    # Now, we want to convert to torch
    elif 'torch' in repr(data_type):
        # If the variable is torch in itself
        if 'torch' in repr(var_type):
            x = x.type(data_type)
        # But, if it's numpy
        elif 'numpy' in repr(type(x)):
            x = torch.tensor(x, dtype=data_type)

    # This only converts between numpy and torch. Any other thing is ignored
    return x


def invert_tensor_ew(x):
    # Elementwise inversion of a tensor where the 0 elements are kept as zero.
    # Warning: Creates a copy of the tensor
    x_inv = x.copy()  # Copy the matrix to invert
    # Replace zeros for ones.
    x_inv[x < zero_tolerance] = 1.  # Replace zeros for ones
    x_inv = 1. / x_inv  # Now we can invert safely
    x_inv[x < zero_tolerance] = 0.  # Put back the zeros

    return x_inv


class _data:
    # Internal supraclass from which all data sets will inherit.
    # There are certain methods that all Data classes must have:
    #   get_samples(), expand_dims(), to() and astype().
    # To avoid coding this methods over and over again, we create a class from
    # which the data can inherit this basic methods.

    # All the signals are always assumed to be graph signals that are written
    #   nDataPoints (x nFeatures) x nNodes
    # If we have one feature, we have the expand_dims() that adds a x1 so that
    # it can be readily processed by architectures/functions that always assume
    # a 3-dimensional signal.

    def __init__(self):
        # Minimal set of attributes that all data classes should have
        self.data_type = None
        self.device = None
        self.n_train = None
        self.n_valid = None
        self.n_test = None
        self.samples = {}
        self.samples['train'] = {}
        self.samples['train']['signals'] = None
        self.samples['train']['targets'] = None
        self.samples['valid'] = {}
        self.samples['valid']['signals'] = None
        self.samples['valid']['targets'] = None
        self.samples['test'] = {}
        self.samples['test']['signals'] = None
        self.samples['test']['targets'] = None

    def get_samples(self, samples_type, *args):
        # samples_type: train, valid, test
        # args: 0 args, give back all
        # args: 1 arg: if int, give that number of samples, chosen at random
        # args: 1 arg: if list, give those samples precisely.
        # Check that the type is one of the possible ones
        assert samples_type == 'train' or samples_type == 'valid' \
               or samples_type == 'test'
        # Check that the number of extra arguments fits
        assert len(args) <= 1
        # If there are no arguments, just return all the desired samples
        x = self.samples[samples_type]['signals']
        y = self.samples[samples_type]['targets']
        # If there's an argument, we have to check whether it is an int or a
        # list
        if len(args) == 1:
            # If it is an int, just return that number of randomly chosen
            # samples.
            if type(args[0]) == int:
                n_samples = x.shape[0]  # total number of samples
                # We can't return more samples than there are available
                assert args[0] <= n_samples
                # Randomly choose args[0] indices
                selected_indices = np.random.choice(n_samples, size=args[0],
                                                    replace=False)
                # Select the corresponding samples
                x_selected = x[selected_indices]
                y = y[selected_indices]
            else:
                # The fact that we put else here instead of elif type()==list
                # allows for np.array to be used as indices as well. In general,
                # any variable with the ability to index.
                x_selected = x[args[0]]
                # And assign the labels
                y = y[args[0]]

            # If we only selected a single element, then the nDataPoints dim
            # has been left out. So if we have less dimensions, we have to
            # put it back
            if len(x_selected.shape) < len(x.shape):
                if 'torch' in self.data_type:
                    x = x_selected.unsqueeze(0)
                else:
                    x = np.expand_dims(x_selected, axis=0)
            else:
                x = x_selected

        return x, y

    def expand_dims(self):

        # For each data set partition
        for key in self.samples.keys():
            # If there's something in them
            if self.samples[key]['signals'] is not None:
                # And if it has only two dimensions
                #   (shape: nDataPoints x nNodes)
                if len(self.samples[key]['signals'].shape) == 2:
                    # Then add a third dimension in between so that it ends
                    # up with shape
                    #   nDataPoints x 1 x nNodes
                    # and it respects the 3-dimensional format that is taken
                    # by many of the processing functions
                    if 'torch' in repr(self.data_type):
                        self.samples[key]['signals'] = \
                            self.samples[key]['signals'].unsqueeze(1)
                    else:
                        self.samples[key]['signals'] = np.expand_dims(
                            self.samples[key]['signals'],
                            axis=1)
                elif len(self.samples[key]['signals'].shape) == 3:
                    if 'torch' in repr(self.data_type):
                        self.samples[key]['signals'] = \
                            self.samples[key]['signals'].unsqueeze(2)
                    else:
                        self.samples[key]['signals'] = np.expand_dims(
                            self.samples[key]['signals'],
                            axis=2)

    def astype(self, data_type):
        # This changes the type for the minimal attributes (samples). This 
        # methods should still be initialized within the data classes, if more
        # attributes are used.

        # The labels could be integers as created from the dataset, so if they
        # are, we need to be sure they are integers also after conversion. 
        # To do this we need to match the desired data_type to its int 
        # counterpart. Typical examples are:
        #   numpy.float64 -> numpy.int64
        #   numpy.float32 -> numpy.int32
        #   torch.float64 -> torch.int64
        #   torch.float32 -> torch.int32

        target_type = str(self.samples['train']['targets'].dtype)
        if 'int' in target_type:
            if 'numpy' in repr(data_type):
                if '64' in target_type:
                    target_type = np.int64
                elif '32' in target_type:
                    target_type = np.int32
            elif 'torch' in repr(data_type):
                if '64' in target_type:
                    target_type = torch.int64
                elif '32' in target_type:
                    target_type = torch.int32
        else:  # If there is no int, just stick with the given data_type
            target_type = data_type

        # Now that we have selected the data_type, and the corresponding
        # labelType, we can proceed to convert the data into the corresponding
        # type
        for key in self.samples.keys():
            self.samples[key]['signals'] = change_data_type(
                self.samples[key]['signals'],
                data_type)
            self.samples[key]['targets'] = change_data_type(
                self.samples[key]['targets'],
                target_type)

        # Update attribute
        if data_type is not self.data_type:
            self.data_type = data_type

    def to(self, device):
        # This changes the type for the minimal attributes (samples). This 
        # methods should still be initialized within the data classes, if more
        # attributes are used.
        # This can only be done if they are torch tensors
        if 'torch' in repr(self.data_type):
            for key in self.samples.keys():
                for second_key in self.samples[key].keys():
                    self.samples[key][second_key] \
                        = self.samples[key][second_key].to(device)

            # If the device changed, save it.
            if device is not self.device:
                self.device = device


class Flocking(_data):
    """
    Flocking: Creates synthetic trajectories for the problem of coordinating
        a team of robots to fly together while avoiding collisions. See the
        following  paper for details
        
        E. Tolstaya, F. Gama, J. Paulos, G. Pappas, V. Kumar, and A. Ribeiro, 
        "Learning Decentralized Controllers for Robot Swarms with Graph Neural
        Networks," in Conf. Robot Learning 2019. Osaka, Japan: Int. Found.
        Robotics Res., 30 Oct.-1 Nov. 2019.
    
    Initialization:
        
    Input:
        n_agents (int): Number of agents
        comm_radius (float): communication radius (in meters)
        repel_dist (float): minimum target separation of agents (in meters)
        n_train (int): number of training trajectories
        n_valid (int): number of validation trajectories
        n_test (int): number of testing trajectories
        duration (float): duration of each trajectory (in seconds)
        sampling_time (float): time between consecutive time instants (in sec)
        init_geometry ('circular', 'rectangular'): initial positioning geometry
            (default: 'circular')
        init_vel_value (float): maximum initial velocity (in meters/seconds,
            default: 3.)
        init_min_dist (float): minimum initial distance between agents (in
            meters, default: 0.1)
        accel_max (float): maximum possible acceleration (in meters/seconds^2,
            default: 10.)
        normalize_graph (bool): if True normalizes the communication graph
            adjacency matrix by the maximum eigenvalue (default: True)
        do_print (bool): If True prints messages (default: True)
        data_type (dtype): datatype for the samples created (default: np.float64)
        device (device): if torch.Tensor datatype is selected, this is on what
            device the data is saved (default: 'cpu')
            
    Methods:
        
    signals, targets = .get_samples(samples_type[, optional_arguments])
        Input:
            samples_type (string): 'train', 'valid' or 'test' to determine from
                which dataset to get the samples from
            optional_arguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        Output:
            signals (dtype.array): number_samples x 6 x number_nodes
            targets (dtype.array): number_samples x 2 x number_nodes
            'signals' are the state variables as described in the corresponding
            paper; 'targets' is the 2-D acceleration for each node
            
    cost = .evaluate(vel = None, accel = None, init_vel = None,
                     sampling_time = None)
        Input:
            vel (array): velocities; n_samples x t_samples x 2 x n_agents
            accel (array): accelerations; n_samples x t_samples x 2 x n_agents
            init_vel (array): initial velocities; n_samples x 2 x n_agents
            sampling_time (float): sampling time
            >> Obs.: Either vel or (accel and init_vel) have to be specified
            for the cost to be computed, if all of them are specified, only
            vel is used
        Output:
            cost (float): flocking cost as specified in eq. (13)

    .astype(data_type): change the type of the data matrix arrays.
        Input:
            data_type (dtype): target type of the variables (e.g. torch.float64,
                numpy.float64, etc.)

    .to(device): if dtype is torch.tensor, move them to the specified device.
        Input:
            device (string): target device to move the variables to (e.g. 
                'cpu', 'cuda:0', etc.)

    state = .compute_states(pos, vel, graph_matrix, ['do_print'])
        Input:
            pos (array): positions; n_samples x t_samples x 2 x n_agents
            vel (array): velocities; n_samples x t_samples x 2 x n_agents
            graph_matrix (array): matrix description of communication graph;
                n_samples x t_samples x n_agents x n_agents
            'do_print' (bool): optional argument to print outputs; if not used
                uses the same status set for the entire class in the
                initialization
        Output:
            state (array): states; n_samples x t_samples x 6 x n_agents
    
    graph_matrix = .compute_communication_graph(pos, comm_radius, normalize_graph,
                    ['kernel_type' = 'gaussian', 'weighted' = False, 'do_print'])
        Input:
            pos (array): positions; n_samples x t_samples x 2 x n_agents
            comm_radius (float): communication radius (in meters)
            normalize_graph (bool): if True normalize adjacency matrix by 
                largest eigenvalue
            'kernel_type' ('gaussian'): kernel to apply to the distance in order
                to compute the weights of the adjacency matrix, default is
                the 'gaussian' kernel; other kernels have to be coded, and also
                the parameters of the kernel have to be included as well, in
                the case of the gaussian kernel, 'kernelScale' determines the
                scale (default: 1.)
            'weighted' (bool): if True the graph is weighted according to the
                kernel type; if False, it's just a binary adjacency matrix
            'do_print' (bool): optional argument to print outputs; if not used
                uses the same status set for the entire class in the
                initialization
        Output:
            graph_matrix (array): adjacency matrix of the communication graph;
                n_samples x t_samples x n_agents x n_agents
    
    this_data = .get_data(name, samples_type[, optional_arguments])
        Input:
            name (string): variable name to get (for example, 'pos', 'vel', 
                etc.)
            samples_type ('train', 'test' or 'valid')
            optional_arguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        Output:
            this_data (array): specific type of data requested
    
    pos, vel[, accel, state, graph] = compute_trajectory(init_pos, init_vel,
                                            duration[, 'archit', 'accel',
                                            'do_print'])
        Input:
            init_pos (array): initial positions; n_samples x 2 x n_agents
            init_vel (array): initial velocities; n_samples x 2 x n_agents
            duration (float): duration of trajectory (in seconds)
            Optional arguments: (either 'accel' or 'archit' have to be there)
            'archit' (nn.Module): torch architecture that computes the output
                from the states
            'accel' (array): accelerations; n_samples x t_samples x 2 x n_agents
            'do_print' (bool): optional argument to print outputs; if not used
                uses the same status set for the entire class in the
                initialization
        Output:
            pos (array): positions; n_samples x t_samples x 2 x n_agents
            vel (array): velocities; n_samples x t_samples x 2 x n_agents
            Optional outputs (only if 'archit' was used)
            accel (array): accelerations; n_samples x t_samples x 2 x n_agents
            state (array): state; n_samples x t_samples x 6 x n_agents
            graph (array): adjacency matrix of communication graph;
                n_samples x t_samples x n_agents x n_agents
            
    u_diff, u_diff_sq = .compute_differences (u):
        Input:
            u (array): n_samples (x t_samples) x 2 x n_agents
        Output:
            u_diff (array): pairwise differences between the agent entries of u;
                n_samples (x t_samples) x 2 x n_agents x n_agents
            u_diff_sq (array): squared distances between agent entries of u;
                n_samples (x t_samples) x n_agents x n_agents
    
    pos, vel, accel = .compute_optimal_trajectory(init_pos, init_vel, duration, 
                                                sampling_time, repel_dist,
                                                accel_max = 100.)
        Input:
            init_pos (array): initial positions; n_samples x 2 x n_agents
            init_vel (array): initial velocities; n_samples x 2 x n_agents
            duration (float): duration of trajectory (in seconds)
            sampling_time (float): time elapsed between consecutive time 
                instants (in seconds)
            repel_dist (float): minimum desired distance between agents (in m)
            accel_max (float, default = 100.): maximum possible acceleration
        Output:
            pos (array): positions; n_samples x t_samples x 2 x n_agents
            vel (array): velocities; n_samples x t_samples x 2 x n_agents
            accel (array): accelerations; n_samples x t_samples x 2 x n_agents
            
    init_pos, init_vel = .compute_initial_positions(n_agents, n_samples, comm_radius,
                                                min_dist = 0.1,
                                                geometry = 'rectangular',
                                                x_max_init_vel = 3.,
                                                y_max_init_vel = 3.)
        Input:
            n_agents (int): number of agents
            n_samples (int): number of sample trajectories
            comm_radius (float): communication radius (in meters)
            min_dist (float): minimum initial distance between agents (in m)
            geometry ('rectangular', 'circular'): initial geometry
            x_max_init_vel (float): maximum velocity in the x-axis
            y_max_init_vel (float): maximum velocity in the y-axis
        Output:
            init_pos (array): initial positions; n_samples x 2 x n_agents
            init_vel (array): initial velocities; n_samples x 2 x n_agents
    
    .save_video(save_dir, pos, [, optional_arguments], comm_graph = None,
               [optional_key_arguments])
        Input:
            save_dir (os.path, string): directory where to save the trajectory
                videos
            pos (array): positions; n_samples x t_samples x 2 x n_agents
            optional_arguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
            comm_graph (array): adjacency matrix of communication graph;
                n_samples x t_samples x n_agents x n_agents
                if not None, then this array is used to produce snapshots of
                the video that include the communication graph at that time
                instant
            'do_print' (bool): optional argument to print outputs; if not used
                uses the same status set for the entire class in the
                initialization
            'video_speed' (float): how faster or slower the video is reproduced
                (default: 1.)
            'show_video_speed' (bool): if True shows the legend with the video
                speed in the video; by default it will show it whenever the
                video speed is different from 1.
            'vel' (array): velocities; n_samples x t_samples x 2 x n_agents
            'show_cost' (bool): if True and velocities are set, the snapshots
                will show the instantaneous cost (default: True)
            'show_arrows' (bool): if True and velocities are set, the snapshots
                will show the arrows of the velocities (default: True)
            
            
    """

    def __init__(self, n_agents, comm_radius, repel_dist,
                 n_train, n_valid, n_test,
                 duration, sampling_time,
                 init_geometry='circular', init_vel_value=3., init_min_dist=0.1,
                 accel_max=10.,
                 normalize_graph=True, do_print=True,
                 data_type=np.float64, device='cpu'):

        # Initialize parent class
        super().__init__()
        # Save the relevant input information
        #   Number of nodes
        self.n_agents = n_agents
        self.comm_radius = comm_radius
        self.repel_dist = repel_dist
        #   Number of samples
        self.n_train = n_train
        self.n_valid = n_valid
        self.n_test = n_test
        n_samples = n_train + n_valid + n_test
        #   Geometry
        self.map_width = None
        self.map_height = None
        #   Agents
        self.init_geometry = init_geometry
        self.init_vel_value = init_vel_value
        self.init_min_dist = init_min_dist
        self.accel_max = accel_max
        #   Duration of the trajectory
        self.duration = float(duration)
        self.sampling_time = sampling_time
        #   Data
        self.normalize_graph = normalize_graph
        self.data_type = data_type
        self.device = device
        #   Options
        self.do_print = do_print

        #   Places to store the data
        self.init_pos = None
        self.init_vel = None
        self.pos = None
        self.vel = None
        self.accel = None
        self.comm_graph = None
        self.state = None

        if self.do_print:
            print("\tComputing initial conditions...", end=' ', flush=True)

        # Compute the initial positions
        init_pos_all, init_vel_all = self.compute_initial_positions(
            self.n_agents, n_samples, self.comm_radius,
            min_dist=self.init_min_dist,
            geometry=self.init_geometry,
            x_max_init_vel=self.init_vel_value,
            y_max_init_vel=self.init_vel_value
        )
        #   Once we have all positions and velocities, we will need to split 
        #   them in the corresponding datasets (train, valid and test)
        self.init_pos = {}
        self.init_vel = {}

        if self.do_print:
            print("OK", flush=True)
            # Erase the label first, then print it
            print("\tComputing the optimal trajectories...",
                  end=' ', flush=True)

        # Compute the optimal trajectory
        pos_all, vel_all, accel_all = self.compute_optimal_trajectory(
            init_pos_all, init_vel_all, self.duration,
            self.sampling_time, self.repel_dist,
            accel_max=self.accel_max)

        self.pos = {}
        self.vel = {}
        self.accel = {}

        if self.do_print:
            print("OK", flush=True)
            # Erase the label first, then print it
            print("\tComputing the communication graphs...",
                  end=' ', flush=True)

        # Compute communication graph
        comm_graph_all = self.compute_communication_graph(pos_all, self.comm_radius,
                                                          self.normalize_graph)

        self.comm_graph = {}

        if self.do_print:
            print("OK", flush=True)
            # Erase the label first, then print it
            print("\tComputing the agent states...", end=' ', flush=True)

        # Compute the states
        state_all = self.compute_states(pos_all, vel_all, comm_graph_all)

        self.state = {}

        if self.do_print:
            # Erase the label
            print("OK", flush=True)

        # Separate the states into training, validation and testing samples
        # and save them
        #   Training set
        self.samples['train']['signals'] = state_all[0:self.n_train].copy()
        self.samples['train']['targets'] = accel_all[0:self.n_train].copy()
        self.init_pos['train'] = init_pos_all[0:self.n_train]
        self.init_vel['train'] = init_vel_all[0:self.n_train]
        self.pos['train'] = pos_all[0:self.n_train]
        self.vel['train'] = vel_all[0:self.n_train]
        self.accel['train'] = accel_all[0:self.n_train]
        self.comm_graph['train'] = comm_graph_all[0:self.n_train]
        self.state['train'] = state_all[0:self.n_train]
        #   Validation set
        start_sample = self.n_train
        end_sample = self.n_train + self.n_valid
        self.samples['valid']['signals'] = state_all[start_sample:end_sample].copy()
        self.samples['valid']['targets'] = accel_all[start_sample:end_sample].copy()
        self.init_pos['valid'] = init_pos_all[start_sample:end_sample]
        self.init_vel['valid'] = init_vel_all[start_sample:end_sample]
        self.pos['valid'] = pos_all[start_sample:end_sample]
        self.vel['valid'] = vel_all[start_sample:end_sample]
        self.accel['valid'] = accel_all[start_sample:end_sample]
        self.comm_graph['valid'] = comm_graph_all[start_sample:end_sample]
        self.state['valid'] = state_all[start_sample:end_sample]
        #   Testing set
        start_sample = self.n_train + self.n_valid
        end_sample = self.n_train + self.n_valid + self.n_test
        self.samples['test']['signals'] = state_all[start_sample:end_sample].copy()
        self.samples['test']['targets'] = accel_all[start_sample:end_sample].copy()
        self.init_pos['test'] = init_pos_all[start_sample:end_sample]
        self.init_vel['test'] = init_vel_all[start_sample:end_sample]
        self.pos['test'] = pos_all[start_sample:end_sample]
        self.vel['test'] = vel_all[start_sample:end_sample]
        self.accel['test'] = accel_all[start_sample:end_sample]
        self.comm_graph['test'] = comm_graph_all[start_sample:end_sample]
        self.state['test'] = state_all[start_sample:end_sample]

        # Change data to specified type and device
        self.astype(self.data_type)
        self.to(self.device)

    def astype(self, data_type):

        # Change all other signals to the correct place
        dataset_type = ['train', 'valid', 'test']
        for key in dataset_type:
            self.init_pos[key] = change_data_type(self.init_pos[key], data_type)
            self.init_vel[key] = change_data_type(self.init_vel[key], data_type)
            self.pos[key] = change_data_type(self.pos[key], data_type)
            self.vel[key] = change_data_type(self.vel[key], data_type)
            self.accel[key] = change_data_type(self.accel[key], data_type)
            self.comm_graph[key] = change_data_type(self.comm_graph[key], data_type)
            self.state[key] = change_data_type(self.state[key], data_type)

        # And call the parent
        super().astype(data_type)

    def to(self, device):

        # Check the data is actually torch
        if 'torch' in repr(self.data_type):
            dataset_type = ['train', 'valid', 'test']
            # Move the data
            for key in dataset_type:
                self.init_pos[key].to(device)
                self.init_vel[key].to(device)
                self.pos[key].to(device)
                self.vel[key].to(device)
                self.accel[key].to(device)
                self.comm_graph[key].to(device)
                self.state[key].to(device)

            super().to(device)

    def expand_dims(self):
        # Just avoid the 'expandDims' method in the parent class
        pass

    def compute_states(self, pos, vel, graph_matrix, **kwargs):

        # We get the following inputs.
        # positions: n_samples x t_samples x 2 x n_agents
        # velocities: n_samples x t_samples x 2 x n_agents
        # graph_matrix: nSaples x t_samples x n_agents x n_agents

        # And we want to build the state, which is a vector of dimension 6 on 
        # each node, that is, the output shape is
        #   n_samples x t_samples x 6 x n_agents

        # The print for this one can be settled independently, if not, use the
        # default of the data object
        if 'do_print' in kwargs.keys():
            do_print = kwargs['do_print']
        else:
            do_print = self.do_print

        # Check correct dimensions
        assert len(pos.shape) == len(vel.shape) == len(graph_matrix.shape) == 4
        n_samples = pos.shape[0]
        t_samples = pos.shape[1]
        assert pos.shape[2] == 2
        n_agents = pos.shape[3]
        assert vel.shape[0] == graph_matrix.shape[0] == n_samples
        assert vel.shape[1] == graph_matrix.shape[1] == t_samples
        assert vel.shape[2] == 2
        assert vel.shape[3] == graph_matrix.shape[2] == graph_matrix.shape[3] \
               == n_agents

        # If we have a lot of batches and a particularly long sequence, this
        # is bound to fail, memory-wise, so let's do it time instant by time
        # instant if we have a large number of time instants, and split the
        # batches
        max_time_samples = 200  # Set the maximum number of t.Samples before
        # which to start doing this time by time.
        max_batch_size = 100  # Maximum number of samples to process at a given
        # time

        # Compute the number of samples, and split the indices accordingly
        if n_samples < max_batch_size:
            n_batches = 1
            batch_size = [n_samples]
        elif n_samples % max_batch_size != 0:
            # If we know it's not divisible, then we do floor division and
            # add one more batch
            n_batches = n_samples // max_batch_size + 1
            batch_size = [max_batch_size] * n_batches
            # But the last batch is actually smaller, so just add the 
            # remaining ones
            batch_size[-1] = n_samples - sum(batch_size[0:-1])
        # If they fit evenly, then just do so.
        else:
            n_batches = int(n_samples / max_batch_size)
            batch_size = [max_batch_size] * n_batches
        # batch_index is used to determine the first and last element of each
        # batch. We need to add the 0 because it's the first index.
        batch_index = np.cumsum(batch_size).tolist()
        batch_index = [0] + batch_index

        # Create the output state variable
        state = np.zeros((n_samples, t_samples, 6, n_agents))

        for b in range(n_batches):

            # Pick the batch elements
            pos_batch = pos[batch_index[b]:batch_index[b + 1]]
            vel_batch = vel[batch_index[b]:batch_index[b + 1]]
            graph_matrix_batch = graph_matrix[batch_index[b]:batch_index[b + 1]]

            if t_samples > max_time_samples:

                # For each time instant
                for t in range(t_samples):

                    # Now, we need to compute the differences, in velocities and in 
                    # positions, for each agent, for each time instant
                    pos_diff, pos_dist_sq = \
                        self.compute_differences(pos_batch[:, t, :, :])
                    #   pos_diff: batch_size[b] x 2 x n_agents x n_agents
                    #   pos_dist_sq: batch_size[b] x n_agents x n_agents
                    vel_diff, _ = self.compute_differences(vel_batch[:, t, :, :])
                    #   vel_diff: batch_size[b] x 2 x n_agents x n_agents

                    # Next, we need to get ride of all those places where there are
                    # no neighborhoods. That is given by the nonzero elements of the 
                    # graph matrix.
                    graph_matrix_time = (np.abs(graph_matrix_batch[:, t, :, :]) \
                                         > zero_tolerance) \
                        .astype(pos.dtype)
                    #   graph_matrix: batch_size[b] x n_agents x n_agents
                    # We also need to invert the squares of the distances
                    pos_dist_sq_inv = invert_tensor_ew(pos_dist_sq)
                    #   pos_dist_sq_inv: batch_size[b] x n_agents x n_agents

                    # Now we add the extra dimensions so that all the 
                    # multiplications are adequate
                    graph_matrix_time = np.expand_dims(graph_matrix_time, 1)
                    #   graph_matrix: batch_size[b] x 1 x n_agents x n_agents

                    # Then, we can get rid of non-neighbors
                    pos_diff = pos_diff * graph_matrix_time
                    pos_dist_sq_inv = np.expand_dims(pos_dist_sq_inv, 1) \
                                      * graph_matrix_time
                    vel_diff = vel_diff * graph_matrix_time

                    # Finally, we can compute the states
                    state_vel = np.sum(vel_diff, axis=3)
                    #   state_vel: batch_size[b] x 2 x n_agents
                    state_pos_fourth = np.sum(pos_diff * (pos_dist_sq_inv ** 2),
                                              axis=3)
                    #   state_pos_fourth: batch_size[b] x 2 x n_agents
                    state_pos_sq = np.sum(pos_diff * pos_dist_sq_inv, axis=3)
                    #   state_pos_sq: batch_size[b] x 2 x n_agents

                    # Concatentate the states and return the result
                    state[batch_index[b]:batch_index[b + 1], t, :, :] = \
                        np.concatenate((state_vel,
                                        state_pos_fourth,
                                        state_pos_sq),
                                       axis=1)
                    #   batch_size[b] x 6 x n_agents

                    if do_print:
                        # Sample percentage count
                        percentage_count = int(100 * (t + 1 + b * t_samples) \
                                               / (n_batches * t_samples))

                        if t == 0 and b == 0:
                            # It's the first one, so just print it
                            print("%3d%%" % percentage_count,
                                  end='', flush=True)
                        else:
                            # Erase the previous characters
                            print('\b \b' * 4 + "%3d%%" % percentage_count,
                                  end='', flush=True)

            else:

                # Now, we need to compute the differences, in velocities and in 
                # positions, for each agent, for each time instante
                pos_diff, pos_dist_sq = self.compute_differences(pos_batch)
                #   pos_diff: batch_size[b] x t_samples x 2 x n_agents x n_agents
                #   pos_dist_sq: batch_size[b] x t_samples x n_agents x n_agents
                vel_diff, _ = self.compute_differences(vel_batch)
                #   vel_diff: batch_size[b] x t_samples x 2 x n_agents x n_agents

                # Next, we need to get ride of all those places where there are
                # no neighborhoods. That is given by the nonzero elements of the 
                # graph matrix.
                graph_matrix_batch = (np.abs(graph_matrix_batch) > zero_tolerance) \
                    .astype(pos.dtype)
                #   graph_matrix: batch_size[b] x t_samples x n_agents x n_agents
                # We also need to invert the squares of the distances
                pos_dist_sq_inv = invert_tensor_ew(pos_dist_sq)
                #   pos_dist_sq_inv: batch_size[b] x t_samples x n_agents x n_agents

                # Now we add the extra dimensions so that all the multiplications
                # are adequate
                graph_matrix_batch = np.expand_dims(graph_matrix_batch, 2)
                #   graph_matrix:batch_size[b] x t_samples x 1 x n_agents x n_agents

                # Then, we can get rid of non-neighbors
                pos_diff = pos_diff * graph_matrix_batch
                pos_dist_sq_inv = np.expand_dims(pos_dist_sq_inv, 2) \
                                  * graph_matrix_batch
                vel_diff = vel_diff * graph_matrix_batch

                # Finally, we can compute the states
                state_vel = np.sum(vel_diff, axis=4)
                #   state_vel: batch_size[b] x t_samples x 2 x n_agents
                state_pos_fourth = np.sum(pos_diff * (pos_dist_sq_inv ** 2), axis=4)
                #   state_pos_fourth: batch_size[b] x t_samples x 2 x n_agents
                state_pos_sq = np.sum(pos_diff * pos_dist_sq_inv, axis=4)
                #   state_pos_sq: batch_size[b] x t_samples x 2 x n_agents

                # Concatentate the states and return the result
                state[batch_index[b]:batch_index[b + 1]] = \
                    np.concatenate((state_vel,
                                    state_pos_fourth,
                                    state_pos_sq),
                                   axis=2)
                #   state: batch_size[b] x t_samples x 6 x n_agents

                if do_print:
                    # Sample percentage count
                    percentage_count = int(100 * (b + 1) / n_batches)

                    if b == 0:
                        # It's the first one, so just print it
                        print("%3d%%" % percentage_count,
                              end='', flush=True)
                    else:
                        # Erase the previous characters
                        print('\b \b' * 4 + "%3d%%" % percentage_count,
                              end='', flush=True)

        # Print
        if do_print:
            # Erase the percentage
            print('\b \b' * 4, end='', flush=True)

        return state

    def compute_communication_graph(self, pos, comm_radius, normalize_graph,
                                    **kwargs):

        # Take in the position and the communication radius, and return the
        # trajectory of communication graphs
        # Input will be of shape
        #   n_samples x t_samples x 2 x n_agents
        # Output will be of shape
        #   n_samples x t_samples x n_agents x n_agents

        assert comm_radius > 0
        assert len(pos.shape) == 4
        n_samples = pos.shape[0]
        t_samples = pos.shape[1]
        assert pos.shape[2] == 2
        n_agents = pos.shape[3]

        # Graph type options
        #   Kernel type (only Gaussian implemented so far)
        if 'kernel_type' in kwargs.keys():
            kernel_type = kwargs['kernel_type']
        else:
            kernel_type = 'gaussian'
        #   Decide if the graph is weighted or not
        if 'weighted' in kwargs.keys():
            weighted = kwargs['weighted']
        else:
            weighted = False

        # If it is a Gaussian kernel, we need to determine the scale
        if kernel_type == 'gaussian':
            if 'kernel_scale' in kwargs.keys():
                kernel_scale = kwargs['kernel_scale']
            else:
                kernel_scale = 1.

        # The print for this one can be settled independently, if not, use the
        # default of the data object
        if 'do_print' in kwargs.keys():
            do_print = kwargs['do_print']
        else:
            do_print = self.do_print

        # If we have a lot of batches and a particularly long sequence, this
        # is bound to fail, memory-wise, so let's do it time instant by time
        # instant if we have a large number of time instants, and split the
        # batches
        max_time_samples = 200  # Set the maximum number of t.Samples before
        # which to start doing this time by time.
        max_batch_size = 100  # Maximum number of samples to process at a given
        # time

        # Compute the number of samples, and split the indices accordingly
        if n_samples < max_batch_size:
            n_batches = 1
            batch_size = [n_samples]
        elif n_samples % max_batch_size != 0:
            # If we know it's not divisible, then we do floor division and
            # add one more batch
            n_batches = n_samples // max_batch_size + 1
            batch_size = [max_batch_size] * n_batches
            # But the last batch is actually smaller, so just add the 
            # remaining ones
            batch_size[-1] = n_samples - sum(batch_size[0:-1])
        # If they fit evenly, then just do so.
        else:
            n_batches = int(n_samples / max_batch_size)
            batch_size = [max_batch_size] * n_batches
        # batch_index is used to determine the first and last element of each
        # batch. We need to add the 0 because it's the first index.
        batch_index = np.cumsum(batch_size).tolist()
        batch_index = [0] + batch_index

        # Create the output state variable
        graph_matrix = np.zeros((n_samples, t_samples, n_agents, n_agents))

        for b in range(n_batches):

            # Pick the batch elements
            pos_batch = pos[batch_index[b]:batch_index[b + 1]]

            if t_samples > max_time_samples:
                # If the trajectories are longer than 200 points, then do it 
                # time by time.

                # For each time instant
                for t in range(t_samples):

                    # Let's start by computing the distance squared
                    _, dist_sq = self.compute_differences(pos_batch[:, t, :, :])
                    # Apply the Kernel
                    if kernel_type == 'gaussian':
                        graph_matrix_time = np.exp(-kernel_scale * dist_sq)
                    else:
                        graph_matrix_time = dist_sq
                    # Now let's place zeros in all places whose distance is greater
                    # than the radius
                    graph_matrix_time[dist_sq > (comm_radius ** 2)] = 0.
                    # Set the diagonal elements to zero
                    graph_matrix_time[:, \
                    np.arange(0, n_agents), np.arange(0, n_agents)] \
                        = 0.
                    # If it is unweighted, force all nonzero values to be 1
                    if not weighted:
                        graph_matrix_time = (graph_matrix_time > zero_tolerance) \
                            .astype(dist_sq.dtype)

                    if normalize_graph:
                        is_symmetric = np.allclose(graph_matrix_time,
                                                   np.transpose(graph_matrix_time,
                                                                axes=[0, 2, 1]))
                        # Tries to make the computation faster, only the 
                        # eigenvalues (while there is a cost involved in 
                        # computing whether the matrix is symmetric, 
                        # experiments found that it is still faster to use the
                        # symmetric algorithm for the eigenvalues)
                        if is_symmetric:
                            W = np.linalg.eigvalsh(graph_matrix_time)
                        else:
                            W = np.linalg.eigvals(graph_matrix_time)
                        max_eigenvalue = np.max(np.real(W), axis=1)
                        #   batch_size[b]
                        # Reshape to be able to divide by the graph matrix
                        max_eigenvalue = max_eigenvalue.reshape((batch_size[b], 1, 1))
                        # Normalize
                        graph_matrix_time = graph_matrix_time / max_eigenvalue

                    # And put it in the corresponding time instant
                    graph_matrix[batch_index[b]:batch_index[b + 1], t, :, :] = \
                        graph_matrix_time

                    if do_print:
                        # Sample percentage count
                        percentage_count = int(100 * (t + 1 + b * t_samples) \
                                               / (n_batches * t_samples))

                        if t == 0 and b == 0:
                            # It's the first one, so just print it
                            print("%3d%%" % percentage_count,
                                  end='', flush=True)
                        else:
                            # Erase the previous characters
                            print('\b \b' * 4 + "%3d%%" % percentage_count,
                                  end='', flush=True)

            else:
                # Let's start by computing the distance squared
                _, dist_sq = self.compute_differences(pos_batch)
                # Apply the Kernel
                if kernel_type == 'gaussian':
                    graph_matrix_batch = np.exp(-kernel_scale * dist_sq)
                else:
                    graph_matrix_batch = dist_sq
                # Now let's place zeros in all places whose distance is greater
                # than the radius
                graph_matrix_batch[dist_sq > (comm_radius ** 2)] = 0.
                # Set the diagonal elements to zero
                graph_matrix_batch[:, :,
                np.arange(0, n_agents), np.arange(0, n_agents)] = 0.
                # If it is unweighted, force all nonzero values to be 1
                if not weighted:
                    graph_matrix_batch = (graph_matrix_batch > zero_tolerance) \
                        .astype(dist_sq.dtype)

                if normalize_graph:
                    is_symmetric = np.allclose(graph_matrix_batch,
                                               np.transpose(graph_matrix_batch,
                                                            axes=[0, 1, 3, 2]))
                    # Tries to make the computation faster
                    if is_symmetric:
                        W = np.linalg.eigvalsh(graph_matrix_batch)
                    else:
                        W = np.linalg.eigvals(graph_matrix_batch)
                    max_eigenvalue = np.max(np.real(W), axis=2)
                    #   batch_size[b] x t_samples
                    # Reshape to be able to divide by the graph matrix
                    max_eigenvalue = max_eigenvalue.reshape((batch_size[b],
                                                             t_samples,
                                                             1, 1))
                    # Normalize
                    graph_matrix_batch = graph_matrix_batch / max_eigenvalue

                # Store
                graph_matrix[batch_index[b]:batch_index[b + 1]] = graph_matrix_batch

                if do_print:
                    # Sample percentage count
                    percentage_count = int(100 * (b + 1) / n_batches)

                    if b == 0:
                        # It's the first one, so just print it
                        print("%3d%%" % percentage_count,
                              end='', flush=True)
                    else:
                        # Erase the previous characters
                        print('\b \b' * 4 + "%3d%%" % percentage_count,
                              end='', flush=True)

        # Print
        if do_print:
            # Erase the percentage
            print('\b \b' * 4, end='', flush=True)

        return graph_matrix

    def get_data(self, name, samples_type, *args):

        # samples_type: train, valid, test
        # args: 0 args, give back all
        # args: 1 arg: if int, give that number of samples, chosen at random
        # args: 1 arg: if list, give those samples precisely.

        # Check that the type is one of the possible ones
        assert samples_type == 'train' or samples_type == 'valid' \
               or samples_type == 'test'
        # Check that the number of extra arguments fits
        assert len(args) <= 1

        # Check that the name is actually an attribute
        assert name in dir(self)

        # Get the desired attribute
        this_data_dict = getattr(self, name)

        # Check it's a dictionary and that it has the corresponding key
        assert type(this_data_dict) is dict
        assert samples_type in this_data_dict.keys()

        # Get the data now
        this_data = this_data_dict[samples_type]
        # Get the dimension length
        this_data_dims = len(this_data.shape)

        # Check that it has at least two dimension, where the first one is
        # always the number of samples
        assert this_data_dims > 1

        if len(args) == 1:
            # If it is an int, just return that number of randomly chosen
            # samples.
            if type(args[0]) == int:
                n_samples = this_data.shape[0]  # total number of samples
                # We can't return more samples than there are available
                assert args[0] <= n_samples
                # Randomly choose args[0] indices
                selected_indices = np.random.choice(n_samples, size=args[0],
                                                    replace=False)
                # Select the corresponding samples
                this_data = this_data[selected_indices]
            else:
                # The fact that we put else here instead of elif type()==list
                # allows for np.array to be used as indices as well. In general,
                # any variable with the ability to index.
                this_data = this_data[args[0]]

            # If we only selected a single element, then the nDataPoints dim
            # has been left out. So if we have less dimensions, we have to
            # put it back
            if len(this_data.shape) < this_data_dims:
                if 'torch' in repr(this_data.dtype):
                    this_data = this_data.unsqueeze(0)
                else:
                    this_data = np.expand_dims(this_data, axis=0)

        return this_data

    def evaluate(self, vel=None, accel=None, init_vel=None,
                 sampling_time=None):

        # It is optional to add a different sampling time, if not, it uses
        # the internal one
        if sampling_time is None:
            # If there's no argument use the internal sampling time
            sampling_time = self.sampling_time

        # Check whether we have vel, or accel and init_vel (i.e. we are either
        # given the velocities, or we are given the elements to compute them)
        if vel is not None:
            assert len(vel.shape) == 4
            n_samples = vel.shape[0]
            t_samples = vel.shape[1]
            assert vel.shape[2] == 2
            n_agents = vel.shape[3]
        elif accel is not None and init_vel is not None:
            assert len(accel.shape) == 4 and len(init_vel.shape) == 3
            n_samples = accel.shape[0]
            t_samples = accel.shape[1]
            assert accel.shape[2] == 2
            n_agents = accel.shape[3]
            assert init_vel.shape[0] == n_samples
            assert init_vel.shape[1] == 2
            assert init_vel.shape[2] == n_agents

            # Now that we know we have a accel and init velocity, compute the
            # velocity trajectory
            # Compute the velocity trajectory
            if 'torch' in repr(accel.dtype):
                # Check that init_vel is also torch
                assert 'torch' in repr(init_vel.dtype)
                # Create the tensor to save the velocity trajectory
                vel = torch.zeros(n_samples, t_samples, 2, n_agents,
                                  dtype=accel.dtype, device=accel.device)
                # Add the initial velocity
                vel[:, 0, :, :] = init_vel.clone().detach()
            else:
                # Create the space
                vel = np.zeros((n_samples, t_samples, 2, n_agents),
                               dtype=accel.dtype)
                # Add the initial velocity
                vel[:, 0, :, :] = init_vel.copy()

            # Go over time
            for t in range(1, t_samples):
                # Compute velocity
                vel[:, t, :, :] = accel[:, t - 1, :, :] * sampling_time + vel[:, t - 1, :, :]

        # Check that I did enter one of the if clauses
        assert vel is not None

        # And now that we have the velocities, we can compute the cost
        if 'torch' in repr(vel.dtype):
            # Average velocity for time t, averaged across agents
            avg_vel = torch.mean(vel, dim=3)  # n_samples x t_samples x 2
            # Compute the difference in velocity between each agent and the
            # mean velocity
            diff_vel = vel - avg_vel.unsqueeze(3)
            #   n_samples x t_samples x 2 x n_agents
            # Compute the MSE velocity
            diff_vel_norm = torch.sum(diff_vel ** 2, dim=2)
            #   n_samples x t_samples x n_agents
            # Average over agents
            diff_vel_avg = torch.mean(diff_vel_norm, dim=2)  # n_samples x t_samples
            # Sum over time
            cost_per_sample = torch.sum(diff_vel_avg, dim=1)  # n_samples
            # Final average cost
            cost = torch.mean(cost_per_sample)
        else:
            # Repeat for numpy
            avg_vel = np.mean(vel, axis=3)  # n_samples x t_samples x 2
            diff_vel = vel - np.tile(np.expand_dims(avg_vel, 3),
                                     (1, 1, 1, n_agents))
            #   n_samples x t_samples x 2 x n_agents
            diff_vel_norm = np.sum(diff_vel ** 2, axis=2)
            #   n_samples x t_samples x n_agents
            diff_vel_avg = np.mean(diff_vel_norm, axis=2)  # n_samples x t_samples
            cost_per_sample = np.sum(diff_vel_avg, axis=1)  # n_samples
            cost = np.mean(cost_per_sample)  # scalar

        return cost

    def compute_trajectory(self, init_pos, init_vel, duration, **kwargs):

        # Check init_pos is of shape batch_size x 2 x n_agents
        assert len(init_pos.shape) == 3
        batch_size = init_pos.shape[0]
        assert init_pos.shape[1]
        n_agents = init_pos.shape[2]

        # Check init_vel is of shape batch_size x 2 x n_agents
        assert len(init_vel.shape) == 3
        assert init_vel.shape[0] == batch_size
        assert init_vel.shape[1] == 2
        assert init_vel.shape[2] == n_agents

        # Check what kind of data it is
        #   This is because all the functions are numpy, but if this was
        #   torch, we need to return torch, to make it consistent
        if 'torch' in repr(init_pos.dtype):
            assert 'torch' in repr(init_vel.dtype)
            use_torch = True
            device = init_pos.device
            assert init_vel.device == device
        else:
            use_torch = False

        # Create time line
        time = np.arange(0, duration, self.sampling_time)
        t_samples = len(time)

        # Here, we have two options, or we're given the acceleration or the
        # architecture
        assert 'archit' in kwargs.keys() or 'accel' in kwargs.keys()
        # Flags to determine which method to use
        use_archit = False
        use_accel = False

        if 'archit' in kwargs.keys():
            archit = kwargs['archit']  # This is a torch.nn.Module architecture
            archit_device = list(archit.parameters())[0].device
            use_archit = True
        elif 'accel' in kwargs.keys():
            accel = kwargs['accel']
            # accel has to be of shape batch_size x t_samples x 2 x n_agents
            assert len(accel.shape) == 4
            assert accel.shape[0] == batch_size
            assert accel.shape[1] == t_samples
            assert accel.shape[2] == 2
            assert accel.shape[3] == n_agents
            if use_torch:
                assert 'torch' in repr(accel.dtype)
            use_accel = True

        # Decide on printing or not:
        if 'do_print' in kwargs.keys():
            do_print = kwargs['do_print']
        else:
            do_print = self.do_print  # Use default

        # Now create the outputs that will be filled afterwards
        pos = np.zeros((batch_size, t_samples, 2, n_agents), dtype=np.float)
        vel = np.zeros((batch_size, t_samples, 2, n_agents), dtype=np.float)
        if use_archit:
            accel = np.zeros((batch_size, t_samples, 2, n_agents), dtype=np.float)
            state = np.zeros((batch_size, t_samples, 6, n_agents), dtype=np.float)
            graph = np.zeros((batch_size, t_samples, n_agents, n_agents),
                             dtype=np.float)

        # Assign the initial positions and velocities
        if use_torch:
            pos[:, 0, :, :] = init_pos.cpu().numpy()
            vel[:, 0, :, :] = init_vel.cpu().numpy()
            if use_accel:
                accel = accel.cpu().numpy()
        else:
            pos[:, 0, :, :] = init_pos.copy()
            vel[:, 0, :, :] = init_vel.copy()

        if do_print:
            # Sample percentage count
            percentage_count = int(100 / t_samples)
            # Print new value
            print("%3d%%" % percentage_count, end='', flush=True)

        # Now, let's get started:
        for t in range(1, t_samples):

            # If it is architecture-based, we need to compute the state, and
            # for that, we need to compute the graph
            if use_archit:
                # Adjust pos value for graph computation
                this_pos = np.expand_dims(pos[:, t - 1, :, :], 1)
                # Compute graph
                this_graph = self.compute_communication_graph(this_pos,
                                                              self.comm_radius,
                                                              True,
                                                              doPrint=False)
                # Save graph
                graph[:, t - 1, :, :] = this_graph.squeeze(1)
                # Adjust vel value for state computation
                this_vel = np.expand_dims(vel[:, t - 1, :, :], 1)
                # Compute state
                this_state = self.compute_states(this_pos, this_vel, this_graph,
                                                 doPrint=False)
                # Save state
                state[:, t - 1, :, :] = this_state.squeeze(1)

                # Compute the output of the architecture
                #   Note that we need the collection of all time instants up
                #   to now, because when we do the communication exchanges,
                #   it involves past times.
                x = torch.tensor(state[:, 0:t, :, :], device=archit_device)
                S = torch.tensor(graph[:, 0:t, :, :], device=archit_device)
                with torch.no_grad():
                    this_accel = archit(x, S)
                # Now that we have computed the acceleration, we only care 
                # about the last element in time
                this_accel = this_accel.cpu().numpy()[:, -1, :, :]
                this_accel[this_accel > self.accel_max] = self.accel_max
                this_accel[this_accel < -self.accel_max] = self.accel_max
                # And save it
                accel[:, t - 1, :, :] = this_accel

            # Now that we have the acceleration, we can update position and
            # velocity
            vel[:, t, :, :] = accel[:, t - 1, :, :] * self.sampling_time + vel[:, t - 1, :, :]
            pos[:, t, :, :] = accel[:, t - 1, :, :] * (self.sampling_time ** 2) / 2 + \
                              vel[:, t - 1, :, :] * self.sampling_time + pos[:, t - 1, :, :]

            if do_print:
                # Sample percentage count
                percentage_count = int(100 * (t + 1) / t_samples)
                # Erase previous value and print new value
                print('\b \b' * 4 + "%3d%%" % percentage_count,
                      end='', flush=True)

        # And we're missing the last values of graph, state and accel, so
        # let's compute them for completeness
        #   Graph
        this_pos = np.expand_dims(pos[:, -1, :, :], 1)
        this_graph = self.compute_communication_graph(this_pos, self.comm_radius,
                                                      True, doPrint=False)
        graph[:, -1, :, :] = this_graph.squeeze(1)
        #   State
        this_vel = np.expand_dims(vel[:, -1, :, :], 1)
        this_state = self.compute_states(this_pos, this_vel, this_graph,
                                         doPrint=False)
        state[:, -1, :, :] = this_state.squeeze(1)
        #   Accel
        x = torch.tensor(state).to(archit_device)
        S = torch.tensor(graph).to(archit_device)
        with torch.no_grad():
            this_accel = archit(x, S)
        this_accel = this_accel.cpu().numpy()[:, -1, :, :]
        this_accel[this_accel > self.accel_max] = self.accel_max
        this_accel[this_accel < -self.accel_max] = self.accel_max
        # And save it
        accel[:, -1, :, :] = this_accel

        # Print
        if do_print:
            # Erase the percentage
            print('\b \b' * 4, end='', flush=True)

        # After we have finished, turn it back into tensor, if required
        if use_torch:
            pos = torch.tensor(pos).to(device)
            vel = torch.tensor(vel).to(device)
            accel = torch.tensor(accel).to(device)

        # And return it
        if use_archit:
            return pos, vel, accel, state, graph
        elif use_accel:
            return pos, vel

    def compute_differences(self, u):

        # Takes as input a tensor of shape
        #   n_samples x t_samples x 2 x n_agents
        # or of shape
        #   n_samples x 2 x n_agents
        # And returns the elementwise difference u_i - u_j of shape
        #   n_samples (x t_samples) x 2 x n_agents x n_agents
        # And the distance squared ||u_i - u_j||^2 of shape
        #   n_samples (x t_samples) x n_agents x n_agents

        # Check dimensions
        assert len(u.shape) == 3 or len(u.shape) == 4
        # If it has shape 3, which means it's only a single time instant, then
        # add the extra dimension so we move along assuming we have multiple
        # time instants
        if len(u.shape) == 3:
            u = np.expand_dims(u, 1)
            has_time_dim = False
        else:
            has_time_dim = True

        # Now we have that pos always has shape
        #   n_samples x t_samples x 2 x n_agents
        n_samples = u.shape[0]
        t_samples = u.shape[1]
        assert u.shape[2] == 2
        n_agents = u.shape[3]

        # Compute the difference along each axis. For this, we subtract a
        # column vector from a row vector. The difference tensor on each
        # position will have shape n_samples x t_samples x n_agents x n_agents
        # and then we add the extra dimension to concatenate and obtain a final
        # tensor of shape n_samples x t_samples x 2 x n_agents x n_agents
        # First, axis x
        #   Reshape as column and row vector, respectively
        u_col_x = u[:, :, 0, :].reshape((n_samples, t_samples, n_agents, 1))
        u_row_x = u[:, :, 0, :].reshape((n_samples, t_samples, 1, n_agents))
        #   Subtract them
        u_diff_x = u_col_x - u_row_x  # n_samples x t_samples x n_agents x n_agents
        # Second, for axis y
        u_col_y = u[:, :, 1, :].reshape((n_samples, t_samples, n_agents, 1))
        u_row_y = u[:, :, 1, :].reshape((n_samples, t_samples, 1, n_agents))
        u_diff_y = u_col_y - u_row_y  # n_samples x t_samples x n_agents x n_agents
        # Third, compute the distance tensor of shape
        #   n_samples x t_samples x n_agents x n_agents
        u_dist_sq = u_diff_x ** 2 + u_diff_y ** 2
        # Finally, concatenate to obtain the tensor of differences
        #   Add the extra dimension in the position
        u_diff_x = np.expand_dims(u_diff_x, 2)
        u_diff_y = np.expand_dims(u_diff_y, 2)
        #   And concatenate them
        u_diff = np.concatenate((u_diff_x, u_diff_y), 2)
        #   n_samples x t_samples x 2 x n_agents x n_agents

        # Get rid of the time dimension if we don't need it
        if not has_time_dim:
            # (This fails if t_samples > 1)
            u_dist_sq = u_dist_sq.squeeze(1)
            #   n_samples x n_agents x n_agents
            u_diff = u_diff.squeeze(1)
            #   n_samples x 2 x n_agents x n_agents

        return u_diff, u_dist_sq

    def compute_optimal_trajectory(self, init_pos, init_vel, duration,
                                   sampling_time, repel_dist,
                                   accel_max=100.):

        # The optimal trajectory is given by
        # u_{i} = - \sum_{j=1}^{N} (v_{i} - v_{j})
        #         + 2 \sum_{j=1}^{N} (r_{i} - r_{j}) *
        #                                 (1/\|r_{i}\|^{4} + 1/\|r_{j}\|^{2}) *
        #                                 1{\|r_{ij}\| < R}
        # for each agent i=1,...,N, where v_{i} is the velocity and r_{i} the
        # position.

        # Check that init_pos and init_vel as n_samples x 2 x n_agents arrays
        assert len(init_pos.shape) == len(init_vel.shape) == 3
        n_samples = init_pos.shape[0]
        assert init_pos.shape[1] == init_vel.shape[1] == 2
        n_agents = init_pos.shape[2]
        assert init_vel.shape[0] == n_samples
        assert init_vel.shape[2] == n_agents

        # time
        time = np.arange(0, duration, sampling_time)
        t_samples = len(time)  # number of time samples

        # Create arrays to store the trajectory
        pos = np.zeros((n_samples, t_samples, 2, n_agents))
        vel = np.zeros((n_samples, t_samples, 2, n_agents))
        accel = np.zeros((n_samples, t_samples, 2, n_agents))

        # Initial settings
        pos[:, 0, :, :] = init_pos
        vel[:, 0, :, :] = init_vel

        if self.do_print:
            # Sample percentage count
            percentage_count = int(100 / t_samples)
            # Print new value
            print("%3d%%" % percentage_count, end='', flush=True)

        # For each time instant
        for t in range(1, t_samples):

            # Compute the optimal acceleration
            #   Compute the distance between all elements (positions)
            ij_diff_pos, ij_dist_sq = self.compute_differences(pos[:, t - 1, :, :])
            #       ij_diff_pos: n_samples x 2 x n_agents x n_agents
            #       ij_dist_sq:  n_samples x n_agents x n_agents
            #   And also the difference in velocities
            ij_diff_vel, _ = self.compute_differences(vel[:, t - 1, :, :])
            #       ij_diff_vel: n_samples x 2 x n_agents x n_agents
            #   The last element we need to compute the acceleration is the
            #   gradient. Note that the gradient only counts when the distance 
            #   is smaller than the repel distance
            #       This is the mask to consider each of the differences
            repel_mask = (ij_dist_sq < (repel_dist ** 2)).astype(ij_diff_pos.dtype)
            #       Apply the mask to the relevant differences
            ij_diff_pos = ij_diff_pos * np.expand_dims(repel_mask, 1)
            #       Compute the constant (1/||r_ij||^4 + 1/||r_ij||^2)
            ij_dist_sq_inv = invert_tensor_ew(ij_dist_sq)
            #       Add the extra dimension
            ij_dist_sq_inv = np.expand_dims(ij_dist_sq_inv, 1)
            #   Compute the acceleration
            accel[:, t - 1, :, :] = \
                -np.sum(ij_diff_vel, axis=3) \
                + 2 * np.sum(ij_diff_pos * (ij_dist_sq_inv ** 2 + ij_dist_sq_inv),
                             axis=3)

            # Finally, note that if the agents are too close together, the
            # acceleration will be very big to get them as far apart as
            # possible, and this is physically impossible.
            # So let's add a limitation to the maximum aceleration

            # Find the places where the acceleration is big
            this_accel = accel[:, t - 1, :, :].copy()
            # Values that exceed accel_max, force them to be accel_max
            this_accel[accel[:, t - 1, :, :] > accel_max] = accel_max
            # Values that are smaller than -accel_max, force them to be accel_max
            this_accel[accel[:, t - 1, :, :] < -accel_max] = -accel_max
            # And put it back
            accel[:, t - 1, :, :] = this_accel

            # Update the values
            #   Update velocity
            vel[:, t, :, :] = accel[:, t - 1, :, :] * sampling_time + vel[:, t - 1, :, :]
            #   Update the position
            pos[:, t, :, :] = accel[:, t - 1, :, :] * (sampling_time ** 2) / 2 + \
                              vel[:, t - 1, :, :] * sampling_time + pos[:, t - 1, :, :]

            if self.do_print:
                # Sample percentage count
                percentage_count = int(100 * (t + 1) / t_samples)
                # Erase previous pecentage and print new value
                print('\b \b' * 4 + "%3d%%" % percentage_count,
                      end='', flush=True)

        # Print
        if self.do_print:
            # Erase the percentage
            print('\b \b' * 4, end='', flush=True)

        return pos, vel, accel

    def compute_initial_positions(self, n_agents, n_samples, comm_radius,
                                  min_dist=0.1, geometry='rectangular',
                                  **kwargs):

        # It will always be uniform. We can select whether it is rectangular
        # or circular (or some other shape) and the parameters respecting
        # that
        assert geometry == 'rectangular' or geometry == 'circular'
        assert min_dist * (1. + zero_tolerance) <= comm_radius * (1. - zero_tolerance)
        # We use a zero_tolerance buffer zone, just in case
        min_dist = min_dist * (1. + zero_tolerance)
        comm_radius = comm_radius * (1. - zero_tolerance)

        # If there are other keys in the kwargs argument, they will just be
        # ignored

        # We will first create the grid, whether it is rectangular or
        # circular.

        # Let's start by setting the fixed position
        if geometry == 'rectangular':

            # This grid has a distance that depends on the desired min_dist and
            # the comm_radius
            dist_fixed = (comm_radius + min_dist) / (2. * np.sqrt(2))
            #   This is the fixed distance between points in the grid
            dist_perturb = (comm_radius - min_dist) / (4. * np.sqrt(2))
            #   This is the standard deviation of a uniform perturbation around
            #   the fixed point.
            # This should guarantee that, even after the perturbations, there
            # are no agents below min_dist, and that all agents have at least
            # one other agent within comm_radius.

            # How many agents per axis
            n_agents_per_axis = int(np.ceil(np.sqrt(n_agents)))

            axis_fixed_pos = np.arange(-(n_agents_per_axis * dist_fixed) / 2,
                                       (n_agents_per_axis * dist_fixed) / 2,
                                       step=dist_fixed)

            # Repeat the positions in the same order (x coordinate)
            x_fixed_pos = np.tile(axis_fixed_pos, n_agents_per_axis)
            # Repeat each element (y coordinate)
            y_fixed_pos = np.repeat(axis_fixed_pos, n_agents_per_axis)

            # Concatenate this to obtain the positions
            fixed_pos = np.concatenate((np.expand_dims(x_fixed_pos, 0),
                                        np.expand_dims(y_fixed_pos, 0)),
                                       axis=0)

            # Get rid of unnecessary agents
            fixed_pos = fixed_pos[:, 0:n_agents]
            # And repeat for the number of samples we want to generate
            fixed_pos = np.repeat(np.expand_dims(fixed_pos, 0), n_samples,
                                  axis=0)
            #   n_samples x 2 x n_agents

            # Now generate the noise
            perturb_pos = np.random.uniform(low=-dist_perturb,
                                            high=dist_perturb,
                                            size=(n_samples, 2, n_agents))

            # Initial positions
            init_pos = fixed_pos + perturb_pos

        elif geometry == 'circular':

            # Radius for the grid
            r_fixed = (comm_radius + min_dist) / 2.
            r_perturb = (comm_radius - min_dist) / 4.
            fixed_radius = np.arange(0, r_fixed * n_agents, step=r_fixed) + r_fixed

            # Angles for the grid
            a_fixed = (comm_radius / fixed_radius + min_dist / fixed_radius) / 2.
            for a in range(len(a_fixed)):
                # How many times does a_fixed[a] fits within 2pi?
                n_agents_per_circle = 2 * np.pi // a_fixed[a]
                # And now divide 2*np.pi by this number
                a_fixed[a] = 2 * np.pi / n_agents_per_circle
            #   Fixed angle difference for each value of fixed_radius

            # Now, let's get the radius, angle coordinates for each agents
            init_radius = np.empty((0))
            init_angles = np.empty((0))
            agents_so_far = 0  # Number of agents located so far
            n = 0  # Index for radius
            while agents_so_far < n_agents:
                this_radius = fixed_radius[n]
                this_angles = np.arange(0, 2 * np.pi, step=a_fixed[n])
                agents_so_far += len(this_angles)
                init_radius = np.concatenate((init_radius,
                                              np.repeat(this_radius,
                                                        len(this_angles))))
                init_angles = np.concatenate((init_angles, this_angles))
                n += 1
                assert len(init_radius) == agents_so_far

            # Restrict to the number of agents we need
            init_radius = init_radius[0:n_agents]
            init_angles = init_angles[0:n_agents]

            # Add the number of samples
            init_radius = np.repeat(np.expand_dims(init_radius, 0), n_samples,
                                    axis=0)
            init_angles = np.repeat(np.expand_dims(init_angles, 0), n_samples,
                                    axis=0)

            # Add the noise
            #   First, to the angles
            for n in range(n_agents):
                # Get the radius (the angle noise depends on the radius); so
                # far the radius is the same for all samples
                this_radius = init_radius[0, n]
                a_perturb = (comm_radius / this_radius - min_dist / this_radius) / 4.
                # Add the noise to the angles
                init_angles[:, n] += np.random.uniform(low=-a_perturb,
                                                       high=a_perturb,
                                                       size=(n_samples))
            #   Then, to the radius
            init_radius += np.random.uniform(low=-r_perturb,
                                             high=r_perturb,
                                             size=(n_samples, n_agents))

            # And finally, get the positions in the cartesian coordinates
            init_pos = np.zeros((n_samples, 2, n_agents))
            init_pos[:, 0, :] = init_radius * np.cos(init_angles)
            init_pos[:, 1, :] = init_radius * np.sin(init_angles)

        # Now, check that the conditions are met:
        #   Compute square distances
        _, dist_sq = self.compute_differences(np.expand_dims(init_pos, 1))
        #   Get rid of the "time" dimension that arises from using the 
        #   method to compute distances
        dist_sq = dist_sq.squeeze(1)
        #   Compute the minimum distance (don't forget to add something in
        #   the diagonal, which otherwise is zero)
        min_dist_sq = np.min(dist_sq + \
                             2 * comm_radius \
                             * np.eye(dist_sq.shape[1]).reshape(1,
                                                                dist_sq.shape[1],
                                                                dist_sq.shape[2])
                             )

        assert min_dist_sq >= min_dist ** 2

        #   Now the number of neighbors
        graph_matrix = self.compute_communication_graph(np.expand_dims(init_pos, 1),
                                                        self.comm_radius,
                                                        False,
                                                        doPrint=False)
        graph_matrix = graph_matrix.squeeze(1)  # n_samples x n_agents x n_agents

        #   Binarize the matrix
        graph_matrix = (np.abs(graph_matrix) > zero_tolerance) \
            .astype(init_pos.dtype)

        #   And check that we always have initially connected graphs
        for n in range(n_samples):
            assert graph.is_connected(graph_matrix[n, :, :])

        # We move to compute the initial velocities. Velocities can be
        # either positive or negative, so we do not need to determine
        # the lower and higher, just around zero
        if 'x_max_init_vel' in kwargs.keys():
            x_max_init_vel = kwargs['x_max_init_vel']
        else:
            x_max_init_vel = 3.
            #   Takes five seconds to traverse half the map
        # Same for the other axis
        if 'y_max_init_vel' in kwargs.keys():
            y_max_init_vel = kwargs['y_max_init_vel']
        else:
            y_max_init_vel = 3.

        # And sample the velocities
        x_init_vel = np.random.uniform(low=-x_max_init_vel, high=x_max_init_vel,
                                       size=(n_samples, 1, n_agents))
        y_init_vel = np.random.uniform(low=-y_max_init_vel, high=y_max_init_vel,
                                       size=(n_samples, 1, n_agents))
        # Add bias
        x_vel_bias = np.random.uniform(low=-x_max_init_vel, high=x_max_init_vel,
                                       size=(n_samples))
        y_vel_bias = np.random.uniform(low=-y_max_init_vel, high=y_max_init_vel,
                                       size=(n_samples))

        # And concatenate them
        vel_bias = np.concatenate((x_vel_bias, y_vel_bias)).reshape((n_samples, 2, 1))
        init_vel = np.concatenate((x_init_vel, y_init_vel), axis=1) + vel_bias
        #   n_samples x 2 x n_agents

        return init_pos, init_vel

    def save_video(self, save_dir, pos, *args,
                   comm_graph=None, **kwargs):

        # Check that pos is a position of shape n_samples x t_samples x 2 x n_agents
        assert len(pos.shape) == 4
        n_samples = pos.shape[0]
        t_samples = pos.shape[1]
        assert pos.shape[2] == 2
        n_agents = pos.shape[3]
        if 'torch' in repr(pos.dtype):
            pos = pos.cpu().numpy()

        # Check if there's the need to plot a graph
        if comm_graph is not None:
            # If there's a communication graph, then it has to have shape
            #   n_samples x t_samples x n_agents x n_agents
            assert len(comm_graph.shape) == 4
            assert comm_graph.shape[0] == n_samples
            assert comm_graph.shape[1] == t_samples
            assert comm_graph.shape[2] == comm_graph.shape[3] == n_agents
            if 'torch' in repr(comm_graph.dtype):
                comm_graph = comm_graph.cpu().numpy()
            show_graph = True
        else:
            show_graph = False

        if 'do_print' in kwargs.keys():
            do_print = kwargs['do_print']
        else:
            do_print = self.do_print

        # This number determines how faster or slower to reproduce the video
        if 'video_speed' in kwargs.keys():
            video_speed = kwargs['video_speed']
        else:
            video_speed = 1.

        if 'show_video_speed' in kwargs.keys():
            show_video_speed = kwargs['show_video_speed']
        else:
            if video_speed != 1:
                show_video_speed = True
            else:
                show_video_speed = False

        if 'vel' in kwargs.keys():
            vel = kwargs['vel']
            if 'show_cost' in kwargs.keys():
                show_cost = kwargs['show_cost']
            else:
                show_cost = True
            if 'show_arrows' in kwargs.keys():
                show_arrows = kwargs['show_arrows']
            else:
                show_arrows = True
        else:
            show_cost = False
            show_arrows = False

        # Check that the number of extra arguments fits
        assert len(args) <= 1
        # If there's an argument, we have to check whether it is an int or a
        # list
        if len(args) == 1:
            # If it is an int, just return that number of randomly chosen
            # samples.
            if type(args[0]) == int:
                # We can't return more samples than there are available
                assert args[0] <= n_samples
                # Randomly choose args[0] indices
                selected_indices = np.random.choice(n_samples, size=args[0],
                                                    replace=False)
            else:
                # The fact that we put else here instead of elif type()==list
                # allows for np.array to be used as indices as well. In general,
                # any variable with the ability to index.
                selected_indices = args[0]

            # Select the corresponding samples
            pos = pos[selected_indices]

            # Finally, observe that if pos has shape only 3, then that's 
            # because we selected a single sample, so we need to add the extra
            # dimension back again
            if len(pos.shape) < 4:
                pos = np.expand_dims(pos, 0)

            if show_graph:
                comm_graph = comm_graph[selected_indices]
                if len(comm_graph.shape) < 4:
                    comm_graph = np.expand_dims(comm_graph, 0)

        # Where to save the video
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        video_name = 'sample_trajectory'

        x_min_map = np.min(pos[:, :, 0, :]) * 1.2
        x_max_map = np.max(pos[:, :, 0, :]) * 1.2
        y_min_map = np.min(pos[:, :, 1, :]) * 1.2
        y_max_map = np.max(pos[:, :, 1, :]) * 1.2

        # Create video object

        video_metadata = dict(title='Sample Trajectory', artist='Flocking',
                              comment='Flocking example')
        video_writer = FFMpegWriter(fps=video_speed / self.sampling_time,
                                    metadata=video_metadata)

        if do_print:
            print("\tSaving video(s)...", end=' ', flush=True)

        # For each sample now
        for n in range(pos.shape[0]):

            # If there's more than one video to create, enumerate them
            if pos.shape[0] > 1:
                this_video_name = video_name + '%03d.mp4' % n
            else:
                this_video_name = video_name + '.mp4'

            # Select the corresponding position trajectory
            this_pos = pos[n]

            # Create figure
            video_fig = plt.figure(figsize=(5, 5))

            # Set limits
            plt.xlim((x_min_map, x_max_map))
            plt.ylim((y_min_map, y_max_map))
            plt.axis('equal')

            if show_video_speed:
                plt.text(x_min_map, y_min_map, r'Speed: $%.2f$' % video_speed)

            # Create plot handle
            plot_agents, = plt.plot([], [],
                                    marker='o',
                                    markersize=3,
                                    linewidth=0,
                                    color='#01256E',
                                    scalex=False,
                                    scaley=False)

            # Create the video
            with video_writer.saving(video_fig,
                                     os.path.join(save_dir, this_video_name),
                                     t_samples):

                for t in range(t_samples):

                    # Plot the agents
                    plot_agents.set_data(this_pos[t, 0, :], this_pos[t, 1, :])
                    video_writer.grab_frame()

                    # Print
                    if do_print:
                        # Sample percentage count
                        percentage_count = int(
                            100 * (t + 1 + n * t_samples) / (t_samples * pos.shape[0])
                        )

                        if n == 0 and t == 0:
                            print("%3d%%" % percentage_count,
                                  end='', flush=True)
                        else:
                            print('\b \b' * 4 + "%3d%%" % percentage_count,
                                  end='', flush=True)

            plt.close(fig=video_fig)

        # Print
        if do_print:
            # Erase the percentage and the label
            print('\b \b' * 4 + "OK", flush=True)

        if show_graph:

            # Normalize velocity
            if show_arrows:
                # vel is of shape n_samples x t_samples x 2 x n_agents
                vel_norm_sq = np.sum(vel ** 2, axis=2)
                #   n_samples x t_samples x n_agents
                max_vel_norm_sq = np.max(np.max(vel_norm_sq, axis=2), axis=1)
                #   n_samples
                max_vel_norm_sq = max_vel_norm_sq.reshape((n_samples, 1, 1, 1))
                #   n_samples x 1 x 1 x 1
                norm_vel = 2 * vel / np.sqrt(max_vel_norm_sq)

            if do_print:
                print("\tSaving graph snapshots...", end=' ', flush=True)

            # Essentially, we will print nGraphs snapshots and save them
            # as images with the graph. This is the best we can do in a
            # reasonable processing time (adding the graph to the video takes
            # forever).
            time = np.arange(0, self.duration, step=self.sampling_time)
            assert len(time) == t_samples

            n_snapshots = 5  # The number of snapshots we will consider
            t_snapshots = np.linspace(0, t_samples - 1, num=n_snapshots)
            #   This gives us n_snapshots equally spaced in time. Now, we need
            #   to be sure these are integers
            t_snapshots = np.unique(t_snapshots.astype(np.int)).astype(np.int)

            # Directory to save the snapshots
            snapshot_dir = os.path.join(save_dir, 'graph_snapshots')
            # Base name of the snapshots
            snapshot_name = 'graph_snapshot'

            for n in range(pos.shape[0]):

                if pos.shape[0] > 1:
                    this_snapshot_dir = snapshot_dir + '%03d' % n
                    this_snapshot_name = snapshot_name + '%03d' % n
                else:
                    this_snapshot_dir = snapshot_dir
                    this_snapshot_name = snapshot_name

                if not os.path.exists(this_snapshot_dir):
                    os.mkdir(this_snapshot_dir)

                # Get the corresponding positions
                this_pos = pos[n]
                this_comm_graph = comm_graph[n]

                for t in t_snapshots:

                    # Get the edge pairs
                    #   Get the graph for this time instant
                    this_comm_graph_time = this_comm_graph[t]
                    #   Check if it is symmetric
                    is_symmetric = np.allclose(this_comm_graph_time,
                                               this_comm_graph_time.T)
                    if is_symmetric:
                        #   Use only half of the matrix
                        this_comm_graph_time = np.triu(this_comm_graph_time)

                    #   Find the position of all edges
                    out_edge, in_edge = np.nonzero(np.abs(this_comm_graph_time) \
                                                   > zero_tolerance)

                    # Create the figure
                    this_graph_snapshot_fig = plt.figure(figsize=(5, 5))

                    # Set limits (to be the same as the video)
                    plt.xlim((x_min_map, x_max_map))
                    plt.ylim((y_min_map, y_max_map))
                    plt.axis('equal')

                    # Plot the edges
                    plt.plot([this_pos[t, 0, out_edge], this_pos[t, 0, in_edge]],
                             [this_pos[t, 1, out_edge], this_pos[t, 1, in_edge]],
                             color='#A8AAAF', linewidth=0.75,
                             scalex=False, scaley=False)

                    # Plot the arrows
                    if show_arrows:
                        for i in range(n_agents):
                            plt.arrow(this_pos[t, 0, i], this_pos[t, 1, i],
                                      norm_vel[n, t, 0, i], norm_vel[n, t, 1, i])

                    # Plot the nodes
                    plt.plot(this_pos[t, 0, :], this_pos[t, 1, :],
                             marker='o', markersize=3, linewidth=0,
                             color='#01256E', scalex=False, scaley=False)

                    # Add the cost value
                    if show_cost:
                        total_cost = self.evaluate(vel=vel[:, t:t + 1, :, :])
                        plt.text(x_min_map, y_min_map, r'Cost: $%.4f$' % total_cost)

                    # Add title
                    plt.title("Time $t=%.4f$s" % time[t])

                    # Save figure
                    this_graph_snapshot_fig.savefig(os.path.join(this_snapshot_dir,
                                                                 this_snapshot_name + '%03d.pdf' % t))

                    # Close figure
                    plt.close(fig=this_graph_snapshot_fig)

                    # Print percentage completion
                    if do_print:
                        # Sample percentage count
                        percentage_count = int(
                            100 * (t + 1 + n * t_samples) / (t_samples * pos.shape[0])
                        )
                        if n == 0 and t == 0:
                            # Print new value
                            print("%3d%%" % percentage_count,
                                  end='', flush=True)
                        else:
                            # Erase the previous characters
                            print('\b \b' * 4 + "%3d%%" % percentage_count,
                                  end='', flush=True)

            # Print
            if do_print:
                # Erase the percentage and the label
                print('\b \b' * 4 + "OK", flush=True)
