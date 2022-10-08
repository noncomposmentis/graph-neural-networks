# 2018/10/15~
# Fernando Gama, fgama@seas.upenn.edu.
# Luana Ruiz, rubruiz@seas.upenn.edu.
"""
misc_tools Miscellaneous Tools module

num2filename: change a numerical value into a string usable as a filename
save_seed: save the random state of generators
load_seed: load the number of random state of generators
write_var_values: write the specified values in the specified txt file
"""

import os
import pickle

import numpy as np
import torch


def num2filename(x, d):
    """
    Takes a number and returns a string with the value of the number, but in a
    format that is writable into a filename.

    s = num2filename(x,d) Gets rid of decimal points which are usually
        inconvenient to have in a filename.
        If the number x is an integer, then s = str(int(x)).
        If the number x is a decimal number, then it replaces the '.' by the
        character specified by d. Setting d = '' erases the decimal point,
        setting d = '.' simply returns a string with the exact same number.

    Example:
        >> num2filename(2,'d')
        >> '2'

        >> num2filename(3.1415,'d')
        >> '3d1415'

        >> num2filename(3.1415,'')
        >> '31415'

        >> num2filename(3.1415,'.')
        >> '3.1415'
    """
    if x == int(x):
        return str(int(x))
    else:
        return str(x).replace('.', d)


def save_seed(random_states, save_dir):
    """
    Takes a list of dictionaries of random generator states of different modules
    and saves them in a .pkl format.
    
    Inputs:
        random_states (list): The length of this list is equal to the number of
            modules whose states want to be saved (torch, numpy, etc.). Each
            element in this list is a dictionary. The dictionary has three keys:
            'module' with the name of the module in string format ('numpy' or
            'torch', for example), 'state' with the saved generator state and,
            if corresponds, 'seed' with the specific seed for the generator
            (note that torch has both state and seed, but numpy only has state)
        save_dir (path): where to save the seed, it will be saved under the 
            filename 'random_seed_used.pkl'
    """
    path_to_seed = os.path.join(save_dir, 'random_seed_used.pkl')
    with open(path_to_seed, 'wb') as seedFile:
        pickle.dump({'random_states': random_states}, seedFile)


def load_seed(load_dir):
    """
    Loads the states and seed saved in a specified path
    
    Inputs:
        load_dir (path): where to look for thee seed to load; it is expected that
            the appropriate file within load_dir is named 'random_seed_used.pkl'
    
    Obs.: The file 'random_seed_used.pkl' should contain a list structured as
        follows. The length of this list is equal to the number of modules whose
        states were saved (torch, numpy, etc.). Each element in this list is a
        dictionary. The dictionary has three keys: 'module' with the name of 
        the module in string format ('numpy' or 'torch', for example), 'state' 
        with the saved generator state and, if corresponds, 'seed' with the 
        specific seed for the generator (note that torch has both state and 
        seed, but numpy only has state)
    """
    path_to_seed = os.path.join(load_dir, 'random_seed_used.pkl')
    with open(path_to_seed, 'rb') as seedFile:
        random_states = pickle.load(seedFile)
        random_states = random_states['random_states']
    for module in random_states:
        this_module = module['module']
        if this_module == 'numpy':
            np.random.RandomState().set_state(module['state'])
        elif this_module == 'torch':
            torch.set_rng_state(module['state'])
            torch.manual_seed(module['seed'])


def write_var_values(file_to_write, var_values):
    """
    Write the value of several string variables specified by a dictionary into
    the designated .txt file.
    
    Input:
        file_to_write (os.path): text file to save the specified variables
        var_values (dictionary): values to save in the text file. They are
            saved in the format "key = value".
    """
    with open(file_to_write, 'a+') as file:
        for key in var_values.keys():
            file.write('%s = %s\n' % (key, var_values[key]))
        file.write('\n')
