# 2018/10/02~
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu
"""
model.py Model Module

Utilities useful for working on the model

Model: binds together the architecture, the loss function, the optimizer,
       the trainer, and the evaluator.
"""

import os
import torch


class Model:
    """
    Model: binds together the architecture, the loss function, the optimizer,
        the trainer, and the evaluator.
        
    Initialization:
        
        architecture (nn.Module)
        loss (nn.modules.loss._Loss)
        optimizer (nn.optim)
        trainer (Modules.training)
        evaluator (Modules.evaluation)
        device (string or device)
        name (string)
        save_dir (string or path)
        
    .train(data, n_epochs, batch_side, **kwargs): train the model for n_epochs 
        epochs, using batches of size batch_side and running over data data 
        class; see the specific selected trainer for extra options
    
    .evaluate(data): evaluate the model over data data class; see the specific
        selected evaluator for extra options
        
    .save(label = '', [save_dir=dir_path]): save the model parameters under the
        name given by label, if the save_dir is different from the one specified
        in the initialization, it needs to be specified now
        
    .load(label = '', [loadFiles=(archit_load_file, optim_load_file)]): loads the
        model parameters under the specified name inside the specific save_dir,
        unless they are provided externally through the keyword 'loadFiles'.
        
    .get_training_options(): get a dict with the options used during training; it
        returns None if it hasn't been trained yet.'
    """

    def __init__(self,
                 # Architecture (nn.Module)
                 architecture,
                 # Loss Function (nn.modules.loss._Loss)
                 loss,
                 # Optimization Algorithm (nn.optim)
                 optimizer,
                 # Training Algorithm (Modules.training)
                 trainer,
                 # Evaluating Algorithm (Modules.evaluation)
                 evaluator,
                 # Other
                 device, name, save_dir):

        # \\\ ARCHITECTURE
        # Store
        self.archit = architecture
        # Move it to device
        self.archit.to(device)
        # Count parameters (doesn't work for EdgeVarying)
        self.n_parameters = 0
        for param in list(self.archit.parameters()):
            if len(param.shape) > 0:
                this_n_param = 1
                for p in range(len(param.shape)):
                    this_n_param *= param.shape[p]
                self.n_parameters += this_n_param
            else:
                pass
        # \\\ LOSS FUNCTION
        self.loss = loss
        # \\\ OPTIMIZATION ALGORITHM
        self.optim = optimizer
        # \\\ TRAINING ALGORITHM
        self.trainer = trainer
        # \\\ EVALUATING ALGORITHM
        self.evaluator = evaluator
        # \\\ OTHER
        # Device
        self.device = device
        # Model name
        self.name = name
        # Saving directory
        self.save_dir = save_dir

    def train(self, data, n_epochs, batch_side, **kwargs):

        self.trainer = self.trainer(self, data, n_epochs, batch_side, **kwargs)

        return self.trainer.train()

    def evaluate(self, data, **kwargs):

        return self.evaluator(self, data, **kwargs)

    def save(self, label='', **kwargs):
        if 'save_dir' in kwargs.keys():
            save_dir = kwargs['save_dir']
        else:
            save_dir = self.save_dir
        save_model_dir = os.path.join(save_dir, 'savedModels')
        # Create directory savedModels if it doesn't exist yet:
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)
        save_file = os.path.join(save_model_dir, self.name)
        torch.save(self.archit.state_dict(), save_file + 'Archit' + label + '.ckpt')
        torch.save(self.optim.state_dict(), save_file + 'Optim' + label + '.ckpt')

    def load(self, label='', **kwargs):
        if 'loadFiles' in kwargs.keys():
            (archit_load_file, optim_load_file) = kwargs['loadFiles']
        else:
            save_model_dir = os.path.join(self.save_dir, 'savedModels')
            archit_load_file = os.path.join(save_model_dir,
                                            self.name + 'Archit' + label + '.ckpt')
            optim_load_file = os.path.join(save_model_dir,
                                           self.name + 'Optim' + label + '.ckpt')
        self.archit.load_state_dict(torch.load(archit_load_file))
        self.optim.load_state_dict(torch.load(optim_load_file))

    def get_training_options(self):

        return self.trainer.training_options \
            if 'training_options' in dir(self.trainer) \
            else None

    def __repr__(self):
        repr_string = "Name: %s\n" % (self.name)
        repr_string += "Number of learnable parameters: %d\n" % (self.n_parameters)
        repr_string += "\n"
        repr_string += "Model architecture:\n"
        repr_string += "----- -------------\n"
        repr_string += "\n"
        repr_string += repr(self.archit) + "\n"
        repr_string += "\n"
        repr_string += "Loss function:\n"
        repr_string += "---- ---------\n"
        repr_string += "\n"
        repr_string += repr(self.loss) + "\n"
        repr_string += "\n"
        repr_string += "Optimizer:\n"
        repr_string += "----------\n"
        repr_string += "\n"
        repr_string += repr(self.optim) + "\n"
        repr_string += "Training algorithm:\n"
        repr_string += "-------- ----------\n"
        repr_string += "\n"
        repr_string += repr(self.trainer) + "\n"
        repr_string += "Evaluation algorithm:\n"
        repr_string += "---------- ----------\n"
        repr_string += "\n"
        repr_string += repr(self.evaluator) + "\n"
        return repr_string
