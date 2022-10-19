# 2020/02/25~
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu
"""
training.py Training Module

Trainer classes

Trainer: general trainer that just computes a loss over a training set and
    runs an evaluation on a validation test
TrainerSingleNode: trainer class that computes a loss over the training set and
    runs an evaluation on a validation set, but assuming that the architectures
    involved have a single node forward structure and that the data involved
    has a method for identifying the target nodes
TrainerFlocking: traininer class that computes a loss over the training set,
    suited for the problem of flocking (i.e. it involves specific uses of
    the data, like computing trajectories or using DAGger)

"""

import datetime
import os
import pickle

import numpy as np
import torch

from alegnn.utils.dataTools import invert_tensor_ew


class Trainer:
    """
    Trainer: general trainer that just computes a loss over a training set and
        runs an evaluation on a validation test
        
    Initialization:
        
        model (Modules.model class): model to train
        data (Utils.data class): needs to have a get_samples and an evaluate
            method
        n_epochs (int): number of epochs (passes over the dataset)
        batch_size (int): size of each minibatch

        Optional (keyword) arguments:
            
        validation_interval (int): interval of training (number of training
            steps) without running a validation stage.

        learning_rate_decay_rate (float): float that multiplies the latest learning
            rate used.
        learning_rate_decay_period (int): how many training steps before 
            multiplying the learning rate decay rate by the actual learning
            rate.
        > Obs.: Both of these have to be defined for the learningRateDecay
              scheduler to be activated.
        logger (Visualizer): save tensorboard logs.
        save_dir (string): path to the directory where to save relevant training
            variables.
        print_interval (int): how many training steps after which to print
            partial results (0 means do not print)
        graph_no (int): keep track of what graph realization this is
        realitization_no (int): keep track of what data realization this is
        >> Alternatively, these last two keyword arguments can be used to keep
            track of different trainings of the same model
            
    Training:
        
        .train(): trains the model and returns train_vars dict with the keys
            'n_epochs': number of epochs (int)
            'n_batches': number of batches (int)
            'validation_interval': number of training steps in between 
                validation steps (int)
            'batch_size': batch size of each training step (np.array)
            'batch_index': indices for the start sample and end sample of each
                batch (np.array)
            'loss_train': loss function on the training samples for each training
                step (np.array)
            'eval_train': evaluation function on the training samples for each
                training step (np.array)
            'loss_valid': loss function on the validation samples for each
                validation step (np.array)
            'eval_valid': evaluation function on the validation samples for each
                validation step (np.array)
    """

    def __init__(self, model, data, n_epochs, batch_size, **kwargs):

        # \\\ Store model

        self.model = model
        self.data = data

        ####################################
        # ARGUMENTS (Store chosen options) #
        ####################################

        # Training Options:
        if 'do_logging' in kwargs.keys():
            do_logging = kwargs['do_logging']
        else:
            do_logging = False

        if 'do_save_vars' in kwargs.keys():
            do_save_vars = kwargs['do_save_vars']
        else:
            do_save_vars = True

        if 'print_interval' in kwargs.keys():
            print_interval = kwargs['print_interval']
            if print_interval > 0:
                do_print = True
            else:
                do_print = False
        else:
            do_print = True
            print_interval = (data.n_train // batch_size) // 5

        if 'learning_rate_decay_rate' in kwargs.keys() and \
                'learning_rate_decay_period' in kwargs.keys():
            do_learning_rate_decay = True
            learning_rate_decay_rate = kwargs['learning_rate_decay_rate']
            learning_rate_decay_period = kwargs['learning_rate_decay_period']
        else:
            do_learning_rate_decay = False

        if 'validation_interval' in kwargs.keys():
            validation_interval = kwargs['validation_interval']
        else:
            validation_interval = data.n_train // batch_size

        if 'early_stopping_lag' in kwargs.keys():
            do_early_stopping = True
            early_stopping_lag = kwargs['early_stopping_lag']
        else:
            do_early_stopping = False
            early_stopping_lag = 0

        if 'graph_no' in kwargs.keys():
            graph_no = kwargs['graph_no']
        else:
            graph_no = -1

        if 'realization_no' in kwargs.keys():
            if 'graph_no' in kwargs.keys():
                realization_no = kwargs['realization_no']
            else:
                graph_no = kwargs['realization_no']
                realization_no = -1
        else:
            realization_no = -1

        if do_logging:
            from alegnn.utils.visual_tools import Visualizer
            logsTB = os.path.join(self.save_dir, self.name + '-logsTB')
            logger = Visualizer(logsTB, name='visual_results')
        else:
            logger = None

        # No training case:
        if n_epochs == 0:
            do_save_vars = False
            do_logging = False
            # If there's no training happening, there's nothing to report about
            # training losses and stuff.

        ###########################################
        # DATA INPUT (pick up on data parameters) #
        ###########################################

        n_train = data.n_train  # size of the training set

        # Number of batches: If the desired number of batches does not split the
        # dataset evenly, we reduce the size of the last batch (the number of
        # samples in the last batch).
        # The variable batch_size is a list of length n_batches (number of
        # batches), where each element of the list is a number indicating the
        # size of the corresponding batch.
        if n_train < batch_size:
            n_batches = 1
            batch_size = [n_train]
        elif n_train % batch_size != 0:
            n_batches = np.ceil(n_train / batch_size).astype(np.int64)
            batch_size = [batch_size] * n_batches
            # If the sum of all batches so far is not the total number of
            # graphs, start taking away samples from the last batch (remember
            # that we used ceiling, so we are overshooting with the estimated
            # number of batches)
            while sum(batch_size) != n_train:
                batch_size[-1] -= 1
        # If they fit evenly, then just do so.
        else:
            n_batches = np.int(n_train / batch_size)
            batch_size = [batch_size] * n_batches
        # batch_index is used to determine the first and last element of each
        # batch.
        # If batch_size is, for example [20,20,20] meaning that there are three
        # batches of size 20 each, then cumsum will give [20,40,60] which
        # determines the last index of each batch: up to 20, from 20 to 40, and
        # from 40 to 60. We add the 0 at the beginning so that
        # batch_index[b]:batch_index[b+1] gives the right samples for batch b.
        batch_index = np.cumsum(batch_size).tolist()
        batch_index = [0] + batch_index

        ###################
        # SAVE ATTRIBUTES #
        ###################

        self.training_options = {}
        self.training_options['do_logging'] = do_logging
        self.training_options['logger'] = logger
        self.training_options['do_save_vars'] = do_save_vars
        self.training_options['do_print'] = do_print
        self.training_options['print_interval'] = print_interval
        self.training_options['do_learning_rate_decay'] = do_learning_rate_decay
        if do_learning_rate_decay:
            self.training_options['learning_rate_decay_rate'] = \
                learning_rate_decay_rate
            self.training_options['learning_rate_decay_period'] = \
                learning_rate_decay_period
        self.training_options['validation_interval'] = validation_interval
        self.training_options['do_early_stopping'] = do_early_stopping
        self.training_options['early_stopping_lag'] = early_stopping_lag
        self.training_options['batch_index'] = batch_index
        self.training_options['batch_size'] = batch_size
        self.training_options['n_epochs'] = n_epochs
        self.training_options['n_batches'] = n_batches
        self.training_options['graph_no'] = graph_no
        self.training_options['realization_no'] = realization_no

    def train_batch(self, this_batch_indices):

        # Get the samples
        x_train, y_train = self.data.get_samples('train', this_batch_indices)
        x_train = x_train.to(self.model.device)
        y_train = y_train.to(self.model.device)

        # Start measuring time
        start_time = datetime.datetime.now()

        # Reset gradients
        self.model.archit.zero_grad()

        # Obtain the output of the GNN
        y_hat_train = self.model.archit(x_train)

        # Compute loss
        loss_value_train = self.model.loss(y_hat_train, y_train)

        # Compute gradients
        loss_value_train.backward()

        # Optimize
        self.model.optim.step()

        # Finish measuring time
        end_time = datetime.datetime.now()

        time_elapsed = abs(end_time - start_time).total_seconds()

        # Compute the accuracy
        #   Note: Using y_hat_train.data creates a new tensor with the
        #   same value, but detaches it from the gradient, so that no
        #   gradient operation is taken into account here.
        #   (Alternatively, we could use a with torch.no_grad():)
        cost_train = self.data.evaluate(y_hat_train.data, y_train)

        return loss_value_train.item(), cost_train.item(), time_elapsed

    def validation_step(self):

        # Validation:
        x_valid, y_valid = self.data.get_samples('valid')
        x_valid = x_valid.to(self.model.device)
        y_valid = y_valid.to(self.model.device)

        # Start measuring time
        start_time = datetime.datetime.now()

        # Under torch.no_grad() so that the computations carried out
        # to obtain the validation accuracy are not taken into
        # account to update the learnable parameters.
        with torch.no_grad():
            # Obtain the output of the GNN
            y_hat_valid = self.model.archit(x_valid)

            # Compute loss
            loss_value_valid = self.model.loss(y_hat_valid, y_valid)

            # Finish measuring time
            end_time = datetime.datetime.now()

            time_elapsed = abs(end_time - start_time).total_seconds()

            # Compute accuracy:
            cost_valid = self.data.evaluate(y_hat_valid, y_valid)

        return loss_value_valid.item(), cost_valid.item(), time_elapsed

    def train(self):

        # Get back the training options
        assert 'training_options' in dir(self)
        assert 'do_logging' in self.training_options.keys()
        do_logging = self.training_options['do_logging']
        assert 'logger' in self.training_options.keys()
        logger = self.training_options['logger']
        assert 'do_save_vars' in self.training_options.keys()
        do_save_vars = self.training_options['do_save_vars']
        assert 'do_print' in self.training_options.keys()
        do_print = self.training_options['do_print']
        assert 'print_interval' in self.training_options.keys()
        print_interval = self.training_options['print_interval']
        assert 'do_learning_rate_decay' in self.training_options.keys()
        do_learning_rate_decay = self.training_options['do_learning_rate_decay']
        if do_learning_rate_decay:
            assert 'learning_rate_decay_rate' in self.training_options.keys()
            learning_rate_decay_rate = self.training_options['learning_rate_decay_rate']
            assert 'learning_rate_decay_period' in self.training_options.keys()
            learning_rate_decay_period = self.training_options['learning_rate_decay_period']
        assert 'validation_interval' in self.training_options.keys()
        validation_interval = self.training_options['validation_interval']
        assert 'do_early_stopping' in self.training_options.keys()
        do_early_stopping = self.training_options['do_early_stopping']
        assert 'early_stopping_lag' in self.training_options.keys()
        early_stopping_lag = self.training_options['early_stopping_lag']
        assert 'batch_index' in self.training_options.keys()
        batch_index = self.training_options['batch_index']
        assert 'batch_size' in self.training_options.keys()
        batch_size = self.training_options['batch_size']
        assert 'n_epochs' in self.training_options.keys()
        n_epochs = self.training_options['n_epochs']
        assert 'n_batches' in self.training_options.keys()
        n_batches = self.training_options['n_batches']
        assert 'graph_no' in self.training_options.keys()
        graph_no = self.training_options['graph_no']
        assert 'realization_no' in self.training_options.keys()
        realization_no = self.training_options['realization_no']

        # Learning rate scheduler:
        if do_learning_rate_decay:
            learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(
                self.model.optim, learning_rate_decay_period, learning_rate_decay_rate)

        # Initialize counters (since we give the possibility of early stopping,
        # we had to drop the 'for' and use a 'while' instead):
        epoch = 0  # epoch counter
        lag_count = 0  # lag counter for early stopping

        # Store the training variables
        loss_train = []
        cost_train = []
        loss_valid = []
        cost_valid = []
        time_train = []
        time_valid = []

        while epoch < n_epochs \
                and (lag_count < early_stopping_lag or (not do_early_stopping)):
            # The condition will be zero (stop), whenever one of the items of
            # the 'and' is zero. Therefore, we want this to stop only for epoch
            # counting when we are NOT doing early stopping. This can be
            # achieved if the second element of the 'and' is always 1 (so that
            # the first element, the epoch counting, decides). In order to
            # force the second element to be one whenever there is not early
            # stopping, we have an or, and force it to one. So, when we are not
            # doing early stopping, the variable 'not do_early_stopping' is 1,
            # and the result of the 'or' is 1 regardless of the lag_count. When
            # we do early stopping, then the variable 'not do_early_stopping' is
            # 0, and the value 1 for the 'or' gate is determined by the lag
            # count.
            # ALTERNATIVELY, we could just keep 'and lag_count<early_stopping_lag'
            # and be sure that lag_count can only be increased whenever
            # do_early_stopping is True. But I somehow figured out that would be
            # harder to maintain (more parts of the code to check if we are
            # accidentally increasing lag_count).

            # Randomize dataset for each epoch
            random_permutation = np.random.permutation(self.data.n_train)
            # Convert a numpy.array of numpy.int into a list of actual int.
            id_x_epoch = [int(i) for i in random_permutation]

            # Learning decay
            if do_learning_rate_decay:
                learning_rate_scheduler.step()

                if do_print:
                    # All the optimization have the same learning rate, so just
                    # print one of them
                    # TODO: Actually, they might be different, so I will need to
                    # print all of them.
                    print("Epoch %d, learning rate = %.8f" % (epoch + 1,
                                                              learning_rate_scheduler.optim.param_groups[0]['lr']))

            # Initialize counter
            batch = 0  # batch counter
            while batch < n_batches \
                    and (lag_count < early_stopping_lag or (not do_early_stopping)):

                # Extract the adequate batch
                this_batch_indices = id_x_epoch[batch_index[batch]
                                                : batch_index[batch + 1]]

                loss_value_train, cost_value_train, time_elapsed = \
                    self.train_batch(this_batch_indices)

                # Logging values
                if do_logging:
                    loss_train_TB = loss_value_train
                    cost_train_TB = cost_value_train
                # Save values
                loss_train += [loss_value_train]
                cost_train += [cost_value_train]
                time_train += [time_elapsed]

                # Print:
                if do_print:
                    if (epoch * n_batches + batch) % print_interval == 0:
                        print("\t(E: %2d, B: %3d) %6.4f / %7.4f - %6.4fs" % (
                            epoch + 1, batch + 1, cost_value_train,
                            loss_value_train, time_elapsed),
                              end=' ')
                        if graph_no > -1:
                            print("[%d" % graph_no, end='')
                            if realization_no > -1:
                                print("/%d" % realization_no,
                                      end='')
                            print("]", end='')
                        print("")

                # \\\\\\\
                # \\\ TB LOGGING (for each batch)
                # \\\\\\\

                if do_logging:
                    logger.scalar_summary(mode='Training',
                                          epoch=epoch * n_batches + batch,
                                          **{'loss_train': loss_train_TB,
                                             'cost_train': cost_train_TB})

                # \\\\\\\
                # \\\ VALIDATION
                # \\\\\\\

                if (epoch * n_batches + batch) % validation_interval == 0:

                    loss_value_valid, cost_value_valid, time_elapsed = \
                        self.validation_step()

                    # Logging values
                    if do_logging:
                        loss_valid_TB = loss_value_valid
                        cost_valid_TB = cost_value_valid
                    # Save values
                    loss_valid += [loss_value_valid]
                    cost_valid += [cost_value_valid]
                    time_valid += [time_elapsed]

                    # Print:
                    if do_print:
                        print("\t(E: %2d, B: %3d) %6.4f / %7.4f - %6.4fs" % (
                            epoch + 1, batch + 1,
                            cost_value_valid,
                            loss_value_valid,
                            time_elapsed), end=' ')
                        print("[VALIDATION", end='')
                        if graph_no > -1:
                            print(".%d" % graph_no, end='')
                            if realization_no > -1:
                                print("/%d" % realization_no, end='')
                        print(" (%s)]" % self.model.name)

                    if do_logging:
                        logger.scalar_summary(mode='Validation',
                                              epoch=epoch * n_batches + batch,
                                              **{'loss_valid': loss_valid_TB,
                                                 'cost_valid': cost_valid_TB})

                    # No previous best option, so let's record the first trial
                    # as the best option
                    if epoch == 0 and batch == 0:
                        best_score = cost_value_valid
                        best_epoch, best_batch = epoch, batch
                        # Save this model as the best (so far)
                        self.model.save(label='best')
                        # Start the counter
                        if do_early_stopping:
                            initial_best = True
                    else:
                        this_valid_score = cost_value_valid
                        if this_valid_score < best_score:
                            best_score = this_valid_score
                            best_epoch, best_batch = epoch, batch
                            if do_print:
                                print("\t=> New best achieved: %.4f" % \
                                      (best_score))
                            self.model.save(label='best')
                            # Now that we have found a best that is not the
                            # initial one, we can start counting the lag (if
                            # needed)
                            initial_best = False
                            # If we achieved a new best, then we need to reset
                            # the lag count.
                            if do_early_stopping:
                                lag_count = 0
                        # If we didn't achieve a new best, increase the lag
                        # count.
                        # Unless it was the initial best, in which case we
                        # haven't found any best yet, so we shouldn't be doing
                        # the early stopping count.
                        elif do_early_stopping and not initial_best:
                            lag_count += 1

                # \\\\\\\
                # \\\ END OF BATCH:
                # \\\\\\\

                # \\\ Increase batch count:
                batch += 1

            # \\\\\\\
            # \\\ END OF EPOCH:
            # \\\\\\\

            # \\\ Increase epoch count:
            epoch += 1

        # \\\ Save models:
        self.model.save(label='last')

        #################
        # TRAINING OVER #
        #################

        # We convert the lists into np.arrays
        loss_train = np.array(loss_train)
        cost_train = np.array(cost_train)
        loss_valid = np.array(loss_valid)
        cost_valid = np.array(cost_valid)
        # And we would like to save all the relevant information from
        # training
        train_vars = {'n_epochs': n_epochs,
                      'n_batches': n_batches,
                      'validation_interval': validation_interval,
                      'batch_size': np.array(batch_size),
                      'batch_index': np.array(batch_index),
                      'loss_train': loss_train,
                      'cost_train': cost_train,
                      'loss_valid': loss_valid,
                      'cost_valid': cost_valid
                      }

        if do_save_vars:
            save_dir_vars = os.path.join(self.model.save_dir, 'train_vars')
            if not os.path.exists(save_dir_vars):
                os.makedirs(save_dir_vars)
            path_to_file = os.path.join(save_dir_vars,
                                        self.model.name + 'train_vars.pkl')
            with open(path_to_file, 'wb') as train_vars_file:
                pickle.dump(train_vars, train_vars_file)

        # Now, if we didn't do any training (i.e. n_epochs = 0), then the last is
        # also the best.
        if n_epochs == 0:
            self.model.save(label='best')
            self.model.save(label='last')
            if do_print:
                print("WARNING: No training. Best and Last models are the same.")

        # After training is done, reload best model before proceeding to
        # evaluation:
        self.model.load(label='best')

        # \\\ Print out best:
        if do_print and n_epochs > 0:
            print("=> Best validation achieved (E: %d, B: %d): %.4f" % (
                best_epoch + 1, best_batch + 1, best_score))

        return train_vars


class TrainerFlocking(Trainer):
    """
    Trainer: trains flocking models, following the appropriate evaluation of
        the cost, and has options for different DAGger alternatives
        
    Initialization:
        
        model (Modules.model class): model to train
        data (Utils.data class): needs to have a get_samples and an evaluate
            method
        n_epochs (int): number of epochs (passes over the dataset)
        batch_size (int): size of each minibatch

        Optional (keyword) arguments:
        
        prob_expert (float): initial probability of choosing the expert
        DAGger_type ('fixed_batch', 'random_epoch', 'replace_time_batch'):
            'fixed_batch' (default if 'prob_expert' is defined): doubles the batch
                samples by considering the same initial velocities and 
                positions, a trajectory given by the latest trained
                architecture, and the corresponding correction given by the
                optimal acceleration (i.e. for each position and velocity we 
                give what would be the optimal acceleration, even though the
                next position and velocity won't reflect this decision, but the
                one taken by the learned policy)
            'random_epoch':  forms a new training set for each epoch consisting,
                with probability prob_expert, of samples of the original dataset
                (optimal trajectories) and with probability 1-prob_expert, with
                trajectories following the latest trained dataset.
            'replace_time_batch': creates a fixed number of new trajectories
                following randomly at each time step either the optimal control
                or the learned control; then, replaces this fixed number of new
                trajectores into the training set (then these might, or might 
                not get selected by the next batch)
            
        validation_interval (int): interval of training (number of training
            steps) without running a validation stage.

        learning_rate_decay_rate (float): float that multiplies the latest learning
            rate used.
        learning_rate_decay_period (int): how many training steps before 
            multiplying the learning rate decay rate by the actual learning
            rate.
        > Obs.: Both of these have to be defined for the learningRateDecay
              scheduler to be activated.
        logger (Visualizer): save tensorboard logs.
        save_dir (string): path to the directory where to save relevant training
            variables.
        print_interval (int): how many training steps after which to print
            partial results (0 means do not print)
        graph_no (int): keep track of what graph realization this is
        realitization_no (int): keep track of what data realization this is
        >> Alternatively, these last two keyword arguments can be used to keep
            track of different trainings of the same model
            
    Training:
        
        .train(): trains the model and returns train_vars dict with the keys
            'n_epochs': number of epochs (int)
            'n_batches': number of batches (int)
            'validation_interval': number of training steps in between 
                validation steps (int)
            'batch_size': batch size of each training step (np.array)
            'batch_index': indices for the start sample and end sample of each
                batch (np.array)
            'best_batch': batch index at which the best model was achieved (int)
            'best_epoch': epoch at which the best model was achieved (int)
            'best_score': evaluation measure on the validation sample that 
                achieved the best model (i.e. minimum achieved evaluation
                measure on the validation set)
            'loss_train': loss function on the training samples for each training
                step (np.array)
            'time_train': time elapsed at each training step (np.array)
            'eval_valid': evaluation function on the validation samples for each
                validation step (np.array)
            'time_valid': time elapsed at each validation step (np.array)
    """

    def __init__(self, model, data, n_epochs, batch_size, **kwargs):

        # Initialize supraclass
        super().__init__(model, data, n_epochs, batch_size, **kwargs)

        # Add the specific options

        if 'prob_expert' in kwargs.keys():
            do_DAGer = True
            prob_expert = kwargs['prob_expert']
        else:
            do_DAGer = False

        if 'DAGger_type' in kwargs.keys():
            DAGger_type = kwargs['DAGger_type']
        else:
            DAGger_type = 'fixed_batch'

        self.training_options['do_DAGer'] = do_DAGer
        if do_DAGer:
            self.training_options['prob_expert'] = prob_expert
            self.training_options['DAGger_type'] = DAGger_type

    def train(self):

        # Get back the training options
        assert 'training_options' in dir(self)
        assert 'do_logging' in self.training_options.keys()
        do_logging = self.training_options['do_logging']
        assert 'logger' in self.training_options.keys()
        logger = self.training_options['logger']
        assert 'do_save_vars' in self.training_options.keys()
        do_save_vars = self.training_options['do_save_vars']
        assert 'do_print' in self.training_options.keys()
        do_print = self.training_options['do_print']
        assert 'print_interval' in self.training_options.keys()
        print_interval = self.training_options['print_interval']
        assert 'do_learning_rate_decay' in self.training_options.keys()
        do_learning_rate_decay = self.training_options['do_learning_rate_decay']
        if do_learning_rate_decay:
            assert 'learning_rate_decay_rate' in self.training_options.keys()
            learning_rate_decay_rate = self.training_options['learning_rate_decay_rate']
            assert 'learning_rate_decay_period' in self.training_options.keys()
            learning_rate_decay_period = self.training_options['learning_rate_decay_period']
        assert 'validation_interval' in self.training_options.keys()
        validation_interval = self.training_options['validation_interval']
        assert 'do_early_stopping' in self.training_options.keys()
        do_early_stopping = self.training_options['do_early_stopping']
        assert 'early_stopping_lag' in self.training_options.keys()
        early_stopping_lag = self.training_options['early_stopping_lag']
        assert 'batch_index' in self.training_options.keys()
        batch_index = self.training_options['batch_index']
        assert 'batch_size' in self.training_options.keys()
        batch_size = self.training_options['batch_size']
        assert 'n_epochs' in self.training_options.keys()
        n_epochs = self.training_options['n_epochs']
        assert 'n_batches' in self.training_options.keys()
        n_batches = self.training_options['n_batches']
        assert 'graph_no' in self.training_options.keys()
        graph_no = self.training_options['graph_no']
        assert 'realization_no' in self.training_options.keys()
        realization_no = self.training_options['realization_no']
        assert 'do_DAGer' in self.training_options.keys()
        do_DAGer = self.training_options['do_DAGer']
        if do_DAGer:
            assert 'DAGger_type' in self.training_options.keys()
            DAGger_type = self.training_options['DAGger_type']

        # Get the values we need
        n_train = self.data.n_train
        this_archit = self.model.archit
        thisLoss = self.model.loss
        this_optim = self.model.optim
        this_device = self.model.device

        # Learning rate scheduler:
        if do_learning_rate_decay:
            learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(self.optim,
                                                                      learning_rate_decay_period,
                                                                      learning_rate_decay_rate)

        # Initialize counters (since we give the possibility of early stopping,
        # we had to drop the 'for' and use a 'while' instead):
        epoch = 0  # epoch counter
        lag_count = 0  # lag counter for early stopping

        if do_save_vars:
            loss_train = []
            eval_valid = []
            time_train = []
            time_valid = []

        # Get original dataset
        x_train_orig, y_train_orig = self.data.get_samples('train')
        S_train_orig = self.data.get_data('comm_graph', 'train')
        init_vel_train_all = self.data.get_data('init_vel', 'train')
        if do_DAGer:
            init_pos_train_all = self.data.get_data('init_pos', 'train')

        # And save it as the original "all samples"
        x_train_all = x_train_orig
        y_train_all = y_train_orig
        S_train_all = S_train_orig

        # If it is:
        #   'random_epoch' assigns always the original training set at the
        #       beginning of each epoch, so it is reset by using the variable
        #       Orig, instead of the variable all
        #   'replace_time_batch' keeps working only in the All variables, so
        #       every epoch updates the previous dataset, and never goes back
        #       to the original dataset (i.e. there is no Orig involved in
        #       the 'replace_time_batch' DAGger)
        #   'fixed_batch': it takes All = Orig from the beginning and then it
        #       doesn't matter becuase it always acts by creating a new
        #       batch with "corrected" trajectories for the learned policies

        while epoch < n_epochs \
                and (lag_count < early_stopping_lag or (not do_early_stopping)):
            # The condition will be zero (stop), whenever one of the items of
            # the 'and' is zero. Therefore, we want this to stop only for epoch
            # counting when we are NOT doing early stopping. This can be
            # achieved if the second element of the 'and' is always 1 (so that
            # the first element, the epoch counting, decides). In order to
            # force the second element to be one whenever there is not early
            # stopping, we have an or, and force it to one. So, when we are not
            # doing early stopping, the variable 'not do_early_stopping' is 1,
            # and the result of the 'or' is 1 regardless of the lag_count. When
            # we do early stopping, then the variable 'not do_early_stopping' is
            # 0, and the value 1 for the 'or' gate is determined by the lag
            # count.
            # ALTERNATIVELY, we could just keep 'and lag_count<early_stopping_lag'
            # and be sure that lag_count can only be increased whenever
            # do_early_stopping is True. But I somehow figured out that would be
            # harder to maintain (more parts of the code to check if we are
            # accidentally increasing lag_count).

            # Randomize dataset for each epoch
            random_permutation = np.random.permutation(n_train)
            # Convert a numpy.array of numpy.int into a list of actual int.
            id_x_epoch = [int(i) for i in random_permutation]

            # Learning decay
            if do_learning_rate_decay:
                learning_rate_scheduler.step()

                if do_print:
                    # All the optimization have the same learning rate, so just
                    # print one of them
                    # TODO: Actually, they might be different, so I will need to
                    # print all of them.
                    print("Epoch %d, learning rate = %.8f" % (epoch + 1,
                                                              learning_rate_scheduler.optim.param_groups[0]['lr']))

            # \\\\\\\\\\\\\\\\
            # \\\ Start DAGGER: random_epoch
            # \\\
            if do_DAGer and epoch > 0 and DAGger_type == 'random_epoch':
                # The 'random_epoch' option forms a new training set for each
                # epoch consisting, with probability prob_expert, of samples
                # of the original dataset (optimal trajectories) and with
                # probability 1-prob_expert, with trajectories following the
                # latest trained dataset.

                x_train_all, y_train_all, S_train_all = \
                    self.random_epoch_DAGger(epoch, x_train_orig, y_train_orig,
                                             S_train_orig, init_pos_train_all,
                                             init_vel_train_all)
            # \\\
            # \\\ Finished DAGGER
            # \\\\\\\\\\\\\\\\\\\

            # Initialize counter
            batch = 0  # batch counter
            while batch < n_batches \
                    and (lag_count < early_stopping_lag or (not do_early_stopping)):

                # \\\\\\\\\\\\\\\\
                # \\\ Start DAGGER: replace_time_batch
                # \\\
                if do_DAGer and (batch > 0 or epoch > 0) \
                        and DAGger_type == 'replace_time_batch':
                    # The option 'replace_time_batch' creates a fixed number of
                    # new trajectories following randomly at each time step
                    # either the optimal control or the learned control
                    # Then, replaces this fixed number of new trajectores into
                    # the training set (then these might, or might not get
                    # selected by the next batch)

                    x_train_all, y_train_all, S_train_all = \
                        self.replace_time_batch_DAGger(epoch, x_train_all, y_train_all,
                                                       S_train_all, init_pos_train_all,
                                                       init_vel_train_all)
                # \\\
                # \\\ Finished DAGGER
                # \\\\\\\\\\\\\\\\\\\

                # Extract the adequate batch
                this_batch_indices = id_x_epoch[batch_index[batch]
                                                : batch_index[batch + 1]]
                # Get the samples
                x_train = x_train_all[this_batch_indices]
                y_train = y_train_all[this_batch_indices]
                S_train = S_train_all[this_batch_indices]
                init_vel_train = init_vel_train_all[this_batch_indices]
                if do_DAGer and DAGger_type == 'fixed_batch':
                    init_pos_train = init_pos_train_all[this_batch_indices]

                # \\\\\\\\\\\\\\\\
                # \\\ Start DAGGER: fixed_batch
                # \\\
                if do_DAGer and (batch > 0 or epoch > 0) \
                        and DAGger_type == 'fixed_batch':
                    # The 'fixed_batch' option, doubles the batch samples
                    # by considering the same initial velocities and
                    # positions, a trajectory given by the latest trained
                    # architecture, and the corresponding correction
                    # given by the optimal acceleration (i.e. for each
                    # position and velocity we give what would be the
                    # optimal acceleration, even though the next position
                    # and velocity won't reflect this decision, but the
                    # one taken by the learned policy)

                    x_DAG, y_DAG, S_DAG = self.fixed_batchDAGger(init_pos_train,
                                                                 init_vel_train)

                    x_train = np.concatenate((x_train, x_DAG), axis=0)
                    S_train = np.concatenate((S_train, S_DAG), axis=0)
                    y_train = np.concatenate((y_train, y_DAG), axis=0)
                    init_vel_train = np.tile(init_vel_train, (2, 1, 1))
                # \\\
                # \\\ Finished DAGGER
                # \\\\\\\\\\\\\\\\\\\

                # Now that we have our dataset, move it to tensor and device
                # so we can use it
                x_train = torch.tensor(x_train, device=this_device)
                S_train = torch.tensor(S_train, device=this_device)
                y_train = torch.tensor(y_train, device=this_device)
                init_vel_train = torch.tensor(init_vel_train, device=this_device)

                # Start measuring time
                start_time = datetime.datetime.now()

                # Reset gradients
                this_archit.zero_grad()

                # Obtain the output of the GNN
                y_hat_train = this_archit(x_train, S_train)

                # Compute loss
                loss_value_train = thisLoss(y_hat_train, y_train)

                # Compute gradients
                loss_value_train.backward()

                # Optimize
                this_optim.step()

                # Finish measuring time
                end_time = datetime.datetime.now()

                time_elapsed = abs(end_time - start_time).total_seconds()

                # Logging values
                if do_logging:
                    loss_train_TB = loss_value_train.item()
                # Save values
                if do_save_vars:
                    loss_train += [loss_value_train.item()]
                    time_train += [time_elapsed]

                # Print:
                if do_print and print_interval > 0:
                    if (epoch * n_batches + batch) % print_interval == 0:
                        print("\t(E: %2d, B: %3d) %7.4f - %6.4fs" % (
                            epoch + 1, batch + 1,
                            loss_value_train.item(), time_elapsed),
                              end=' ')
                        if graph_no > -1:
                            print("[%d" % graph_no, end='')
                            if realization_no > -1:
                                print("/%d" % realization_no,
                                      end='')
                            print("]", end='')
                        print("")

                # Delete variables to free space in CUDA memory
                del x_train
                del S_train
                del y_train
                del init_vel_train
                del loss_value_train

                # \\\\\\\
                # \\\ TB LOGGING (for each batch)
                # \\\\\\\

                if do_logging:
                    logger.scalar_summary(mode='Training',
                                          epoch=epoch * n_batches + batch,
                                          **{'loss_train': loss_train_TB})

                # \\\\\\\
                # \\\ VALIDATION
                # \\\\\\\

                if (epoch * n_batches + batch) % validation_interval == 0:

                    # Start measuring time
                    start_time = datetime.datetime.now()

                    # Create trajectories

                    # Initial data
                    init_pos_valid = self.data.get_data('init_pos', 'valid')
                    init_vel_valid = self.data.get_data('init_vel', 'valid')

                    # Compute trajectories
                    _, vel_test_valid, _, _, _ = self.data.compute_trajectory(
                        init_pos_valid, init_vel_valid, self.data.duration,
                        archit=this_archit, do_print=False)

                    # Compute evaluation
                    acc_valid = self.data.evaluate(vel=vel_test_valid)

                    # Finish measuring time
                    end_time = datetime.datetime.now()

                    time_elapsed = abs(end_time - start_time).total_seconds()

                    # Logging values
                    if do_logging:
                        eval_valid_TB = acc_valid
                    # Save values
                    if do_save_vars:
                        eval_valid += [acc_valid]
                        time_valid += [time_elapsed]

                    # Print:
                    if do_print:
                        print("\t(E: %2d, B: %3d) %8.4f - %6.4fs" % (
                            epoch + 1, batch + 1,
                            acc_valid,
                            time_elapsed), end=' ')
                        print("[VALIDATION", end='')
                        if graph_no > -1:
                            print(".%d" % graph_no, end='')
                            if realization_no > -1:
                                print("/%d" % realization_no, end='')
                        print(" (%s)]" % self.model.name)

                    if do_logging:
                        logger.scalar_summary(mode='Validation',
                                              epoch=epoch * n_batches + batch,
                                              **{'eval_valid': eval_valid_TB})

                    # No previous best option, so let's record the first trial
                    # as the best option
                    if epoch == 0 and batch == 0:
                        best_score = acc_valid
                        best_epoch, best_batch = epoch, batch
                        # Save this model as the best (so far)
                        self.model.save(label='best')
                        # Start the counter
                        if do_early_stopping:
                            initial_best = True
                    else:
                        this_valid_score = acc_valid
                        if this_valid_score < best_score:
                            best_score = this_valid_score
                            best_epoch, best_batch = epoch, batch
                            if do_print:
                                print("\t=> New best achieved: %.4f" % \
                                      (best_score))
                            self.model.save(label='best')
                            # Now that we have found a best that is not the
                            # initial one, we can start counting the lag (if
                            # needed)
                            initial_best = False
                            # If we achieved a new best, then we need to reset
                            # the lag count.
                            if do_early_stopping:
                                lag_count = 0
                        # If we didn't achieve a new best, increase the lag
                        # count.
                        # Unless it was the initial best, in which case we
                        # haven't found any best yet, so we shouldn't be doing
                        # the early stopping count.
                        elif do_early_stopping and not initial_best:
                            lag_count += 1

                    # Delete variables to free space in CUDA memory
                    del init_vel_valid
                    del init_pos_valid

                # \\\\\\\
                # \\\ END OF BATCH:
                # \\\\\\\

                # \\\ Increase batch count:
                batch += 1

            # \\\\\\\
            # \\\ END OF EPOCH:
            # \\\\\\\

            # \\\ Increase epoch count:
            epoch += 1

        # \\\ Save models:
        self.model.save(label='last')

        #################
        # TRAINING OVER #
        #################

        if do_save_vars:
            # We convert the lists into np.arrays
            loss_train = np.array(loss_train)
            eval_valid = np.array(eval_valid)
            # And we would like to save all the relevant information from
            # training
            train_vars = {'n_epochs': n_epochs,
                          'n_batches': n_batches,
                          'validation_interval': validation_interval,
                          'batch_size': np.array(batch_size),
                          'batch_index': np.array(batch_index),
                          'best_batch': best_batch,
                          'best_epoch': best_epoch,
                          'best_score': best_score,
                          'loss_train': loss_train,
                          'time_train': time_train,
                          'eval_valid': eval_valid,
                          'time_valid': time_valid
                          }
            save_dir_vars = os.path.join(self.model.save_dir, 'train_vars')
            if not os.path.exists(save_dir_vars):
                os.makedirs(save_dir_vars)
            path_to_file = os.path.join(save_dir_vars, self.model.name + 'train_vars.pkl')
            with open(path_to_file, 'wb') as train_vars_file:
                pickle.dump(train_vars, train_vars_file)

        # Now, if we didn't do any training (i.e. n_epochs = 0), then the last is
        # also the best.
        if n_epochs == 0:
            self.model.save(label='best')
            self.model.save(label='last')
            if do_print:
                print("\nWARNING: No training. Best and Last models are the same.\n")

        # After training is done, reload best model before proceeding to
        # evaluation:
        self.model.load(label='best')

        # \\\ Print out best:
        if do_print and n_epochs > 0:
            print("\t=> Best validation achieved (E: %d, B: %d): %.4f" % (
                best_epoch + 1, best_batch + 1, best_score))

        return train_vars

    def random_epoch_DAGger(self, epoch, x_train_orig, y_train_orig, S_train_orig,
                            init_pos_train_all, init_vel_train_all):

        # The 'random_epoch' option forms a new training set for each
        # epoch consisting, with probability prob_expert, of samples
        # of the original dataset (optimal trajectories) and with
        # probability 1-prob_expert, with trajectories following the
        # latest trained dataset.

        assert 'prob_expert' in self.training_options.kwargs()
        prob_expert = self.training_options['prob_expert']
        n_train = x_train_orig.shape[0]

        # Compute the prob expert
        choose_expert_prob = np.max((prob_expert ** epoch, 0.5))

        # What we will pass to the actual training epoch are:
        # x_train, S_train and y_train for computation
        x_DAG = np.zeros(x_train_orig.shape)
        y_DAG = np.zeros(y_train_orig.shape)
        S_DAG = np.zeros(S_train_orig.shape)
        # init_vel_train is needed for evaluation, but doesn't change

        # For each sample, choose whether we keep the optimal
        # trajectory or we add the learned trajectory
        for s in range(n_train):

            if np.random.binomial(1, choose_expert_prob) == 1:

                # If we choose the expert, we just get the values of
                # the optimal trajectory

                x_DAG[s] = x_train_orig[s]
                y_DAG[s] = y_train_orig[s]
                S_DAG[s] = S_train_orig[s]

            else:

                # If not, we compute a new trajectory based on the
                # given architecture
                pos_DAG, vel_DAG, _, _, _ = self.data.compute_trajectory(
                    init_pos_train_all[s:s + 1], init_vel_train_all[s:s + 1],
                    self.data.duration, archit=self.model.archit,
                    do_print=False)

                # Now that we have the position and velocity trajectory
                # that we would get based on the learned controller,
                # we need to compute what the optimal acceleration
                # would actually be in each case.
                # And since this could be a large trajectory, we need
                # to split it based on how many samples

                max_time_samples = 200

                if pos_DAG.shape[1] > max_time_samples:

                    # Create the space
                    y_DAG_aux = np.zeros((1,  # batch_size
                                          pos_DAG.shape[1],  # tSamples
                                          2,
                                          pos_DAG.shape[3]))  # nAgents

                    for t in range(pos_DAG.shape[1]):
                        # Compute the expert on the corresponding
                        # trajectory
                        #   First, we need the difference in positions
                        ij_diff_pos, ij_dist_sq = \
                            self.data.compute_differences(pos_DAG[:, t, :, :])
                        #   And in velocities
                        ij_diff_vel, _ = \
                            self.data.compute_differences(vel_DAG[:, t, :, :])
                        #   Now, the second term (the one that depends
                        #   on the positions) only needs to be computed
                        #   for nodes thatare within repel distance, so
                        #   let's compute a mask to find these nodes.
                        repel_mask = (ij_dist_sq < (self.data.repel_dist ** 2)) \
                            .astype(ij_diff_pos.dtype)
                        #   Apply this mask to the position difference
                        #   (we need not apply it to the square
                        #   differences since these will be multiplied
                        #   by the position differences which already
                        #   will be zero)
                        #   Note that we need to add the dimension of axis
                        #   to properly multiply it
                        ij_diff_pos = ij_diff_pos * \
                                      np.expand_dims(repel_mask, 1)
                        #   Invert the tensor elementwise (avoiding the
                        #   zeros)
                        ij_dist_sq_inv = invert_tensor_ew(ij_dist_sq)
                        #   Add an extra dimension, also across the
                        #   axis
                        ij_dist_sq_inv = np.expand_dims(ij_dist_sq_inv, 1)
                        #   Compute the optimal solution
                        this_accel = -np.sum(ij_diff_vel, axis=3) \
                                     + 2 * np.sum(ij_diff_pos * \
                                                  (ij_dist_sq_inv ** 2 + ij_dist_sq_inv),
                                                  axis=3)
                        # And cap it
                        this_accel[this_accel > self.data.accel_max] = \
                            self.data.accel_max
                        this_accel[this_accel < -self.data.accel_max] = \
                            -self.data.accel_max

                        # Store it
                        y_DAG_aux[:, t, :, :] = this_accel

                else:
                    # Compute the expert on the corresponding
                    # trajectory
                    #   First, we need the difference in positions
                    ij_diff_pos, ij_dist_sq = self.data.compute_differences(pos_DAG)
                    #   And in velocities
                    ij_diff_vel, _ = self.data.compute_differences(vel_DAG)
                    #   Now, the second term (the one that depends on
                    #   the positions) only needs to be computed for
                    #   nodes that are within repel distance, so let's
                    #   compute a mask to find these nodes.
                    repel_mask = (ij_dist_sq < (self.data.repel_dist ** 2)) \
                        .astype(ij_diff_pos.dtype)
                    #   Apply this mask to the position difference (we
                    #   need not apply it to the square differences,
                    #   since these will be multiplied by the position
                    #   differences, which already will be zero)
                    #   Note that we need to add the dimension of axis
                    #   to properly multiply it
                    ij_diff_pos = ij_diff_pos * np.expand_dims(repel_mask, 2)
                    #   Invert the tensor elementwise (avoiding the
                    #   zeros)
                    ij_dist_sq_inv = invert_tensor_ew(ij_dist_sq)
                    #   Add an extra dimension, also across the axis
                    ij_dist_sq_inv = np.expand_dims(ij_dist_sq_inv, 2)
                    #   Compute the optimal solution
                    y_DAG_aux = -np.sum(ij_diff_vel, axis=4) \
                                + 2 * np.sum(ij_diff_pos * \
                                             (ij_dist_sq_inv ** 2 + ij_dist_sq_inv),
                                             axis=4)
                    # And cap it
                    y_DAG_aux[y_DAG_aux > self.data.accel_max] = self.data.accel_max
                    y_DAG_aux[y_DAG_aux < -self.data.accel_max] = -self.data.accel_max

                # Finally, compute the corresponding graph of states
                # (pos) visited by the policy
                S_DAG_aux = self.data.compute_communications_graph(
                    pos_DAG, self.data.comm_radius, True, do_print=False)
                x_DAG_aux = self.data.compute_states(pos_DAG, vel_DAG, S_DAG_aux,
                                                     do_print=False)

                # And save them
                x_DAG[s] = x_DAG_aux[0]
                y_DAG[s] = y_DAG_aux[0]
                S_DAG[s] = S_DAG_aux[0]

        # And now that we have created the DAGger alternatives, we
        # just need to consider them as the basic training variables
        return x_DAG, y_DAG, S_DAG

    def replace_time_batch_DAGger(self, epoch, x_train_all, y_train_all, S_train_all,
                                  init_pos_train_all, init_vel_train_all, n_replace=10):

        # The option 'replace_time_batch' creates a fixed number of
        # new trajectories following randomly at each time step
        # either the optimal control or the learned control
        # Then, replaces this fixed number of new trajectores into
        # the training set (then these might, or might not get
        # selected by the next batch)

        assert 'prob_expert' in self.training_options.kwargs()
        prob_expert = self.training_options['prob_expert']
        n_train = x_train_all.shape[0]

        if n_replace > n_train:
            n_replace = n_train

        # Select the indices of the samples to replace
        replace_indices = np.random.permutation(n_train)[0:n_replace]

        # Get the corresponding initial velocities and positions
        init_pos_train_this = init_pos_train_all[replace_indices]
        init_vel_train_this = init_vel_train_all[replace_indices]

        # Save the resulting trajectories
        x_DAG = np.zeros((n_replace,
                          x_train_all.shape[1],
                          6,
                          x_train_all.shape[3]))
        y_DAG = np.zeros((n_replace,
                          y_train_all.shape[1],
                          2,
                          y_train_all.shape[3]))
        S_DAG = np.zeros((n_replace,
                          S_train_all.shape[1],
                          S_train_all.shape[2],
                          S_train_all.shape[3]))
        pos_DAG = np.zeros(y_DAG.shape)
        vel_DAG = np.zeros(y_DAG.shape)

        # Initialize first elements
        pos_DAG[:, 0, :, :] = init_pos_train_this
        vel_DAG[:, 0, :, :] = init_vel_train_this
        S_DAG[:, 0, :, :] = S_train_all[replace_indices, 0]
        x_DAG[:, 0, :, :] = x_train_all[replace_indices, 0]

        # Compute the prob expert
        choose_expert_prob = np.max((prob_expert ** (epoch + 1), 0.5))

        # Now, for each sample
        for s in range(n_replace):

            # For each time instant
            for t in range(1, x_train_all.shape[1]):

                # Decide whether we apply the learned or the
                # optimal controller
                if np.random.binomial(1, choose_expert_prob) == 1:

                    # Compute the optimal acceleration
                    ij_diff_pos, ij_dist_sq = \
                        self.data.compute_differences(pos_DAG[s:s + 1, t - 1, :, :])
                    ij_diff_vel, _ = \
                        self.data.compute_differences(vel_DAG[s:s + 1, t - 1, :, :])
                    repel_mask = (ij_dist_sq < (self.data.repel_dist ** 2)) \
                        .astype(ij_diff_pos.dtype)
                    ij_diff_pos = ij_diff_pos * \
                                  np.expand_dims(repel_mask, 1)
                    ij_dist_sq_inv = invert_tensor_ew(ij_dist_sq)
                    ij_dist_sq_inv = np.expand_dims(ij_dist_sq_inv, 1)
                    this_accel = -np.sum(ij_diff_vel, axis=3) \
                                 + 2 * np.sum(ij_diff_pos * \
                                              (ij_dist_sq_inv ** 2 + ij_dist_sq_inv),
                                              axis=3)
                else:

                    # Compute the learned acceleration
                    #   Add the sample dimension
                    x_this = np.expand_dims(x_DAG[s, 0:t, :, :], 0)
                    S_this = np.expand_dims(S_DAG[s, 0:t, :, :], 0)
                    #   Convert to tensor
                    x_this = torch.tensor(x_this, device=self.model.device)
                    S_this = torch.tensor(S_this, device=self.model.device)
                    #   Compute the acceleration
                    with torch.no_grad():
                        this_accel = self.model.archit(x_this, S_this)
                    #   Get only the last acceleration
                    this_accel = this_accel.cpu().numpy()[:, -1, :, :]

                # Cap the acceleration
                this_accel[this_accel > self.data.accel_max] = self.data.accel_max
                this_accel[this_accel < -self.data.accel_max] = -self.data.accel_max
                # Save it
                y_DAG[s, t - 1, :, :] = this_accel.squeeze(0)

                # Update the position and velocity
                vel_DAG[s, t, :, :] = \
                    y_DAG[s, t - 1, :, :] * self.data.sampling_time \
                    + vel_DAG[s, t - 1, :, :]
                pos_DAG[s, t, :, :] = \
                    vel_DAG[s, t - 1, :, :] * self.data.sampling_time \
                    + pos_DAG[s, t - 1, :, :]
                # Update the state and the graph
                this_graph = self.data.compute_communications_graph(
                    pos_DAG[s:s + 1, t:t + 1, :, :], self.data.comm_radius,
                    True, do_print=False)
                S_DAG[s, t, :, :] = this_graph.squeeze(1).squeeze(0)
                this_state = self.data.compute_states(
                    pos_DAG[s:s + 1, t:t + 1, :, :],
                    vel_DAG[s:s + 1, t:t + 1, :, :],
                    S_DAG[s:s + 1, t:t + 1, :, :],
                    do_print=False)
                x_DAG[s, t, :, :] = this_state.squeeze(1).squeeze(0)

            # And now compute the last acceleration step

            if np.random.binomial(1, choose_expert_prob) == 1:

                # Compute the optimal acceleration
                ij_diff_pos, ij_dist_sq = \
                    self.data.compute_differences(pos_DAG[s:s + 1, -1, :, :])
                ij_diff_vel, _ = \
                    self.data.compute_differences(vel_DAG[s:s + 1, -1, :, :])
                repel_mask = (ij_dist_sq < (self.data.repel_dist ** 2)) \
                    .astype(ij_diff_pos.dtype)
                ij_diff_pos = ij_diff_pos * \
                              np.expand_dims(repel_mask, 1)
                ij_dist_sq_inv = invert_tensor_ew(ij_dist_sq)
                ij_dist_sq_inv = np.expand_dims(ij_dist_sq_inv, 1)
                this_accel = -np.sum(ij_diff_vel, axis=3) \
                             + 2 * np.sum(ij_diff_pos * \
                                          (ij_dist_sq_inv ** 2 + ij_dist_sq_inv),
                                          axis=3)
            else:

                # Compute the learned acceleration
                #   Add the sample dimension
                x_this = np.expand_dims(x_DAG[s], 0)
                S_this = np.expand_dims(S_DAG[s], 0)
                #   Convert to tensor
                x_this = torch.tensor(x_this, device=self.model.device)
                S_this = torch.tensor(S_this, device=self.model.device)
                #   Compute the acceleration
                with torch.no_grad():
                    this_accel = self.model.archit(x_this, S_this)
                #   Get only the last acceleration
                this_accel = this_accel.cpu().numpy()[:, -1, :, :]

            # Cap the acceleration
            this_accel[this_accel > self.data.accel_max] = self.data.accel_max
            this_accel[this_accel < -self.data.accel_max] = -self.data.accel_max
            # Save it
            y_DAG[s, -1, :, :] = this_accel.squeeze(0)

        # And now that we have done this for all the samples in
        # the replacement set, just replace them

        x_train_all[replace_indices] = x_DAG
        y_train_all[replace_indices] = y_DAG
        S_train_all[replace_indices] = S_DAG

        return x_train_all, y_train_all, S_train_all

    def fixed_batchDAGger(self, init_pos_train, init_vel_train):

        # The 'fixed_batch' option, doubles the batch samples
        # by considering the same initial velocities and
        # positions, a trajectory given by the latest trained
        # architecture, and the corresponding correction
        # given by the optimal acceleration (i.e. for each
        # position and velocity we give what would be the
        # optimal acceleration, even though the next position
        # and velocity won't reflect this decision, but the
        # one taken by the learned policy)

        # Note that there's no point on doing it randomly here,
        # since the optimal trajectory is already considered in
        # the batch anyways.

        # \\\\\\\\\\\\\\\\
        # \\\ Start DAGGER

        # Always apply DAGger on the trained policy
        pos_pol, vel_pol, _, _, _ = \
            self.data.compute_trajectory(init_pos_train,
                                         init_vel_train,
                                         self.data.duration,
                                         archit=self.model.archit,
                                         do_print=False)

        # Compute the optimal acceleration on the trajectory given
        # by the trained policy

        max_time_samples = 200

        if pos_pol.shape[1] > max_time_samples:

            # Create the space to store this
            y_DAG = np.zeros(pos_pol.shape)

            for t in range(pos_pol.shape[1]):
                # Compute the expert on the corresponding trajectory
                #   First, we need the difference in positions
                ij_diff_pos, ij_dist_sq = \
                    self.data.compute_differences(pos_pol[:, t, :, :])
                #   And in velocities
                ij_diff_vel, _ = \
                    self.data.compute_differences(vel_pol[:, t, :, :])
                #   Now, the second term (the one that depends on
                #   the positions) only needs to be computed for
                #   nodes thatare within repel distance, so let's
                #   compute a mask to find these nodes.
                repel_mask = (ij_dist_sq < (self.data.repel_dist ** 2)) \
                    .astype(ij_diff_pos.dtype)
                #   Apply this mask to the position difference (we
                #   need not apply it to the square differences,
                #   since these will be multiplied by the position
                #   differences which already will be zero)
                #   Note that we need to add the dimension of axis
                #   to properly multiply it
                ij_diff_pos = ij_diff_pos * np.expand_dims(repel_mask, 1)
                #   Invert the tensor elementwise (avoiding the
                #   zeros)
                ij_dist_sq_inv = invert_tensor_ew(ij_dist_sq)
                #   Add an extra dimension, also across the axis
                ij_dist_sq_inv = np.expand_dims(ij_dist_sq_inv, 1)
                #   Compute the optimal solution
                this_accel = -np.sum(ij_diff_vel, axis=3) \
                             + 2 * np.sum(ij_diff_pos * \
                                          (ij_dist_sq_inv ** 2 + ij_dist_sq_inv),
                                          axis=3)
                # And cap it
                this_accel[this_accel > self.data.accel_max] = self.data.accel_max
                this_accel[this_accel < -self.data.accel_max] = -self.data.accel_max

                # Store it
                y_DAG[:, t, :, :] = this_accel

        else:
            # Compute the expert on the corresponding trajectory
            #   First, we need the difference in positions
            ij_diff_pos, ij_dist_sq = self.data.compute_differences(pos_pol)
            #   And in velocities
            ij_diff_vel, _ = self.data.compute_differences(vel_pol)
            #   Now, the second term (the one that depends on the
            #   positions) only needs to be computed for nodes that
            #   are within repel distance, so let's compute a mask
            #   to find these nodes.
            repel_mask = (ij_dist_sq < (self.data.repel_dist ** 2)) \
                .astype(ij_diff_pos.dtype)
            #   Apply this mask to the position difference (we need
            #   not apply it to the square differences, since these
            #   will be multiplied by the position differences,
            #   which already will be zero)
            #   Note that we need to add the dimension of axis to
            #   properly multiply it
            ij_diff_pos = ij_diff_pos * np.expand_dims(repel_mask, 2)
            #   Invert the tensor elementwise (avoiding the zeros)
            ij_dist_sq_inv = invert_tensor_ew(ij_dist_sq)
            #   Add an extra dimension, also across the axis
            ij_dist_sq_inv = np.expand_dims(ij_dist_sq_inv, 2)
            #   Compute the optimal solution
            y_DAG = -np.sum(ij_diff_vel, axis=4) \
                    + 2 * np.sum(ij_diff_pos * \
                                 (ij_dist_sq_inv ** 2 + ij_dist_sq_inv),
                                 axis=4)
            # And cap it
            y_DAG[y_DAG > self.data.accel_max] = self.data.accel_max
            y_DAG[y_DAG < -self.data.accel_max] = -self.data.accel_max

        # Finally, compute the corresponding graph of states
        # (pos) visited by the policy
        graph_DAG = self.data.compute_communications_graph(pos_pol,
                                                           self.data.comm_radius,
                                                           True,
                                                           do_print=False)
        x_DAG = self.data.compute_states(pos_pol, vel_pol, graph_DAG,
                                         do_print=False)

        # Add it to the existing batch

        return x_DAG, y_DAG, graph_DAG
