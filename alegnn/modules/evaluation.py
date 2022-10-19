# 2020/02/25~
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu
"""
evaluation.py Evaluation Module

Methods for evaluating the models.

evaluate: evaluate a model
evaluate_single_node: evaluate a model that has a single node forward
evaluate_flocking: evaluate a model using the flocking cost
"""

import os
import pickle

import torch


def evaluate(model, data, **kwargs):
    """
    evaluate: evaluate a model using classification error
    
    Input:
        model (model class): class from Modules.model
        data (data class): a data class from the Utils.dataTools; it needs to
            have a get_samples method and an evaluate method.
        do_print (optional, bool): if True prints results
    
    Output:
        eval_vars (dict): 'errorBest' contains the error rate for the best
            model, and 'errorLast' contains the error rate for the last model
    """

    # Get the device we're working on
    device = model.device

    do_save_vars = kwargs.get("do_save_vars", True)

    ########
    # DATA #
    ########
    x_test, y_test = data.get_samples('test')
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    ##############
    # BEST MODEL #
    ##############
    model.load(label='Best')

    with torch.no_grad():
        y_hat_test = model.archit(x_test)  # Process the samples (y_hat_test is of shape [testSize x numberOfClasses])
        cost_best = data.evaluate(y_hat_test, y_test)  # We compute the error

    ##############
    # LAST MODEL #
    ##############
    model.load(label='Last')

    with torch.no_grad():
        y_hat_test = model.archit(x_test)  # Process the samples (y_hat_test is of shape [testSize x numberOfClasses])
        cost_last = data.evaluate(y_hat_test, y_test)  # We compute the error

    eval_vars = {"cost_best": cost_best.item(), "cost_last": cost_last.item()}

    if do_save_vars:
        save_dir_vars = os.path.join(model.save_dir, 'eval_vars')
        if not os.path.exists(save_dir_vars): os.makedirs(save_dir_vars)
        path_to_file = os.path.join(save_dir_vars, model.name + 'eval_vars.pkl')
        with open(path_to_file, 'wb') as eval_varsFile:
            pickle.dump(eval_vars, eval_varsFile)

    return eval_vars


def evaluate_flocking(model, data, **kwargs):
    """
    evaluateClassif: evaluate a model using the flocking cost of velocity
        variacne of the team

    Input:
        model (model class): class from Modules.model
        data (data class): the data class that generates the flocking data
        do_print (optional; bool, default: True): if True prints results
        n_videos (optional; int, default: 3): number of videos to save
        graph_no (optional): identify the run with a number
        realization_no (optional): identify the run with another number

    Output:
        eval_vars (dict):
            'cost_bestFull': cost of the best model over the full trajectory
            'cost_bestEnd': cost of the best model at the end of the trajectory
            'cost_lastFull': cost of the last model over the full trajectory
            'cost_lastEnd': cost of the last model at the end of the trajectory
    """
    do_print = kwargs.get("do_print", True)
    n_videos = kwargs.get("n_videos", 3)
    graph_no = kwargs.get("graph_no", -1)

    if 'realization_no' in kwargs.keys():
        if 'graph_no' in kwargs.keys():
            realization_no = kwargs['realization_no']
        else:
            graph_no = kwargs['realization_no']
            realization_no = -1
    else:
        realization_no = -1

    # \\\\\\\\\\\\\\\\\\\\
    # \\\ TRAJECTORIES \\\
    # \\\\\\\\\\\\\\\\\\\\

    ########
    # DATA #
    ########

    # Initial data
    init_pos_test = data.get_data('init_pos', 'test')
    init_vel_test = data.get_data('init_vel', 'test')

    ##############
    # BEST MODEL #
    ##############

    model.load(label='Best')

    if do_print: print("\tComputing learned trajectory for best model...", end=' ', flush=True)

    pos_test_best, vel_test_best, accel_test_best, state_test_best, comm_graph_test_best = \
        data.compute_trajectory(init_pos_test, init_vel_test, data.duration, archit=model.archit)

    if do_print: print("OK")

    ##############
    # LAST MODEL #
    ##############

    model.load(label='Last')

    if do_print: print("\tComputing learned trajectory for last model...", end=' ', flush=True)

    pos_test_last, vel_test_last, accel_test_last, state_test_last, comm_graph_test_last = \
        data.compute_trajectory(init_pos_test, init_vel_test, data.duration, archit=model.archit)

    if do_print: print("OK")

    ###########
    # PREVIEW #
    ###########

    learned_trajectories_dir = os.path.join(model.save_dir,
                                            'learnedTrajectories')

    if not os.path.exists(learned_trajectories_dir):
        os.mkdir(learned_trajectories_dir)

    if graph_no > -1:
        learned_trajectories_dir = os.path.join(learned_trajectories_dir,
                                                '%03d' % graph_no)
        if not os.path.exists(learned_trajectories_dir):
            os.mkdir(learned_trajectories_dir)
    if realization_no > -1:
        learned_trajectories_dir = os.path.join(learned_trajectories_dir,
                                                '%03d' % realization_no)
        if not os.path.exists(learned_trajectories_dir):
            os.mkdir(learned_trajectories_dir)

    learned_trajectories_dir = os.path.join(learned_trajectories_dir, model.name)

    if not os.path.exists(learned_trajectories_dir):
        os.mkdir(learned_trajectories_dir)

    if do_print: print("\tPreview data...", end=' ', flush=True)

    data.save_video(os.path.join(learned_trajectories_dir, 'Best'),
                    pos_test_best,
                    n_videos,
                    commGraph=comm_graph_test_best,
                    vel=vel_test_best,
                    videoSpeed=0.5,
                    do_print=False)

    data.save_video(os.path.join(learned_trajectories_dir, 'Last'),
                    pos_test_last,
                    n_videos,
                    commGraph=comm_graph_test_last,
                    vel=vel_test_last,
                    videoSpeed=0.5,
                    do_print=False)

    if do_print: print("OK", flush=True)

    # \\\\\\\\\\\\\\\\\\
    # \\\ EVALUATION \\\
    # \\\\\\\\\\\\\\\\\\

    eval_vars = {
        "cost_best_full": data.evaluate(vel=vel_test_best),
        "cost_best_end": data.evaluate(vel=vel_test_best[:, -1:, :, :]),
        "cost_last_full": data.evaluate(vel=vel_test_last),
        "cost_last_end": data.evaluate(vel=vel_test_last[:, -1:, :, :])
    }

    return eval_vars
