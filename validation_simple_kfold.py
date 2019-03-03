# -*- coding: utf-8 -*-
"""Automatically generated by Colaboratory.
"""

# This version implements a simple k-fold scheme to generate a performance metric for a NN

import time
import numpy as np
from deeplearning_kandidat_data import normalizer as norm


def validate_nn(model_fcn=None,
                seeds=[],
                data=[],
                results=[],
                data_type='subtask',
                model_name="Generic NN",
                loss_fcn="binary crossentropy",
                optimizer_fcn="rmsprop",
                epochs=24,
                batch_size=8,
                train_size=100,
                test_size=0,
                verbose=1,
                multi_input=False,
                multi_input_data = {}):

    # Todo add callbacks for tesnorboard

    if verbose == 1:
        print("*********** Validating model: " + model_name + " ***********")
    start_total = time.time()
    acc_matrix = [[], []]  # min, max
    loss_matrix = [[],[]]  # min, max

    if verbose == 1:
        print("Training model: " + model_name)
    for idx, seed in enumerate(seeds):
        if verbose == 1:
            print("\tProgress: " + str(idx+1) + "/" + str(len(seeds)) + ".", end=" ")
        start_seed = time.time()
        # ***************** Normalize data *******************
        np.random.seed(seed)  # Set a seed for randomization - to control output of np.random
        random_users = np.random.randint(0, data.shape[0] - test_size, size=data.shape[0] - test_size) # Shuffle data

        # ******* Results data ******
        # Results are the same for multi input and regular input NN
        shuffled_float_results = results[random_users]
        norm_float_results = norm.normalize_results(shuffled_float_results)
        y_val = norm_float_results[train_size:]
        y_train = norm_float_results[:train_size]

        # ******* Training data ******
        if multi_input:
            # ************* Special case - multi input NN *******************
            multi_global = multi_input_data['global']
            multi_subtask = multi_input_data['subtask']

            # Shuffle data
            shuffled_multi_global = multi_global[random_users]
            shuffled_multi_subtask = multi_subtask[random_users]

            # Normalize data
            norm_multi_global = norm.normalize_global_data(global_data_tensor=shuffled_multi_global,
                                                         train_data_size=train_size)
            norm_multi_subtask = norm.normalize_data(data_tensor=shuffled_multi_subtask, train_data_size=train_size)

            # Split into validation set and training set
            multi_train_data_x = {
                'global': norm_multi_global[:train_size],
                'subtask': norm_multi_subtask[:train_size]
            }

            multi_val_data_x = {
                'global': norm_multi_global[train_size:],
                'subtask': norm_multi_subtask[train_size:]
            }
        else:
            shuffled_float_data = data[random_users]
            # Normalize the now shuffled data and results matrices
            if data_type == 'subtask':
                norm_float_data = norm.normalize_subtask_data_new(shuffled_float_data, train_size)

            elif data_type == 'global':
                norm_float_data = norm.normalize_global_data(global_data_tensor=data, train_data_size=train_size)
            else:
                norm_float_data = []
            x_train = norm_float_data[:train_size]
            x_val = norm_float_data[train_size:]
        # ******************** Train NN ***********************
        if multi_input:
            model = model_fcn(multi_train_data_x, optimizer_fcn=optimizer_fcn, loss_fcn=loss_fcn)
            out = model.fit(multi_train_data_x, y_train,
                            validation_data=[multi_val_data_x, y_val],
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=0)
        else:
            model = model_fcn(data=x_train, optimizer_fcn=optimizer_fcn, loss_fcn=loss_fcn)
            out = model.fit(x_train, y_train,
                            validation_data=[x_val, y_val],
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=0)
        val_acc = out.history['val_acc']
        val_loss = out.history['val_loss']

        acc_matrix[0].append(min(val_acc))
        acc_matrix[1].append(max(val_acc))

        loss_matrix[0].append(min(val_loss))
        loss_matrix[1].append(max(val_loss))
        end_seed = time.time()
        if verbose == 1:
            print("Seed time: " + str(end_seed-start_seed) + "s")
    end_total = time.time()
    total_time = end_total-start_total
    if verbose == 1:
        print("Total time: " + str(total_time) + "s")
        print()

    print("********************** RESULTS ************************")
    print("Number of Seeds: " + str(len(seeds)))
    print("Loss function: " + loss_fcn)
    print("Optimizer function: " + optimizer_fcn)
    if multi_input:
        print('Multi input: ' + str(multi_input))
        print("Number of global features: " + str(multi_global.shape[1]))
        print("Number of subtask features: " + str(multi_subtask.shape[1]))
    else:
        print("Number of features: " + str(data.shape[1]))
    print("Epochs: " + str(epochs))
    print("Batch size: " + str(batch_size))
    print("Total run time of script: " + str(total_time) + "s")
    #print("*******************************************************")
    print("**************** Model: " + model_name + " ****************")
    val_accs = np.array(acc_matrix)
    val_losses = np.array(loss_matrix)
    # Calculate accuracy score
    val_acc_max_mean = np.mean(val_accs[1])
    val_acc_min_mean = np.mean(val_accs[0])
    val_acc_max_std = np.std(val_accs[1])
    val_acc_min_std = np.std(val_accs[0])
    # Calculate loss score
    val_loss_max_mean = np.mean(val_losses[1])
    val_loss_min_mean = np.mean(val_losses[0])
    val_loss_max_std = np.std(val_losses[1])
    val_loss_min_std = np.std(val_losses[0])

    print("\tMax validation acc (mean +-std): " + str(val_acc_max_mean) + " +- " + str(val_acc_max_std))
    print("\tMin validation acc (mean +-std): " + str(val_acc_min_mean) + " +- " + str(val_acc_min_std))

    print("\tMax validation loss (mean +-std): " + str(val_loss_max_mean) + " +- " + str(val_loss_max_std))
    print("\tMin validation loss (mean +-std): " + str(val_loss_min_mean) + " +- " + str(val_loss_min_std))
    print("****************************************************")