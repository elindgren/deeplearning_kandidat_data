# -*- coding: utf-8 -*-
'''Automatically generated by Colaboratory.
'''

# This version implements a simple k-fold scheme to generate a performance metric for a NN

import time
import numpy as np
import random as rn
import copy

from keras.models import load_model
from keras.layers import Dense, Flatten, LSTM
from keras.layers import Input, ReLU, Dropout, concatenate
from keras.models import Model
from keras.losses import mse, mae
from keras.utils import plot_model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import Callback
from keras import backend as K

from tqdm import tqdm_notebook as tqdm


def validate_model(model_fcn=None,
                    seeds=[],
                    data = {},
                    data_type='subtask',
                    problem_type='UG',
                    keras_parameters={},
                    verbose=1):
    # Reference
    # data = {
    #     'FFM516': {
    #         input_data: [],
    #         result_data: [],
    #         grade_points: [],
    #         train_size: 100,
    #         test_size: 0
    #     }
    #
    # }
    # keras_parameters = {
    #         model_name: 'Generic NN',
    #         epochs: 24,
    #         batch_size: 8,
    #         loss_function: 'binary_crossentropy',
    #         optimizer_function: 'rmsprop'
    # }

    # Declare arrays to hold min, max of accuracy and loss
    acc_matrix = np.zeros((len(seeds), 2))  # min, max
    loss_matrix = np.zeros((len(seeds), 2))  # min, max

    # Iterate over seeds
    for idx, seed in enumerate(tqdm(seeds)):
        # Tuple that will contain training and validation data
        x_train_tuple, y_train_tuple, x_val_tuple, y_val_tuple = (), (), (), ()
        # iterate over courses in input data
        for course in data:
            # Copy data and results
            copy_data = np.copy(course['input_data'])
            copy_result = np.copy(course['result_data'])
            # Generate fit data
            x_train, y_train, x_val, y_val, x_test, y_test, _, _ = gen_fit_data(input_data=copy_data,
                                                                                    input_results=copy_result,
                                                                                    grade_points=course['grade_points'],
                                                                                    train_size=course['train_size'],
                                                                                    test_size=course['test_size'],
                                                                                    data_type=data_type,
                                                                                    problem_type=problem_type,
                                                                                    seed=seed,
                                                                                    normalize_results=True)
            # Concatenate to tuples
            x_train_tuple = x_train_tuple + x_train
            y_train_tuple = y_train_tuple + y_train
            x_val_tuple = x_val_tuple + x_val
            y_val_tuple = y_val_tuple + y_val

        # Concatenate all tensors in the tuples
        x_train = np.concatenate(x_train_tuple, axis=0)
        y_train = np.concatenate(y_train_tuple, axis=0)
        x_val = np.concatenate(x_val_tuple, axis=0)
        y_val = np.concatenate(y_val_tuple, axis=0)

        # Fit NN
        model = model_fcn(data=x_train,
                          optimizer_fcn=keras_parameters['optimizer_function'],
                          loss_fcn=keras_parameters['loss_function'])

        out = model.fit(x_train, y_train,
                        validation_data=[x_val, y_val],
                        epochs=keras_parameters['epochs'],
                        batch_size=keras_parameters['batch_size'],
                        verbose=0)

        # Retreive accuracy and loss information
        val_acc = out.history['val_acc']
        val_loss = out.history['val_loss']
        # Save to matrices
        acc_matrix[idx][0] = min(val_acc)
        acc_matrix[idx][1] = max(val_acc)
        loss_matrix[idx][0] = min(val_loss)
        loss_matrix[idx][1] = max(val_loss)

    # Calculate accuracy score
    val_acc_min_median = np.median(acc_matrix[:,0])
    val_acc_max_median = np.median(acc_matrix[:,1])
    val_acc_min_mean = np.mean(acc_matrix[:,0])
    val_acc_max_mean = np.mean(acc_matrix[:,1])
    val_acc_min_std = np.std(acc_matrix[:,0])
    val_acc_max_std = np.std(acc_matrix[:,1])

    # Calculate loss score
    val_loss_min_median = np.median(loss_matrix[:,0])
    val_loss_max_median = np.median(loss_matrix[:,1])
    val_loss_min_mean = np.mean(loss_matrix[:,0])
    val_loss_max_mean = np.mean(loss_matrix[:,1])
    val_loss_min_std = np.std(loss_matrix[:,0])
    val_loss_max_std = np.std(loss_matrix[:,1])

    # Print results
    print('********************** RESULTS ************************')
    print('Number of Seeds: ' + str(len(seeds)))
    print('**************** Model: ' + keras_parameters['model_name'] + ' parameters ****************')
    print('Epochs: ' + keras_parameters['epochs'])
    print('Batch size: ' + keras_parameters['batch_size'])
    print('Loss function: ' + keras_parameters['loss_function'])
    print('Optimizer function: ' + keras_parameters['optimizer_function'])
    print('**************** Model: ' + keras_parameters['model_name'] + ' results ****************')

    print('\tMax validation acc (mean +-std, median): ' + str(val_acc_max_mean) + ' +- ' + str(
        val_acc_max_std) + ', median: ' + str(val_acc_max_median))
    print('\tMin validation acc (mean +-std, median): ' + str(val_acc_min_mean) + ' +- ' + str(
        val_acc_min_std) + ', median: ' + str(val_acc_min_median))

    print('\tMax validation loss (mean +-std, median): ' + str(val_loss_max_mean) + ' +- ' + str(
        val_loss_max_std) + ', median: ' + str(val_loss_max_median))
    print('\tMin validation loss (mean +-std, median): ' + str(val_loss_min_mean) + ' +- ' + str(
        val_loss_min_std) + ', median: ' + str(val_loss_min_median))
    print('****************************************************')


def gen_fit_data(input_data=[],
                 input_results=[],
                 train_size=100,
                 test_size=0,
                 seed=100,
                 data_type='subtask',
                 problem_type='UG',
                 normalize_results=True,
                 grade_points=[]):

    data = np.copy(input_data)
    results = np.copy(input_results)

    rn.seed(seed)  # Set a seed for randomization - to control output of np.random
    # random_users = np.random.randint(0, data.shape[0] - test_size, size=data.shape[0] - test_size)  # Shuffle data
    random_users = rn.sample(range(0, data.shape[0]-test_size), data.shape[0]-test_size)  # Shuffle data
    shuffled_float_data = data[random_users]
    # Normalize the now shuffled data and results matrices
    if data_type == 'subtask' or data_type == 'exercise':
        norm_float_data = normalize_tensor_data_new(shuffled_float_data, train_size)
    elif data_type == 'global':
        norm_float_data = normalize_global_data(global_data_tensor=data, train_data_size=train_size)
    else:
        norm_float_data = []
    x_train = norm_float_data[:train_size]
    x_val = norm_float_data[train_size:]
    # ******* Results data ******
    # Results are the same for multi input and regular input NN
    shuffled_float_results = results[random_users]
    if normalize_results:
        if problem_type == 'U5':
            norm_float_results = normalize_results_u5(shuffled_float_results, grade_points)
        elif problem_type == 'UG':
            norm_float_results = normalize_results(shuffled_float_results, grade_points[0])
        else:
            sys.exit("Incorrect problem_type")
    else:
        norm_float_results = shuffled_float_results
    y_val = norm_float_results[train_size:]
    y_train = norm_float_results[:train_size]

    # return a non-normalized version of y_val as reference
    results_reference = shuffled_float_results[train_size:]
    # return a non-normalized version of x_val as reference
    data_reference = shuffled_float_data[train_size:]

    x_test = []
    y_test = []
    return x_train, y_train, x_val, y_val, x_test, y_test, data_reference, results_reference


def normalize_tensor_data_new(data_tensor, train_data_size):
    tmp = np.copy(data_tensor)
    train_subset = tmp[:train_data_size]  # Normalize all data from the training data
    mean_matrix = np.zeros(
        (train_subset.shape[1], train_subset.shape[2]))  # A matrix containing mean values for each dataset
    std_matrix = np.zeros((train_subset.shape[1], train_subset.shape[2]))
    for i, val in enumerate(train_subset[0, 0, :]):
        #user_matrix = subtask_data_tensor[:, :, i]
        for j in range(len(train_subset[0, :, 0])):
            mean_matrix[j][i] = train_subset[:, j, i].mean(axis=0)
            tmp[:, j, i] -= mean_matrix[j][i]
            std_matrix[j][i] = train_subset[:, j, i].std()
            if std_matrix[j][i] != 0:
                tmp[:, j, i] /= std_matrix[j][i]
    return tmp


def normalize_global_data(global_data_tensor, train_data_size):
    tmp = np.copy(global_data_tensor)
    train_subset = tmp[:train_data_size]  # Normalize all data from the training data
    # Normalize for each feature along the user dimension.
    mean_vector = np.zeros((train_subset.shape[1]))
    std_vector = np.zeros((train_subset.shape[1]))
    for i in range(len(tmp[0, :])):
        mean_vector[i] = train_subset[:, i].mean(axis=0)
        tmp[:, i] -= mean_vector[i]
        std_vector[i] = train_subset[:, i].std()
        if std_vector[i] != 0:
            tmp[:, i] /= std_vector[i]
    return tmp


def normalize_results(results, passing_points):  # As of now only converts to 0 or 1
    tmp = np.copy(results)
    # Normalize results vector - simplify output space to 0 or 1 for passed or not passed
    for i, result in enumerate(tmp):
        if float(tmp[i]) >= passing_points:  # If more than 4 points, the user passed the exam
            tmp[i] = float(1)
        else:
            tmp[i] = float(0)
    return tmp


def normalize_results_u5(results, grade_points):
    tmp = np.copy(results)
    one_hot = np.zeros((tmp.shape[0], 4))  # 4 grades - U, 3, 4, 5
    for i, result in enumerate(tmp):
        if result >= grade_points[2]:
            one_hot[i, 3] = float(1)
        elif (result < grade_points[2]) and (result >= grade_points[1]):
            one_hot[i, 2] = float(1)
        elif (result < grade_points[1]) and (result >= grade_points[0]):
            one_hot[i, 1] = float(1)
        else:
            one_hot[i, 0] = float(1)
    return one_hot
