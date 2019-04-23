# -*- coding: utf-8 -*-
"""Automatically generated by Colaboratory.
"""

# This version implements a simple k-fold scheme to generate a performance metric for a NN

import time
import numpy as np
from deeplearning_kandidat_data import normalizer as norm
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


# Simons' losses **********
def loss_laplace(y_true, y_pred):
    loss = K.abs(y_true - y_pred)
    loss = K.exp(-s) * loss
    loss = s + loss
    loss = K.mean(loss)

    return loss


def loss_normal(y_true, y_pred):
    loss = K.square(y_true - y_pred)
    loss = K.exp(-s) * loss
    loss = s + loss
    loss = K.mean(loss)

    return loss


def custom_loss_laplace(s):
    return loss_laplace


def custom_loss_normal(s):
    def loss_normal(y_true, y_pred):
        loss = K.square(y_true - y_pred)
        loss = K.exp(-s) * loss
        loss = s + loss
        loss = K.mean(loss)

        return loss
    return loss_normal
# *******************


def validate_nn(model_fcn=None,
                seeds=[],
                input_data=[[]],
                input_results=[[]],
                data_type='subtask',
                model_name="Generic NN",
                loss_fcn="binary crossentropy",
                optimizer_fcn="rmsprop",
                epochs=24,
                batch_size=8,
                train_size=[100],
                test_size=[0],
                verbose=1,
                multi_input=False,
                multi_input_data={},
                u5=False,
                flip=False,
                grade_points=[[]]):

    # Todo add callbacks for tensorboard
    # Create copies of input data and results data - to avoid overwriting them
    data = np.copy(input_data)
    multi_data = copy.deepcopy(multi_input_data)  # Create a deep copy - all sub arrays has to be new as well
    results = np.copy(input_results)
    if verbose == 1:
        print("*********** Validating model: " + model_name + " ***********")
    start_total = time.time()

    acc_matrix = np.zeros((len(seeds), 2))  # min, max
    loss_matrix = np.zeros((len(seeds), 2))  # min, max

    if verbose == 1:
        print("Training model: " + model_name)
    for idx, seed in enumerate(seeds):
        if verbose == 1:
            print("\tProgress: " + str(idx+1) + "/" + str(len(seeds)) + ".", end=" ")
        start_seed = time.time()

        # Loop over all tensors in data
        for i, course_data in enumerate(data):
            tr_size = train_size[i]
            te_size = test_size[i]
            # ***************** Normalize data *******************
            np.random.seed(seed)  # Set a seed for randomization - to control output of np.random
            random_users = np.random.randint(0, course_data.shape[0] - te_size, size=course_data.shape[0] - te_size)  # Shuffle data

            # ******* Results data ******
            course_results = results[i]
            # Results are the same for multi input and regular input NN
            shuffled_float_results = course_results[random_users]
            if u5:
                norm_float_results = norm.normalize_results_u5(shuffled_float_results, grade_points[i])
            else:
                norm_float_results = norm.normalize_results(shuffled_float_results, grade_points[i][0])
            # Declare or append to tensors
            if i == 0:
                y_train = norm_float_results[:tr_size]
                y_val = norm_float_results[tr_size:]
            else:
                y_train = np.append(y_train, norm_float_results[:tr_size], axis=0)
                y_val = np.append(y_val, norm_float_results[tr_size:], axis=0)
            # ******* Training data ******
            if multi_input:
                # ************* Special case - multi input NN *******************
                multi_train_data_x = {}
                multi_val_data_x = {}
                shuffled_data = {}
                normalized_data = {}
                for key in multi_data:
                    # shuffle data
                    shuffled_data[key] = multi_data[key][random_users]
                    # normalize data
                    if key == 'subtask' or key == 'exercise':
                        normalized_data[key] = norm.normalize_tensor_data_new(data_tensor=shuffled_data[key],
                                                                        train_data_size=train_size)
                    elif key == 'global':
                        normalized_data[key] = norm.normalize_global_data(global_data_tensor=shuffled_data[key],
                                                                    train_data_size=train_size)
                    else:
                        normalized_data[key] = []
                    # Split into validation set and training set
                    multi_train_data_x[key] = normalized_data[key][:train_size]
                    multi_val_data_x[key] = normalized_data[key][train_size:]
            else:
                shuffled_float_data = course_data[random_users]
                # Normalize the now shuffled data and results matrices
                if data_type == 'subtask' or data_type == 'exercise':
                    norm_float_data = norm.normalize_tensor_data_new(shuffled_float_data, train_size[i])
                elif data_type == 'global':
                    norm_float_data = norm.normalize_global_data(global_data_tensor=shuffled_float_data, train_data_size=train_size[i])
                else:
                    norm_float_data = []
                if i == 0:
                    x_train = norm_float_data[:tr_size]
                    x_val = norm_float_data[tr_size:]
                else:
                    x_train = np.append(x_train, norm_float_data[:tr_size], axis=0)
                    x_val = np.append(x_val, norm_float_data[tr_size:], axis=0)
        # Debug
        # print("Shape x_train: " + str(x_train.shape))
        # print("Shape y_train: " + str(y_train.shape))
        # print("Shape x_val: " + str(x_val.shape))
        # print("Shape y_val: " + str(y_val.shape))
        # Flip axes for RNN
        if flip:
            x_train = np.swapaxes(x_train, 1, 2)
            x_val = np.swapaxes(x_val, 1, 2)
        # # ******************** Train NN ***********************
        if multi_input:
            model = model_fcn(data=multi_train_data_x, optimizer_fcn=optimizer_fcn, loss_fcn=loss_fcn)
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

        acc_matrix[idx][0] = min(val_acc)
        acc_matrix[idx][1] = max(val_acc)

        loss_matrix[idx][0] = min(val_loss)
        loss_matrix[idx][1] = max(val_loss)
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
        for key in multi_data:
            print("Number of " + key + " features: " + str(multi_data[key].shape[1]))
    else:
        print("Number of features: " + str(course_data.shape[1]))
    print("Epochs: " + str(epochs))
    print("Batch size: " + str(batch_size))
    print("Total run time of script: " + str(total_time) + "s")
    #print("*******************************************************")
    print("**************** Model: " + model_name + " ****************")
    # Calculate accuracy score
    #print(acc_matrix)
    val_acc_max_mean = np.mean(acc_matrix[:, 1])
    val_acc_min_mean = np.mean(acc_matrix[:, 0])
    val_acc_max_std = np.std(acc_matrix[:, 1])
    val_acc_min_std = np.std(acc_matrix[:, 0])
    # Calculate loss score
    val_loss_max_mean = np.mean(loss_matrix[:, 1])
    val_loss_min_mean = np.mean(loss_matrix[:, 0])
    val_loss_max_std = np.std(loss_matrix[:, 1])
    val_loss_min_std = np.std(loss_matrix[:, 0])

    print("\tMax validation acc (mean +-std): " + str(val_acc_max_mean) + " +- " + str(val_acc_max_std))
    print("\tMin validation acc (mean +-std): " + str(val_acc_min_mean) + " +- " + str(val_acc_min_std))

    print("\tMax validation loss (mean +-std): " + str(val_loss_max_mean) + " +- " + str(val_loss_max_std))
    print("\tMin validation loss (mean +-std): " + str(val_loss_min_mean) + " +- " + str(val_loss_min_std))
    print("****************************************************")


def validate_aleatoric(model_fcn=None,
                seeds=[],
                input_data=[],
                input_results=[],
                data_type='subtask',
                model_name="Generic NN",
                loss_fcn="binary crossentropy",
                optimizer_fcn="rmsprop",
                epochs=100,
                batch_size=8,
                train_size=100,
                test_size=0,
                verbose=1,
                callback=None,
                custom_objects=None,
                grade_points=[]):

    # Todo add callbacks for tensorboard
    # Create copies of input data and results data - to avoid overwriting them
    data = np.copy(input_data)
    results = np.copy(input_results)
    if verbose == 1:
        print("*********** Validating model: " + model_name + " ***********")
    start_total = time.time()
    val_users = input_data.shape[0]-train_size
    abs_errors = np.zeros((len(seeds), val_users))
    sigma_xs = np.zeros((len(seeds), val_users))

    if verbose == 1:
        print("Training model: " + model_name)
    for idx, seed in enumerate(seeds):
        if verbose == 1:
            print("\tProgress: " + str(idx+1) + "/" + str(len(seeds)) + ".", end=" ")
        start_seed = time.time()

        tr_size = train_size
        te_size = test_size
        # ***************** Normalize data *******************
        np.random.seed(seed)  # Set a seed for randomization - to control output of np.random
        random_users = np.random.randint(0, data.shape[0] - te_size, size=data.shape[0] - te_size)  # Shuffle data

        # ******* Results data ******
        # Shuffle
        shuffled_float_results = results[random_users]
        y_train = shuffled_float_results[:tr_size]
        y_val = shuffled_float_results[tr_size:]
        # ******* Training data ******
        shuffled_float_data = data[random_users]
        # Normalize the now shuffled data and results matrices
        if data_type == 'subtask' or data_type == 'exercise':
            norm_float_data = norm.normalize_tensor_data_new(shuffled_float_data, tr_size)
        elif data_type == 'global':
            norm_float_data = norm.normalize_global_data(global_data_tensor=shuffled_float_data, train_data_size=tr_size)
        else:
            norm_float_data = []
        x_train = norm_float_data[:tr_size]
        x_val = norm_float_data[tr_size:]
        # # ******************** Train NN ***********************
        # MODEL
        input_shape = (data.shape[1], data.shape[2])
        inputs = Input(shape=input_shape, name='DNN_input')
        x = Flatten()(inputs)
        x = Dropout(0.45)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(64, activation='relu')(x)
        y = Dense(1)(x)  # predict exam point
        s = Dense(1)(x)  # predict aletoric uncertainty

        train_outputs = y
        predict_outputs = [y, s]

        train_model = Model(inputs, train_outputs, name='train_model')
        predict_model = Model(inputs, predict_outputs, name='predict_model')
        train_model.compile(loss=custom_loss_normal(s), optimizer='rmsprop')
        #
        _ = train_model.fit(x_train, y_train,
                            validation_data=[x_val, y_val],
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=0,
                            callbacks=[callback(filepath="best_predict_model_s" + str(seed) + ".h5", verbose=0, predict_model=predict_model)])
        best_predict_model = load_model(filepath="best_predict_model_s" + str(seed) + ".h5", custom_objects=custom_objects)
        predict = best_predict_model.predict(x_val)
        exam_scores_pred = predict[0]
        exam_scores_s = predict[1]
        exam_scores_sigma_x = np.exp(exam_scores_s / 2)

        diff = y_val - exam_scores_pred[:,0]
        abs_errors[idx,:] = np.abs(diff)
        sigma_xs[idx, :] = exam_scores_sigma_x

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
    #print("Loss function: " + loss_fcn.toString())
    #print("Optimizer function: " + optimizer_fcn.toString())
    print("Number of features: " + str(data.shape[1]))
    print("Epochs: " + str(epochs))
    print("Batch size: " + str(batch_size))
    print("Total run time of script: " + str(total_time) + "s")
    #print("*******************************************************")
    print("**************** Model: " + model_name + " ****************")
    # Calculate accuracy score
    mean_abs_error = np.mean(abs_errors)
    std_abs_error = np.std(abs_errors)
    mean_sigma_x = np.mean(sigma_xs)
    std_sigma_x = np.std(sigma_xs)

    print("\tAbs error (mean+-std): " + str(mean_abs_error) + " +- " + str(std_abs_error))
    print("\tSigma_x(mean+-std): " + str(mean_sigma_x) + " +- " + str(std_sigma_x))
    print("****************************************************")