import numpy as np
import copy


def normalize_data(data_tensor, train_data_size):
    train_data = data_tensor[:train_data_size]  # Normalize all data from the training data
    mean_matrix = np.zeros(
        (train_data.shape[1], train_data.shape[2]))  # A matrix containing mean values for each dataset
    std_matrix = np.zeros((train_data.shape[1], train_data.shape[2]))
    for i, val in enumerate(train_data[0, 0, :]):
        # Normalize results dimension
        mean_matrix[0][i] = train_data[:, 0, i].mean(axis=0)
        data_tensor[:, 0, i] -= mean_matrix[0][i]
        std_matrix[0][i] = train_data[:, 0, i].std()
        if std_matrix[0][i] != 0:
            data_tensor[:, 0, i] /= std_matrix[0][i]
        # Normalize tries dimension
        mean_matrix[1][i] = train_data[:, 1, i].mean(axis=0)
        data_tensor[:, 1, i] -= mean_matrix[1][i]
        std_matrix[1][i] = train_data[:, 1, i].std()
        if std_matrix[1][i] != 0:
            data_tensor[:, 1, i] /= std_matrix[1][i]
        # Normalize time from course start dimension
        mean_matrix[2][i] = train_data[:, 2, i].mean(axis=0)
        data_tensor[:, 2, i] -= mean_matrix[2][i]
        std_matrix[2][i] = train_data[:, 2, i].std()
        if std_matrix[2][i] != 0:
            data_tensor[:, 2, i] /= std_matrix[2][i]
        # Normalize time until solved from first try dimension
        mean_matrix[3][i] = train_data[:, 3, i].mean(axis=0)
        data_tensor[:, 3, i] -= mean_matrix[3][i]
        std_matrix[3][i] = train_data[:, 3, i].std()
        if std_matrix[3][i] != 0:
            data_tensor[:, 3, i] /= std_matrix[3][i]
        # Normalize mean time between tries
        mean_matrix[4][i] = train_data[:, 4, i].mean(axis=0)
        data_tensor[:, 4, i] -= mean_matrix[4][i]
        std_matrix[4][i] = train_data[:, 4, i].std()
        if std_matrix[4][i] != 0:
            data_tensor[:, 4, i] /= std_matrix[4][i]
    return data_tensor


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
