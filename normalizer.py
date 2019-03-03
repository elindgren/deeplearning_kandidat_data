import numpy as np


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


def normalize_subtask_data_new(subtask_data_tensor, train_data_size):
    train_subset = subtask_data_tensor[:train_data_size]  # Normalize all data from the training data
    mean_matrix = np.zeros(
        (train_subset.shape[1], train_subset.shape[2]))  # A matrix containing mean values for each dataset
    std_matrix = np.zeros((train_subset.shape[1], train_subset.shape[2]))
    for i, val in enumerate(train_subset[0, 0, :]):
        #user_matrix = subtask_data_tensor[:, :, i]
        for j in range(len(train_subset[0, :, 0])):
            mean_matrix[j][i] = train_subset[:, j, i].mean(axis=0)
            subtask_data_tensor[:, j, i] -= mean_matrix[j][i]
            std_matrix[j][i] = train_subset[:, j, i].std()
            if std_matrix[j][i] != 0:
                subtask_data_tensor[:, j, i] /= std_matrix[j][i]
    return subtask_data_tensor


def normalize_global_data(global_data_tensor, train_data_size):
    train_subset = global_data_tensor[:train_data_size]  # Normalize all data from the training data
    # Normalize for each feature along the user dimension.
    mean_vector = np.zeros((train_subset.shape[1]))
    std_vector = np.zeros((train_subset.shape[1]))
    for i in range(len(global_data_tensor[0,:])):
        mean_vector[i] = train_subset[:, i].mean(axis=0)
        global_data_tensor[:, i] -= mean_vector[i]
        std_vector[i] = train_subset[:, i].std()
        if std_vector[i] != 0:
            global_data_tensor[:, i] /= std_vector[i]
    return global_data_tensor


def normalize_results(results):  # As of now only converts to 0 or 1
    # Normalize results vector - simplify output space to 0 or 1 for passed or not passed
    for i, result in enumerate(results):
        if float(results[i]) >= 5:  # If more than 4 points, the user passed the exam
            results[i] = float(1)
        else:
            results[i] = float(0)
    return results
