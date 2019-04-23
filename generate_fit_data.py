import numpy as np
from deeplearning_kandidat_data import normalizer as norm


def gen_fit_data(input_data= [],
                 input_results=[],
                 train_size=100,
                 test_size=0,
                 seed=100,
                 data_type='subtask',
                 normalize_results=True,
                 u5=False,
                 grade_points=[]):
    data = np.copy(input_data)
    results = np.copy(input_results)

    np.random.seed(seed)  # Set a seed for randomization - to control output of np.random
    random_users = np.random.randint(0, data.shape[0] - test_size, size=data.shape[0] - test_size)  # Shuffle data
    shuffled_float_data = data[random_users]
    # Normalize the now shuffled data and results matrices
    if data_type == 'subtask' or data_type == 'exercise':
        norm_float_data = norm.normalize_tensor_data_new(shuffled_float_data, train_size)
    elif data_type == 'global':
        norm_float_data = norm.normalize_global_data(global_data_tensor=data, train_data_size=train_size)
    else:
        norm_float_data = []
    x_train = norm_float_data[:train_size]
    x_val = norm_float_data[train_size:]
    # ******* Results data ******
    # Results are the same for multi input and regular input NN
    shuffled_float_results = results[random_users]
    if normalize_results:
        if u5:
            norm_float_results = norm.normalize_results_u5(shuffled_float_results, grade_points)
        else:
            norm_float_results = norm.normalize_results(shuffled_float_results, grade_points[0])
    else:
        norm_float_results = shuffled_float_results
    y_val = norm_float_results[train_size:]
    y_train = norm_float_results[:train_size]

    # return a non-normalized version of y_val as reference
    results_reference = shuffled_float_results[train_size:]

    x_test = []
    y_test = []
    return x_train, y_train, x_val, y_val, x_test, y_test, results_reference
