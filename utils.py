import csv
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def get_model_parameters_linear(model):
    """
    Returns the parameters of a sklearn LogisticRegression model.
    """
    if model.fit_intercept:
        params = [model.coef_, model.intercept_]
    else:
        params = [model.coef_, ]
    return params

def set_model_params_linear(model: LogisticRegression, params) -> LogisticRegression:
    """
    Sets the parameters of a sklearn LogisticRegression model.
    """
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

def set_initial_params_linear(model: LogisticRegression, n_classes, n_features):
    """
    Sets initial parameters as zeros.
    """
    model.classes_ = np.array([i for i in range(n_classes)])
    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))

def partition(X: np.ndarray, y: np.ndarray, num_partitions: int):
    """Splits X and y into a number of partitions."""
    return list(zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions)))

def load_mnist_linear():
    """
    Loads the MNIST dataset for linear models.
    """
    # Load MNIST dataset from https://www.openml.org/d/554
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Shuffling the training data
    indices = np.random.permutation(len(x_train))
    x_train = x_train[indices]
    y_train = y_train[indices]

    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))

    # Standardize the features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return (x_train, y_train), (x_test, y_test)

def load_mnist_nonlinear():
    """
    Loads the MNIST dataset for nonlinear models.
    """
    # Load MNIST dataset from https://www.openml.org/d/554
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Shuffling the training data
    indices = np.random.permutation(len(x_train))
    x_train = x_train[indices]
    y_train = y_train[indices]

    x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
    return (x_train, y_train), (x_test, y_test)


def load_kinases(client_id=-1):
    """
    Loads the kinases dataset from multiple CSV files.
    """
    file_paths = ['kinases_FLT3_dataset/ChemDB_FLT3_processed.csv', 'kinases_FLT3_dataset/PKIS_FLT3_processed.csv', 'kinases_FLT3_dataset/Tang_FLT3_processed.csv']
    inputs = []
    labels = []
    sets = []

    # Iterate over each file
    for file_path in file_paths:
        # Read the CSV file
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header row

            # Process each row
            for row in csv_reader:
                input_data = list(map(float, row[:8192]))
                inputs.append(input_data)
                label = int(float(row[8192]))
                labels.append(label)
                set_type = row[8193]
                sets.append(set_type)

    X = np.array(inputs)
    y = np.array(labels)

    train_indices = np.array(sets) == 'train'
    test_indices = np.array(sets) == 'test'

    x_train = X[train_indices]
    y_train = y[train_indices]
    x_test = X[test_indices]
    y_test = y[test_indices]

    # Shuffle the training data
    indices = np.random.permutation(len(x_train))
    x_train = x_train[indices]
    y_train = y_train[indices]

    if client_id != -1:
        print(f"Client ID: {client_id}")
        (x_train, y_train) = partition(x_train, y_train, 3)[client_id]

    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))

    return (x_train, y_train), (x_test, y_test)

def load_kinases_niid(file_index):
    """
    Loads a specific kinase dataset from a CSV file.
    Args:
        file_index (int): Index of the CSV file to load (0, 1, or 2).
    Returns:
        Tuple: Tuple containing training and testing data (x_train, y_train), (x_test, y_test).
    """
    file_paths = [
        'kinases_FLT3_dataset/ChemDB_FLT3_processed.csv',
        'kinases_FLT3_dataset/PKIS_FLT3_processed.csv',
        'kinases_FLT3_dataset/Tang_FLT3_processed.csv'
    ]

    if file_index not in range(len(file_paths)):
        raise ValueError(f"Invalid file_index. Expected a value between 0 and {len(file_paths)-1}.")

    inputs = []
    labels = []
    sets = []

    # Read the CSV file
    with open(file_paths[file_index], 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row

        # Process each row
        for row in csv_reader:
            input_data = list(map(float, row[:8192]))
            inputs.append(input_data)
            label = int(float(row[8192]))
            labels.append(label)
            set_type = row[8193]
            sets.append(set_type)

    X = np.array(inputs)
    y = np.array(labels)

    train_indices = np.array(sets) == 'train'
    test_indices = np.array(sets) == 'test'

    x_train = X[train_indices]
    y_train = y[train_indices]
    x_test = X[test_indices]
    y_test = y[test_indices]

    # Shuffle the training data
    indices = np.random.permutation(len(x_train))
    x_train = x_train[indices]
    y_train = y_train[indices]

    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))

    (x_train2, y_train2), (x_test2, y_test2) = load_kinases()

    return (x_train, y_train), (x_test2, y_test2)

loader = {
    "logregmnist": load_mnist_linear(),
    "svmmnist": load_mnist_linear(),
    "mlpmnist": load_mnist_nonlinear(),
    "cnnmnist": load_mnist_nonlinear(),
    "rnnmnist": load_mnist_nonlinear()
}

def load_data(model: str, dataset: str):
    """
    Loads the specified model and dataset.
    """
    return loader.get(model + dataset, load_mnist_linear())


def create_sample_with_custom_distribution(x, y, dist, total_samples):
    """
    Creates a sample with a custom distribution.
    """
    num_samples = {class_label: int(dist[class_label] * total_samples) for class_label in dist}

    sampled_x = []
    sampled_y = []

    # Iterate over each class and sample the desired number of examples
    for class_label, num_samples_class in num_samples.items():
        # Get indices of data points for the current class
        class_indices = np.where(y == int(class_label))[0]

        # Reduce the sample size if there are not enough samples available
        if len(class_indices) < num_samples_class:
            num_samples_class = len(class_indices)

        # Sample from the class indices without replacement
        sampled_indices = np.random.choice(class_indices, size=num_samples_class, replace=False)

        # Append the sampled data and labels
        sampled_x.extend(x[sampled_indices])
        sampled_y.extend(y[sampled_indices])

    # Convert the sampled data and labels back to numpy arrays
    sampled_x = np.array(sampled_x)
    sampled_y = np.array(sampled_y)

    # Return the sampled data and labels
    return sampled_x, sampled_y

def load_mnist_skewed(num, linear):
    """
    Loads the MNIST dataset with a skewed data distribution.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    dist = {'0': 0.02, '1': 0.02, '2': 0.02, '3': 0.02, '4': 0.02, '5': 0.02, '6': 0.02, '7': 0.02, '8': 0.02, '9': 0.02}
    dist[str(num)] = 0.8
    sampled_x_train, sampled_y_train = create_sample_with_custom_distribution(x_train, y_train, dist, len(y_train)/10)

    if linear:
        sampled_x_train = np.reshape(sampled_x_train, (sampled_x_train.shape[0], -1))
        x_test = np.reshape(x_test, (x_test.shape[0], -1))
    else:
        sampled_x_train, x_test = sampled_x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

    return (sampled_x_train, sampled_y_train), (x_test, y_test)

def load_mnist_niid(num, linear):
    """
    Loads the MNIST dataset with a non-IID data distribution.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    dist = {'0': 0.1, '1': 0.1, '2': 0.1, '3': 0.1, '4': 0.1, '5': 0.1, '6': 0.1, '7': 0.1, '8': 0.1, '9': 0.1}
    dist[str(num)] = 0.5
    sampled_x_train, sampled_y_train = create_sample_with_custom_distribution(x_train, y_train, dist, len(y_train)/10)
    if linear:
        sampled_x_train = np.reshape(sampled_x_train, (sampled_x_train.shape[0], -1))
        x_test = np.reshape(x_test, (x_test.shape[0], -1))
    else:
        sampled_x_train, x_test = sampled_x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

    return (sampled_x_train, sampled_y_train), (x_test, y_test)

