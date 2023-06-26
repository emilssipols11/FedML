import utils
import sklearn
from sklearn import svm
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LogisticRegression, SGDClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, GRU

def LogReg(n_classes, n_features):
    # Create LogisticRegression Model
    model = LogisticRegression(
        penalty="l2",
        max_iter=1, # local epoch
        warm_start=True, # prevent refreshing weights when fitting
    )
    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params_linear(model, n_classes, n_features)
    return model

def SVM(n_classes, n_features, loss="hinge", learning_rate=0.00001):
    # Create SVM Model
    model = SGDClassifier(
        loss=loss,
        penalty="l2",
        max_iter=1, # local epoch
        learning_rate='constant',
        eta0=learning_rate,
        warm_start=True, # prevent refreshing weights when fitting
    )
    
    # Setting initial parameters, akin to model.compile for keras models
    if n_classes == 2:
        utils.set_initial_params_linear(model, 1, n_features)
    else:
        utils.set_initial_params_linear(model, n_classes, n_features)
    return model

def MLP(input_shape, num_classes, learning_rate=0.0001, loss="sparse_categorical_crossentropy"):
    # Load and compile Keras model
    if num_classes > 2:
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=input_shape),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
    else:
        model = Sequential([
            Dense(64, activation='relu',input_shape=input_shape),
            Dropout(0.7),
            Dense(32, activation='relu'),
            Dropout(0.7),
            Dense(num_classes, activation='softmax')
        ])
        # learning_rate = 0.0001
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model

def CNN(input_shape, num_classes, learning_rate=0.0001, loss="sparse_categorical_crossentropy"):
    # Load and compile Keras model
    if num_classes > 2:
        model = keras.Sequential([
            keras.layers.Reshape(input_shape + (1,), input_shape=input_shape),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
    else:
        model = Sequential([
            Conv1D(32, kernel_size=3, activation='relu', input_shape=(8192, 1)),
            MaxPooling1D(pool_size=2),
            Conv1D(64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.7),
            Dense(32, activation='relu'),
            Dropout(0.7),
            Dense(num_classes, activation='sigmoid')
        ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


def RNN(input_shape, num_classes, learning_rate=0.00001, loss="sparse_categorical_crossentropy"):
    # reshaped_input_shape = 128
    if num_classes == 2:
        # Load and compile Keras model
        model = keras.Sequential([
        keras.layers.Reshape((input_shape[0], 1), input_shape=input_shape),
        keras.layers.SimpleRNN(128),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    else:
        model = keras.Sequential([
            keras.layers.SimpleRNN(128, input_shape=input_shape),
            keras.layers.Dense(num_classes, activation='sigmoid')
        ])
        # learning_rate = 0.0001
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model

loader = {
    "logregmnist": LogReg(10, 784),
    "svmmnist": SVM(10, 784),
    "mlpmnist": MLP((28,28), 10),
    "cnnmnist": CNN((28,28), 10),
    "rnnmnist": RNN((28,28), 10),
    "logregkinases": SVM(2, 8192, "log_loss", 0.0001),
    "svmkinases": SVM(2, 8192),
    "mlpkinases": MLP((8192,), 2, loss='binary_crossentropy'),
    "cnnkinases": CNN((8192,), 2, loss='binary_crossentropy'),
    "rnnkinases": RNN((8192,), 2, loss='binary_crossentropy')

}

def load_model(model: str, dataset: str):
    print(model+dataset)
    return loader.get(model+dataset, None)
