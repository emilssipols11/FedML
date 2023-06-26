import utils
import warnings

import flwr as fl
import numpy as np

from sklearn.metrics import log_loss
from tensorflow import keras


# Define Linear client
class LinearClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.data = []

    def get_parameters(self, config):
        return utils.get_model_parameters_linear(self.model)

    def fit(self, parameters, config):
        utils.set_model_params_linear(self.model, parameters)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.x_train, self.y_train)
            # print(f"Training finished for round {config['rnd']}")
  
        # Return updated model parameters and results
        parameters_prime = utils.get_model_parameters_linear(self.model)
        num_examples_train = self.x_train.shape[0]
        results = {}

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        utils.set_model_params_linear(self.model, parameters)
        # Obtain the predicted probabilities for X_test
        y_pred_prob =self.model.decision_function(self.x_test)
        # Calculate the loss
        loss = log_loss(self.y_test, y_pred_prob, labels=self.model.classes_)
        accuracy = self.model.score(self.x_test, self.y_test)
        self.data.append(accuracy)
        return loss, self.x_train.shape[0], {"accuracy": accuracy}


# Define NN client
class NNCliennt(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, batch_size):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.data = []

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=self.batch_size, validation_data=(self.x_test, self.y_test), verbose=0)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print("Eval accuracy : ", accuracy)
        self.data.append(accuracy)
        return loss, len(self.x_test), {"accuracy": accuracy}
