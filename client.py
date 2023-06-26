import os
import json
import utils
import clientConfig

import flwr as fl
from models import load_model
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
num_clients = int(os.environ.get("NUM_CLIENTS","1"))
dataset = os.environ.get("DATASET", 'mnist')
model_name = os.environ.get("MODEL", 'logreg')
partition_id = int(os.environ.get("CLIENT_ID"))
skewed = os.environ.get("SKEWED", "false")  == "true"
iid = os.environ.get("IID", "false") == "true"

if dataset == "mnist":
    if not skewed:
        (x_train, y_train), (x_test, y_test) = utils.load_data(model_name, dataset)
        if num_clients > 1:
            if iid:
                print("Loading iid data")
                (x_train, y_train) = utils.partition(x_train, y_train, num_clients)[partition_id]
            else:
                print("Loading non-iid data")
                if not (model_name == "logreg" or model_name == "svm"):
                    (x_train, y_train), (x_test, y_test) = utils.load_mnist_niid(partition_id, False)
                else:
                    (x_train, y_train), (x_test, y_test) = utils.load_mnist_niid(partition_id, True)
    else:
        print("Loading skewed data")
        (x_train, y_train), (x_test, y_test) = utils.load_mnist_skewed(partition_id)
else:
    if num_clients == 1:
        print("Loading central data")
        (x_train, y_train), (x_test, y_test) = utils.load_kinases()
    elif iid:
        print("Loading iid data")
        (x_train, y_train), (x_test, y_test) = utils.load_kinases(partition_id)
    else:
        print("Loading non-iid data")
        (x_train, y_train), (x_test, y_test) = utils.load_kinases_niid(partition_id)
    
    if not (model_name == "logreg" or model_name == "svm"):
        y_train = keras.utils.to_categorical(y_train, num_classes=2)
        y_test = keras.utils.to_categorical(y_test, num_classes=2)

label_counts = np.bincount(y_train)

# Convert counts to dictionary
label_counts_dict = {str(label): int(count) for label, count in enumerate(label_counts)}

# Write label distributions to a JSON file
with open(f'client{partition_id}.json', 'w') as file:
    json.dump(label_counts_dict, file)

model = load_model(model_name, dataset)

if model_name == "logreg" or model_name == "svm":
    client = clientConfig.LinearClient(model=model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
else:
    client = clientConfig.NNCliennt(model=model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, batch_size=128)

# Start Flower client
fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)

print(client.data)

with open(f"Data/{model_name}/{dataset}/{model_name}_{dataset}_1C.json", "r") as file:
    # Read the JSON data from the file and parse it into a dictionary
    central = json.load(file)

if(partition_id == 0):
    # Plotting the accuracy values
    plt.plot(client.data, label=f"Federated {model_name}")
    # plt.plot(client.data, label="Centralized CNN")
    plt.plot(central, label=f"Centralized {model_name}")
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy") 
    plt.title(f"Non-IID Kinases Dataset, {num_clients} Clients")
    plt.legend()
 
    num_rounds = len(client.data)
    ticks = range(0, num_rounds, 25)

    plt.show()


