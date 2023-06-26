# FedML
Implementation of various ML Models in the context of Federated Learning

## Setup
1. virtualenv venv
2. source venv/bin/activate
3. pip install -r requirements.txt


## Running locally
Execute ```python main.py``` with the following arguments

- `-c`, `--num_clients`: Number of clients (default: 1)
- `-m`, `--model`: ML Model to use for training (default: "logreg")
- `-d`, `--dataset`: Data to train client models on (default: "mnist")
- `-s`, `--skewed`: Flag indicating whether or not to skew training data for MNIST (default: "false")
- `-i`, `--iid`: Flag indicating whether or not to use IID or non-IID data (default: "false")


## Examples

```
python main.py -c 1 -m svm -d mnist                         Running SVM on MNIST, centralized, 1 client
python main.py -c 10 -m svm -d mnist -i true                Running SVM on MNIST, federated, 10 clients, IID data
python main.py -c 10 -m logreg -d mnist -i false            Running LogReg on MNIST, federated, 10 clients, non-IID data
python main.py -c 10 -m mlp -d mnist -s true                Running MLP on MNIST, federated, 10 clients, aggressive non-IID data
```
