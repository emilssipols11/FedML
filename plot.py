import json

model = "RNN"
dataset = "MNIST"

if dataset == "MNIST":
    with open(f"Data/{model}/{dataset}/{model}_{dataset}_1C.json", "r") as file:
        # Read the JSON data from the file and parse it into a dictionary
        central = json.load(file)

    with open(f"Data/{model}/{dataset}/IID_{model}_{dataset}_10C.json", "r") as file:
        # Read the JSON data from the file and parse it into a dictionary
        iid2 = json.load(file)

    with open(f"Data/{model}/{dataset}/NIID_{model}_{dataset}_10C.json", "r") as file:
        # Read the JSON data from the file and parse it into a dictionary
        iid3 = json.load(file)

    with open(f"Data/{model}/{dataset}/NIID_SKEWED_{model}_{dataset}_10C.json", "r") as file:
        # Read the JSON data from the file and parse it into a dictionary
        iid4 = json.load(file)


    #plotting the accuracy values
    import matplotlib.pyplot as plt
    plt.plot(central, label=f"Centralized")
    plt.plot(iid2, label=f"IID")
    plt.plot(iid3, label=f"non-IID")
    plt.plot(iid4, label=f"aggressive non-IID")
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy")
    plt.title(f"{model} on {dataset}")
    plt.legend()
    steps = 10
    xticks = range(0, len(iid3)+steps, steps)
    plt.xticks(xticks)
    plt.grid(True)
    plt.show()
else:
    with open(f"Data/{model}/{dataset}/{model}_{dataset}_1C.json", "r") as file:
        # Read the JSON data from the file and parse it into a dictionary
        central = json.load(file)

    with open(f"Data/{model}/{dataset}/{model}_{dataset}_3C.json", "r") as file:
        # Read the JSON data from the file and parse it into a dictionary
        niid = json.load(file)

    #plotting the accuracy values
    import matplotlib.pyplot as plt
    plt.plot(central, label=f"Centralized")
    plt.plot(niid, label=f"Federated 3 Clients")
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy")
    plt.title(f"{model} on {dataset}")
    plt.legend()
    steps = 100
    xticks = range(0, len(niid)+steps, steps)
    plt.xticks(xticks)
    plt.grid(True)
    plt.show()
