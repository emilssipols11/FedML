import os
import flwr as fl

strategy = fl.server.strategy.FedAvg(
    min_available_clients=int(os.environ.get("MAC", 1)),
    min_evaluate_clients=int(os.environ.get("MEC", 1)),
    min_fit_clients=int(os.environ.get("MFC", 1)),
)

fl.server.start_server(
        server_address = "0.0.0.0:8080", 
        config=fl.server.ServerConfig(num_rounds=10),
        strategy = strategy,
)
