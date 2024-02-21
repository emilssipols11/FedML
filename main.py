import os
import argparse
import subprocess
import threading

def read_output(process):
    # # Read and print the server's output in real-time
    # while True:
    #     output = process.stdout.readline().decode().rstrip()
    #     if output == '' and process.poll() is not None:
    #         break
    #     if output:
    #         print(output)

    # Read and print the server's error in real-time
    while True:
        error = process.stderr.readline().decode().rstrip()
        if error == '' and process.poll() is not None:
            break
        if error:
            print(error)


def main(num_clients, available_clients, eval_clients, fit_clients, model, dataset, skewed, iid):
    my_env = os.environ.copy()
    my_env["MAC"] = f"{available_clients}"
    my_env["MEC"] = f"{eval_clients}"
    my_env["MFC"] = f"{fit_clients}"
    my_env["NUM_CLIENTS"] = f"{num_clients}"
    my_env["MODEL"] = model
    my_env["DATASET"] = dataset
    my_env["SKEWED"] = skewed
    my_env["IID"] = iid

    process = subprocess.Popen(['python', 'server.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env)

    # Create a separate thread to read and print the server's output in real-time
    output_thread = threading.Thread(target=read_output, args=(process,))
    output_thread.start()

    # Start the client scripts
    for i in range(num_clients):
        my_env["CLIENT_ID"] = str(i)
        process_client = subprocess.Popen(['python', 'client.py'], env=my_env)

    # Wait for the clients to finish
    process_client.wait()

    # Wait for the server to finish
    process.wait()

    # Join the output thread to ensure it finishes
    output_thread.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--num_clients', type=int, default=1, help='Number of clients')
    parser.add_argument('-a', '--available_clients', type=int, default=1, help='Number of minimum necessary available clients')
    parser.add_argument('-f', '--fit_clients', type=int, default=1, help='Number of minimum necessary clients for fitting')
    parser.add_argument('-e', '--evaluate_clients', type=int, default=1, help='Number of minimum necessary clients for evaluating')
    parser.add_argument('-m', '--model', type=str, default="logreg", help='ML Model to use for training')
    parser.add_argument('-d', '--dataset', type=str, default="mnist", help='Data to train client models on')
    parser.add_argument('-s', '--skewed', type=str, default="false", help='Flag weather or not to skew training data for MNIST')
    parser.add_argument('-i', '--iid', type=str, default="false", help='Flag weather or not to  use iid or non-iid data')
    args = parser.parse_args()

    main(args.num_clients, args.available_clients, args.fit_clients, args.evaluate_clients, args.model, args.dataset, args.skewed, args.iid)
