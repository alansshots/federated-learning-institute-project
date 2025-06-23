import random
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# --- Settings ---
NUM_CLIENTS = 5
LOCAL_EPOCHS = 5
CLIENT_TIMEOUT = 3.0
LEARNING_RATE = 0.01
BATCH_SIZE = 16
INPUT_DIM = 28 * 28
NUM_CLASSES = 10
ROUNDS = 10

# Seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# --- Model ---
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# --- Load MNIST ---
def generate_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Flatten images and reduce size for speed
    X_train = train_dataset.data[:5000].reshape(-1, 28*28).float().numpy() / 255.0
    y_train = train_dataset.targets[:5000].numpy()

    X_test = test_dataset.data.reshape(-1, 28*28).float().numpy() / 255.0
    y_test = test_dataset.targets.numpy()

    return train_test_split(X_train, y_train, test_size=0.2, random_state=1), (X_test, y_test)

def split_data(X, y, num_clients):
    data_per_client = len(X) // num_clients
    datasets = []
    for i in range(num_clients):
        start = i * data_per_client
        end = (i + 1) * data_per_client
        x_tensor = torch.tensor(X[start:end], dtype=torch.float32)
        y_tensor = torch.tensor(y[start:end], dtype=torch.long)
        datasets.append(TensorDataset(x_tensor, y_tensor))
    return datasets

# --- Client ---
class Client:
    def __init__(self, client_id, dataset):
        self.client_id = client_id
        self.dataset = dataset
        self.speed_factor = 1.0  # will change each round

    def train(self, global_model):
        model = LogisticRegression(INPUT_DIM, NUM_CLASSES)
        model.load_state_dict(global_model.state_dict())
        model.train()

        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True)

        start_time = time.time()
        epochs_done = 0

        for epoch in range(LOCAL_EPOCHS):
            for batch_x, batch_y in loader:
                time.sleep(0.01 * self.speed_factor)

                if time.time() - start_time > CLIENT_TIMEOUT:
                    print(f"Client {self.client_id} timed out in epoch {epoch}, returning partial update.")
                    return model.state_dict(), epochs_done

                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
            epochs_done += 1

        print(f"Client {self.client_id} completed {epochs_done} epochs.")
        return model.state_dict(), epochs_done

# --- FedAvg Weighted ---
def average_models_weighted(client_states, client_epochs):
    total_epochs = sum(client_epochs)
    avg_model = {}

    for key in client_states[0].keys():
        weighted_sum = sum([
            (epochs / total_epochs) * state[key]
            for state, epochs in zip(client_states, client_epochs)
        ])
        avg_model[key] = weighted_sum

    return avg_model

# --- Evaluation ---
def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_tensor = torch.tensor(y_test, dtype=torch.long)
        logits = model(X_tensor)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, y_tensor).item()
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(y_tensor, preds)
        f1 = f1_score(y_tensor, preds, average='macro')
    return acc, f1, loss

# --- Federated Learning ---
def federated_learning():
    (X_train, X_val, y_train, y_val), (X_test, y_test) = generate_mnist_data()
    datasets = split_data(X_train, y_train, NUM_CLIENTS)
    clients = [Client(i, datasets[i]) for i in range(NUM_CLIENTS)]
    global_model = LogisticRegression(INPUT_DIM, NUM_CLASSES)

    acc_list, f1_list, loss_list = [], [], []

    for r in range(ROUNDS):
        print(f"\n--- Federated Learning Round {r + 1} ---")
        client_states = []
        client_epochs = []

        for client in clients:
            client.speed_factor = random.choice([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        print("Client speed factors:", [c.speed_factor for c in clients])

        with ThreadPoolExecutor(max_workers=NUM_CLIENTS) as executor:
            futures = [executor.submit(client.train, global_model) for client in clients]
            for future in as_completed(futures):
                try:
                    result, epochs_done = future.result()
                    if result is not None and epochs_done > 0:
                        client_states.append(result)
                        client_epochs.append(epochs_done)
                except Exception as e:
                    print("Client failed:", e)

        if client_states:
            avg_state = average_models_weighted(client_states, client_epochs)
            global_model.load_state_dict(avg_state)
        else:
            print("No client updates this round.")

        acc, f1, loss = evaluate(global_model, X_test, y_test)
        acc_list.append(acc)
        f1_list.append(f1)
        loss_list.append(loss)
        print(f"Round {r+1} - Accuracy: {acc:.4f}, F1: {f1:.4f}, Loss: {loss:.4f}")

    # Plot results
    plt.plot(acc_list, label="Accuracy")
    plt.plot(f1_list, label="F1 Score")
    plt.plot(loss_list, label="Loss")
    plt.title("Federated Learning on MNIST")
    plt.xlabel("Rounds")
    plt.ylabel("Metric")
    plt.legend()
    plt.grid(True)
    plt.show()

# Run it
federated_learning()
