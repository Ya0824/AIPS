import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import joblib
import time


# Set seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed(1)


def load_and_flatten_features(feature_path):
    cell_feature = np.load(feature_path)
    flattened_feature = cell_feature.flatten().reshape(-1, 1131)
    scaler_feature = StandardScaler()
    feature_normalized = scaler_feature.fit_transform(flattened_feature)
    feature_tensor = torch.tensor(feature_normalized, dtype=torch.float32)
    return feature_tensor, scaler_feature


def load_and_normalize_data(power_path, vcd_path, feature_tensor):
    # Load and normalize VCD data
    vcd_trace = np.load(vcd_path)[:, 1:]
    scaler_vcd = StandardScaler()
    vcd_trace_normalized = scaler_vcd.fit_transform(vcd_trace)
    vcd_trace_tensor = torch.tensor(vcd_trace_normalized, dtype=torch.float32)

    # If power_path is provided, load and normalize power data
    if power_path is not None:
        power_trace = np.load(power_path)[:, :159]
        scaler_power = StandardScaler()
        power_trace_normalized = scaler_power.fit_transform(power_trace)
        power_trace_tensor = torch.tensor(power_trace_normalized, dtype=torch.float32)
    else:
        power_trace_tensor = None  # No power data for prediction
        scaler_power = None

    return power_trace_tensor, vcd_trace_tensor, feature_tensor, scaler_power, scaler_vcd



def add_noise(data, noise_level):
    noise = torch.randn_like(data) * noise_level
    return data + noise


class DiffusionModel(nn.Module):
    def __init__(self, input_dim_combined, hidden_dim):
        super(DiffusionModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim_combined, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 159)
        )

    def forward(self, combined_data):
        encoded = self.encoder(combined_data)
        decoded = self.decoder(encoded)
        return decoded


def train_model(model, power_data, combined_data, noise_levels, scaler_power, batch_size=64, epochs=3000, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    loss_history = []

    dataset = TensorDataset(combined_data, power_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_combined, batch_power in dataloader:
            t = random.randint(0, len(noise_levels) - 1)
            noisy_combined = add_noise(batch_combined, noise_levels[t])
            decoded = model(noisy_combined)
            loss = criterion(decoded, batch_power)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        average_epoch_loss = epoch_loss / len(dataloader)
        loss_history.append(average_epoch_loss)

        if average_epoch_loss < best_loss:
            best_loss = average_epoch_loss
            torch.save(model.state_dict(), 'best_diffusion_model.pth')

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {average_epoch_loss}')

    # Save the scaler after training
    joblib.dump(scaler_power, 'scaler_power.pkl')

    return model, loss_history



def predict(model, combined_data, noise_levels):
    model.eval()
    with torch.no_grad():
        t = random.randint(0, len(noise_levels) - 1)
        noisy_combined = add_noise(combined_data, noise_levels[t])
        predicted_data = model(noisy_combined)

    # Load the saved scaler
    scaler_power = joblib.load('scaler_power.pkl')

    predicted_power_data = scaler_power.inverse_transform(predicted_data.numpy())
    return predicted_power_data



def plot_loss(loss_history):
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.show()


def main(args):
    feature_tensor, scaler_feature = load_and_flatten_features(args.feature_path)

    if args.mode == 'train':
        start_time = time.time()
        power_trace_tensor, vcd_trace_tensor, feature_tensor, scaler_power, scaler_vcd = \
            load_and_normalize_data(args.power_path, args.train_vcd_path, feature_tensor)

        input_dim_vcd = 159
        input_dim_feature = feature_tensor.shape[1]
        hidden_dim = args.hidden_dim

        combined_data = torch.cat((vcd_trace_tensor, feature_tensor.repeat(vcd_trace_tensor.shape[0], 1)), dim=1)
        noise_levels = np.linspace(0, args.noise_level, num=10)

        model = DiffusionModel(input_dim_vcd + input_dim_feature, hidden_dim)
        trained_model, loss_history = train_model(
            model,
            power_trace_tensor,
            combined_data,
            noise_levels,
            scaler_power,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate
        )

        plot_loss(loss_history)
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Training time: {elapsed_time:.6f} s")

    elif args.mode == 'predict':
        start_time = time.time()
        _, vcd_trace_tensor, feature_tensor, _, _ = \
            load_and_normalize_data(None, args.predict_vcd_path, feature_tensor)

        input_dim_vcd = 159
        input_dim_feature = feature_tensor.shape[1]
        hidden_dim = args.hidden_dim

        combined_test_data = torch.cat((vcd_trace_tensor, feature_tensor.repeat(vcd_trace_tensor.shape[0], 1)), dim=1)
        model = DiffusionModel(input_dim_vcd + input_dim_feature, hidden_dim)
        model.load_state_dict(torch.load('best_diffusion_model.pth', weights_only=True))


        noise_levels = np.linspace(0, args.noise_level, num=10)

        predicted_power_trace = predict(model, combined_test_data, noise_levels)
        np.save(args.output_path, predicted_power_trace)
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Predict time: {elapsed_time:.6f} s")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict'], default='train')
    parser.add_argument('--feature_path', type=str, default='../1-Data Preparation/cell_feature.npy')
    parser.add_argument('--power_path', type=str, default='../1-Data Preparation/power_trace_train.npy')
    parser.add_argument('--train_vcd_path', type=str, default='../1-Data Preparation/pin_switch_mean_160_train.npy')
    parser.add_argument('--predict_vcd_path', type=str, default='../1-Data Preparation/pin_switch_mean_160_train.npy')
    parser.add_argument('--hidden_dim', type=int, default=48)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--noise_level', type=float, default=0.001)
    parser.add_argument('--output_path', type=str, default='power_trace_train_pre.npy')

    args = parser.parse_args()
    main(args)
