import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def ensure_dir(p):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


set_seed(44)


# ----------------------------
# Load data
# ----------------------------
def load_power(power_path: str):
    power = np.load(power_path)
    power = power[:, :159]
    print(f"[INFO] Loaded power from {power_path}: {power.shape}")
    return power


def load_vcd(vcd_path: str):
    vcd = np.load(vcd_path)
    vcd = vcd[:, 1:]
    print(f"[INFO] Loaded vcd from {vcd_path}: {vcd.shape}")
    return vcd


def load_features(feature_path: str):
    feat = np.load(feature_path)
    print(f"[INFO] Loaded feature: {feat.shape}")
    return feat


def process_features(feat_np, N):
    """
    Flatten features
    """
    feat_flat = feat_np.reshape(-1)
    feat_rep = np.tile(feat_flat, (N, 1))
    return feat_rep


# ----------------------------
# Diffusion-style scheduler
# ----------------------------
class SimpleScheduler:
    def __init__(self, num_steps=10, beta_start=1e-3, beta_end=1e-2, device="cpu"):
        self.num_steps = num_steps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def q_sample(self, z0, t, noise):
        sb = self.sqrt_alpha_bars[t].reshape(-1, 1)
        so = self.sqrt_one_minus_alpha_bars[t].reshape(-1, 1)
        return sb * z0 + so * noise


# ----------------------------
# Model parts
# ----------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, t_idx):
        t = t_idx.float().unsqueeze(-1)
        return self.net(t)


class FusionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, power_dim=159):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, power_dim)
        )

    def forward(self, z):
        return self.net(z)


class RefinementBlock(nn.Module):
    def __init__(self, latent_dim, cond_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z_t, t_emb, cond_emb):
        x = torch.cat([z_t, t_emb, cond_emb], dim=1)
        delta = self.net(x)
        return z_t + delta


class LatentDiffusionAED(nn.Module):
    def __init__(self, vcd_dim, feat_dim, latent_dim=128, hidden_dim=256, power_dim=159, num_steps=10):
        super().__init__()
        self.power_dim = power_dim
        self.latent_dim = latent_dim
        self.num_steps = num_steps

        self.fusion = FusionMLP(vcd_dim + feat_dim, hidden_dim, hidden_dim)
        self.encoder = Encoder(hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, power_dim)

        self.time_emb = TimeEmbedding(hidden_dim)
        self.refine = RefinementBlock(latent_dim, cond_dim=hidden_dim, hidden_dim=hidden_dim)

    def forward(self, vcd, feat, scheduler: SimpleScheduler, force_add_noise=False):

        B = vcd.size(0)
        cond_in = torch.cat([vcd, feat], dim=1)
        cond_emb = self.fusion(cond_in)

        z0 = self.encoder(cond_emb)

        if force_add_noise:

            t0 = torch.randint(0, scheduler.num_steps, (B,), device=vcd.device, dtype=torch.long)
            eps = torch.randn_like(z0)
            z = scheduler.q_sample(z0, t0, eps)
            for step in range(scheduler.num_steps - 1, -1, -1):
                mask = (t0 >= step).float().unsqueeze(-1)
                t_emb = self.time_emb(torch.full((B,), step, device=vcd.device))
                z = z + mask * (self.refine(z, t_emb, cond_emb) - z)
        else:

            z = z0
            for step in range(scheduler.num_steps - 1, -1, -1):
                t_emb = self.time_emb(torch.full((B,), step, device=vcd.device))
                z = self.refine(z, t_emb, cond_emb)

        pred_power = self.decoder(z)
        return pred_power


# ----------------------------
# Train & Predict
# ----------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- 1. Load dataset ----------
    power_train_all = load_power(args.train_power_path)
    vcd_train_all   = load_vcd(args.train_vcd_path)

    power_test_np   = load_power(args.test_power_path)
    vcd_test_np     = load_vcd(args.test_vcd_path)

    feat_np = load_features(args.feature_path)

    N_train_all = power_train_all.shape[0]
    N_test      = power_test_np.shape[0]

    # ---------- 1.1 Split dataset ----------
    indices = np.arange(N_train_all)
    train_idx, val_idx = train_test_split(
        indices, test_size=0.1, random_state=44, shuffle=True
    )

    power_train_np = power_train_all[train_idx]
    power_val_np   = power_train_all[val_idx]

    vcd_train_np   = vcd_train_all[train_idx]
    vcd_val_np     = vcd_train_all[val_idx]


    feat_train_all = process_features(feat_np, N_train_all)
    feat_train_np  = feat_train_all[train_idx]
    feat_val_np    = feat_train_all[val_idx]


    feat_test_np   = process_features(feat_np, N_test)

    # ---------- 2. Scaler dataset ----------
    scaler_power = StandardScaler()
    scaler_vcd   = StandardScaler()
    scaler_feat  = StandardScaler()

    power_train_norm = scaler_power.fit_transform(power_train_np)
    power_val_norm   = scaler_power.transform(power_val_np)
    power_test_norm  = scaler_power.transform(power_test_np)

    vcd_train_norm   = scaler_vcd.fit_transform(vcd_train_np)
    vcd_val_norm     = scaler_vcd.transform(vcd_val_np)
    vcd_test_norm    = scaler_vcd.transform(vcd_test_np)

    feat_train_norm  = scaler_feat.fit_transform(feat_train_np)
    feat_val_norm    = scaler_feat.transform(feat_val_np)
    feat_test_norm   = scaler_feat.transform(feat_test_np)

    # ---------- 3. Trans to tensor ----------
    power_train = torch.tensor(power_train_norm, dtype=torch.float32, device=device)
    power_val   = torch.tensor(power_val_norm,   dtype=torch.float32, device=device)
    power_test  = torch.tensor(power_test_norm,  dtype=torch.float32, device=device)

    vcd_train   = torch.tensor(vcd_train_norm,   dtype=torch.float32, device=device)
    vcd_val     = torch.tensor(vcd_val_norm,     dtype=torch.float32, device=device)
    vcd_test    = torch.tensor(vcd_test_norm,    dtype=torch.float32, device=device)

    feat_train  = torch.tensor(feat_train_norm,  dtype=torch.float32, device=device)
    feat_val    = torch.tensor(feat_val_norm,    dtype=torch.float32, device=device)
    feat_test   = torch.tensor(feat_test_norm,   dtype=torch.float32, device=device)

    # ---------- 4. DataLoader ----------
    train_dataset = TensorDataset(power_train, vcd_train, feat_train)
    val_dataset   = TensorDataset(power_val,   vcd_val,   feat_val)
    test_dataset  = TensorDataset(power_test,  vcd_test,  feat_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    # ---------- 5. Model ----------
    model = LatentDiffusionAED(
        vcd_dim=vcd_train.shape[1],
        feat_dim=feat_train.shape[1],
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        power_dim=power_train.shape[1],
        num_steps=args.num_steps
    ).to(device)

    scheduler = SimpleScheduler(
        num_steps=args.num_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device
    )

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    # ---------- 6. Train the model ----------
    best_test_loss = float('inf')
    train_loss_hist = []
    val_loss_hist   = []
    test_loss_hist  = []
    patience_counter = 0

    for epoch in range(args.epochs):
        # ----- 6.1 Training process -----
        model.train()
        running_train = 0.0

        for power_b, vcd_b, feat_b in train_loader:
            pred_power = model(vcd_b, feat_b, scheduler, force_add_noise=True)
            loss = criterion(pred_power, power_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train += loss.item()

        epoch_train_loss = running_train / len(train_loader)
        train_loss_hist.append(epoch_train_loss)

        # ----- 6.2 Validating process -----
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for power_b, vcd_b, feat_b in val_loader:
                pred_power = model(vcd_b, feat_b, scheduler, force_add_noise=False)
                loss = criterion(pred_power, power_b)
                running_val += loss.item()

        epoch_val_loss = running_val / len(val_loader)
        val_loss_hist.append(epoch_val_loss)

        # ----- 6.3 Testing process -----
        running_test = 0.0
        with torch.no_grad():
            for power_b, vcd_b, feat_b in test_loader:
                pred_power = model(vcd_b, feat_b, scheduler, force_add_noise=False)
                loss = criterion(pred_power, power_b)
                running_test += loss.item()

        epoch_test_loss = running_test / len(test_loader)
        test_loss_hist.append(epoch_test_loss)

        if (epoch % 1 == 0) or (epoch == args.epochs - 1):
            print(f"[Epoch {epoch}/{args.epochs}] "
                  f"train_loss={epoch_train_loss:.6f}  "
                  f"val_loss={epoch_val_loss:.6f}  "
                  f"test_loss={epoch_test_loss:.6f}")

        # ----- 6.4 Save the best model -----
        if epoch_test_loss < best_test_loss:
            best_test_loss = epoch_test_loss
            patience_counter = 0
            ensure_dir(args.checkpoint_path)
            torch.save({
                "model_state": model.state_dict(),
                "meta": {
                    "vcd_dim": vcd_train.shape[1],
                    "feat_dim": feat_train.shape[1],
                    "latent_dim": args.latent_dim,
                    "hidden_dim": args.hidden_dim,
                    "power_dim": power_train.shape[1],
                    "num_steps": args.num_steps,
                    "beta_start": args.beta_start,
                    "beta_end": args.beta_end
                }
            }, args.checkpoint_path)
            joblib.dump({
                "scaler_power": scaler_power,
                "scaler_vcd": scaler_vcd,
                "scaler_feat": scaler_feat
            }, args.scaler_path)
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"[INFO] Early stopping at epoch {epoch}, "
                  f"no improvement on test loss for {args.patience} epochs.")
            break

    print(f"[INFO] Training done. Best test_loss={best_test_loss:.6f}")

    return train_loss_hist, val_loss_hist, test_loss_hist


@torch.no_grad()
def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint_path, map_location=device)
    meta = ckpt["meta"]
    scalers = joblib.load(args.scaler_path)
    scaler_power = scalers["scaler_power"]
    scaler_vcd = scalers["scaler_vcd"]
    scaler_feat = scalers["scaler_feat"]

    model = LatentDiffusionAED(
        vcd_dim=meta["vcd_dim"],
        feat_dim=meta["feat_dim"],
        latent_dim=meta["latent_dim"],
        hidden_dim=meta["hidden_dim"],
        power_dim=meta["power_dim"],
        num_steps=meta["num_steps"]
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scheduler = SimpleScheduler(
        num_steps=meta["num_steps"],
        beta_start=meta["beta_start"],
        beta_end=meta["beta_end"],
        device=device
    )

    vcd_np = load_vcd(args.predict_vcd_path)
    feat_np = load_features(args.feature_path)
    M = vcd_np.shape[0]
    feat_rep = process_features(feat_np, M)

    vcd_norm = scaler_vcd.transform(vcd_np)
    feat_norm = scaler_feat.transform(feat_rep)

    vcd = torch.tensor(vcd_norm, dtype=torch.float32, device=device)
    feat = torch.tensor(feat_norm, dtype=torch.float32, device=device)

    pred_power_norm = model(vcd, feat, scheduler, force_add_noise=False)
    pred_power = scaler_power.inverse_transform(pred_power_norm.cpu().numpy())

    ensure_dir(args.output_path)
    np.save(args.output_path, pred_power)
    print(f"[INFO] Saved prediction: {args.output_path}, shape={pred_power.shape}")
    return pred_power


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict'], default='train')

    # paths
    parser.add_argument('--feature_path',      type=str, default='../1-Data Preparation/cell_feature.npy')

    parser.add_argument('--train_power_path', type=str, default='power_trace_train.npy')
    parser.add_argument('--train_vcd_path',   type=str, default='pin_switch_mean_train.npy')

    parser.add_argument('--test_power_path',  type=str, default='power_trace_test.npy')
    parser.add_argument('--test_vcd_path',    type=str, default='pin_switch_mean_test.npy')

    parser.add_argument('--predict_vcd_path', type=str, default='pin_switch_mean_test.npy')

    parser.add_argument('--checkpoint_path',  type=str, default='ckpts/latent_dae.pth')
    parser.add_argument('--scaler_path',      type=str, default='ckpts/scalers.pkl')
    parser.add_argument('--output_path',      type=str, default='power_trace_test_pre.npy')

    # model
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=256)

    # diffusion-like schedule
    parser.add_argument('--num_steps',  type=int,   default=10)
    parser.add_argument('--beta_start', type=float, default=1e-3)
    parser.add_argument('--beta_end',   type=float, default=1e-2)

    # train
    parser.add_argument('--batch_size',     type=int,   default=64)
    parser.add_argument('--epochs',         type=int,   default=300)
    parser.add_argument('--learning_rate',  type=float, default=1e-3)
    parser.add_argument('--patience',       type=int,   default=10)

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
