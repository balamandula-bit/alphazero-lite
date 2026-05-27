"""
Training Pipeline for AlphaZero-lite
Run locally for testing, run on Google Colab for real training.

Loss = policy_loss + value_loss
policy_loss = cross-entropy(predicted_policy, mcts_visit_distribution)
value_loss  = MSE(predicted_value, actual_game_outcome)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time
from model import AlphaZeroNet, save_model, load_model, POLICY_SIZE
from selfplay import generate_games, load_games, DATA_DIR

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
EPOCHS_PER_ITER = 5
MODEL_PATH = r"C:\Chess\alphazero\model.pt"
CHECKPOINT_DIR = r"C:\Chess\alphazero\checkpoints"

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────


class ChessDataset(Dataset):
    def __init__(self, tensors, policies, values):
        self.tensors = torch.FloatTensor(tensors)
        self.policies = torch.FloatTensor(policies)
        self.values = torch.FloatTensor(values).unsqueeze(1)

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx], self.policies[idx], self.values[idx]

# ─────────────────────────────────────────────
# TRAINING STEP
# ─────────────────────────────────────────────


def train_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = total_policy = total_value = 0
    batches = 0

    for tensors, policies, values in loader:
        tensors = tensors.to(device)
        policies = policies.to(device)
        values = values.to(device)

        # Forward pass
        pred_policy, pred_value = model(tensors)

        # Policy loss: cross entropy
        # pred_policy = raw logits, policies = target distribution
        log_probs = torch.log_softmax(pred_policy, dim=1)
        policy_loss = -(policies * log_probs).sum(dim=1).mean()

        # Value loss: MSE
        value_loss = nn.MSELoss()(pred_value, values)

        # Combined loss
        loss = policy_loss + value_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_policy += policy_loss.item()
        total_value += value_loss.item()
        batches += 1

    avg_loss = total_loss / max(batches, 1)
    avg_policy = total_policy / max(batches, 1)
    avg_value = total_value / max(batches, 1)

    print(f"  Epoch {epoch}: loss={avg_loss:.4f} "
          f"policy={avg_policy:.4f} value={avg_value:.4f}")
    return avg_loss

# ─────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────


def train(num_iterations=10, games_per_iter=50, sims_per_move=200):
    """
    AlphaZero training loop:
    1. Generate self-play games with current model
    2. Train model on generated data
    3. Repeat
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load or create model
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print("Loaded existing model")
    else:
        model = AlphaZeroNet(num_res_blocks=10, channels=128)
        print("Created new model")

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(),
                           lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_loss = float('inf')

    for iteration in range(1, num_iterations + 1):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}/{num_iterations}")
        print(f"{'='*60}")

        # ── SELF-PLAY ───────────────────────
        print(f"\nGenerating {games_per_iter} self-play games "
              f"({sims_per_move} sims/move)...")
        model_cpu = model.cpu()  # Self-play on CPU
        data_path = generate_games(model_cpu, num_games=games_per_iter)
        model = model.to(device)

        # ── LOAD DATA ───────────────────────
        tensors, policies, values = load_games()
        if tensors is None:
            print("No training data found, skipping training")
            continue

        print(f"\nTraining on {len(tensors)} positions...")
        dataset = ChessDataset(tensors, policies, values)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=0)

        # ── TRAIN ───────────────────────────
        for epoch in range(1, EPOCHS_PER_ITER + 1):
            loss = train_epoch(model, loader, optimizer, device, epoch)

        scheduler.step()

        # ── SAVE CHECKPOINT ─────────────────
        model_cpu = model.cpu()
        save_model(model_cpu, MODEL_PATH)

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_iter_{iteration}.pt")
        save_model(model_cpu, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

        model = model.to(device)

        if loss < best_loss:
            best_loss = loss
            print(f"New best loss: {best_loss:.4f}")

    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    print(f"Model saved to {MODEL_PATH}")

# ─────────────────────────────────────────────
# QUICK LOCAL TEST (small scale)
# ─────────────────────────────────────────────


def quick_test():
    """Test training pipeline with tiny data — runs in 1-2 minutes."""
    print("Quick training test (2 games, 1 iteration, 25 sims)...")
    print("For real training use Google Colab!")
    print("=" * 50)

    device = torch.device("cpu")
    model = AlphaZeroNet(num_res_blocks=4, channels=64)  # Smaller for test
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Generate tiny dataset
    from selfplay import play_game
    from mcts import MCTS

    mcts = MCTS(model, num_simulations=25)

    all_t, all_p, all_v = [], [], []
    for i in range(2):
        print(f"Generating game {i+1}/2...")
        samples, result = play_game(model, verbose=False)
        for t, p, v in samples[:50]:   # max 50 positions per game
            all_t.append(t.numpy())
            all_p.append(p)
            all_v.append(v)

    print(f"\nTraining on {len(all_t)} positions...")
    dataset = ChessDataset(
        np.array(all_t), np.array(all_p), np.array(all_v)
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(1, 4):
        loss = train_epoch(model, loader, optimizer, device, epoch)

    print("\nQuick test passed!")
    print("\nTo train properly:")
    print("1. Upload this folder to Google Colab")
    print("2. Run: train(num_iterations=20, games_per_iter=100, sims_per_move=400)")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if "--full" in sys.argv:
        train(num_iterations=20, games_per_iter=100, sims_per_move=400)
    else:
        quick_test()
