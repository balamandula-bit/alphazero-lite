"""
Self-Play Game Generator for AlphaZero-lite
Plays games against itself and stores training data.
"""

import chess
import numpy as np
import torch
import os
import time
import random
from model import AlphaZeroNet, board_to_tensor, move_to_index, POLICY_SIZE
from mcts import MCTS

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
SIMULATIONS_PER_MOVE = 200   # Higher = stronger but slower
TEMPERATURE_MOVES = 30    # Use temperature=1 for first N moves
MAX_GAME_MOVES = 300   # Abort draw if game too long
DATA_DIR = r"C:\Chess\alphazero\data"

# ─────────────────────────────────────────────
# SINGLE SELF-PLAY GAME
# ─────────────────────────────────────────────


def play_game(model, game_id=0, verbose=False):
    """
    Play one complete self-play game.
    Returns list of (board_tensor, policy_target, value_target) tuples.
    """
    mcts = MCTS(model, num_simulations=SIMULATIONS_PER_MOVE)
    board = chess.Board()

    # Store: (board_tensor, visit_distribution, player_to_move)
    history = []
    move_count = 0

    while not board.is_game_over() and move_count < MAX_GAME_MOVES:
        # Temperature: explore early, exploit later
        temp = 1.0 if move_count < TEMPERATURE_MOVES else 0.0

        # Run MCTS
        move, visit_dist = mcts.run(board, temperature=temp)

        # Store board state and policy
        tensor = board_to_tensor(board)
        flip = board.turn == chess.BLACK

        # Build policy target vector
        policy_target = np.zeros(POLICY_SIZE, dtype=np.float32)
        for m, prob in visit_dist.items():
            idx = move_to_index(m, flip)
            policy_target[idx] = prob

        history.append((tensor, policy_target, board.turn))

        board.push(move)
        move_count += 1

        if verbose and move_count % 10 == 0:
            print(f"  Move {move_count}: {move.uci()}")

    # ── DETERMINE RESULT ────────────────────
    result = board.result()
    if result == "1-0":
        winner = chess.WHITE
    elif result == "0-1":
        winner = chess.BLACK
    else:
        winner = None  # Draw

    if verbose:
        print(f"  Result: {result} in {move_count} moves")

    # ── BUILD TRAINING SAMPLES ──────────────
    # Assign value targets based on game outcome
    samples = []
    for tensor, policy_target, player in history:
        if winner is None:
            value_target = 0.0
        elif player == winner:
            value_target = 1.0
        else:
            value_target = -1.0
        samples.append((tensor, policy_target, value_target))

    return samples, result

# ─────────────────────────────────────────────
# GENERATE MULTIPLE GAMES
# ─────────────────────────────────────────────


def generate_games(model, num_games=100, save=True):
    """
    Generate num_games self-play games and save training data.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    all_tensors = []
    all_policies = []
    all_values = []
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}

    start = time.time()

    for i in range(num_games):
        game_start = time.time()
        print(f"Game {i+1}/{num_games}...", end=" ", flush=True)

        samples, result = play_game(model, game_id=i, verbose=False)

        results[result] = results.get(result, 0) + 1
        elapsed = time.time() - game_start

        for tensor, policy, value in samples:
            all_tensors.append(tensor.numpy())
            all_policies.append(policy)
            all_values.append(value)

        print(f"result={result} moves={len(samples)} time={elapsed:.1f}s "
              f"positions_total={len(all_tensors)}")

    total_time = time.time() - start
    print(f"\nGenerated {num_games} games in {total_time:.1f}s")
    print(f"Results: {results}")
    print(f"Total positions: {len(all_tensors)}")

    if save and all_tensors:
        timestamp = int(time.time())
        path = os.path.join(DATA_DIR, f"games_{timestamp}.npz")
        np.savez_compressed(
            path,
            tensors=np.array(all_tensors,  dtype=np.float32),
            policies=np.array(all_policies, dtype=np.float32),
            values=np.array(all_values,   dtype=np.float32),
        )
        print(f"Data saved to {path}")
        return path

    return None

# ─────────────────────────────────────────────
# LOAD SAVED GAMES
# ─────────────────────────────────────────────


def load_games(data_dir=DATA_DIR):
    """Load all saved game data for training."""
    tensors = []
    policies = []
    values = []

    if not os.path.exists(data_dir):
        return None, None, None

    files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    if not files:
        return None, None, None

    for fname in files:
        path = os.path.join(data_dir, fname)
        data = np.load(path)
        tensors.append(data['tensors'])
        policies.append(data['policies'])
        values.append(data['values'])
        print(f"Loaded {len(data['tensors'])} positions from {fname}")

    return (np.concatenate(tensors),
            np.concatenate(policies),
            np.concatenate(values))


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing self-play with 2 quick games (50 sims/move)...")
    print("=" * 50)

    model = AlphaZeroNet(num_res_blocks=10, channels=128)

    # Use fewer sims for quick test
    mcts = MCTS(model, num_simulations=50)

    for i in range(2):
        print(f"\nGame {i+1}:")
        samples, result = play_game(model, game_id=i, verbose=True)
        print(f"  Training samples: {len(samples)}")
        print(f"  Sample tensor shape: {samples[0][0].shape}")
        print(f"  Sample policy sum: {samples[0][1].sum():.3f}")

    print("\nSelf-play test passed!")
    print("\nFor full training, run generate_games() with num_games=500+")
    print("on Google Colab for GPU acceleration.")
