"""
AlphaZero-lite UCI Engine
Run: r"python C:\Chess\alphazero\play.py"
Add to CuteChess/Arena as a UCI engine.
"""

from mcts import MCTS
from model import AlphaZeroNet, POLICY_SIZE
import sys
import os
import chess
import torch
import time

# Add alphazero folder to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH = r"C:\Chess\alphazero\model_final.pt"
SIMULATIONS = 200   # Higher = stronger but slower
DEVICE = torch.device("cpu")  # CPU for local play

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────


def load_model():
    model = AlphaZeroNet(num_res_blocks=10, channels=128)
    if os.path.exists(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state)
        sys.stderr.write(f"Model loaded from {MODEL_PATH}\n")
    else:
        sys.stderr.write(
            f"WARNING: No model found at {MODEL_PATH}, using random weights\n")
    model.eval()
    return model

# ─────────────────────────────────────────────
# UCI LOOP
# ─────────────────────────────────────────────


def uci_loop():
    global SIMULATIONS
    model = load_model()
    mcts = MCTS(model, num_simulations=SIMULATIONS)
    board = chess.Board()

    print("id name Antigravity-Zero", flush=True)
    print("id author Bala", flush=True)
    print("option name Simulations type spin default 200 min 50 max 2000", flush=True)
    print("uciok", flush=True)

    while True:
        try:
            line = input().strip()
        except EOFError:
            break
        if not line:
            continue

        if line == "uci":
            print("id name Antigravity-Zero", flush=True)
            print("uciok", flush=True)

        elif line == "isready":
            print("readyok", flush=True)

        elif line == "ucinewgame":
            board = chess.Board()

        elif line.startswith("setoption"):
            parts = line.split()
            if "name" in parts and "Simulations" in parts and "value" in parts:
                vi = parts.index("value") + 1
                SIMULATIONS = int(parts[vi])
                mcts = MCTS(model, num_simulations=SIMULATIONS)

        elif line.startswith("position"):
            parts = line.split()
            if "startpos" in parts:
                board = chess.Board()
                if "moves" in parts:
                    for uci in parts[parts.index("moves")+1:]:
                        board.push_uci(uci)
            elif "fen" in parts:
                fi = parts.index("fen") + 1
                mi = parts.index("moves") if "moves" in parts else len(parts)
                board = chess.Board(" ".join(parts[fi:mi]))
                if "moves" in parts:
                    for uci in parts[mi+1:]:
                        board.push_uci(uci)

        elif line.startswith("go"):
            # Parse time
            parts = line.split()
            think_time = 5.0
            if "movetime" in parts:
                think_time = int(parts[parts.index("movetime")+1]) / 1000.0
            elif board.turn == chess.WHITE and "wtime" in parts:
                wt = int(parts[parts.index("wtime")+1])
                think_time = max(0.5, min(wt/1000.0/30, 10.0))
            elif board.turn == chess.BLACK and "btime" in parts:
                bt = int(parts[parts.index("btime")+1])
                think_time = max(0.5, min(bt/1000.0/30, 10.0))

            # Adjust simulations based on time
            sims = max(50, min(int(think_time * 40), 800))
            mcts.num_simulations = sims

            if board.is_game_over():
                moves = list(board.legal_moves)
                move = moves[0] if moves else None
            else:
                start = time.time()
                move, visit_dist = mcts.run(board, temperature=0.0)
                elapsed = time.time() - start

                # Print info
                if visit_dist:
                    best_visits = max(visit_dist.values())
                    print(f"info depth {sims} time {int(elapsed*1000)} "
                          f"nodes {sims} score cp 0 "
                          f"pv {move.uci() if move else ''}",
                          flush=True)

            if move:
                print(f"bestmove {move.uci()}", flush=True)
            else:
                legal = list(board.legal_moves)
                if legal:
                    print(f"bestmove {legal[0].uci()}", flush=True)

        elif line == "quit":
            break


if __name__ == "__main__":
    uci_loop()
