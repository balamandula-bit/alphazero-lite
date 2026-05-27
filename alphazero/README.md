# Antigravity-Zero — AlphaZero-Style Chess Engine

> A deep reinforcement learning chess engine built from scratch in Python.
> No handcrafted evaluation. No opening books. No endgame tables.
> The engine learns entirely through **self-play**.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org)
[![Platform](https://img.shields.io/badge/Training-Google%20Colab%20T4-F9AB00?style=flat&logo=googlecolab)](https://colab.research.google.com)
[![Protocol](https://img.shields.io/badge/Protocol-UCI-4CAF50?style=flat)](https://www.chessprogramming.org/UCI)

---

## What This Is

This project is a faithful reimplementation of DeepMind's **AlphaZero** algorithm applied to chess —
built entirely from scratch without using any existing chess AI libraries for the intelligence layer.

The engine combines:
- A **ResNet policy-value network** that evaluates board positions and suggests moves
- **Monte Carlo Tree Search (MCTS)** guided by the neural network to plan ahead
- A **self-play training loop** where the engine generates its own training data by playing against itself

It is a **personal skill-building project** for AI/ML, demonstrating hands-on implementation of
deep reinforcement learning, neural architecture design, and training infrastructure.

**Training status:** 100 iterations complete → targeting 500+ iterations until the engine can defeat a 1600 Elo player.

---

## Demo — UCI Play

```
$ python play.py

uci
  id name Antigravity-Zero
  id author Bala
  option name Simulations type spin default 200 min 50 max 2000
  uciok
isready
  readyok
position startpos
go movetime 1000
  info depth 200 time 987 nodes 200 score cp 0 pv e2e4
  bestmove e2e4
```

**Tune strength from your GUI:**
```
setoption name Simulations value 400
```
Higher simulations = stronger play, slower response. Default is 200 (~1 move/sec on CPU).

Plug directly into **Arena**, **CuteChess**, or **Lichess-bot** via UCI protocol.

---

## Architecture — Neural Network

The network takes a board state and outputs **what to play** (policy) and **who is winning** (value).

```
Input: 19 × 8 × 8 tensor
  ├── Planes  0–11 : piece positions (6 piece types × 2 colors)
  ├── Planes 12–15 : castling rights (K/Q for both sides)
  ├── Plane  16    : side to move
  ├── Plane  17    : en passant file
  └── Plane  18    : fifty-move counter (normalized 0→1)

Residual Tower:
  Input Conv  (19 → 128 channels, 3×3, BN, ReLU)
  └── 10 × ResBlock
        ├── Conv2d(128→128, 3×3) → BN → ReLU
        └── Conv2d(128→128, 3×3) → BN → skip connection → ReLU

Policy Head:
  Conv(128→32, 1×1) → BN → ReLU → Flatten → Linear → 4,672 move logits
  (Softmax masked to legal moves only)

Value Head:
  Conv(128→1, 1×1) → BN → ReLU → Flatten → Linear(64) → ReLU → Linear(1) → Tanh
  Output: scalar in [-1, +1]  (−1 = losing, 0 = drawn, +1 = winning)
```

| Attribute | Value |
|---|---|
| Parameters | ~6.8 million |
| Input planes | 19 × 8 × 8 |
| Policy output | 4,672 move probabilities |
| Value output | scalar ∈ [−1, +1] |
| Residual blocks | 10 |
| Channels | 128 |
| Board perspective | Always current player's POV (flipped for Black) |

---

## Algorithm — MCTS

Each move is chosen by running **200 simulations** through the game tree.
The neural network guides which paths are worth exploring.

```
repeat N=200 times:

  SELECT   Walk down the tree, at each node pick the child maximising:
           UCB = Q(s,a)  +  C_puct × P(s,a) × √N(s) / (1 + N(s,a))
                └─ exploit ─┘  └──────────── explore ─────────────┘
           C_puct = 1.5

  EXPAND   At the leaf, call the neural network:
             policy P(s, ·) → prior probabilities for all legal moves
             value  V(s)    → position evaluation

  BACKUP   Propagate V back up the path, alternating sign at each ply

After N simulations:
  π(a) = N(s,a)^(1/τ) / Σ N(s,·)^(1/τ)
  τ = 1.0 for first 30 moves (exploration), 0.0 after (exploitation)
  Pick move ~ π
```

**Dirichlet noise** (α=0.3, ε=0.25) is injected at the root during self-play
to guarantee every legal move is explored regardless of network priors.

---

## Training Pipeline

The engine learns by playing millions of positions against itself, with no human knowledge injected.

```
Iteration loop:
  ┌─────────────────────────────────────────────────────┐
  │  1. SELF-PLAY                                        │
  │     Run current model vs itself for N games          │
  │     Each move: 200 MCTS simulations                  │
  │     Store: (board_tensor, π_mcts, z_outcome)         │
  │                                                      │
  │  2. TRAIN                                            │
  │     Sample mini-batches (size 256) from replay buffer│
  │     Loss = CrossEntropy(π_pred, π_mcts)              │
  │           + MSE(v_pred, z_outcome)                   │
  │     Optimizer: Adam  lr=1e-3  weight_decay=1e-4      │
  │     LR scheduler: ×0.5 every 5 iterations            │
  │     Gradient clipping: max_norm=1.0                  │
  │                                                      │
  │  3. CHECKPOINT                                       │
  │     Save model_latest.pt + milestone every 10 iters  │
  └─────────────────────────────────────────────────────┘
  Repeat →
```

| Hyperparameter | Value |
|---|---|
| Simulations per move | 200 |
| Temperature threshold | 30 moves |
| Batch size | 256 |
| Learning rate | 0.001 (decays ×0.5 every 5 iters) |
| Epochs per iteration | 5 |
| Training platform | Google Colab T4 GPU |
| Checkpoint storage | Google Drive (auto-resumes on disconnect) |

---

## Performance

| Metric | Value |
|---|---|
| Training iterations completed | 100 (targeting 500+) |
| Move speed | ~1 move per second (200 sims, CPU) |
| Target strength | Beat 1600 Elo player |
| UCI compatible | ✅ Arena / CuteChess / Lichess-bot |

> Training is ongoing. The model improves every iteration as it accumulates
> more self-play data. Milestones at every 10 iterations are preserved for comparison.

---

## Project Structure

```
alphazero/
├── model.py          # ResNet architecture, board encoding, move indexing
├── mcts.py           # Monte Carlo Tree Search with UCB + Dirichlet noise
├── selfplay.py       # Self-play game generation & training data storage
├── train.py          # Training loop: self-play → train → checkpoint → repeat
├── play.py           # UCI wrapper — connects engine to any chess GUI
├── run_zero.bat      # One-click launcher for Windows chess GUIs
├── README.md
├── data/             # Auto-created: self-play games saved as .npz arrays
└── checkpoints/      # Auto-created: model snapshot per training iteration
```

> **Note on model files:** `train.py` saves the latest model as `model.pt`.
> `play.py` loads `model_final.pt`. After training, copy or rename:
> ```bash
> copy model.pt model_final.pt
> ```

---

## Quick Start

### Install dependencies
```bash
pip install torch python-chess numpy tqdm
```

### Play against the engine (UCI)
```bash
cd alphazero
python play.py
```

### Run a training iteration locally (small scale)
```bash
python train.py             # Quick test: 2 games, 1 iteration
python train.py --full      # Full run: 20 iterations, 100 games each
```

### Train on Google Colab (recommended)
```python
# 1. Runtime → Change runtime type → T4 GPU
# 2. Upload the alphazero/ folder
# 3. Mount Google Drive for persistent checkpoints
# 4. Run the training loop — auto-resumes after disconnect

from train import train
train(num_iterations=100, games_per_iter=100, sims_per_move=400)
```

### Add to CuteChess / Arena
| Field | Value |
|---|---|
| Command | `python` |
| Arguments | `C:\path\to\alphazero\play.py` |
| Protocol | UCI |

Or use the included `run_zero.bat` directly as the engine executable.

### Time control support
`play.py` handles both fixed-time and clock-based time controls:
- `go movetime N` — think for N milliseconds
- `go wtime W btime B` — auto-scales simulations from **50 to 800** based on remaining clock
  (formula: `sims = clamp(clock_seconds / 30 × 40, 50, 800)`)

### `.gitignore` recommendation
Model weights (`.pt`) and training data (`.npz`) can reach hundreds of MB.
Add these to `.gitignore` and share models separately (e.g. Google Drive / HuggingFace):
```
alphazero/data/
alphazero/checkpoints/
alphazero/model.pt
alphazero/model_final.pt
```

---

## Tech Stack

| Technology | Role |
|---|---|
| **PyTorch 2.x** | ResNet model, training, GPU inference |
| **python-chess** | Board representation, legal move generation, UCI protocol |
| **NumPy** | Board tensor encoding, training data as `.npz` arrays |
| **Google Colab T4** | GPU training platform |
| **Google Drive** | Fault-tolerant checkpoint storage across Colab sessions |

---

## Key Implementation Details

- **Perspective-invariant encoding** — the board is always flipped to the current player's point of view,
  so the network never needs to learn "I am White" vs "I am Black" separately.
- **Illegal move masking** — policy logits for illegal moves are set to `-inf` before softmax,
  guaranteeing the engine only ever considers legal moves.
- **Fault-tolerant training** — every iteration saves a checkpoint to Google Drive.
  If Colab disconnects, training resumes automatically from the last saved state.
- **Temperature annealing** — τ=1 for the first 30 moves (encouraging opening variety in training data),
  τ=0 after (deterministic exploitation), matching the original AlphaZero paper.
- **Dirichlet root noise** — α=0.3 ensures all legal moves have a non-zero prior at the root,
  preventing the network from collapsing into repetitive lines during self-play.

---

## Skills Demonstrated

This project was built to develop and demonstrate practical AI/ML engineering skills:

| Skill | Where |
|---|---|
| Deep learning architecture design | `model.py` — ResNet with dual heads |
| Reinforcement learning from scratch | `selfplay.py` + `train.py` — no supervised labels |
| Search algorithms (MCTS + UCB) | `mcts.py` — tree search with neural guidance |
| PyTorch: custom training loop | `train.py` — loss, optimizer, scheduler, grad clipping |
| GPU training infrastructure | Google Colab T4 with Drive fault-tolerance |
| Systems integration (UCI protocol) | `play.py` — standard chess engine protocol |
| Data pipeline design | `.npz` replay buffer, batched DataLoader |

---

## Roadmap

- [x] ResNet policy-value network (19×8×8 input, 10 blocks, 128 channels)
- [x] MCTS with UCB exploration and Dirichlet root noise
- [x] Self-play game generation with temperature annealing
- [x] Full training pipeline (self-play → train → checkpoint loop)
- [x] UCI protocol integration (Arena / CuteChess / Lichess-bot)
- [x] Fault-tolerant Google Colab training with Drive checkpoints
- [ ] 500+ training iterations (in progress)
- [ ] Defeat a 1600 Elo player (goal)
- [ ] Lichess bot deployment
- [ ] Web interface for browser play

---

## Author

**Bala Mandula**  
AI/ML Engineer · B.Tech 2026  
[balamandula@gmail.com](mailto:balamandula@gmail.com)

---

## License

MIT — free to use, modify, and distribute.