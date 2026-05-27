"""
Monte Carlo Tree Search for AlphaZero-lite
Uses the neural network for position evaluation and move priors.
"""

import chess
import math
import numpy as np
from model import AlphaZeroNet, board_to_tensor, move_to_index

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
C_PUCT = 1.5   # Exploration constant (higher = more exploration)
DIRICHLET_ALPHA = 0.3   # Noise for root node exploration
DIRICHLET_EPSILON = 0.25  # How much noise to mix in at root

# ─────────────────────────────────────────────
# MCTS NODE
# ─────────────────────────────────────────────


class Node:
    def __init__(self, prior=0.0):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior    # P(s,a) from neural network
        self.children = {}       # move → Node
        self.is_expanded = False

    @property
    def value(self):
        """Average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits):
        """
        UCB score = Q(s,a) + C_PUCT * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        Balances exploitation (Q) and exploration (P * sqrt(N))
        """
        q = self.value
        u = C_PUCT * self.prior * \
            math.sqrt(parent_visits) / (1 + self.visit_count)
        return q + u

# ─────────────────────────────────────────────
# MCTS
# ─────────────────────────────────────────────


class MCTS:
    def __init__(self, model, num_simulations=200):
        self.model = model
        self.num_simulations = num_simulations

    def run(self, board, temperature=1.0):
        """
        Run MCTS from the given board position.
        Returns:
            move: best move
            pi:   visit count distribution (for training)
        """
        root = Node(prior=1.0)
        self._expand(root, board)
        self._add_dirichlet_noise(root)

        for _ in range(self.num_simulations):
            node = root
            brd = board.copy()
            path = [node]

            # ── SELECT ──────────────────────────
            # Traverse tree using UCB until we reach an unexpanded node
            while node.is_expanded and not brd.is_game_over():
                move, node = self._select(node)
                brd.push(move)
                path.append(node)

            # ── EVALUATE ────────────────────────
            if brd.is_game_over():
                result = brd.result()
                if result == "1-0":
                    value = 1.0 if brd.turn == chess.BLACK else -1.0
                elif result == "0-1":
                    value = 1.0 if brd.turn == chess.WHITE else -1.0
                else:
                    value = 0.0
            else:
                # Expand and evaluate with neural network
                value = self._expand(node, brd)

            # ── BACKUP ──────────────────────────
            # Update visit counts and values along the path
            for i, n in enumerate(reversed(path)):
                sign = 1 if i % 2 == 0 else -1
                n.visit_count += 1
                n.value_sum += sign * value

        # ── SELECT MOVE ─────────────────────────
        return self._pick_move(root, temperature)

    def _select(self, node):
        """Select child with highest UCB score."""
        best_score = -float('inf')
        best_move = None
        best_child = None

        for move, child in node.children.items():
            score = child.ucb_score(node.visit_count)
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    def _expand(self, node, board):
        """
        Expand a node using the neural network.
        Returns the value estimate from the network.
        """
        if board.is_game_over():
            return 0.0

        policy, value = self.model.predict(board)

        # Create children for all legal moves
        for move, prob in policy.items():
            node.children[move] = Node(prior=prob)

        node.is_expanded = True
        return value

    def _add_dirichlet_noise(self, root):
        """
        Add Dirichlet noise to root node priors.
        This ensures the engine explores all moves at the root,
        not just what the network thinks is best.
        """
        moves = list(root.children.keys())
        if not moves:
            return
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(moves))
        for move, n in zip(moves, noise):
            child = root.children[move]
            child.prior = (1 - DIRICHLET_EPSILON) * \
                child.prior + DIRICHLET_EPSILON * n

    def _pick_move(self, root, temperature):
        """
        Pick a move based on visit counts.
        temperature=1.0  → proportional to visits (exploration, used in training)
        temperature=0.0  → always pick most visited (exploitation, used in play)
        """
        moves = list(root.children.keys())
        visits = np.array(
            [root.children[m].visit_count for m in moves], dtype=np.float32)

        if temperature == 0 or visits.sum() == 0:
            best_idx = np.argmax(visits)
            pi = np.zeros(len(moves))
            pi[best_idx] = 1.0
        else:
            visits_temp = visits ** (1.0 / temperature)
            pi = visits_temp / visits_temp.sum()

        chosen_idx = np.random.choice(len(moves), p=pi)
        chosen_move = moves[chosen_idx]

        # Return visit distribution for all moves (used in training)
        visit_dist = {m: visits[i] / max(visits.sum(), 1)
                      for i, m in enumerate(moves)}

        return chosen_move, visit_dist


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import time
    print("Testing MCTS...")

    model = AlphaZeroNet(num_res_blocks=10, channels=128)
    mcts = MCTS(model, num_simulations=50)

    board = chess.Board()
    start = time.time()
    move, visit_dist = mcts.run(board, temperature=1.0)
    elapsed = time.time() - start

    print(f"Best move: {move.uci()}")
    print(f"Time: {elapsed:.2f}s for 50 simulations")
    print(f"Simulations/sec: {50/elapsed:.1f}")
    print(f"\nTop 5 moves by visit count:")
    top5 = sorted(visit_dist.items(), key=lambda x: x[1], reverse=True)[:5]
    for m, v in top5:
        print(f"  {m.uci()}: {v:.3f}")
    print("\nMCTS test passed!")
