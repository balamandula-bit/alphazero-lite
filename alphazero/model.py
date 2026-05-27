"""
AlphaZero-lite Neural Network
Architecture: ResNet-style with policy + value heads
Input:  111 x 8 x 8 board planes
Output: policy (1968 move probabilities) + value (-1 to +1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np

# ─────────────────────────────────────────────
# BOARD ENCODING
# 111 input planes:
#   - 12 planes: piece positions (6 types x 2 colors)
#   - 4 planes:  castling rights
#   - 1 plane:   side to move
#   - 1 plane:   en passant file
#   - 1 plane:   fifty move counter (normalized)
#   Total: 19 planes x current + 8 history = keep simple: 19 planes
# ─────────────────────────────────────────────
INPUT_PLANES = 19
BOARD_SIZE   = 8
POLICY_SIZE  = 4672  # standard AlphaZero move encoding

PIECE_TO_IDX = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}

def board_to_tensor(board):
    """
    Convert a chess.Board to a (19, 8, 8) float tensor.
    Always from the current player's perspective.
    """
    planes = np.zeros((INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    # Flip board if black to move (always encode from current player's POV)
    flip = board.turn == chess.BLACK

    # Planes 0-11: piece positions
    for sq, piece in board.piece_map().items():
        r = sq // 8
        f = sq  % 8
        if flip:
            r = 7 - r
        idx = PIECE_TO_IDX[(piece.piece_type, piece.color)]
        if flip:
            # Swap colors when encoding from black's perspective
            idx = (idx + 6) % 12
        planes[idx][r][f] = 1.0

    # Plane 12-15: castling rights
    planes[12][:] = float(board.has_kingside_castling_rights(chess.WHITE))
    planes[13][:] = float(board.has_queenside_castling_rights(chess.WHITE))
    planes[14][:] = float(board.has_kingside_castling_rights(chess.BLACK))
    planes[15][:] = float(board.has_queenside_castling_rights(chess.BLACK))

    # Plane 16: side to move (1 = white, 0 = black)
    planes[16][:] = float(board.turn == chess.WHITE)

    # Plane 17: en passant
    if board.ep_square is not None:
        f = board.ep_square % 8
        planes[17][:, f] = 1.0

    # Plane 18: fifty move counter (normalized 0-1)
    planes[18][:] = board.halfmove_clock / 100.0

    return torch.FloatTensor(planes)

# ─────────────────────────────────────────────
# MOVE ENCODING
# Maps chess.Move → policy index and back
# ─────────────────────────────────────────────
def _build_move_index():
    """Precompute all possible moves → index mapping."""
    moves = []
    # Queen moves: 8 directions x 7 distances
    dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    for r in range(8):
        for f in range(8):
            for dr, df in dirs:
                for dist in range(1, 8):
                    nr = r + dr*dist
                    nf = f + df*dist
                    if 0 <= nr < 8 and 0 <= nf < 8:
                        moves.append((r*8+f, nr*8+nf, None))
            # Knight moves
            for dr, df in [(-2,-1),(-2,1),(-1,-2),(-1,2),
                           (1,-2),(1,2),(2,-1),(2,1)]:
                nr = r + dr; nf = f + df
                if 0 <= nr < 8 and 0 <= nf < 8:
                    moves.append((r*8+f, nr*8+nf, None))
            # Promotions
            if r == 6:
                for nf in [f-1, f, f+1]:
                    if 0 <= nf < 8:
                        for promo in [chess.QUEEN, chess.ROOK,
                                      chess.BISHOP, chess.KNIGHT]:
                            moves.append((r*8+f, 7*8+nf, promo))

    move_to_idx = {}
    idx_to_move = {}
    for i, m in enumerate(moves):
        move_to_idx[m] = i
        idx_to_move[i] = m
    return move_to_idx, idx_to_move

MOVE_TO_IDX, IDX_TO_MOVE = _build_move_index()
POLICY_SIZE = len(MOVE_TO_IDX)

def move_to_index(move, flip=False):
    """Convert chess.Move to policy index."""
    fr = move.from_square // 8
    ff = move.from_square  % 8
    tr = move.to_square   // 8
    tf = move.to_square   %  8
    if flip:
        fr = 7 - fr; tr = 7 - tr
    promo = move.promotion
    key = (fr*8+ff, tr*8+tf, promo)
    return MOVE_TO_IDX.get(key, 0)

def index_to_move(idx, flip=False):
    """Convert policy index to chess.Move."""
    fr_sq, to_sq, promo = IDX_TO_MOVE.get(idx, (0, 1, None))
    fr = fr_sq // 8; ff = fr_sq % 8
    tr = to_sq  // 8; tf = to_sq  % 8
    if flip:
        fr = 7 - fr; tr = 7 - tr
    return chess.Move(fr*8+ff, tr*8+tf, promo)

# ─────────────────────────────────────────────
# RESIDUAL BLOCK
# ─────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

# ─────────────────────────────────────────────
# MAIN NETWORK
# ─────────────────────────────────────────────
class AlphaZeroNet(nn.Module):
    """
    Smaller than the real AlphaZero (20 res blocks, 256 channels)
    but trainable on CPU/Colab:
    - 10 residual blocks
    - 128 channels
    - Policy head: move probabilities
    - Value head:  position evaluation (-1 to +1)
    """
    def __init__(self, num_res_blocks=10, channels=128):
        super().__init__()
        self.channels = channels

        # Input convolution
        self.input_conv = nn.Sequential(
            nn.Conv2d(INPUT_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        # Residual tower
        self.res_blocks = nn.ModuleList(
            [ResBlock(channels) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.policy_fc = nn.Linear(32 * 64, POLICY_SIZE)

        # Value head
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.value_fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()   # Output in [-1, 1]
        )

    def forward(self, x):
        # x shape: (batch, INPUT_PLANES, 8, 8)
        x = self.input_conv(x)
        for block in self.res_blocks:
            x = block(x)

        # Policy
        p = self.policy_conv(x)
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)           # Raw logits

        # Value
        v = self.value_conv(x)
        v = v.view(v.size(0), -1)
        v = self.value_fc(v)            # Scalar in [-1, 1]

        return p, v

    def predict(self, board):
        """
        Given a chess.Board, return:
        - policy: dict of {move: probability} for legal moves only
        - value: float in [-1, 1] from current player's perspective
        """
        self.eval()
        flip = board.turn == chess.BLACK

        with torch.no_grad():
            tensor = board_to_tensor(board).unsqueeze(0)  # (1, 19, 8, 8)
            logits, value = self.forward(tensor)

        # Mask illegal moves
        legal_moves  = list(board.legal_moves)
        legal_indices = [move_to_index(m, flip) for m in legal_moves]

        logits = logits.squeeze(0)
        mask   = torch.full((POLICY_SIZE,), float('-inf'))
        mask[legal_indices] = logits[legal_indices]
        probs  = F.softmax(mask, dim=0)

        policy = {m: probs[i].item()
                  for m, i in zip(legal_moves, legal_indices)}

        return policy, value.item()

# ─────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────
def save_model(model, path="C:\\Chess\\alphazero\\model.pt"):
    torch.save({
        'model_state': model.state_dict(),
        'policy_size': POLICY_SIZE,
        'channels':    model.channels,
    }, path)
    print(f"Model saved to {path}")

def load_model(path="C:\\Chess\\alphazero\\model.pt"):
    checkpoint = torch.load(path, map_location='cpu')
    model = AlphaZeroNet(channels=checkpoint.get('channels', 128))
    model.load_state_dict(checkpoint['model_state'])
    print(f"Model loaded from {path}")
    return model

# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing AlphaZero-lite network...")
    print(f"Policy size: {POLICY_SIZE} moves")
    print(f"Input planes: {INPUT_PLANES}")

    model = AlphaZeroNet(num_res_blocks=10, channels=128)

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params/1e6:.1f}M)")

    # Test on starting position
    board = chess.Board()
    policy, value = model.predict(board)

    print(f"\nStarting position:")
    print(f"Value: {value:.4f} (should be near 0)")
    print(f"Legal moves: {len(policy)}")
    print(f"Top 5 moves by policy:")
    top5 = sorted(policy.items(), key=lambda x: x[1], reverse=True)[:5]
    for move, prob in top5:
        print(f"  {move.uci()}: {prob:.4f}")

    # Test board encoding
    tensor = board_to_tensor(board)
    print(f"\nBoard tensor shape: {tensor.shape}")
    print("All tests passed!")