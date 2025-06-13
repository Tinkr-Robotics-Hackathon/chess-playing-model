import chess
import cv2
from src.libs.detect_board import load_model, detect_corners
from src.libs.warp_board import warp_board
from src.libs.classify_squares import split_board_into_squares
from src.libs.classify_piece import load_piece_model, classify_piece
from src.libs.classify_color import load_color_model, classify_color

# Initialize the chess board(8x8 grid with pieces named as W_P, B_P, etc.)
initial_board = [
    ["B_R", "B_N", "B_B", "B_Q", "B_K", "B_B", "B_N", "B_R"],  # Rank 8
    ["B_P", "B_P", "B_P", "B_P", "B_P",   "B_P", "B_P", "B_P"],  # Rank 7
    ["E",   "E",   "E",   "E",   "E", "E",   "E",   "E"],    # Rank 6
    ["E",   "E",   "E",   "E",   "E",   "E",   "E",   "E"],    # Rank 5
    ["E",   "E",   "E",   "E",   "E",   "E",   "E",   "E"],    # Rank 4
    ["E",   "E",   "E",   "E",   "E",   "E",   "E",   "E"],    # Rank 3
    ["W_P", "W_P", "W_P", "W_P", "W_P", "W_P", "W_P", "W_P"],  # Rank 2
    ["W_R", "W_N", "W_B", "W_Q", "W_K", "W_B", "W_N", "W_R"],  # Rank 1
]

navigation = {
    f"{r}-{c}": f"{chr(ord('a') + c)}{8 - r}" 
    for r in range(8) for c in range(8)
}

def decode_uci_to_json(fen: str, uci_move: str):
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci_move)

    piece = board.piece_at(move.from_square)
    if piece is None:
        raise ValueError("No piece found on the from-square")

    piece_name = f"{'W' if piece.color == chess.WHITE else 'B'}_{piece.symbol().upper()}"

    return {
        "piece": piece_name,
        "from": chess.square_name(move.from_square),
        "to": chess.square_name(move.to_square)
    }

class ChessBoard:
    def __init__(self):
        self.board_matrix = [row[:] for row in initial_board]  # deep copy
        self.chess_board = chess.Board()  # tracks FEN + legality
        # add initial move
        self.chess_board.push(chess.Move.from_uci("e2e4"))
        self.model = load_model()
        self.piece_model = load_piece_model()
        self.color_model = load_color_model()

    def analyse_board(self, image_url: str):
        image = cv2.imread(image_url)
        if image is None:
            raise ValueError("Image not found or could not be read.")

        # Detect corners
        corners = detect_corners(image, self.model)
        if corners is None or len(corners) != 4:
            raise ValueError("Could not detect board corners")

        warped_image = warp_board(image, corners)
        squares = split_board_into_squares(warped_image)

        new_board_state = [["E" for _ in range(8)] for _ in range(8)]

        for square in squares:
            row, col = square["row"], square["col"]
            img = square["image"]
            piece_or_empty = classify_piece(img, self.piece_model)
            
            if piece_or_empty == "piece":
                color = classify_color(img, self.color_model)
                code = "W_P" if color == "white" else "B_P"
                # Fallback as pawn until we have a type detector
                new_board_state[row][col] = code
            else:
                new_board_state[row][col] = "E"
        
        

        move_uci = self._detect_move(self.board_matrix, new_board_state)
        if move_uci:
            try:
                move = chess.Move.from_uci(move_uci)
                if move in self.chess_board.legal_moves:
                    self.chess_board.push(move)
                    # Update board matrix
                    fr, fc = self._coord_from_square(move_uci[:2])
                    tr, tc = self._coord_from_square(move_uci[2:])
                    self.board_matrix[tr][tc] = self.board_matrix[fr][fc]
                    self.board_matrix[fr][fc] = "E"
                    print(f"Move detected: {move_uci}")
                    return self.chess_board.fen()
                else:
                    print(f"Illegal move detected: {move_uci}")
            except Exception as e:
                print(f"Error processing move: {e}")
        else:
            print("No valid move detected")
        return self.chess_board.fen()

    def _detect_move(self, old_board, new_board):
        from_sq = None
        to_sq = None

        for r in range(8):
            for c in range(8):
                old = old_board[r][c]
                new = new_board[r][c]

                if old != "E" and new == "E":
                    from_sq = (r, c)
                if (old == "E" and new != "E") or \
                   (old.startswith("W") and new.startswith("B")) or \
                   (old.startswith("B") and new.startswith("W")):
                    to_sq = (r, c)
        print(from_sq, to_sq)
        if from_sq and to_sq:
            from_square = navigation[f"{from_sq[0]}-{from_sq[1]}"]
            to_square = navigation[f"{to_sq[0]}-{to_sq[1]}"]
            return from_square + to_square
        return None

    def _coord_from_square(self, square):
        col = ord(square[0]) - ord('a')
        row = 8 - int(square[1])
        return (row, col)
    
    def make_move(self, color: str, uci_move: str):
        try:
            move = chess.Move.from_uci(uci_move)
        except:
            raise ValueError(f"Invalid UCI format: {uci_move}")
        
        if move not in self.chess_board.legal_moves:
            raise ValueError(f"Illegal move: {uci_move}")

        # Update python-chess board
        self.chess_board.push(move)

        # Update our matrix
        from_sq = move.from_square
        to_sq = move.to_square
        from_row = 7 - chess.square_rank(from_sq)
        from_col = chess.square_file(from_sq)
        to_row = 7 - chess.square_rank(to_sq)
        to_col = chess.square_file(to_sq)

        piece = self.state_matrix[from_row][from_col]

        # Check that the right color is moving
        if color == "W" and not piece.startswith("W_"):
            raise ValueError(f"No white piece at {uci_move[:2]}")
        if color == "B" and not piece.startswith("B_"):
            raise ValueError(f"No black piece at {uci_move[:2]}")

        self.state_matrix[to_row][to_col] = piece
        self.state_matrix[from_row][from_col] = "E"

        print(f"Move applied: {piece} {uci_move[:2]} -> {uci_move[2:]}")
        print(f"FEN: {self.chess_board.fen()}")
    
    