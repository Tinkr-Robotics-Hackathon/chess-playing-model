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

    # Determine move type
    if board.is_castling(move):
        move_type = "castle"
    elif board.is_capture(move):
        move_type = "capture"
    else:
        move_type = "move"
    

    return {
        "piece": piece_name,
        "from": chess.square_name(move.from_square),
        "to": chess.square_name(move.to_square),
        "type": move_type
    }

class ChessBoard:
    def __init__(self):
        self.board_matrix = [row[:] for row in initial_board]  # deep copy
        self.chess_board = chess.Board()  
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
        
        #  write new and old board state to a .txt file for comparison
        with open("old_board_state.txt", "w") as f:
            for row in self.board_matrix:
                f.write(" ".join(row) + "\n")
        with open("new_board_state.txt", "w") as f:
            for row in new_board_state:
                f.write(" ".join(row) + "\n")
        
        

        move_uci = self._detect_move(self.board_matrix, new_board_state)
        if move_uci:
            try:
                move = chess.Move.from_uci(move_uci)
                if move in self.chess_board.legal_moves:
                    self.chess_board.push(move)
                    # Update board matrix
                    if move_uci in ["e1g1", "e1c1", "e8g8", "e8c8"]:
                        print(f"Castling detected: {move_uci}")
                        # Handle castling
                        if move_uci == "e1g1":
                            # White kingside castling
                            self.board_matrix[7][5] = "W_R"
                            self.board_matrix[7][6] = "W_K"
                            self.board_matrix[7][4] = "E"
                            self.board_matrix[7][7] = "E"
                        elif move_uci == "e1c1":
                            # White queenside castling
                            self.board_matrix[7][2] = "W_K"
                            self.board_matrix[7][3] = "W_R"
                            self.board_matrix[7][4] = "E"
                            self.board_matrix[7][0] = "E"
                        elif move_uci == "e8g8":
                            # Black kingside castling
                            self.board_matrix[0][5] = "B_R"
                            self.board_matrix[0][6] = "B_K"
                            self.board_matrix[0][4] = "E"
                            self.board_matrix[0][7] = "E"
                        elif move_uci == "e8c8":
                            # Black queenside castling
                            self.board_matrix[0][2] = "B_K"
                            self.board_matrix[0][3] = "B_R"
                            self.board_matrix[0][4] = "E"
                            self.board_matrix[0][0] = "E"
                    else:
                    # Regular move
                    # Update the board matrix with the move
                        fr, fc = self._coord_from_square(move_uci[:2])
                        tr, tc = self._coord_from_square(move_uci[2:])
                        self.board_matrix[tr][tc] = self.board_matrix[fr][fc]
                        self.board_matrix[fr][fc] = "E"
                    print(f"Move detected: {move_uci}")
                    return self.chess_board.fen()
                else:
                    print(f"Illegal move detected: {move_uci}")
                    raise ValueError(f"Illegal move detected: {move_uci}")
            except Exception as e:
                print(e.args[0])
                raise ValueError(e.args[0])
        else:
            print("No valid move detected")
        return self.chess_board.fen()

    def _detect_move(self, old_board, new_board):
        from_sqs = []
        to_sqs = []

        for r in range(8):
            for c in range(8):
                old = old_board[r][c]
                new = new_board[r][c]

                if old != "E" and new == "E":
                    from_sqs.append((r, c))
                if (old == "E" and new != "E") or \
                   (old.startswith("W") and new.startswith("B")) or \
                   (old.startswith("B") and new.startswith("W")):
                    to_sqs.append((r, c))

        print(f"from_sqs: {from_sqs}, to_sqs: {to_sqs}")

        # Handle castling: two froms and two tos, try to find a valid king move
        if len(from_sqs) == 2 and len(to_sqs) == 2:
            # Try all from-to pairs, look for a king move that is a valid castle
            for f in from_sqs:
                for t in to_sqs:
                    piece = self.board_matrix[f[0]][f[1]]
                    if not piece.endswith("K"):
                        continue
                    from_square = navigation[f"{f[0]}-{f[1]}"]
                    to_square = navigation[f"{t[0]}-{t[1]}"]
                    uci = from_square + to_square
                    try:
                        move = chess.Move.from_uci(uci)
                        if move in self.chess_board.legal_moves and self.chess_board.is_castling(move):
                            return uci
                    except Exception:
                        continue
        # Normal move (single from and to)
        if len(from_sqs) == 1 and len(to_sqs) == 1:
            from_square = navigation[f"{from_sqs[0][0]}-{from_sqs[0][1]}"]
            to_square = navigation[f"{to_sqs[0][0]}-{to_sqs[0][1]}"]
            return from_square + to_square
        # Fallback: try to return the first possible move
        if from_sqs and to_sqs:
            from_square = navigation[f"{from_sqs[0][0]}-{from_sqs[0][1]}"]
            to_square = navigation[f"{to_sqs[0][0]}-{to_sqs[0][1]}"]
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
        
        # Handle castling
        if uci_move in ["e1g1", "e1c1", "e8g8", "e8c8"]:
            print(f"Castling detected: {uci_move}")
            # Exchange the rook and king positions
            if uci_move == "e1g1":
                # White kingside castling
                # self.chess_board.set_piece_at(chess.E1, chess.Piece(chess.ROOK, chess.WHITE))
                # self.chess_board.set_piece_at(chess.G1, chess.Piece(chess.KING, chess.WHITE))
                # Update the board matrix
                self.board_matrix[7][5] = "W_R"  # F1
                self.board_matrix[7][6] = "W_K"  # G1
                self.board_matrix[7][4] = "E"
                self.board_matrix[7][7] = "E"
            elif uci_move == "e1c1":
                # White queenside castling
                # self.chess_board.set_piece_at(chess.D1, chess.Piece(chess.ROOK, chess.WHITE))
                # self.chess_board.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
                # Update the board matrix
                self.board_matrix[7][2] = "W_K"
                self.board_matrix[7][3] = "W_R"
                self.board_matrix[7][4] = "E"
                self.board_matrix[7][0] = "E"
            elif uci_move == "e8g8":
                # Black kingside castling
                # self.chess_board.set_piece_at(chess.F8, chess.Piece(chess.ROOK, chess.BLACK))
                # self.chess_board.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
                # Update the board matrix
                self.board_matrix[0][5] = "B_R"
                self.board_matrix[0][6] = "B_K"
                self.board_matrix[0][4] = "E"
                self.board_matrix[0][7] = "E"
            elif uci_move == "e8c8":
                # Black queenside castling
                # self.chess_board.set_piece_at(chess.D8, chess.Piece(chess.ROOK, chess.BLACK))
                # self.chess_board.set_piece_at(chess.A8, chess.Piece(chess.KING, chess.BLACK))
                # Update the board matrix
                self.board_matrix[0][2] = "B_K"
                self.board_matrix[0][3] = "B_R"
                self.board_matrix[0][4] = "E"
                self.board_matrix[0][0] = "E"
        else:
        # Update our matrix
            from_sq = move.from_square
            to_sq = move.to_square
            from_row = 7 - chess.square_rank(from_sq)
            from_col = chess.square_file(from_sq)
            to_row = 7 - chess.square_rank(to_sq)
            to_col = chess.square_file(to_sq)

            piece = self.board_matrix[from_row][from_col]

            # Check that the right color is moving
            if color == "W" and not piece.startswith("W_"):
                raise ValueError(f"No white piece at {uci_move[:2]}")
            if color == "B" and not piece.startswith("B_"):
                raise ValueError(f"No black piece at {uci_move[:2]}")

            self.board_matrix[to_row][to_col] = piece
            self.board_matrix[from_row][from_col] = "E"

        print(f"Move applied: {piece} {uci_move[:2]} -> {uci_move[2:]}")
        # write the board to a .txt file
        with open("chess_board.txt", "w") as f:
            for row in self.board_matrix:
                f.write(" ".join(row) + "\n")
        print(f"FEN: {self.chess_board.fen()}")
    
    