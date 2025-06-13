# main.py
from predictor import ChessMovePredictor
from utils import decode_uci_to_json, ChessBoard


predictor = ChessMovePredictor()
board = ChessBoard()
fen = board.analyse_board("chess1.jpeg")
best_move = predictor.predict_best_move(fen)
uci_move = best_move.uci()
move_json = decode_uci_to_json(fen, uci_move)

print(f"Best move: {move_json}")
predictor.close()
