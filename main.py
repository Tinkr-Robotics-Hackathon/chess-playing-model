# main.py
import chess
from predictor import ChessMovePredictor
from utils import decode_uci_to_json, ChessBoard
import cv2


predictor = ChessMovePredictor()
board = ChessBoard()

cap = cv2.VideoCapture(2)


def play():
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        error_message = (
            "Failed to capture frame from the feed. Check the URL and try again."
        )
        raise RuntimeError(error_message)
    image_path = "chess_board_capture.jpeg"
    cv2.imwrite(image_path, frame)
    board_fen = board.analyse_board(image_path)
    best_move = predictor.predict_best_move(board_fen)
    board.make_move(
        "W" if board.chess_board.turn == chess.WHITE else "B",
        best_move.uci(),
    )
    move_json = decode_uci_to_json(board_fen, best_move.uci())
    return move_json
