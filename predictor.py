# predictor.py
import chess
import chess.engine

class ChessMovePredictor:
    def __init__(self, stockfish_path="engine/stockfish-ubuntu-x86-64-avx2", depth=15):
        print("Loading Stockfish...")
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.depth = depth
        print("Stockfish loaded.")

    def predict_best_move(self, fen):
        board = chess.Board(fen)
        result = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
        return result["pv"][0]  # Best move

    def close(self):
        self.engine.quit()
