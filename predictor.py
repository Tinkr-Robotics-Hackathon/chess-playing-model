import os
import chess.engine

class ChessMovePredictor:
    def __init__(self, stockfish_path=None, depth=15):
        if stockfish_path is None:
            stockfish_path = os.path.join(
                os.path.dirname(__file__), 
                'engine', 
                'stockfish-ubuntu-x86-64-avx2'
            )
        print(f"Loading Stockfish from: {stockfish_path}")
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.depth = depth
        print("Stockfish loaded.")

    def predict_best_move(self, fen):
        board = chess.Board(fen)
        result = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
        return result["pv"][0]

    def close(self):
        self.engine.quit()
