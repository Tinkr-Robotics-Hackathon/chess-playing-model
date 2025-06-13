
# Chess Move Predictor

A computer vision-based chessboard analyzer and move predictor. This project uses deep learning models to detect a chessboard from an image, identify the pieces and their colors, and update the board state accordingly. It can be used as a foundation for chess automation, digital board tracking, or chess AI applications.

---

## Features

- **Chessboard Detection:** Uses a YOLO-based model to detect the four corners of a chessboard in an image.
- **Board Warping:** Warps the detected chessboard to a standard square for further analysis.
- **Square Segmentation:** Splits the warped board into 64 individual squares.
- **Piece Classification:** Classifies each square as empty or containing a piece using a CNN model.
- **Color Classification:** Determines the color (white/black) of each detected piece.
- **Board State Update:** Compares the detected board state with the internal board and updates it accordingly.
- **Move Analysis:** (Planned/Optional) Can be extended to predict moves or generate FEN strings for chess engines.

---

## Project Structure

```
chess-move-predictor/
│
├── main.py
├── predictor.py
├── utils.py
├── requirements.txt
├── README.md
├── chess9.png
├── engine/
│   └── stockfish-ubuntu-x86-64-avx2
├── src/
│   └── libs/
│       ├── classify_color.py
│       ├── classify_piece.py
│       ├── classify_squares.py
│       ├── detect_board.py
│       ├── fen_generator.py
│       ├── main.py
│       ├── warp_board.py
│       └── models/
│           ├── cnn_color.onnx
│           ├── cnn_piece.onnx
│           └── yolo_corner.onnx
│       └── utils/
│           ├── drawing.py
│           └── image_utils.py
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Tinkr-Robotics-Hackathon/chess-playing-model.git
   cd chess-playing-model
   ```

2. **Set up a Python virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download or place the required ONNX models:**
   - `src/libs/models/yolo_corner.onnx` (for board detection)
   - `src/libs/models/cnn_piece.onnx` (for piece detection)
   - `src/libs/models/cnn_color.onnx` (for color detection)

---

## Usage

### Analyze a Chessboard Image

You can use the provided scripts to analyze a chessboard image and get the board state.

```python
from utils import ChessBoard

cb = ChessBoard()
board_state = cb.analyse_board('chess9.png')
print(board_state)
```

- The `analyse_board` method will:
  - Detect the board and corners in the image.
  - Warp and segment the board.
  - Classify each square as empty or containing a piece, and determine the color.
  - Update the internal board state.

### Main Script

You can run the main script:

```bash
python main.py
```

---

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- onnxruntime
- chess (python-chess)
- torch (if using PyTorch models)
- seaborn (for some YOLOv5 visualizations, optional)

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Models

- **YOLOv5/YOLOv8 ONNX model** for board corner detection.
- **CNN ONNX models** for piece and color classification.

You can train your own models or use the provided ones.

---

## Extending the Project

- **Move Prediction:** Integrate with Stockfish or another chess engine for move prediction.
- **FEN Generation:** Use the detected board state to generate FEN strings.
- **Webcam/Video Support:** Extend to process live video or webcam streams.
- **GUI:** Build a graphical interface for easier use.

---

## Troubleshooting

- Ensure all model files are present in the correct directories.
- If you encounter missing package errors, install them using `pip`.
- For OpenCV errors, ensure your image paths are correct and images are readable.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/yolov5)
- [python-chess](https://python-chess.readthedocs.io/)
- [Stockfish Chess Engine](https://stockfishchess.org/)

---
