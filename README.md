# Chess Move Predictor

A computer vision-based chessboard analyzer and move predictor. This project uses deep learning models to detect a chessboard from an image, identify the pieces and their colors, and update the board state accordingly. It can be used as a foundation for chess automation, digital board tracking, or chess AI applications. The application can be run as a command-line tool or through an interactive web interface hosted with Streamlit.

---

## Features

- **Chessboard Detection:** Uses a YOLO-based model to detect the four corners of a chessboard in an image.
- **Board Warping:** Warps the detected chessboard to a standard square for further analysis.
- **Square Segmentation:** Splits the warped board into 64 individual squares.
- **Piece Classification:** Classifies each square as empty or containing a piece using a CNN model.
- **Color Classification:** Determines the color (white/black) of each detected piece.
- **Board State Update:** Compares the detected board state with the internal board and updates it accordingly.
- **Interactive Web Interface:** A Streamlit-based web app to upload an image and visualize the board analysis.
- **Move Analysis:** (Planned/Optional) Can be extended to predict moves or generate FEN strings for chess engines.

---

## Project Structure

```
chess-move-predictor/
│
├── main.py
├── app.py                  # Streamlit web application
├── predictor.py
├── utils.py
├── requirements.txt
├── README.md
├── chess9.png
├── engine/
│   └── stockfish-ubuntu-x86-64-avx2
└── src/
    └── libs/
        ├── classify_color.py
        ├── classify_piece.py
        ├── classify_squares.py
        ├── detect_board.py
        ├── fen_generator.py
        ├── main.py
        ├── warp_board.py
        └── models/
        |   ├── cnn_color.onnx
        |   ├── cnn_piece.onnx
        |   └── yolo_corner.onnx
        └── utils/
            ├── drawing.py
            └── image_utils.py
```

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Tinkr-Robotics-Hackathon/chess-playing-model.git](https://github.com/Tinkr-Robotics-Hackathon/chess-playing-model.git)
    cd chess-playing-model
    ```

2.  **Set up a Python virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Make sure your `requirements.txt` file includes `streamlit`, then run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download or place the required ONNX models:**
    -   `src/libs/models/yolo_corner.onnx` (for board detection)
    -   `src/libs/models/cnn_piece.onnx` (for piece detection)
    -   `src/libs/models/cnn_color.onnx` (for color detection)

---

## Usage

You can run the application using either the interactive Streamlit web interface or the command-line script.

### Running the Web App with Streamlit

To make the application interactive, we use Streamlit to host a simple web-based GUI. This allows you to upload a chessboard image and see the analysis in your browser.

1.  **Create the Streamlit app file:**
    Create a file named `app.py` in the root directory and add your Streamlit interface code. A basic example might look like this:

    ```python
    # app.py
    import streamlit as st
    from PIL import Image
    from utils import ChessBoard
    import numpy as np

    st.title("Chess Move Predictor")
    st.write("Upload an image of a chessboard to analyze its state.")

    uploaded_file = st.file_uploader("Choose a chessboard image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Chessboard', use_column_width=True)
        st.write("")
        st.write("Analyzing the board...")

        # Convert PIL Image to OpenCV format (NumPy array)
        img_array = np.array(image)

        # Instantiate and analyze the board
        try:
            cb = ChessBoard()
            # The analyse_board method should be able to handle a numpy array
            board_state, analyzed_image = cb.analyse_board(img_array, is_path=False)

            st.write("### Detected Board State:")
            st.text(board_state)

            st.write("### Analyzed Image:")
            st.image(analyzed_image, caption='Processed Board', use_column_width=True)

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
    ```
    *(Note: You may need to adapt your `analyse_board` function to accept an image array instead of just a file path, as shown with the `is_path=False` flag in the example).*

2.  **Run the Streamlit app:**
    Execute the following command in your terminal:
    ```bash
    streamlit run app.py
    ```

3.  **Interact with the app:**
    Your web browser will open with the app running. Upload an image, and the application will display the detected board state and the processed image.

### Command-Line Usage

You can still use the provided scripts to analyze a chessboard image directly from the command line.

```python
# Example from a Python script
from utils import ChessBoard

cb = ChessBoard()
board_state, _ = cb.analyse_board('chess9.png')
print(board_state)
```

-   The `analyse_board` method will:
    -   Detect the board and corners in the image.
    -   Warp and segment the board.
    -   Classify each square as empty or containing a piece, and determine the color.
    -   Return the final board state.

You can also run the main script directly:

```bash
python main.py
```

---

## Requirements

-   Python 3.8+
-   OpenCV
-   NumPy
-   onnxruntime
-   **streamlit**
-   chess (python-chess)
-   torch (if using PyTorch models)
-   seaborn (for some YOLOv5 visualizations, optional)

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Models

-   **YOLOv5/YOLOv8 ONNX model** for board corner detection.
-   **CNN ONNX models** for piece and color classification.

You can train your own models or use the provided ones.

---

## Extending the Project

-   **Move Prediction:** Integrate with Stockfish or another chess engine for move prediction.
-   **FEN Generation:** Use the detected board state to generate FEN strings.
-   **Webcam/Video Support:** Extend the Streamlit app to process live video or webcam streams.
-   **Improve GUI:** Add more features to the Streamlit interface, like displaying the FEN string, suggesting best moves, or showing piece capture history.

---

## Troubleshooting

-   Ensure all model files are present in the correct directories.
-   If you encounter missing package errors, install them using `pip`.
-   For OpenCV errors, ensure your image paths are correct and images are readable.
-   When using Streamlit, make sure your analysis functions can handle image data in the form of NumPy arrays.

---

## License

MIT License. See `LICENSE` for details.

---

## Acknowledgements

-   [Streamlit](https://streamlit.io/)
-   [Ultralytics YOLO](https://github.com/ultralytics/yolov5)
-   [python-chess](https://python-chess.readthedocs.io/)
-   [Stockfish Chess Engine](https://stockfishchess.org/)
