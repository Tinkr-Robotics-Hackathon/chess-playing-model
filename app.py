import streamlit as st
import chess
import chess.svg
from PIL import Image
import numpy as np
import os
import time

cam_url = "http://192.168.206.82:4747/video"

# --- Import your actual classes from your project files ---
# Make sure predictor.py and utils.py are in the same directory as this app.py
try:
    from predictor import ChessMovePredictor
    from utils import ChessBoard, decode_uci_to_json # Assuming ChessBoard is in utils.py
except ImportError as e:
    st.error(f"Failed to import required modules: {e}. Make sure predictor.py and utils.py are in the correct directory.")
    st.stop()


# --- Main Application ---

# --- Page Configuration ---
st.set_page_config(
    page_title="Man Vs Machine Chess",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS for Styling ---
st.markdown("""
<style>
    .main {
        background-color: #2E2E2E;
        color: #FFFFFF;
    }
    .st-emotion-cache-1y4p8pa {
        background-color: #1E1E1E;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
        width: 100%;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    h1, h2, h3 {
        color: #4CAF50;
    }
    .status-box {
        background-color: #333333;
        border-left: 5px solid #4CAF50;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .move-info {
        font-size: 1.2em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# --- State and Model Initialization ---
# Use @st.cache_resource to initialize the models only once.
@st.cache_resource
def load_models():
    predictor = ChessMovePredictor()
    chess_board = ChessBoard()
    return predictor, chess_board

predictor, chess_board = load_models()

# --- Session State Initialization ---
if 'game_started' not in st.session_state:
    st.session_state.game_started = False
if 'board' not in st.session_state:
    st.session_state.board = chess_board  # This is a ChessBoard instance
if 'fen' not in st.session_state:
    st.session_state.fen = st.session_state.board.chess_board.fen()
if 'image_captured' not in st.session_state:
    st.session_state.image_captured = False
if 'ai_move' not in st.session_state:
    st.session_state.ai_move = None
if 'game_over' not in st.session_state:
    st.session_state.game_over = False
if 'last_board_image_path' not in st.session_state:
    st.session_state.last_board_image_path = None
if 'ai_move_json' not in st.session_state:
    st.session_state.ai_move_json = None


# --- Sidebar ---
with st.sidebar:
    st.title("♟️ AI Chess Master")
    st.markdown("---")

    if st.button("New Game"):
        st.session_state.game_started = True
        st.session_state.board = ChessBoard()  # Reset to new ChessBoard instance
        st.session_state.fen = st.session_state.board.chess_board.fen()
        st.session_state.image_captured = False
        st.session_state.ai_move = None
        st.session_state.game_over = False
        st.rerun()

    st.markdown("---")
    st.subheader("Game Info")
    turn = "White" if st.session_state.board.chess_board.turn == chess.WHITE else "Black"
    st.write(f"**Turn:** {turn}")
    st.write(f"**Castling Rights:**")
    st.write(f"- White: {st.session_state.board.chess_board.has_kingside_castling_rights(chess.WHITE)} (K), {st.session_state.board.chess_board.has_queenside_castling_rights(chess.WHITE)} (Q)")
    st.write(f"- Black: {st.session_state.board.chess_board.has_kingside_castling_rights(chess.BLACK)} (k), {st.session_state.board.chess_board.has_queenside_castling_rights(chess.BLACK)} (q)")

    st.markdown("---")
    st.subheader("How to Play")
    st.info("""
    1.  Click **New Game** to start.
    2.  Make your move on the physical board.
    3.  Click on PLAY.
    4.  The AI will make its move.
    5.  The robot arm will materialize the AI move on the physical board.
    6.  Repeat!
    """)

# --- Main Content ---
st.title("Man Vs Machine Chess")
st.markdown("Play chess against our own model. The future of board games is here!")

if not st.session_state.game_started:
    st.warning("Click 'New Game' in the sidebar to begin a match.")
else:
    col1, col2 = st.columns([2, 1.5])

    with col1:
        import cairosvg
        st.header("Chess Board")
        # Highlight the AI's last move on the board
        last_move = st.session_state.ai_move if st.session_state.ai_move else None
        board_svg = chess.svg.board(
            board=st.session_state.board.chess_board,
            lastmove=last_move,
            size=1000
        )
        # Convert SVG to PNG for display
        png_data = cairosvg.svg2png(bytestring=board_svg.encode("utf-8"))
        st.image(png_data, use_container_width=True)

    with col2:
        st.header("Game Controls")

        # Inject JS to trigger a hidden Streamlit button on spacebar press
        st.markdown("""
            <script>
            document.addEventListener('keydown', function(e) {
                if (e.code === 'Space' && !e.repeat) {
                    var btn = window.parent.document.querySelector('button[data-testid="play-btn"]');
                    if (btn && !btn.disabled) { btn.click(); }
                }
            });
            </script>
        """, unsafe_allow_html=True)

        if st.session_state.game_over:
            st.error("Game Over!")
            result = st.session_state.board.result()
            st.metric("Result", result)
        else:
            import cv2
            play_disabled = st.session_state.get("play_disabled", False)
            # Ensure play_disabled is initialized
            if "play_disabled" not in st.session_state:
                st.session_state.play_disabled = False
            play_button = st.button("PLAY", key="play-btn", disabled=play_disabled, use_container_width=True)

            # On click, set disabled and rerun to update UI immediately
            if play_button and not play_disabled:
                st.session_state.play_disabled = True
                st.rerun()

            # If play_disabled is True, process the move (after rerun)
            if play_disabled:
                error_message = None
                warning_message = None
                with st.spinner("Capturing board, analyzing, and calculating AI move..."):
                    try:
                        cap = cv2.VideoCapture(cam_url)
                        ret, frame = cap.read()
                        cap.release()
                        if not ret or frame is None:
                            error_message = "Failed to capture frame from the feed. Check the URL and try again."
                            raise RuntimeError()
                        image_path = "chess_board_capture.jpeg"
                        cv2.imwrite(image_path, frame)
                        st.session_state.board.analyse_board(image_path)
                        if not st.session_state.board.chess_board.is_valid():
                            error_message = "Illegal move detected. Please ensure the board is set up correctly."
                            raise RuntimeError()
                        new_fen = st.session_state.board.chess_board.fen()
                        st.session_state.fen = new_fen
                        if not st.session_state.board.chess_board.is_valid():
                            error_message = "The board state is invalid. Please check the board setup."
                            raise RuntimeError()
                        best_move = predictor.predict_best_move(st.session_state.fen)
                        with open("best_move.txt", "w") as f:
                            f.write(best_move.uci())
                        move_json = decode_uci_to_json(
                            st.session_state.fen, best_move.uci()
                        )
                        st.session_state.ai_move_json = move_json
                        if best_move and best_move in st.session_state.board.chess_board.legal_moves:
                            st.session_state.ai_move = best_move
                            st.session_state.board.make_move(
                                "W" if st.session_state.board.chess_board.turn == chess.WHITE else "B",
                                best_move.uci()
                            )
                            st.session_state.fen = st.session_state.board.chess_board.fen()
                            st.session_state.play_disabled = False
                        else:
                            warning_message = "AI could not determine a valid move. The game might be over or the board state is unusual."
                            st.session_state.game_over = True
                            st.session_state.play_disabled = False
                    except Exception as e:
                        if not error_message:
                            error_message = str(e)
                        warning_message = "Please ensure the image is clear and the board is well-lit and you play a valid move."
                        st.session_state.play_disabled = False
                # Show errors/warnings after spinner closes
                if error_message:
                    st.error(error_message)
                if warning_message:
                    st.warning(warning_message)
                # Give user time to read the error/warning
                time.sleep(2)
                st.rerun()


    st.markdown("---")
    st.header("Game Status")
    status_col1, status_col2 = st.columns(2)

    with status_col1:
        if st.session_state.ai_move:
            st.subheader("AI's Last Move")
            
            st.markdown(
                f"""
                <div class="status-box">
                    <p class="move-info">The AI moved: <b>{st.session_state.ai_move.uci()}</b></p>
                    <p>Piece: <b>{st.session_state.ai_move_json.get('piece', '?')}</b></p>
                    <p>From: <b>{st.session_state.ai_move_json.get('from', '?')}</b> &rarr; To: <b>{st.session_state.ai_move_json.get('to', '?')}</b></p>
                    <p>Type: <b>{st.session_state.ai_move_json.get('type', "Default").capitalize()}</b></p>
                    <p>Make this move on your physical board, then make your own move and capture the new board state.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("The AI is waiting for you to start the game or make a move.")

    with status_col2:
        st.subheader("Current Board State (FEN)")
        st.code(st.session_state.fen, language="text")

        if st.session_state.board.chess_board.is_checkmate():
            st.session_state.game_over = True
            st.error("Checkmate!")
        elif st.session_state.board.chess_board.is_stalemate():
            st.session_state.game_over = True
            st.warning("Stalemate!")
        elif st.session_state.board.chess_board.is_insufficient_material():
            st.session_state.game_over = True
            st.warning("Insufficient Material!")
        elif st.session_state.board.chess_board.is_check():
            st.warning("You are in check!")
