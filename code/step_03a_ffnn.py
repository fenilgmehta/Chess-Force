from typing import Union
from pathlib import Path
import sys
import chess
import numpy as np

import step_02_preprocess as step_02


# Feed Forward Neural Network
class FFNN:
    # TODO
    def __init__(self):
        return

    # TODO
    def generate_model(self):
        return

    # TODO
    def save_model(self, model_path: Union[str, Path]):
        return

    # TODO
    def load_model(self, model_path: Union[str, Path]):
        return

    # TODO
    def train_model(self, x_input, y_output):
        return

    # TODO
    def predict(self, encoded_board: str) -> np.float32:
        return np.float32(0)

    # TODO
    def predict_board(self, board: chess.Board) -> np.float32:
        return self.predict(step_02.encode_board_777(board))
