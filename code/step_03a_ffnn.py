from pathlib import Path
from typing import Union

import chess
import numpy as np
from tensorflow import keras

import step_02_preprocess as step_02


# Feed Forward Neural Network
class FFNN:
    def __init__(self):
        self.model: keras.Sequential = None
        return

    def generate_model(self):
        # define the keras model
        self.model = keras.Sequential()
        self.model.add(keras.Dense(256, input_dim=777, activation='relu'))
        self.model.add(keras.Dense(256, activation='relu'))
        self.model.add(keras.Dense(256, activation='relu'))
        self.model.add(keras.Dense(256, activation='relu'))
        self.model.add(keras.Dense(1, activation='sigmoid'))

        # compile the keras model
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return

    # TODO
    def save_model(self, model_path: Union[str, Path]):
        return

    # TODO
    def load_model(self, model_path: Union[str, Path]):
        return

    def train_model(self, x_input, y_output, epochs, batch_size):
        self.model.fit(x_input, y_output, epochs=epochs, batch_size=batch_size)
        return

    def predict(self, encoded_board: str) -> np.float32:
        return self.model.predict(encoded_board)

    def predict_board(self, board: chess.Board) -> np.float32:
        return self.predict(step_02.encode_board_777(board))

    def predict_fen(self, board_fen: str) -> np.float32:
        return self.predict(step_02.encode_board_777(chess.Board(board_fen)))
