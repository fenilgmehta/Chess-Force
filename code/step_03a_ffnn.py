import multiprocessing
from pathlib import Path
from typing import Union, Tuple, List

import chess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# import tensorflow
from tensorflow.python import keras
from tensorflow.python.client import device_lib

import common_services as cs
import step_02_preprocess as step_02


#########################################################################################################################
# Feed Forward Neural Network
class FFNNTorch(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs = 778
        # Outputs = 1

        # Build a feed-forward network
        self.model = nn.Sequential(nn.Linear(778, 512),  # Layer 1
                                   nn.ReLU(),
                                   nn.Linear(512, 512),  # Layer 2
                                   nn.ReLU(),
                                   nn.Linear(512, 512),  # Layer 3
                                   nn.ReLU(),
                                   nn.Linear(512, 512),  # Layer 4
                                   nn.ReLU(),
                                   nn.Linear(512, 1),  # Output layer
                                   nn.Sigmoid())

        # Define the loss
        self.criterion = nn.NLLLoss()
        # Optimizers require the parameters to optimize and a learning rate
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.003)

        # We will use ``torch.device`` objects to move tensors in and out of GPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")  # a CUDA device object

        return

    # TODO
    def c_save_model(self, model_path: Union[str, Path]):
        return

    # TODO
    def c_load_model(self, model_path: Union[str, Path]):
        return

    def c_train_model(self, x_input, y_output, epochs, batch_size):
        for epoch in range(500):
            # Forward pass: Compute predicted y by passing x to the model
            pred_y = self.model(x_input)

            # Compute and print loss
            loss = self.criterion(pred_y, y_output)

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print('epoch {}, loss {}'.format(epoch, loss.data[0]))

        return

    def c_predict(self, encoded_board: np.ndarray) -> torch.Tensor:
        # Pass the input tensor through each of our operations
        encoded_board = torch.from_numpy(encoded_board)
        return self.model(encoded_board)
        # return self.model.predict(encoded_board)

    def predict_board(self, board: chess.Board) -> torch.Tensor:
        return self.c_predict(step_02.BoardEncoder.encode_board_1_778(board))

    def predict_fen(self, board_fen: str) -> np.float32:
        pass
        # return self.c_predict(step_02.BoardEncoder.encode_board_1_778(chess.Board(board_fen)))


#########################################################################################################################
# NOTE: `c` before each method name means that it is custom
# Feed Forward Neural Network - Keras
class FFNNKeras:
    def __init__(self, model_save_path: str = "../chess_models", generate_model_image=False):
        self.model_save_path = model_save_path
        self.model_save_path_dir = str(Path(model_save_path).parent)
        # Create a callback that saves the model's weights
        self.cp_callback = keras.callbacks.ModelCheckpoint(filepath=self.model_save_path,
                                                           save_weights_only=True,
                                                           verbose=1)

        self.model = self.__generate_model()
        # self.model.summary()

        # Save the model graph
        if generate_model_image:
            for i in range(1, 100):
                if not Path(f'FFNNKeras_{i:03}.png').exists():
                    keras.utils.plot_model(self.model, f'FFNNKeras_{i:03}.png', show_shapes=True)
                    print(f"Saving the image: 'FFNNKeras_{i:03}.pgn'")
                    break
        return

    def __generate_model(self):
        # define the keras model
        model: keras.Sequential = keras.Sequential()
        model.add(keras.layers.Dense(256, activation='relu', input_shape=(778,)))
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dense(1, activation='tanh'))

        # compile the keras model
        # optimizer = keras.optimizers.adam(lr=0.001,
        #                                        beta_1=0.9,
        #                                        beta_2=0.999,
        #                                        epsilon=1e-07,
        #                                        amsgrad=False, )
        # __init__(
        #     learning_rate=0.001,  learning_rate: A Tensor or a floating point value. The learning rate.
        #     beta_1=0.9,    beta_1: A float value or a constant float tensor. The exponential decay rate for the 1st moment estimates.
        #     beta_2=0.999,  beta_2: A float value or a constant float tensor. The exponential decay rate for the 2nd moment estimates.
        #     epsilon=1e-07, epsilon: A small constant for numerical stability. This epsilon is "epsilon hat" in the Kingma
        #                    and Ba paper (in the formula just before Section 2.1), not the epsilon in Algorithm 1 of the paper.
        #     amsgrad=False, amsgrad: boolean. Whether to apply AMSGrad variant of this algorithm from the paper "On the Convergence of Adam and beyond".
        #     name='Adam',   name: Optional name for the operations created when applying gradients. Defaults to "Adam". @compatibility(eager)
        #                    When eager execution is enabled, learning_rate, beta_1, beta_2, and epsilon can each be a callable that takes no
        #                    arguments and returns the actual value to use. This can be useful for changing these values across different
        #                    invocations of optimizer functions. @end_compatibility
        #     **kwargs       **kwargs: keyword arguments. Allowed to be {clipnorm, clipvalue, lr, decay}. clipnorm is clip gradients by norm;
        #                    clipvalue is clip gradients by value, decay is included for backward compatibility to allow time inverse decay of
        #                    learning rate. lr is included for backward compatibility, recommended to use learning_rate instead.
        # )

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])  # mae = Mean Absolute Error
        return model

    def c_save_model(self, model_name: str, model_path: Union[str, Path] = "../chess_models"):
        Path(model_path).mkdir(parents=True, exist_ok=True)
        self.model.save(str(Path(model_path) / model_name), overwrite=True)
        return

    def c_save_weights(self, model_name: str, model_path: Union[str, Path] = "../chess_models"):
        Path(model_path).mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(Path(model_path) / model_name), overwrite=True)
        return

    # TODO: to fix this, not working
    def c_load_model(self, model_name: str, model_path: Union[str, Path] = "../chess_models"):
        if not (Path(model_path) / model_name).exists():
            print(f"ERROR: model does not exists: {Path(model_path) / model_name}")
        self.model = keras.models.load_model(str(Path(model_path) / model_name))
        return

    def c_load_weights(self, model_name: str, model_path: Union[str, Path] = "../chess_models"):
        if not (Path(model_path) / model_name).exists():
            print(f"ERROR: model does not exists: {Path(model_path) / model_name}")
        self.model.load_weights(str(Path(model_path) / model_name))
        return

    def c_train_model(self, x_input: np.ndarray, y_output: np.ndarray, epochs: int, batch_size: int, validation_split: float):
        # compile the keras model
        self.model.trainable = True

        # self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])  # mae = Mean Absolute Error
        self.model.fit(x_input, y_output, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
                       verbose=2, workers=multiprocessing.cpu_count(), use_multiprocessing=True,
                       callbacks=[self.cp_callback])

        self.model.trainable = False
        return

    def c_evaluate_model(self, x_input_test: np.ndarray, y_output_test: np.ndarray, verbose=2):
        loss, mae = self.model.evaluate(x_input_test, y_output_test, verbose=2)
        print(f"Evaluated: Loss = {loss:5.3f}")
        print(f"Evaluated: MAE = {mae:5.3f}")

    def c_predict(self, encoded_board: np.ndarray) -> np.float32:
        return self.model.predict(
            encoded_board,
            verbose=1, workers=multiprocessing.cpu_count(), use_multiprocessing=True
        ).ravel()

    def c_predict_board_1(self, board_1: chess.Board) -> np.float32:
        return self.c_predict(
            step_02.BoardEncoder.encode_board_1_778(
                board_1
            ).reshape(1, -1)
        )

    def c_predict_board_n(self, board_n: Union[List[chess.Board], Tuple[chess.Board]]) -> np.float32:
        return self.c_predict(
            step_02.BoardEncoder.encode_board_n_778(
                board_n
            )
        )

    def c_predict_fen_1(self, board_1_fen: str) -> np.float32:
        return self.c_predict(
            step_02.BoardEncoder.encode_board_1_778(
                chess.Board(board_1_fen)
            ).reshape(1, -1)
        )

    def c_predict_fen_n(self, board_n_fen: Union[List[str], Tuple[str]]) -> np.float32:
        return self.c_predict(
            step_02.BoardEncoder.encode_board_n_778(
                [chess.Board(i) for i in board_n_fen]
            )
        )


#########################################################################################################################
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# GET list of devices
# get_available_gpus()
# device_lib.list_local_devices()  # list all local devices

# TRAINING on GPU
# tensorflow.device("/gpu:0")
# sess = tensorflow.compat.v1.Session(config=tensorflow.compat.v1.ConfigProto(log_device_placement=True))
if __name__ == '__main__':
    # DATA-SET loading
    with cs.ExecutionTime():
        data = pd.read_csv("out_combined_KingBase2019-B00-B19_000000.csv")
        data_x = data[cs.COLUMNS[0]].values
        data_y = data[cs.COLUMNS[1]].values
    with cs.ExecutionTime():
        data_x_encoded = step_02.BoardEncoder.encode_board_n_778_fen(data_x)
    with cs.ExecutionTime():
        data_y = data_y * 10000
        data_y[data_y >= 10] = 10
        data_y[data_y <= -10] = -10
        data_y = data_y / 10

    # MODEL creation and training
    ffnn_keras = FFNNKeras(generate_model_image=True)
    ffnn_keras.c_load_weights("ffnn_keras_v003_000010_weights.h5")
    with cs.ExecutionTime():
        ffnn_keras.c_train_model(data_x_encoded, data_y, 10, 512, 0.2)
    ffnn_keras.c_save_weights("ffnn_keras_v004_000010_weights.h5")

    # MODEL testing
    with cs.ExecutionTime():
        y_predicted = ffnn_keras.c_predict_fen_n(data_x)
        print(f"MAE = {np.sum(np.abs(y_predicted - data_y) / len(y_predicted))}")
    with cs.ExecutionTime():
        ffnn_keras.c_evaluate_model(data_x_encoded, data_y)
