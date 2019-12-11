import multiprocessing
import os
import warnings
from pathlib import Path
from typing import Union, Tuple, List

# WARNING/ERROR: numpy FutureWarning
# SOLUTION: https://github.com/tensorflow/tensorflow/issues/30427
import tensorflow

warnings.filterwarnings('ignore', category=FutureWarning)

# WARNING/ERROR: Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
# EXPLANATION: https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
# SOLUTION 1: https://stackoverflow.com/questions/51681727/tensorflow-on-macos-your-cpu-supports-instructions-that-this-tensorflow-binary?rq=1
# SOLUTION 2: manually compile and install tensorflow 2.0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TENSORFLOW 2.0 installation with GPU support
# Not tested SOLUTION: https://software.intel.com/en-us/articles/intel-optimization-for-tensorflow-installation-guide#Anaconda_main_win
# conda install tensorflow -c anaconda
# TESTED SOLUTION: https://anaconda.org/anaconda/tensorflow-gpu
# conda install -c anaconda tensorflow-gpu

import chess
import numpy as np
import pandas as pd
import torch
import torch.nn
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.client import device_lib
import argparse

import common_services as cs
import step_02_preprocess as step_02

#########################################################################################################################

dtype = torch.cuda.FloatTensor  # Uncomment this to run on GPU

parser = argparse.ArgumentParser(description='FFNN')

parser.add_argument('--pre', metavar='PRETRAINED', default=None, type=str,
                    help='path to the pretrained model')

parser.add_argument('--gpu', metavar='GPU', type=str, default="2",
                    help='GPU id to use.')

parser.add_argument('--task', default="SFANet_bnt2_", metavar='TASK', type=str,
                    help='task id to use.')


# Feed Forward Neural Network
class FFNNTorch(torch.nn.Module):
    def __init__(self, model_generator, board_encoder, learning_rate, use_gpu=True):
        super().__init__()

        self.model_generator = model_generator
        self.board_encoder = board_encoder
        self.learning_rate = learning_rate
        self.use_gpu = use_gpu

        # We will use ``torch.device`` objects to move tensors in and out of GPU
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda", 0)  # a CUDA device object
        else:
            self.device = torch.device("cpu", 0)  # a CPU device object

        self.model: torch.nn.Sequential = model_generator(self.device)

        # global parser
        # self.arg = parser.parse_args()
        # self.arg.lr = 0.001
        # self.arg.batch_size = 64
        # self.arg.momentum = 0.95
        # self.arg.decay = 5e-3
        # self.arg.start_epoch = 1
        # self.arg.epochs = 400
        # self.arg.workers = 1
        # self.arg.seed = time.time()
        # self.arg.print_freq = 4

        # Define the loss
        self.criterion = torch.nn.MSELoss(reduction='mean').to(self.device)  # alternative reduction='mean' /

        # Optimizers require the parameters to optimize and a learning rate
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        return

    def c_save_model(self, model_path: Union[str, Path]):
        # with h5py.File(model_path, 'w') as h5f:
        #     for k, v in self.model.state_dict().items():
        #         h5f.create_dataset(k, data=v.cpu().numpy())
        torch.save(self.model.module.state_dict(), model_path)
        return

    def c_load_model(self, model_path: Union[str, Path]):
        # with h5py.File(model_path, 'r') as h5f:
        #     for k, v in self.model.state_dict().items():
        #         param = torch.from_numpy(np.asarray(h5f[k]))
        #         v.copy_(param)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(device=self.device)
        self.model.eval()
        return

    def c_save_checkpoint(self, checkpoint_path: Union[str, Path], epoch, loss):
        # def c_save_checkpoint(state, is_best, task_id, filename='checkpoint.pth.tar'):
        #     torch.save(state, task_id + filename)
        #     if is_best:
        #         shutil.copyfile(task_id + filename, task_id + 'model_best.pth.tar')

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)

    def c_load_checkpoint(self, checkpoint_path: str, strict=True):
        # def c_load_checkpoint(self, checkpoint_path: str):
        #     if checkpoint_path:
        #         if os.path.isfile(checkpoint_path):
        #             print("=> loading checkpoint '{}'".format(checkpoint_path))
        #             checkpoint = torch.load(checkpoint_path)
        #             self.args.start_epoch = checkpoint['epoch']
        #             best_prec1 = checkpoint['best_prec1']
        #             self.model.load_state_dict(checkpoint['state_dict'])
        #             self.optimizer.load_state_dict(checkpoint['optimizer'])
        #             print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        #         else:
        #             print("=> no checkpoint found at '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        self.model.to(device=self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return epoch, loss,

    def c_train_model(self, x_input, y_output, epochs, batch_size, validation_split):
        x_input = torch.from_numpy(x_input).to(device=self.device).float()
        y_output = torch.from_numpy(y_output).to(device=self.device).float()
        for epoch in range(epochs):
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = self.model(x_input)

            # Compute and print loss
            print(type(y_pred), type(y_output))
            loss = self.criterion(y_pred, y_output)
            print(type)
            # print('epoch {}, loss {}'.format(epoch, loss.data[0]))
            print('epoch {}, loss {}'.format(epoch, loss.item()))

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the Tensors it will update (which are the learnable weights
            # of the model)
            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            # loss.backward()
            loss.mean().backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            self.optimizer.step()

        return

    # TODO: implement this
    def c_evaluate_model(self, x_input_test, y_output_test):
        y_output_test = torch.from_numpy(y_output_test).to(device=self.device).float()
        y_predicted = self.c_predict(x_input_test)
        loss = self.criterion(y_predicted, y_output_test)
        print(f"Loss = {loss}")
        return

    def c_predict(self, encoded_board: np.ndarray) -> torch.Tensor:
        # Pass the input tensor through each of our operations
        encoded_board = torch.from_numpy(encoded_board).to(device=self.device)
        return self.model(encoded_board)
        # return self.model.predict(encoded_board)

    def c_predict_board_1(self, board_1: chess.Board) -> torch.Tensor:
        return self.c_predict(
            self.board_encoder.encode_board_1(
                board_1
            ).reshape(1, -1)
        )[0]

    def c_predict_board_n(self, board_n: Union[List[chess.Board], Tuple[chess.Board]]) -> torch.Tensor:
        return self.c_predict(
            self.board_encoder.encode_board_n(
                board_n
            )
        )

    def c_predict_fen_1(self, board_1_fen: str) -> torch.Tensor:
        return self.c_predict(
            self.board_encoder.encode_board_1_fen(
                board_1_fen
            ).reshape(1, -1)
        )[0]

    def c_predict_fen_n(self, board_n_fen: Union[List[str], Tuple[str]]) -> torch.Tensor:
        return self.c_predict(
            self.board_encoder.encode_board_n(
                [chess.Board(i) for i in board_n_fen]
            )
        )


class TorchModels:
    @staticmethod
    def model_001(device):
        """
        Inputs = 778

        Outputs = 1

        :param device:
        :return:
        """

        model_layers = [
            torch.nn.Linear(778, 512),  # Input layer
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),

            torch.nn.Linear(512, 512),  # Layer 1
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),

            torch.nn.Linear(512, 512),  # Layer 2
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),

            torch.nn.Linear(512, 512),  # Layer 3
            torch.nn.ReLU(),

            torch.nn.Linear(512, 1),  # Layer 4 and Output layer
            torch.nn.Sigmoid()
        ]

        # Build a feed-forward network
        model = torch.nn.Sequential(*model_layers).to(device=device)

        return model

    # @staticmethod
    # def model_002(device):
    #     model =
    #     model = model.to(device)
    #     criterion = [torch.nn.MSELoss(reduction='mean').to(device)]
    #     optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.decay)


#########################################################################################################################
# NOTE: `c` before each method name means that it is custom
# Feed Forward Neural Network - Keras
class FFNNKeras:
    def __init__(self, model_generator, board_encoder, model_save_path: str = "../chess_models/", generate_model_image=False):
        self.model: keras.Sequential = model_generator()
        self.board_encoder = board_encoder
        self.model_save_path = model_save_path
        # self.model_save_path_dir = str(Path(model_save_path).parent)

        # CREATE a callback that saves the model's weights
        self.cp_callback = keras.callbacks.ModelCheckpoint(filepath=str(Path(self.model_save_path) / "weights.{epoch:06d}-{val_loss:.2f}.hdf5"),
                                                           save_best_only=True,
                                                           save_weights_only=True,
                                                           verbose=1)

        # PRINT model summary
        # self.model.summary()

        # SAVE the model graph
        if generate_model_image:
            for i in range(1, 100):
                if not Path(f'FFNNKeras_{i:03}.png').exists():
                    keras.utils.plot_model(self.model, f'FFNNKeras_{i:03}.png', show_shapes=True)
                    print(f"Saving the image: 'FFNNKeras_{i:03}.pgn'")
                    break
        return

    # TODO: to fix this, probably this may not be saving the model correctly/properly due to some parameters or version compatibility problems
    # REFER: https://github.com/tensorflow/tensorflow/issues/28281
    def c_save_model(self, model_name: str, model_path: Union[str, Path] = "../chess_models"):
        Path(model_path).mkdir(parents=True, exist_ok=True)
        self.model.save(str(Path(model_path) / model_name), overwrite=True)
        return

    def c_save_weights(self, model_name: str, model_path: Union[str, Path] = "../chess_models"):
        Path(model_path).mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(Path(model_path) / model_name), overwrite=True)
        return

    # TODO: to fix this, not working
    # REFER: https://github.com/tensorflow/tensorflow/issues/28281
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
        self.model.trainable = True

        # compile the keras model
        # self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])  # mae = Mean Absolute Error
        with tf.device('/gpu:0'):
            self.model.fit(x_input, y_output, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
                           verbose=2, workers=multiprocessing.cpu_count(), use_multiprocessing=True,
                           callbacks=[self.cp_callback])

        self.model.trainable = False
        return

    def c_evaluate_model(self, x_input_test: np.ndarray, y_output_test: np.ndarray, verbose=2):
        loss, mae = self.model.evaluate(x_input_test, y_output_test, verbose=verbose)
        print(f"Evaluated: Loss = {loss:5.3f}")
        print(f"Evaluated: MAE = {mae:5.3f}")

    def c_predict(self, encoded_board: np.ndarray) -> np.ndarray:
        """
        Takes a 2D np.ndarray where each row is a chess board made up of the
        floating point numbers using methods of class `self.board_encoder`
        :param encoded_board:
        :return: np.ndarray - 1D
        """
        return self.model.predict(
            encoded_board,
            verbose=1, workers=multiprocessing.cpu_count(), use_multiprocessing=True
        ).ravel()

    def c_predict_board_1(self, board_1: chess.Board) -> np.float32:
        return self.c_predict(
            self.board_encoder.encode_board_1(
                board_1
            ).reshape(1, -1)
        )[0]

    def c_predict_board_n(self, board_n: Union[List[chess.Board], Tuple[chess.Board]]) -> np.ndarray:
        return self.c_predict(
            self.board_encoder.encode_board_n(
                board_n
            )
        )

    def c_predict_fen_1(self, board_1_fen: str) -> np.float32:
        return self.c_predict(
            self.board_encoder.encode_board_1_fen(
                board_1_fen
            ).reshape(1, -1)
        )[0]

    def c_predict_fen_n(self, board_n_fen: Union[List[str], Tuple[str]]) -> np.ndarray:
        return self.c_predict(
            self.board_encoder.encode_board_n(
                [chess.Board(i) for i in board_n_fen]
            )
        )


class KerasModels:

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

    @staticmethod
    def model_001() -> keras.Sequential:
        # define the keras model
        model: keras.Sequential = keras.Sequential()
        model.add(keras.layers.Dense(256, activation='relu', input_shape=(778,)))
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dense(1, activation='tanh'))

        # compile the keras model
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])  # mae = Mean Absolute Error
        return model

    @staticmethod
    def model_002() -> keras.Sequential:
        # define the keras model
        model: keras.Sequential = keras.Sequential()
        model.add(keras.layers.Dense(512, activation='relu', input_shape=(778,)))
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(1, activation='tanh'))

        # compile the keras model
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])  # mae = Mean Absolute Error
        return model

    @staticmethod
    def model_003() -> keras.Sequential:
        # define the keras model
        model: keras.Sequential = keras.Sequential()
        model.add(keras.layers.Dense(1024, activation='relu', input_shape=(778,)))
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dropout(rate=0.1))
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dropout(rate=0.1))
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dropout(rate=0.1))
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dense(1, activation='tanh'))

        # compile the keras model
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])  # mae = Mean Absolute Error
        return model

    @staticmethod
    def model_004():
        inputs = keras.Input(shape=(778,), name='Encoded Chess Board')
        x = keras.layers.Dense(256, activation='relu')(inputs)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        outputs = keras.layers.Dense(1, activation='tanh')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='chess_778_model_v004')
        return model


#########################################################################################################################
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# GET list of devices
get_available_gpus()
# list all local devices
device_lib.list_local_devices()

# TRAINING on GPU
# sess = tensorflow.compat.v1.Session(config=tensorflow.compat.v1.ConfigProto(log_device_placement=True))
if __name__ == '__main__':
    # DATA-SET loading
    with cs.ExecutionTime():
        data = pd.read_csv("out_combined_KingBase2019-B00-B19_000000.csv")
        data_x = data[cs.COLUMNS[0]].values
        data_y = data[cs.COLUMNS[1]].values
    with cs.ExecutionTime():
        data_x_encoded = step_02.BoardEncoder.Encode778.encode_board_n_fen(data_x)
    with cs.ExecutionTime():
        data_y = step_02.ScoreNormalizer.normalize_001(data_y)

    # MODEL creation and training
    tensorflow.device("/gpu:0")
    ffnn_keras = FFNNKeras(KerasModels.model_001, board_encoder=step_02.BoardEncoder.Encode778)
    ffnn_keras.c_load_weights("ffnn_keras_v004_000010_weights.h5")
    with cs.ExecutionTime():
        ffnn_keras.c_train_model(data_x_encoded, data_y, 1000, 512, 0.2)
    ffnn_keras.c_save_weights("ffnn_keras_v004_000010_weights.h5")

    # MODEL testing
    with cs.ExecutionTime():
        y_predicted = ffnn_keras.c_predict_fen_n(data_x)
        print(f"MAE = {np.sum(np.abs(y_predicted - data_y) / len(y_predicted))}")
    with cs.ExecutionTime():
        ffnn_keras.c_evaluate_model(data_x_encoded, data_y)
