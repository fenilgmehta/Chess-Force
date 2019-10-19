import os
import pickle
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.neural_network import MLPRegressor
from tensorflow.keras import layers

import TicTacToe.step_01_TicTacToe as step_01


# https://www.ritchieng.com/machine-learning-evaluate-linear-regression-model/
def print_model_stats(y_true, y_prediction):
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # calculate MAE, MSE, RMSE
    print(f"Mean absolute error    = {mean_absolute_error(y_true, y_pred)}")
    print(f"Mean square error      = {mean_squared_error(y_true, y_pred)}")
    print(f"Root mean square error = {np.sqrt(mean_squared_error(y_true, y_pred))}")


# Feed Forward Neural Network - Tensorflow
class FFNN_tf:
    def __init__(self):
        self.model = tf.keras.Sequential()
        self.model.add(layers.Dense(20, activation='tanh'))
        self.model.add(layers.Dense(256, activation='tanh'))
        self.model.add(layers.Dense(256, activation='tanh'))
        self.model.add(layers.Dense(256, activation='tanh'))
        self.model.add(layers.Dense(256, activation='tanh'))
        self.model.add(layers.Dense(1, activation='tanh'))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        # loss = 'categorical_crossentropy',
        # metrics = ['accuracy']

        # loss = 'mse',
        # metrics = ['mae']

    # TODO
    def save_model(self, model_path: Union[str, Path] = "TicTacToe_model.h5"):
        return

    # TODO
    def load_model(self, model_path: Union[str, Path] = "TicTacToe_model.h5"):
        return

    def train_model(self, x_input, y_output, epochs=400, batch_size=1024):
        self.model.fit(x_input, y_output, epochs=epochs, batch_size=batch_size)
        # results = cross_validate(self.model, x_input, y_output, cv=10, n_jobs=-1)
        # print(f"\n\nDEBUG: training results\n{results}")

    def predict(self, board) -> np.float32:
        return self.model.predict(board)

    def predict_score(self, board: step_01.TicTacToe) -> np.float32:
        return self.model.predict(board.encode())


# Feed Forward Neural Network - sklearn
class FFNN_sklearn:
    def __init__(self, hidden_layer_sizes=(100,),
                 activation="relu",  # relu, tanh, logistic,
                 solver="adam",
                 learning_rate="constant",  # adaptive
                 learning_rate_init=0.001,
                 max_iter=200,
                 n_iter_no_change=10,
                 tol=0.0001,
                 random_state=None,
                 verbose=False):
        self.model: MLPRegressor = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                                activation=activation,  # relu, tanh, logistic,
                                                solver=solver,
                                                learning_rate=learning_rate,
                                                learning_rate_init=learning_rate_init,
                                                max_iter=max_iter,
                                                n_iter_no_change=n_iter_no_change,
                                                tol=tol,
                                                random_state=random_state,
                                                verbose=verbose)

    # https://www.thoughtco.com/using-pickle-to-save-objects-2813661
    def save_model(self, model_path: Union[str, Path] = 'model_ffnn_sklearn.obj'):
        filehandler_sk = open(str(model_path), 'wb')  # file handler to save and load object
        pickle.dump(self.model, filehandler_sk)  # save object

    def load_model(self, model_path: Union[str, Path] = "model_ffnn_sklearn.obj"):
        if not Path(model_path).exists():
            raise Exception(f"ERROR: Model FileNotFound: {model_path}")

        filehandler_sk = open(str(model_path), 'rb')  # file handler to save and load object
        self.model = pickle.load(filehandler_sk)  # load object

    def train_model(self, x_input, y_output):
        self.model.fit(x_input, y_output)
        # results = cross_validate(self.model, x_input, y_output, cv=10, n_jobs=-1)
        # print(f"\n\nDEBUG: training results\n{results}")

    def predict(self, board) -> np.float32:
        return self.model.predict(board)

    def predict_score(self, board: step_01.TicTacToe) -> np.float32:
        return self.model.predict(board.encode())


def read_and_process_dataset():
    dataset = pd.read_csv("TicTacToe_dataset.csv")
    dataset_processed = pd.DataFrame(data=len(dataset) * [20 * [None]])
    for i in range(len(dataset)):
        dataset_processed.loc[i][:-1] = [int(j) for j in dataset.loc[i][step_01.COLUMN_NAMES[0]].split(" ")]
        dataset_processed.loc[i][19] = dataset.loc[i][step_01.COLUMN_NAMES[1]]
    dataset_processed[19] /= 18
    return dataset_processed


if __name__ == "__main__":
    os.chdir("TicTacToe")

    # dataset_processed = read_and_process_dataset()
    # dataset_processed.to_csv("TicTacToe_dataset_processed.csv", index=False)
    dataset_processed = pd.read_csv("TicTacToe_dataset_processed.csv", dtype=np.float)
    x_input, y_output = dataset_processed.iloc[:, :-1].values, dataset_processed.iloc[:, 19].values

    #####################################################################
    ffnn_tf = FFNN_tf()
    ffnn_tf.train_model(x_input, y_output, 400, 4096)
    ffnn_tf.predict(board=dataset_processed.iloc[1][:-1].values)

    filehandler_tf = open('model_ffnn_tf.obj', 'w')  # file handler to save and load object
    pickle.dump(ffnn_tf, filehandler_tf)  # save object
    ffnn_tf = pickle.load(filehandler_tf)  # load object

    #####################################################################
    ffnn_sk = FFNN_sklearn(hidden_layer_sizes=(256, 256, 256, 256,),
                           activation="relu",  # relu, tanh, logistic,
                           solver="adam",
                           learning_rate="adaptive",
                           learning_rate_init=0.0001,
                           max_iter=512,
                           n_iter_no_change=20,
                           tol=0.000001,
                           random_state=2,
                           verbose=True)
    ffnn_sk.train_model(x_input, y_output)
    y_pred = ffnn_sk.predict(x_input)
    print_model_stats(y_output, y_pred)
