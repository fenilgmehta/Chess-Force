import argparse
import gc
import os
import random
import shutil
import time
import warnings
from pathlib import Path
from typing import Union, List, Callable

import joblib
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

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

# WARNING/ERROR: numpy FutureWarning
# SOLUTION: https://github.com/tensorflow/tensorflow/issues/30427
import tensorflow as tf
from tensorflow.python.client import device_lib

import common_services as cs
import step_02_preprocess as step_02
import step_03a_ffnn as step_03a


########################################################################################################################


def train_on_file(keras_obj: step_03a.FFNNKeras, file_path: str, data_load_transform, y_normalizer, epochs, batch_size, validation_split):
    data_x_encoded, data_y_normalized = data_load_transform(file_path)
    if y_normalizer is not None:
        data_y_normalized = y_normalizer(data_y_normalized)

    keras_obj.c_train_model(x_input=data_x_encoded,
                            y_output=data_y_normalized,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=validation_split)
    del data_x_encoded, data_y_normalized


def train_on_folder(keras_obj: step_03a.FFNNKeras,
                    input_dir: str, move_dir: str, file_suffix: str,
                    data_load_transform, y_normalizer,
                    epochs: int, batch_size: int, validation_split: float):
    Path(move_dir).mkdir(parents=True, exist_ok=True)
    if not Path(input_dir).exists():
        raise FileNotFoundError(f"Source path does not exists: '{input_dir}'")
    if not Path(move_dir).exists():
        raise FileNotFoundError(f"Destination path does not exists: '{move_dir}'")

    # NOTE: glob uses bash like expression expansion, i.e. `*` => any string of any length
    training_files = sorted(Path(input_dir).glob(f"*{file_suffix}"))
    # The order of training files is shuffled randomly so that the model does not get biased
    random.shuffle(training_files)
    with tqdm(training_files) as t:
        print(f"Input files = {len(training_files)}")
        print(f"Processed files = {len(list(Path(move_dir).glob(f'*{file_suffix}')))}")
        if len(list(Path(input_dir).glob("*"))) > 0 and len(list(Path(move_dir).glob("*"))) == 0:
            keras_obj.model_version.epochs += epochs
            keras_obj.model_version.version = 0

        for ith_file in t:
            t.set_description(desc=f"File: {Path(ith_file).name}", refresh=True)
            train_on_file(keras_obj=keras_obj,
                          file_path=str(ith_file),
                          data_load_transform=data_load_transform,
                          y_normalizer=y_normalizer,
                          epochs=epochs,
                          batch_size=batch_size,
                          validation_split=validation_split)
            # keras_obj.update_version()
            keras_obj.c_save_weights(write_name='z_last_model_weight_name.txt')
            shutil.move(src=str(Path(input_dir) / Path(ith_file).name), dst=str(Path(move_dir)))

            gc.collect()
            time.sleep(5)


def get_available_gpus() -> List:
    """
    Available GPU's device names

    :return:
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


########################################################################################################################

def train(gpu_id: int,
          model_version: int,
          version_number: int,
          epochs: int,
          batch_size: int,
          validation_split: float,
          generate_model_image: bool,
          input_dir: str,
          move_dir: str,
          file_suffix: str,
          y_normalizer: str,
          callback: bool,
          name_prefix: str,
          auto_load_new: bool,
          saved_weights_file: str = '',
          weights_save_path: str = '') -> None:
    """
    Train a Keras model
    :param gpu_id: ID of the GPU to be used for training [0,1,...], use -1 if CPU is to be used for training
    :param model_version: Which FFNNBuilder keras model to load
    :param version_number: Set the version number of the model loaded/created
    :param epochs: Number of epochs to be executed on each file
    :param batch_size: Number of rows to be used at one time for weight adjustment, like one of these
                       [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    :param validation_split: What fraction of data to be used for validation of the trained model
    :param generate_model_image: Whether to write the model structure to a pgn file or not ?
    :param input_dir: Path to the directory where input data is stored
    :param move_dir: Path to the directory where processed files are to be moved so that training can be resumed if interrupted
    :param file_suffix: Suffix of the input files, i.e. file extension ['csv', 'pkl']
    :param y_normalizer: 'None' is expected if output(y) is to be used as it is, otherwise use
                          one of ['normalize_001', 'normalize_002'] to normalize the expected output(y)
    :param callback: Whether to save intermediate weights if the results have improved
    :param name_prefix: Prefix to be used while saving the new ANN weights in h5 file
    :param auto_load_new: Whether to automatically load the last saved model or not if `name_prefix` is same ? This overrides `saved_weights_file` option
    :param saved_weights_file: If this file exists, then load it, else create the model with random weights
    :param weights_save_path: Path to directory where new weights of the model shall be saved
    :return: None
    """
    Path(move_dir).mkdir(parents=True, exist_ok=True)
    if not (4 <= model_version <= 5):
        print(f"ERROR: Invalid `model_version={model_version}`, model_version should be `4 <= model_version <= 5`")
        return
    if not (0 <= version_number):
        print(f"ERROR: Invalid `version_number={version_number}`, it should be `0 <= version_number`")
        return
    if not (1 <= epochs):
        print(f"ERROR: Invalid `epochs={epochs}`, it should be `1 <= epochs`")
        return
    if not (1 <= batch_size):
        print(f"ERROR: Invalid `batch_size={batch_size}`, it should be `1 <= batch_size`")
        return
    if not (0.0 <= validation_split <= 1.0):
        print(f"ERROR: Invalid `validation_split={validation_split}`, it should be `0.0 <= validation_split <= 1.0`")
        return
    if not Path(input_dir).exists():
        print(f"ERROR: `input_folder={input_dir}` does NOT exists")
        return
    if not Path(move_dir).exists():
        print(f"ERROR: `output_folder={move_dir}` does NOT exists")
        return

    data_load_transform = None
    y_normalizer_obj: Union[Callable[[np.ndarray], np.ndarray], None] = None

    if file_suffix == 'csv':
        data_load_transform = pd.read_csv
    elif file_suffix == 'pkl':
        data_load_transform = joblib.load
    else:
        print(f"ERROR: only 'csv' and 'pkl' file can be used to read/load data")
        return

    if 0 <= gpu_id < len(get_available_gpus()):
        # NOTE: IMPORTANT: TRAINING on GPU ***
        # tf.device("/gpu:0")
        tf.device(f"/gpu:{gpu_id}")
    elif gpu_id != -1:
        print(f"WARNING: Invalid parameter for `gpu_id={gpu_id}`, using CPU for training")

    if y_normalizer == "normalize_001":
        y_normalizer_obj = step_02.ScoreNormalizer.normalize_001
    elif y_normalizer == "normalize_002":
        y_normalizer_obj = step_02.ScoreNormalizer.normalize_002
    elif y_normalizer == "normalize_003":
        y_normalizer_obj = step_02.ScoreNormalizer.normalize_003
    elif y_normalizer != "None" and not (y_normalizer is None):
        print(type(y_normalizer))
        print(f"WARNING: Invalid parameter for `y_normalizer={y_normalizer}`, using default value `y_normalizer=None`")

    ffnn_keras_obj: Union[step_03a.FFNNKeras, None] = None
    if model_version == 4:
        ffnn_keras_obj = step_03a.FFNNBuilder.build_004(name_prefix=name_prefix, version=version_number, callback=callback,
                                                        generate_model_image=generate_model_image)
    elif model_version == 5:
        ffnn_keras_obj = step_03a.FFNNBuilder.build_005(name_prefix=name_prefix, version=version_number, callback=callback,
                                                        generate_model_image=generate_model_image)
    else:
        print(f"ERROR: Invalid `model_version={model_version}`")
        return
    ffnn_keras_obj.model_save_path = weights_save_path

    saved_weights_file = Path(saved_weights_file)
    if auto_load_new and (saved_weights_file.parent / 'z_last_model_weight_name.txt').exists():
        print(f"INFO: auto_load_new: Trying")
        last_saved_file_name = eval(open(str(saved_weights_file.parent / 'z_last_model_weight_name.txt'), 'r').read().strip())
        # Path(params.saved_weights_file).parent /
        try:
            if name_prefix in last_saved_file_name.keys():
                saved_weights_file = saved_weights_file.parent / last_saved_file_name[name_prefix]
        except Exception as e:
            print(f"ERROR: auto_load_new: {e}")

    if saved_weights_file.exists() and saved_weights_file.is_file():
        print(f"INFO: auto_load_new: loading: '{saved_weights_file}'")
        ffnn_keras_obj.c_load_weights(str(saved_weights_file.name), str(saved_weights_file.parent))
        print("INFO: auto_load_new: Model loaded :)")
    else:
        print("WARNING: auto_load_new: failed")
    print()

    train_on_folder(ffnn_keras_obj,
                    input_dir,
                    move_dir,
                    file_suffix,
                    data_load_transform=data_load_transform,
                    y_normalizer=y_normalizer_obj,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split)


# def main_training():
#     ffnn_keras_v4 = step_03a.FFNNKeras(step_03a.KerasModels.model_004, step_02.BoardEncoder.Encode778, step_02.ScoreNormalizer.normalize_002,
#                                        step_03a.ModelVersion("ffnn_keras", 4, 778, 2, 10, "weights", version=6), callback=False, generate_model_image=False)
#     # ffnn_keras_v4 = step_03a.FFNNBuilder.model_005(version=1, generate_model_image=True)
#     # ffnn_keras_v4 = step_03a.FFNNBuilder.model_005(version=1, generate_model_image=True)
#
#     ffnn_keras_v4.c_load_weights("ffnn_keras-mg004-be00778-sn002-ep00010-weightsv073.h5")
#     ffnn_keras_v4.update_version(74)
#     train_on_folder(
#         ffnn_keras_v4,
#         input_dir="../../aggregated output 03/be00778-sn002-pkl",
#         move_dir="../../aggregated output 03/be00778-sn002-pkl_trained_data",
#         file_suffix=".pkl",
#         data_load_transform=joblib.load,
#         epochs=10, batch_size=8192, validation_split=0.1
#     )  # batch_size=16384, 32768, 65536, 131072 causes resource exhaustion
#
#     # ffnn_keras_v4.c_save_weights()
#     # ffnn_keras_v4.c_save_model(ffnn_keras_v4.model_version.__str__())
#     # ffnn_keras_v4.c_load_weights("ffnn_keras_v005_000010_weights.h5")
#     # ffnn_keras_v4.c_load_model("ffnn_keras_v005_000010_model.h5")
#

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    from docopt import docopt

    doc_string = '''
    Usage:
        step_03b_train.py get_available_gpus
        step_03b_train.py train \
--gpu=N --model_version=N [--version_number=N] --epochs=N --batch_size=N --validation_split=FLOAT [--generate_model_image] \
--input_dir=PATH --move_dir=PATH --file_suffix=STR --y_normalizer=STR \
[--callback] --name_prefix=STR [--auto_load_new] --saved_weights_file=PATH \
--weights_save_path=PATH
        step_03b_train.py (-h | --help)
        step_03b_train.py --version

    Options:
        --gpu=N                     The GPU to use for training. By default CPU is used [default: -1]
        --model_version=N           Which FFNNBuilder keras model to load
        --version_number=N          Set the version number of the model loaded/created
        --epochs=N                  Number of epochs to be executed on each file
        --batch_size=N              Number of rows to be used at one time for weight adjustment, like one of these ---> 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072
        --validation_split=FLOAT    What fraction of data to be used for validation of the trained model
        --generate_model_image      Flag to decide whether to write the model structure to a pgn file or not ?
        --input_dir=PATH            Path to the directory where input data is stored
        --move_dir=PATH             Path to the directory where processed files are to be moved so that training can be resumed if interrupted
        --file_suffix=STR           Suffix of the input files, i.e. file extension like 'csv', 'pkl'
        --y_normalizer=STR          'None' is expected if output(y) is to be used as it is, otherwise use one of ['normalize_001', 'normalize_002'] to normalize the expected output(y)
        --callback                  Flag to decide whether to save intermediate weights if the results have improved
        --name_prefix=STR           Prefix to be used while saving the new ANN weights in h5 file
        --auto_load_new             Flag to decide whether to automatically load the last saved model or not if `name_prefix` is same ? This overrides the `saved_weights_file` option
        --saved_weights_file=PATH   If this file exists, then load it, else create the model with random weights
        --weights_save_path=PATH    Path to directory where new weights of the model shall be saved

        -h --help               Show this
        --version               Show version
    '''
    arguments = docopt(doc_string, argv=None, help=True, version=f"{cs.VERSION} - Training", options_first=False)

    print("\n\n", arguments, "\n\n", sep="")
    if arguments['get_available_gpus']:
        print(get_available_gpus())
    elif arguments['train']:
        train(int(arguments['--gpu']),
              int(arguments['--model_version']),
              int(arguments['--version_number']),
              int(arguments['--epochs']),
              int(arguments['--batch_size']),
              float(arguments['--validation_split']),
              arguments['--generate_model_image'],  # bool
              arguments['--input_dir'],
              arguments['--move_dir'],
              arguments['--file_suffix'],
              arguments['--y_normalizer'],
              arguments['--callback'],  # bool
              arguments['--name_prefix'],
              arguments['--auto_load_new'],  # bool
              arguments['--saved_weights_file'],
              arguments['--weights_save_path'])
        r'''
        python step_03b_train.py \
            --action train \
            --gpu_id=0 \
            --model_version=5 \
            --version_number=1 \
            --epochs=5 \
            --batch_size=8192 \
            --validation_split=0.2 \
            --generate_model_image \
            --input_dir="/home/student/Desktop/fenil_pc/aggregated output 03/be00778_new1_in" \
            --move_dir="/home/student/Desktop/fenil_pc/aggregated output 03/be00778_new1_out" \
            --file_suffix="pkl" \
            --y_normalizer="normalize_003" \
            --callback \
            --name_prefix="ffnn_keras"
            --saved_weights_file="/home/student/Desktop/fenil_pc/Chess-Kesari-Models/ffnn_keras-mg005-be00778-sn002-ep00010-weights-v003.h5" \
            --auto_load_new
        python step_03b_train.py --action train --gpu_id=0 --model_version=5 --version_number=1 --epochs=5 --batch_size=8192 --validation_split=0.2 --generate_model_image=False --input_dir="/home/student/Desktop/fenil_pc/aggregated output 03/be00778_new1_in" --move_dir="/home/student/Desktop/fenil_pc/aggregated output 03/be00778_new1_out" --file_suffix="pkl" --y_normalizer="normalize_003" --callback=False --saved_weights_file="/home/student/Desktop/fenil_pc/Chess-Kesari-Models/ffnn_keras-mg005-be00778-sn002-ep00010-weights-v003.h5" --auto_load_new=True
        '''
    else:
        print("ERROR: invalid option")

    # 10 epochs
    # 548437/548437 - 6s - loss: 0.0101 - mse: 0.0101 - mae: 0.0691 - val_loss: 0.0313 - val_mse: 0.0313 - val_mae: 0.1142
    # Model weights successfully saved: ffnn_keras-mg005-be00778-sn003-ep00010-weights-v031.h5

    # 15 epochs
    # 548437/548437 - 6s - loss: 0.0127 - mse: 0.0127 - mae: 0.0776 - val_loss: 0.0302 - val_mse: 0.0302 - val_mae: 0.1134
    # Model weights successfully saved: ffnn_keras-mg005-be00778-sn003-ep00015-weights-v095.h5

    # 25 epochs
    # 548437/548437 - 6s - loss: 0.0126 - mse: 0.0126 - mae: 0.0776 - val_loss: 0.0296 - val_mse: 0.0296 - val_mae: 0.1119
    # Model weights successfully saved: ffnn_keras-mg005-be00778-sn003-ep00025-weights-v159.h5

    # 26 epochs
    # 548437/548437 - 6s - loss: 0.0196 - mse: 0.0196 - mae: 0.0971 - val_loss: 0.0311 - val_mse: 0.0311 - val_mae: 0.1170
    # Model weights successfully saved: ffnn_keras-mg005-be00778-sn003-ep00026-weights-v223.h5

    # --------------------

    # 32 epochs
    # 7097/7097 - 0s - loss: 0.0345 - mse: 0.0345 - mae: 0.1442 - val_loss: 0.0397 - val_mse: 0.0397 - val_mae: 0.1541
    # Model weights successfully saved: kaufman-mg005-be00778-sn003-ep00032-weights-v001.h5

    # 64 epochs
    # 7097/7097 - 0s - loss: 0.0157 - mse: 0.0157 - mae: 0.0884 - val_loss: 0.0189 - val_mse: 0.0189 - val_mae: 0.0910
    # Model weights successfully saved: kaufman-mg005-be00778-sn003-ep00064-weights-v001.h5

    # 320 epochs
    # 7097/7097 - 0s - loss: 5.4061e-04 - mse: 5.4061e-04 - mae: 0.0176 - val_loss: 0.0124 - val_mse: 0.0124 - val_mae: 0.0709
    # Model weights successfully saved: kaufman-mg005-be00778-sn003-ep00320-weights-v001.h5

    # 512 epochs
    # 7097/7097 - 0s - loss: 4.4207e-04 - mse: 4.4207e-04 - mae: 0.0159 - val_loss: 0.0119 - val_mse: 0.0119 - val_mae: 0.0676
    # Model weights successfully saved: kaufman-mg005-be00778-sn003-ep00512-weights-v001.h5

    # 1024 epochs
    # 7097/7097 - 0s - loss: 1.7559e-04 - mse: 1.7559e-04 - mae: 0.0100 - val_loss: 0.0123 - val_mse: 0.0123 - val_mae: 0.0658
    # Model weights successfully saved: kaufman-mg005-be00778-sn003-ep01024-weights-v001.h5

    # # # GET list of devices
    # # get_available_gpus()
    # # # list all local devices
    # # device_lib.list_local_devices()
    # # # other command, effect not known
    # # sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
