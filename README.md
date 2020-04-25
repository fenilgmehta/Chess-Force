# Chess-Force

Neural Network based chess AI
- [Usage](https://github.com/fenilgmehta/Chess-Force#usage)
    * [Suggested steps to install all the prerequisites](https://github.com/fenilgmehta/Chess-Force#suggested-steps-to-install-all-the-prerequisites)
    * [Steps for training and using a model](https://github.com/fenilgmehta/Chess-Force#steps-for-training-and-using-a-model)
- [Usage help and documentation](https://github.com/fenilgmehta/Chess-Force#usage-help-and-documentation)
- [Architecture diagram](https://github.com/fenilgmehta/Chess-Force#architecture-diagram)

## Usage

#### Suggested steps to install all the prerequisites
```bash
# Open bash and run
cd ~/Desktop
git clone https://github.com/fenilgmehta/Chess-Force
cd Chess-Force

# Install miniconda3
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -bfu # b is batch mode, f is no error if already installed, u is update existing installation

# Initialize miniconda3
PREFIX=$HOME/miniconda3
$PREFIX/bin/conda init

# Create new environment
conda create -y -n fm_chess python=3.7.7

# NOTE: it is highly suggested that a new terminal be opened and then proceeded
conda activate fm_chess

# Install library dependencies
cd ~/Desktop/Chess-Force
pip install -r requirements.txt
```

#### Steps for training and using a model
###### NOTE: run the following commands from the `Chess-Force/code` directory
1. Initialize the folder names
    ```bash
    # Activate the correct environment
    # conda activate fm_chess
    
    # Goto the directory with source code
    # cd ~/Desktop/Chess-Force/code

    DATA_PATH="../../aggregated output 03/kaufman (copy)"   # EDIT this so that it points to the directory used for storing all the PGN/CSV/PKL data files used for training/testing/playing
    PGN_PATH="${DATA_PATH}/01_pgn_data"                     # Paste the PGN files inside folder
    CSV_PATH="${DATA_PATH}/02_csv_data"
    CSV_SCORES_PATH="${DATA_PATH}/03_csv_score_data"
    CSV_SCORES_PATH_CONVERTED="${DATA_PATH}/03_csv_score_data_converted"
    PKL_PATH="${DATA_PATH}/04_pkl_data"
    PKL_PATH_TRAINED="${DATA_PATH}/04_pkl_data_trained"
    ```
2. Convert chess board positions in PGN games to FEN notation and save them to CSV file. 
    ```bash
    python step_02_preprocess.py pgn_to_csv_parallel        \
        --input_dir="${PGN_PATH}"                           \
        --output_dir="${CSV_PATH}"
    ```

3.  * Setup worker nodes for computing the CentiPawn scores
        ```bash
        # Activate the correct Python environment and run the following commands
        # conda activate fm_chess

        # Install dispy library for distributed computing
        pip install dispy==4.11.1
        
        # Run this on each worker so that it is ready to receive work from the master
        dispynode.py --debug --zombie_interval=1 --clean --cpus=1  
        ```

    * Run the following on master node to generate CentiPawn score for the boards generated in step 2
        ```bash
        # Install dispy on the worker nodes and run 
        python step_02a_preprocess_server_dispy.py          \
            --input_dir="${CSV_PATH}"                       \
            --output_dir="${CSV_SCORES_PATH}"               \
            --host_ip="192.168.0.106"                       \
            --worker_processes="-1"                         \
            --jobs_per_process=30
        ```

4. Convert the CSV files with chess boards in FEN notation and CentiPawn score to `pkl` files.
   
   This is done because converting chess board from FEN notation to bit notation is time consuming and costly if performed repeatedly.

   ```bash
   python step_02_preprocess.py convert_fen_to_pkl_folder   \
       --input_dir "${CSV_SCORES_PATH}"                     \
       --output_dir "${PKL_PATH}"                           \
       --move_dir "${CSV_SCORES_PATH_CONVERTED}"            \
       --board_encoder 'BoardEncoder.Encode778'             \
       --score_normalizer 'None'                            \
       --suffix_to_append '-be00778.pkl'                    \
       && mv "${CSV_SCORES_PATH_CONVERTED}/"*.csv "${CSV_SCORES_PATH}"
   ```
   
5. Use the `pkl` files for training the ANN model
    ```bash
    SAVED_WEIGHTS_FILE="../../Chess-Force-Models/ffnn_keras-mg005-be00778-sn003-ep00005-weights-v031.h5"
    MODEL_NAME_PREFIX="ffnn_keras"
    EPOCHS=8
    WEIGHTS_SAVE_PATH="../../Chess-Force-Models"
    
    python step_03b_train.py train                          \
        --gpu=-1                                            \
        --model_version=5                                   \
        --version_number=1                                  \
        --epochs=${EPOCHS}                                  \
        --batch_size=8192                                   \
        --validation_split=0.2                              \
        --generate_model_image                              \
        --input_dir="${PKL_PATH}"                           \
        --move_dir="${PKL_PATH_TRAINED}"                    \
        --file_suffix="pkl"                                 \
        --y_normalizer="normalize_003"                      \
        --callback                                          \
        --name_prefix="${MODEL_NAME_PREFIX}"                \
        --auto_load_new                                     \
        --saved_weights_file="${SAVED_WEIGHTS_FILE}"        \
        --weights_save_path="${WEIGHTS_SAVE_PATH}"          \
        && mv "${PKL_PATH_TRAINED}/"*.pkl "${PKL_PATH}"
    ```

6.  * Play the game
        ```bash
        MODEL_WEIGHTS_FILE="/home/student/Desktop/Chess-Force-Models/ffnn_keras-mg005-be00778-sn003-ep00127-weights-v032.h5"
        
        python step_04_play.py                              \
            play                                            \
            --game_type=mm                                  \
            --model_weights_file="${MODEL_WEIGHTS_FILE}"    \
            --analyze_game                                  \
            --clear_screen                                  \
            --delay=1.0
        ```

    * Predict move and score for chess boards stored in a CSV file
        ```bash
        MODEL_WEIGHTS_FILE="/home/student/Desktop/Chess-Force-Models/ffnn_keras-mg005-be00778-sn003-ep00127-weights-v032.h5"

        # NOTE: edit the input_dir, output_dir and move_dir before running the command
        python step_04_play.py                              \
            predict_move                                    \
            --input_dir=PATH                                \
            --output_dir=PATH                               \
            --move_dir=PATH                                 \
            --model_weights_file="${MODEL_WEIGHTS_FILE}"
        ```

    * Iterate through the moves list
        ```bash
        python step_04_play.py iterate_moves --moves="['e2e4', 'd7d5', 'e4d5', 'd8d5', 'g1f3', 'g8f6', 'd2d4', 'd5e4', 'f1e2', 'c8f5', 'e1g1', 'e4c2', 'b1c3', 'c2d1', 'g2g4', 'd1c2', 'c3d5', 'f5g4', 'd5f6', 'g7f6', 'a2a4', 'c2e2', 'c1e3', 'g4f3', 'f1b1', 'h8g8', 'e3g5', 'g8g5']"       \
        --analyze_game                                     \
        --clear_screen                                     \
        --delay=1.0
        ```

## Usage help and documentation
* `python step_02_preprocess.py --help`
* `python step_02a_preprocess_server_dispy.py --help`
* `python step_03b_train.py --help`
* `python step_04_play.py --help`


## Architecture diagram
![Architecture diagram](images/Architecture%20Diagram.jpg "Architecture diagram")
