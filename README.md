# Chess-Force

Neural Network based chess AI

# Steps for training a model
1. Initialize the folder names
    ```bash
    # NOTE: run the following commands from the `code` directory
    DATA_PATH="../../aggregated output 03/kaufman (copy)"
    PGN_PATH="${DATA_PATH}/01_pgn_data"
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
    SAVED_WEIGHTS_FILE="../../Chess-Kesari-Models/ffnn_keras-mg005-be00778-sn003-ep00005-weights-v031.h5"
    MODEL_NAME_PREFIX="ffnn_keras"
    EPOCHS=8
    WEIGHTS_SAVE_PATH="../../Chess-Kesari-Models"
    
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
        MODEL_WEIGHTS_FILE="/home/student/Desktop/fenil_pc/Chess-Kesari-Models/ffnn_keras-mg005-be00778-sn003-ep00127-weights-v032.h5"
      
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
        MODEL_WEIGHTS_FILE="/home/student/Desktop/fenil/35_Final Year Project/Chess-Kesari-Models/ffnn_keras-mg005-be00778-sn003-ep00127-weights-v032.h5"

        python step_04_play.py                              \
            predict_move                                    \
            --input_dir=PATH                                \
            --output_dir=PATH                               \
            --move_dir=PATH                                 \
            --model_weights_file="${MODEL_WEIGHTS_FILE}"
        ```
 

# Usage
* `python step_02_preprocess.py --help`
* `python step_02a_preprocess_server_dispy.py --help`
* `python step_03b_train.py --help`
* `python step_04_play.py --help`
