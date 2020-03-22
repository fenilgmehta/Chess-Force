# COMMANDS

BASE_PATH="/home/student/Desktop/fenil_pc/aggregated output 03/kingbase2019"

# BASE_PATH="/home/student/Desktop/fenil_pc/aggregated output 03/kaufman"

PGN_PATH="${BASE_PATH}/01_pgn_data"
CSV_PATH="${BASE_PATH}/02_csv_data"
CSV_SCORES_PATH="${BASE_PATH}/03_csv_score_data"
CSV_SCORES_PATH_CONVERTED="${BASE_PATH}/03_csv_score_data_converted"
PKL_PATH="${BASE_PATH}/04_pkl_data"
PKL_PATH_TRAINED="${BASE_PATH}/04_pkl_data_trained"



#------------------------

python step_02_preprocess.py convert_fen_to_pkl_folder   \
    --input_dir "${CSV_SCORES_PATH}"                     \
    --output_dir "${PKL_PATH}"                           \
    --move_dir "${CSV_SCORES_PATH_CONVERTED}"            \
    --board_encoder 'BoardEncoder.Encode778'             \
    --score_normalizer 'None'                            \
    --suffix_to_append '-be00778.pkl'                    \
    && mv "${CSV_SCORES_PATH_CONVERTED}/"*.csv "${CSV_SCORES_PATH}"

#------------------------

SAVED_WEIGHTS_FILE="../../Chess-Kesari-Models/ffnn_keras-mg005-be00778-sn003-ep00001-weights-v001.h5"
MODEL_NAME_PREFIX="ffnn_keras"
EPOCHS=1

# SAVED_WEIGHTS_FILE="../../Chess-Kesari-Models/kaufman-mg005-be00778-sn003-ep00001-weights-v001.h5"
# MODEL_NAME_PREFIX="kaufman"
# EPOCHS=512

python step_03b_train.py                                \
    --action train                                      \
    --gpu_id=0                                          \
    --model_version=5                                   \
    --version_number=1                                  \
    --epochs=${EPOCHS}                                  \
    --batch_size=65536                                  \
    --validation_split=0.2                              \
    --generate_model_image=False                        \
    --input_dir="${PKL_PATH}"                           \
    --move_dir="${PKL_PATH_TRAINED}"                    \
    --file_suffix="pkl"                                 \
    --y_normalizer="normalize_003"                      \
    --callback=False                                    \
    --name_prefix="${MODEL_NAME_PREFIX}"                \
    --auto_load_new=True                                \
    --saved_weights_file="${SAVED_WEIGHTS_FILE}"        \
    && mv "${PKL_PATH_TRAINED}/"*.pkl "${PKL_PATH}"

#------------------------

CSV_kaufman="${BASE_PATH}/05_kaufman"
CSV_kaufman_move="${BASE_PATH}/05_kaufman_processed"
CSV_kaufman_out="${BASE_PATH}/05_kaufman_score"

python step_04_play.py                                  \
    predict_move                                        \
    --input_dir="${CSV_kaufman}"                        \
    --move_dir="${CSV_kaufman_move}"                    \
    --output_dir="${CSV_kaufman_out}"                   \
    --model_weights_file='../../Chess-Kesari-Models/kaufman-mg005-be00778-sn003-ep01024-weights-v001.h5'
