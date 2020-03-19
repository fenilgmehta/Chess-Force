import collections.abc
import copy
import glob
import operator
import shutil
import time
from pathlib import Path
from typing import Tuple, List, Union, Any

import chess
# import chess.pgn
import numpy as np
import pandas as pd
from termcolor import colored
from tqdm.autonotebook import tqdm

import common_services as cs
import step_01_engine as step_01
import step_03a_ffnn as step_03a


########################################################################################################################
# This is for prediction
class ChessPredict:
    def __init__(self,
                 function_to_predict_1_board=None, function_to_predict_n_board=None, to_maximize=True,
                 predict_move_1=None,
                 analyse_board=None):
        if function_to_predict_1_board is not None:
            assert isinstance(function_to_predict_1_board, collections.abc.Callable), '`function_to_predict_1_board` must be callable'
        if function_to_predict_n_board is not None:
            assert isinstance(function_to_predict_n_board, collections.abc.Callable), '`function_to_predict_n_board` must be callable'
        assert isinstance(to_maximize, bool), '`predict_move_1` must be bool'
        if predict_move_1 is not None:
            assert isinstance(predict_move_1, collections.abc.Callable), '`predict_move_1` must be callable'

        self.function_to_predict_1 = function_to_predict_1_board
        self.function_to_predict_n = function_to_predict_n_board
        self.predict_move_1 = predict_move_1
        self.analyse_board = analyse_board
        self.to_maximize = to_maximize

        self.__initial_score = float(((-1) ** to_maximize) * (2 ** 30))
        self.__compare = None
        if self.to_maximize:
            self.__compare = operator.gt
        else:
            self.__compare = operator.lt

    def predict_score_1(self, board: chess.Board) -> np.float:
        """
        Returns the centi-pawn score for `board` it is given as the parameter.

        Note: may be performance can be impacted as model/engine is called repeatedly for each board state evaluation

        :return: np.float
        """
        return self.function_to_predict_1(board)

    def predict_score_n(self, boards: Union[List, Tuple, np.array]) -> np.array:
        """
        Returns centi-pawn score for all the boards passed to it.

        :return: np.array
        """
        return self.function_to_predict_n(boards)

    def predict_best_move_v1(self, board: chess.Board) -> Tuple[chess.Move, np.float]:
        """
        Generate all possible boards at next level from `board`.
        Predict score of all boards.

        Uses: predict_score_1

        Note: may be performance can be impacted as model/engine is called repeatedly for each board state evaluation

        :return: Tuple[chess.Move, np.float]
        """
        if self.function_to_predict_1 is None:
            raise NotImplementedError
        best_move = None
        best_score = self.__initial_score
        for i in board.legal_moves:
            board.push(i)
            possible_best_score = self.predict_score_1(board)
            if self.__compare(possible_best_score, best_score):
                best_score = possible_best_score
                best_move = copy.deepcopy(i)
            board.pop()
        return best_move, best_score

    def predict_best_move_v2(self, board: chess.Board) -> Tuple[chess.Move, np.float]:
        """
        Generate all possible boards at next level from `board`.
        Predict score of all boards.

        Uses: predict_score_n

        :param board:
        :return: Tuple[chess.Move, np.float]
        """
        if self.function_to_predict_n is None:
            raise NotImplementedError
        board_moves_list: List[chess.Move] = list(board.legal_moves)
        board_states_list: List[chess.Board] = []
        for i in board_moves_list:
            board.push(i)
            board_states_list.append(copy.deepcopy(board))
            board.pop()

        board_scores_list: np.ndarray = self.predict_score_n(board_states_list)
        if self.to_maximize:
            best_index = board_scores_list.argmax()
        else:
            best_index = board_scores_list.argmin()

        best_move: chess.Move = board_moves_list[best_index]
        best_score: float = board_scores_list[best_index]

        return best_move, best_score

    def predict_best_move_v3(self, board: chess.Board) -> Tuple[chess.Move, Any]:
        return self.predict_move_1(board), None

    def predict_best_move(self, board: chess.Board) -> Tuple[chess.Move, Any]:
        if self.predict_move_1 is not None:
            return self.predict_best_move_v3(board)
        if self.predict_score_n is not None:
            return self.predict_best_move_v2(board)
        if self.predict_score_1 is not None:
            return self.predict_best_move_v1(board)
        raise NotImplementedError(
            "Neither of the prediction methods are given to ChessPredict: predict_score_1, predict_score_n, predict_move_1")

    def get_next_move(self, board: chess.Board):
        if self.predict_move_1 is not None:
            return self.predict_move_1(board)
        if self.predict_score_n is not None:
            return self.predict_best_move_v2(board)[0]
        if self.predict_score_1 is not None:
            return self.predict_best_move_v1(board)[0]
        raise NotImplementedError(
            "Neither of the prediction methods are given to ChessPredict: predict_score_1, predict_score_n, predict_move_1")


########################################################################################################################
# This if for CLI game interface

class ChessPlayCLI:
    piece_to_cli = {
        'k': '\u2654',
        'q': '\u2655',
        'r': '\u2656',
        'b': '\u2657',
        'n': '\u2658',
        'p': '\u2659',
        'K': '\u265A',
        'Q': '\u265B',
        'R': '\u265C',
        'B': '\u265D',
        'N': '\u265E',
        'P': '\u265F',
    }

    def __init__(self,
                 player1_name: str = "Player 1",
                 player2_name: str = "Player 2",
                 player1_chess_predict: ChessPredict = None,
                 player2_chess_predict: ChessPredict = None,
                 game_analyzer: ChessPredict = None,
                 clear_screen: bool = False,
                 delay: float = 0.0):
        '''
        Note: `game_analyzer` should implement the method `analyse_board`
        '''
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.player1_chess_predict = player1_chess_predict
        self.player2_chess_predict = player2_chess_predict
        self.game_analyzer = game_analyzer
        self.clear_screen = clear_screen
        if not isinstance(delay, (int, float,)):
            delay = 0.0
        self.delay = delay

    @staticmethod
    def pretty_board(board: chess.Board, clear_screen) -> str:
        color_decoration = 'blue'
        color_white = 'white'
        color_black = 'cyan'
        res_out: str = board.__str__()
        res_out: List = res_out.split("\n")
        for i in range(len(res_out)):
            res_out[i]: List = res_out[i].split(' ')  # list(res_out[i])
            for j in range(len(res_out[i])):
                ch = res_out[i][j]
                if ch in ChessPlayCLI.piece_to_cli:
                    ch_is_upper = ch.isupper()
                    ch = ChessPlayCLI.piece_to_cli[res_out[i][j]]
                    ch = colored(ch, color_white if ch_is_upper else color_black)
                res_out[i][j] = ch
            res_out[i] = " ".join(res_out[i])
        for i in range(len(res_out)):
            res_out[i] = colored(f"[{8 - i}] ", color_decoration) + res_out[i]
        res_out.insert(0, colored(f"   [a|b|c|d|e|f|g|h]", color_decoration))
        res_out.insert(0, colored(f"\n### Board state number = {board.fullmove_number}\n", 'red', attrs=['bold', 'reverse']))
        flush_str = "\033c" if clear_screen else ''
        res_out = flush_str + "\n".join(res_out) + "\n"
        return res_out

    def user_play(self, board_play: chess.Board, is_player1: bool) -> chess.Move:
        player_name = self.player2_name
        player_color = "BLACK"
        player_example = "g7g5"
        if is_player1:
            player_name = self.player1_name
            player_color = "WHITE"
            player_example = "g2g4"
        mov = input(f"[{player_name}] please enter your move as you play [{player_color}] (eg: {player_example}): ")
        while True:
            try:
                while chess.Move.from_uci(mov) not in board_play.legal_moves:
                    mov = input("please enter a legal move: ")
                break
            except ValueError:
                mov = input("[ValueError] Please enter a valid and legal UCI move: ")
        return chess.Move.from_uci(mov)

    def play(self):
        print()
        board_play = chess.Board()
        current_player1_turn = True
        # board_play.is_repetition()
        # board_play.can_claim_threefold_repetition()
        # board_play.is_fivefold_repetition()
        while not (
                board_play.is_checkmate() or board_play.is_stalemate() or board_play.is_insufficient_material() or board_play.can_claim_threefold_repetition()):
            print(ChessPlayCLI.pretty_board(board_play, self.clear_screen))
            print()
            print(f"DEBUG: board_play.fen() = {board_play.fen()}")
            print(f"DEBUG: Legal moves = {list(map(str, list(board_play.legal_moves)))}")
            if board_play.is_check():
                print(colored(text="INFO: it is a check.", attrs=['bold', 'reverse']))
            if current_player1_turn:
                # player 1 plays
                if self.player1_chess_predict is None:
                    move_selected = self.user_play(board_play=board_play, is_player1=current_player1_turn)
                else:
                    with cs.ExecutionTime():
                        move_selected, move_score = self.player1_chess_predict.predict_best_move(board=board_play)
                    print(f"DEBUG: [{self.player1_name}] AI's move = {move_selected}")
                    print(f"DEBUG: [{self.player1_name}] AI's move_score = {move_score}")
            else:
                # player 2 plays
                if self.player2_chess_predict is None:
                    move_selected = self.user_play(board_play=board_play, is_player1=current_player1_turn)
                else:
                    with cs.ExecutionTime():
                        move_selected, move_score = self.player2_chess_predict.predict_best_move(board=board_play)
                    print(f"DEBUG: [{self.player2_name}] AI's move = {move_selected}")
                    print(f"DEBUG: [{self.player2_name}] AI's move_score = {move_score}")
            board_play.push(move_selected)
            if self.game_analyzer is not None:
                print(f"\nGame analysis[from white's perspective] = {self.game_analyzer.analyse_board(board_play)}")
            current_player1_turn ^= True
            time.sleep(self.delay)

        print("\n\n" + ChessPlayCLI.pretty_board(board_play, self.clear_screen), end="\n\n")
        if board_play.is_stalemate():
            print(f"RESULTS: draw, as its a stalemate")
        elif board_play.is_checkmate() and board_play.turn == chess.WHITE:
            print(f"RESULTS: {self.player2_name} wins")
        elif board_play.is_checkmate() and board_play.turn == chess.BLACK:
            print(f"RESULTS: {self.player1_name} wins")
        elif board_play.is_insufficient_material():
            print(f"RESULTS: draw, as both players have insufficient winning material")
        elif board_play.can_claim_threefold_repetition():
            print(f"RESULTS: draw, due to three fold repetition")
        else:
            print(f"RESULTS: draw/unknown, board_play.fen() = {board_play.fen()}")
        game_moves = []
        while board_play != chess.Board():
            game_moves.insert(0, str(board_play.pop()))
        print("\n\nGame moves list:")
        print(game_moves)
        print()
        return


########################################################################################################################


def init_engine(cpu_cores=1, analyse_time=0.01):
    engine_sf = step_01.CustomEngine(src_path=None, cp_score_max=8000, mate_score_max=10000,
                                     mate_score_difference=50, cpu_cores=1, hash_size_mb=16,
                                     depth=20, analyse_time=0.01)
    return engine_sf


def play(player1_name: str, player2_name: str, game_type: str, model_weights_file: str, analyze_game: bool = False, clear_screen=False,
         delay=0.0, ):
    ffnn_keras, engine_sf, engine_predict = None, None, None

    def init_model():
        mv = step_03a.ModelVersion.create_obj(Path(model_weights_file).name)
        if mv.model_generator == 4:
            ffnn_keras = step_03a.FFNNBuilder.build_004(name_prefix='', version=1, callback=False, generate_model_image=False)
        else:
            ffnn_keras = step_03a.FFNNBuilder.build_005(name_prefix='', version=1, callback=False, generate_model_image=False)
        ffnn_keras.c_load_weights(model_weights_file)
        return ffnn_keras

    player1_chess_predict, player2_chess_predict = None, None
    if game_type == 'mm':
        ffnn_keras = init_model()
        player1_chess_predict = ChessPredict(ffnn_keras.c_predict_board_1, ffnn_keras.c_predict_board_n, to_maximize=True)
        player2_chess_predict = ChessPredict(ffnn_keras.c_predict_board_1, ffnn_keras.c_predict_board_n, to_maximize=False)
    elif game_type == 'me':
        ffnn_keras = init_model()
        engine_sf = init_engine()
        player1_chess_predict = ChessPredict(ffnn_keras.c_predict_board_1, ffnn_keras.c_predict_board_n, to_maximize=True)
        player2_chess_predict = ChessPredict(predict_move_1=engine_sf.predict_move, to_maximize=False)
    elif game_type == 'em':
        ffnn_keras = init_model()
        engine_sf = init_engine()
        player1_chess_predict = ChessPredict(predict_move_1=engine_sf.predict_move, to_maximize=True)
        player2_chess_predict = ChessPredict(ffnn_keras.c_predict_board_1, ffnn_keras.c_predict_board_n, to_maximize=False)
    elif game_type == 'ee':
        engine_sf = init_engine()
        player1_chess_predict = ChessPredict(predict_move_1=engine_sf.predict_move, to_maximize=True)
        player2_chess_predict = ChessPredict(predict_move_1=engine_sf.predict_move, to_maximize=False)

    chess_predict_4_analyse = None
    if analyze_game is True:
        engine_predict = init_engine(cpu_cores=2, analyse_time=0.5);
        chess_predict_4_analyse = ChessPredict(
            analyse_board=engine_predict.evaluate_normalized_board
        )

    ChessPlayCLI(player1_name=player1_name,
                 player2_name=player2_name,
                 player1_chess_predict=player1_chess_predict,
                 player2_chess_predict=player2_chess_predict,
                 game_analyzer=chess_predict_4_analyse,
                 clear_screen=clear_screen,
                 delay=delay).play()

    if engine_sf is None and engine_predict is None: return
    print("Closing the connection with the engine...", flush=True)
    if engine_sf is not None:
        engine_sf.close()
    if engine_predict is not None:
        engine_predict.close()
    print("Connection closed.", flush=True)


def predict_move(input_dir: str, output_dir: str, move_dir: str, model_weights_file: str):
    if output_dir is None:
        output_dir = input_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(move_dir).mkdir(parents=True, exist_ok=True)
    if not Path(output_dir).exists():
        print("ERROR: `output_dir` does not exists")
        return
    if not Path(move_dir).exists():
        print("ERROR: `move_dir` does not exists")
        return

    ffnn_keras, player1_chess_predict, player2_chess_predict = None, None, None
    if Path(model_weights_file).exists() and Path(model_weights_file).is_file():
        ffnn_keras = step_03a.FFNNBuilder.build(Path(model_weights_file).name)
        player1_chess_predict = ChessPredict(ffnn_keras.c_predict_board_1, ffnn_keras.c_predict_board_n, to_maximize=True)
        player2_chess_predict = ChessPredict(ffnn_keras.c_predict_board_1, ffnn_keras.c_predict_board_n, to_maximize=False)
    else:
        print(f"ERROR: Invalid model_weights_file={model_weights_file}", flush=True)
        return

    for i in tqdm(glob.glob(f"{Path(input_dir)}/*.csv")):
        boards = pd.read_csv(i, header=None)
        result_move_uci: List = []
        result_move_alg: List = []
        result_score: List = []
        for j in tqdm(boards[0]):
            try:
                j = chess.Board(j)
            except Exception as e:
                print(f"ERROR: parsing error: {j}, {e}")
                continue
            if j.turn:
                move_selected, move_score = player1_chess_predict.predict_best_move(j)
            else:
                move_selected, move_score = player2_chess_predict.predict_best_move(j)
            result_move_uci.append(str(move_selected))
            result_move_alg.append(j.san(move_selected))
            result_score.append(str(move_score))
        boards['move_uci'] = result_move_uci
        boards['move_alg'] = result_move_alg
        boards['score'] = result_score
        boards.to_csv(f"{Path(output_dir) / Path(i).stem}_processed.csv", index=False)
        print(f"Successfully processes: '{Path(output_dir) / Path(i).stem}_processed.csv'")
        shutil.move(src=i, dst=move_dir)


def iterate_moves(moves: List[str], analyze_game: bool, clear_screen: bool, delay: float):
    chess_predict_4_analyse = None
    flush_str = "\033c" if clear_screen else ''
    current_board = chess.Board()

    if analyze_game:
        engine_sf = init_engine()
        chess_predict_4_analyse = ChessPredict(
            analyse_board=step_01.CustomEngine(
                src_path=None, cp_score_max=8000, mate_score_max=10000,
                mate_score_difference=50, cpu_cores=1, hash_size_mb=16,
                depth=20, analyse_time=0.5
            ).evaluate_normalized_board
        )

    for mov_i in moves:
        if chess.Move.from_uci(mov_i) in current_board.legal_moves:
            current_board.push(chess.Move.from_uci(mov_i))
        else:
            print("ERROR: invalid move, exiting the program")
            return
        print(ChessPlayCLI.pretty_board(current_board, clear_screen))
        if chess_predict_4_analyse is not None:
            print(f"\nGame analysis[from white's perspective] = {chess_predict_4_analyse.analyse_board(current_board)}")
        time.sleep(delay)
        # print(flush_str)


if __name__ == "__main__":
    from docopt import docopt

    doc_string = '''
    Usage:
        step_04_play.py play [--player1_name=NAME] [--player2_name=NAME] --game_type=TYPE [--model_weights_file=PATH] [--analyze_game] [--clear_screen] [--delay=SECONDS]
        step_04_play.py predict_move --input_dir=PATH [--output_dir=PATH] --move_dir=PATH --model_weights_file=PATH
        step_04_play.py iterate_moves --moves=MOVESLIST [--analyze_game] [--clear_screen] [--delay=SECONDS]
        step_04_play.py (-h | --help)
        step_04_play.py --version

    Options:
        --player1_name=NAME     Player 1 name [default: Player1]
        --player2_name=NAME     Player 2 name [default: Player2]
        --game_type=TYPE        Type of the game to be played (TYPE can be: mm, me, em, ee)
        --model_weights_file=PATH  Path to the neural network model, required if --game_type is either mm, me or em
        --analyze_game          Whether to use stockfish to analyze the game or not ?
        --clear_screen          Whether to clear the terminal output after each move or not ?
        --delay=SECONDS         Number of seconds the game should pause after each move [default: 0.0]
        
        --input_dir=PATH        Path to directory which has CSV files containing board states to predict move
        --output_dir=PATH       Path to directory where results shall be stored, default is input_dir
        --move_dir=PATH         Path to directory where processed files shall be moved

        -h --help               Show this
        --version               Show version
    '''
    arguments = docopt(doc_string, argv=None, help=True, version=f"{cs.VERSION} - Game play", options_first=False)
    # arguments = eval("""{'--analyze_game': False,
    #  '--clear_screen': False,
    #  '--delay': '1.0',
    #  '--game_type': 'ee',
    #  '--help': False,
    #  '--model_weights_file': None,
    #  '--player1_name': 'Player1',
    #  '--player2_name': 'Player2',
    #  '--version': False}
    # """)

    print("\n\n", arguments, "\n\n", sep="")
    if arguments['play']:
        play(
            arguments['--player1_name'],
            arguments['--player2_name'],
            arguments['--game_type'],
            arguments['--model_weights_file'],
            arguments['--analyze_game'],
            arguments['--clear_screen'],
            float(arguments['--delay']),
        )
    elif arguments['predict_move']:
        predict_move(arguments['--input_dir'], arguments['--output_dir'], arguments['--move_dir'], arguments['--model_weights_file'])
    elif arguments['iterate_moves']:
        iterate_moves(eval(arguments['--moves']), arguments['--analyze_game'], arguments['--clear_screen'], float(arguments['--delay']))

    '''
    Example:
        python step_04_play.py --game_type=ee --delay=1.0 --model_weights_file="/home/student/Desktop/fenil_pc/Chess-Kesari-Models/ffnn_keras-mg005-be00778-sn003-ep00127-weights-v032.h5"

        python step_04_play.py iterate_moves --moves="['e2e4', 'd7d5', 'e4d5', 'd8d5', 'g1f3', 'g8f6', 'd2d4', 'd5e4', 'f1e2', 'c8f5', 'e1g1', 'e4c2', 'b1c3', 'c2d1', 'g2g4', 'd1c2', 'c3d5', 'f5g4', 'd5f6', 'g7f6', 'a2a4', 'c2e2', 'c1e3', 'g4f3', 'f1b1', 'h8g8', 'e3g5', 'g8g5']" #--analyze_game --clear_screen --delay=1.0
    '''

    # fire.Fire(play)
