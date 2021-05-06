import copy
import glob
import itertools
import logging
import multiprocessing
import os
import shutil
import sys
from pathlib import Path
from typing import Union, Iterable, List, TextIO, Tuple

import chess.pgn
import joblib
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

import common_services as cs
import step_01_engine

engine_sf = None
MINI_BATCH_SIZE: int = 1000
BATCH_SIZE: int = 10000

# REFER: https://stackoverflow.com/questions/7016056/python-logging-not-outputting-anything
# This sets the root logger to write to stdout (your console)
logging.basicConfig()


########################################################################################################################
class PreprocessPGN:
    def __init__(self, pgn_file_path: Union[str, Path]):
        if not pgn_file_path.endswith(".pgn"):
            raise Exception(f"ERROR: This is not pgn file: {pgn_file_path}")
        self.pgn_file_path: str = str(pgn_file_path)

        self.pgn_text_io: TextIO = self.__load_pgn()
        print(f"DEBUG: pgn file {self.pgn_file_path} successfully loaded", file=sys.stderr)

    def __load_pgn(self) -> Union[TextIO, Exception]:
        # https://python-chess.readthedocs.io/en/latest/pgn.html
        if not Path(self.pgn_file_path).exists():
            print(f"ERROR: {self.pgn_file_path} does not exist", file=sys.stderr)
            raise FileNotFoundError(f"'{self.pgn_file_path}'")

        pgn = open(self.pgn_file_path, mode="rt")
        return pgn

    def iterate_pgn(self) -> Iterable[chess.pgn.Game]:
        game = chess.pgn.read_game(self.pgn_text_io)
        while game is not None:
            yield game
            try:
                game = chess.pgn.read_game(self.pgn_text_io)
            except UnicodeDecodeError as e:
                print(
                    f"WARNING: it seems pgn file has been completely read, UnicodeDecodeError occurred:\n\t{e}",
                    file=sys.stderr
                )
                break
        return

    @staticmethod
    def iterate_game(nth_game: chess.pgn.Game) -> Iterable[chess.Board]:
        board = nth_game.board()
        for move in nth_game.mainline_moves():
            board.push(move)
            yield copy.deepcopy(board)
        return

    @staticmethod
    def generate_boards(nth_board: chess.Board) -> Iterable[chess.Board]:
        legal_move = nth_board.legal_moves
        for move in legal_move:
            nth_board.push(move)
            yield copy.deepcopy(nth_board)
            nth_board.pop()

    @staticmethod
    def generate_boards_list(nth_board: chess.Board) -> List[chess.Board]:
        return list(PreprocessPGN.generate_boards(nth_board))

    def reload_pgn(self):
        self.pgn_text_io: TextIO = self.__load_pgn()
        print(f"DEBUG: pgn file {self.pgn_file_path} successfully re-loaded", file=sys.stderr)

    def get_pgn_game_count(self) -> int:
        game_count = 0
        for i in self.iterate_pgn():
            game_count += 1

        self.reload_pgn()
        return game_count

    def pgn_to_csv_simple(self, output_dir: Union[str, Path]):
        global engine_sf, BATCH_SIZE

        # Generate the output path if it does not exists and don't raise Exception if output_path already present.
        # if not Path(output_path).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if not Path(output_dir).exists():
            print(f"ERROR: `output_path={output_dir}` does NOT exists")
            return

        # Initialize variables
        # INPUT_PGN_NAME : name of the pgn file
        # res_pd         : pd.DataFrame with FEN notation of chess.Board objects and CentiPawn score of the board
        # engine_sf      : CustomEngine object to generate the CentiPawn score of the board
        INPUT_PGN_NAME: str = Path(self.pgn_file_path).name[:-4]
        res_pd = pd.DataFrame(data=[BATCH_SIZE * [None, None]], index=None, columns=cs.COLUMNS)
        # engine_sf = step_01_engine.CustomEngine(src_path=None, mate_score_max=10000, mate_score_difference=50, hash_size_mb=8192, depth=15, analyse_time=0.2)

        file_count = 1
        game_count = 1
        print(f"DEBUG: processing started...\n")
        for i in self.iterate_pgn():
            print(f"\r\t{game_count}", end="")
            for j in PreprocessPGN.iterate_game(i):
                if not j.is_valid():
                    print(f"*** WARNING: invalid board state in the game: {j.fen()}", file=sys.stderr)
                    continue
                res_pd.loc[len(res_pd)] = [j.fen(), None]
            game_count += 1

            if len(res_pd) > BATCH_SIZE:
                output_file = Path(output_dir) / f"{INPUT_PGN_NAME}_{file_count:06}.csv"
                res_pd.to_csv(output_file, index=False)
                print(f"DEBUG: boards successfully written to file: {output_file}", file=sys.stderr)
                file_count += 1
                res_pd = pd.DataFrame(data=[BATCH_SIZE * [None, None]], index=None, columns=cs.COLUMNS)

        if len(res_pd) > 0:
            output_file = Path(output_dir) / f"{INPUT_PGN_NAME}_{file_count:06}.csv"
            res_pd.to_csv(output_file, index=False)
            print(f"DEBUG: boards successfully written to file: {output_file}", file=sys.stderr)
        print(f"DEBUG: processing finished :)")

    @staticmethod
    def pgn_to_csv(pgn_file_path: Union[str, Path], output_dir: Union[str, Path], resume_file_name=None, debug_flag=3):
        global MINI_BATCH_SIZE, BATCH_SIZE
        logger = logging.getLogger("pgn_to_csv")
        logger.setLevel(logging.DEBUG)

        pgn_obj = PreprocessPGN(pgn_file_path)

        # Generate the output path if it does not exists and don't raise Exception if output_path already present.
        # if not Path(output_path).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if not Path(output_dir).exists():
            print(f"ERROR: `output_path={output_dir}` does NOT exists")
            return

        # Initialize variables
        # INPUT_PGN_NAME : name of the pgn file
        # res_pd         : pd.DataFrame with FEN notation of chess.Board objects and CentiPawn score of the board
        INPUT_PGN_NAME: str = Path(pgn_obj.pgn_file_path).stem
        if resume_file_name is None:
            resume_file_name = f"{Path(pgn_file_path).parent}/resume_{INPUT_PGN_NAME}.txt"
            logger.info(f"Using default name fore resume file='{resume_file_name}'")
        res_pd = pd.DataFrame(data=None, index=None, columns=cs.COLUMNS)

        game_count = 1
        # (last game written + 1), (next file number to be written)
        last_line_content = cs.read_last_line(resume_file_name)
        if last_line_content == '-done-':
            logger.info(f"PGN file already processed completely: '{pgn_obj.pgn_file_path}'"
                        f"\n\treturning")
            return

        resume_game_count, file_count = cs.readpoint(resume_file_name, 2)
        logger.debug(f"resume_game_count, file_count = {resume_game_count}, {file_count}")
        if game_count < resume_game_count and (Path(output_dir) / f"{INPUT_PGN_NAME}_{file_count:06}.csv").exists():
            res_pd = pd.read_csv(Path(output_dir) / f"{INPUT_PGN_NAME}_{file_count:06}.csv")
            if debug_flag >= 1:
                logger.info(f"loading partially saved file, lines = {len(res_pd)}")

        for i in pgn_obj.iterate_pgn():
            if game_count < resume_game_count:
                if debug_flag >= 2:
                    print(f"\rskip {game_count}", file=sys.stderr)
                game_count += 1
                continue

            if debug_flag >= 3:
                logger.info(f"processing game_count = {game_count} ({len(res_pd)})")
            board_count = 1
            for j in PreprocessPGN.iterate_game(i):
                print(f"\r\t{board_count}", end="", file=sys.stderr)
                board_count += 1

                if not j.is_valid():
                    logger.warning(f"*** invalid board state in the game: {j.fen()}")
                    continue

                try:
                    # NOTE: change 01: replace None with call to evaluate function if required
                    # NOTE: None -> engine_sf.evaluate_board(j)
                    res_pd.loc[len(res_pd)] = [j.fen(), None]

                    # for k in generate_boards(j):
                    #     res_pd.loc[len(res_pd)] = [k.fen(), None]
                except Exception as e:
                    logger.error(e)
                    logger.info(f"\tis_valid() = {j.is_valid()}, board.fen() = {j.fen()}")

            print("\r", end="", file=sys.stderr)

            game_count += 1
            if len(res_pd) > BATCH_SIZE or len(res_pd) > MINI_BATCH_SIZE * (max(0, len(res_pd) - 1) // MINI_BATCH_SIZE + 1):
                output_file = Path(output_dir) / f"{INPUT_PGN_NAME}_{file_count:06}.csv"
                res_pd.to_csv(output_file, index=False)
                if debug_flag >= 1:
                    logger.info(f"boards successfully written to file: {output_file}")
                if len(res_pd) > BATCH_SIZE:
                    file_count += 1
                    cs.append_secondlast_line(resume_file_name, f"# {len(res_pd)} {game_count} {file_count}")
                    res_pd = pd.DataFrame(data=None, index=None, columns=cs.COLUMNS)
                cs.savepoint(resume_file_name, f"{game_count},{file_count}")

        if len(res_pd) > 0:
            output_file = Path(output_dir) / f"{INPUT_PGN_NAME}_{file_count:06}.csv"
            res_pd.to_csv(output_file, index=False)
            if debug_flag >= 1:
                logger.info(f"\nboards successfully written to file: {output_file}")
            file_count += 1
            cs.savepoint(resume_file_name, f"{game_count},{file_count}")

        cs.savepoint(resume_file_name, f"-done-")

        logger.info(f"execution successfully complete for '{pgn_obj.pgn_file_path}'")


########################################################################################################################
class BoardEncoder:
    @staticmethod
    def is_check(board: chess.Board, side: chess.Color) -> bool:
        king = board.king(side)
        return king is not None and board.is_attacked_by(not side, king)

    @staticmethod
    def is_checkmate(board: chess.Board, side: chess.Color) -> bool:
        return board.is_checkmate() and board.turn == side

    class EncodeBase(object):
        @staticmethod
        def encode_board_1(board_1: chess.Board) -> np.ndarray:
            pass

        @staticmethod
        def encode_board_1_fen(board_1_fen: str) -> np.ndarray:
            pass

        @staticmethod
        def encode_board_n_fen(board_n_fen: Union[List[str], Tuple[str]]) -> np.ndarray:
            """
            Convert list of tuple of chess boards from FEN notation to 778 floating point 0's and 1's

            NOTE: this computation is performed in parallel processes

            :param board_n_fen:
            :return: np.ndarray
            """

        @staticmethod
        def encode_board_n(board_n: Union[List[chess.Board], Tuple[chess.Board]]) -> np.ndarray:
            """
            Convert list of tuple of chess boards to 778 floating point 0's and 1's

            NOTE: this computation is performed in parallel processes

            :param board_n:
            :return: np.ndarray
            """

    class Encode778(EncodeBase):
        @staticmethod
        def encode_board_1(board_1: chess.Board) -> np.ndarray:
            if not board_1.is_valid():
                print(f"ERROR: invalid board state :(", file=sys.stderr)
                raise Exception("Invalid board state")

            board_mat_list: List[List[str]] = [i.split(" ") for i in board_1.__str__().split("\n")]
            board_mat_df: pd.DataFrame = pd.DataFrame(board_mat_list)

            # The following array has 10 bits of manually extracted features/attributes/information
            result_nparray: np.array = np.array([
                board_1.turn,  # True => white's turn, False => black's turn
                board_1.is_checkmate(),  # V.V.V Important feature/attribute/information  # ADDED 20191207T1137
                board_1.has_kingside_castling_rights(chess.WHITE),
                # True => Castling rights present, False => no rights
                board_1.has_queenside_castling_rights(chess.WHITE),  # --------------------||---------------------
                board_1.has_kingside_castling_rights(chess.BLACK),  # ---------------------||---------------------
                board_1.has_queenside_castling_rights(chess.BLACK),  # --------------------||---------------------
                # True => White King has a Check, False => no check to White King
                BoardEncoder.is_check(board_1, chess.WHITE),
                # True => Black King has a Check, False => no check to Black King
                BoardEncoder.is_check(board_1, chess.BLACK),
                # True => White Queen alive, False => White Queen out works if there are more than on Queen
                (sum((board_mat_df.values == 'Q').ravel()) != 0),
                # True => Black Queen alive, False => Black Queen out works if there are more than on Queen
                (sum((board_mat_df.values == 'q').ravel()) != 0),
            ])

            # The following two lines check if Queen is alive or not. However, am not sure if it will work when there are more than one Queens
            # bool((board.occupied_co[chess.WHITE] & board.queens) != 0),
            # bool((board.occupied_co[chess.BLACK] & board.queens) != 0),

            # WHITE side = 64*6 bits = 384 bits
            result_nparray = np.append(result_nparray, (board_mat_df.values == 'K').ravel())
            result_nparray = np.append(result_nparray, (board_mat_df.values == 'Q').ravel())
            result_nparray = np.append(result_nparray, (board_mat_df.values == 'B').ravel())
            result_nparray = np.append(result_nparray, (board_mat_df.values == 'N').ravel())
            result_nparray = np.append(result_nparray, (board_mat_df.values == 'R').ravel())
            result_nparray = np.append(result_nparray, (board_mat_df.values == 'P').ravel())

            # BLACK side = 64*6 bits = 384 bits
            result_nparray = np.append(result_nparray, (board_mat_df.values == 'k').ravel())
            result_nparray = np.append(result_nparray, (board_mat_df.values == 'q').ravel())
            result_nparray = np.append(result_nparray, (board_mat_df.values == 'b').ravel())
            result_nparray = np.append(result_nparray, (board_mat_df.values == 'n').ravel())
            result_nparray = np.append(result_nparray, (board_mat_df.values == 'r').ravel())
            result_nparray = np.append(result_nparray, (board_mat_df.values == 'p').ravel())

            return result_nparray.astype(dtype=np.float32)

        @staticmethod
        def encode_board_1_fen(board_1_fen: str) -> np.ndarray:
            return BoardEncoder.Encode778.encode_board_1(chess.Board(board_1_fen))

        @staticmethod
        def encode_board_n_fen(board_n_fen: Union[List[str], Tuple[str]]) -> np.ndarray:
            """
            Convert list of tuple of chess boards from FEN notation to 778 floating point 0's and 1's

            NOTE: this computation is performed in parallel processes

            :param board_n_fen:
            :return: np.ndarray
            """
            with multiprocessing.Pool() as pool:
                return np.array(
                    pool.map(func=BoardEncoder.Encode778.encode_board_1_fen, iterable=board_n_fen)
                )
            # return BoardEncoder.encode_board_1_778_fen(board_n_fen)

        @staticmethod
        def encode_board_n(board_n: Union[List[chess.Board], Tuple[chess.Board]]) -> np.ndarray:
            """
            Convert list of tuple of chess boards to 778 floating point 0's and 1's

            NOTE: this computation is performed in parallel processes

            :param board_n:
            :return: np.ndarray
            """
            return BoardEncoder.Encode778.encode_board_n_fen(
                [
                    board_i.fen() for board_i in board_n
                ]
            )


class ScoreNormalizer:
    input_range_2 = [(0, 50), (50, 2000), (2000, 6000), (6000, 10000), (-50, 0), (-2000, -50), (-6000, -2000), (-10000, -6000)]
    output_range_2 = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.8), (0.8, 1.0), (-0.2, 0.0), (-0.4, -0.2), (-0.8, -0.4), (-1.0, -0.8)]
    input_range_3 = [(0, 50), (50, 2000), (2000, 10000), (-50, 0), (-2000, -50), (-10000, -2000)]
    output_range_3 = [(0.0, 0.3), (0.3, 0.95), (0.95, 1.0), (-0.3, 0.0), (-0.95, -0.3), (-1.0, -0.95)]

    @staticmethod
    def __scale_limits(np_array: np.ndarray, old_low, old_high, new_low, new_high) -> None:
        np_logical_and = np.logical_and(old_low <= np_array, np_array <= old_high)
        np_array[np_logical_and] = \
            ((np_array[np_logical_and] - old_low) / (old_high - old_low)) * (new_high - new_low) + new_low

    @staticmethod
    def normalize_001(data_np: np.ndarray) -> np.ndarray:
        """
        Following is the cp_scores normalization details:
            • [      0,    10 ] -> [  0.0,  1.0 ]
            • [     10, 10000 ] -> [     1.0    ]
            • [    -10,     0 ] -> [ -1.0,  0.0 ]
            • [ -10000,   -10 ] -> [    -1.0    ]

        NOTE: Results are not good

        :param data_np:
        :return: Normalized `np.ndarray`
        """
        data_np = copy.deepcopy(data_np)
        data_np *= 10000
        data_np[data_np >= 10] = 10
        data_np[data_np <= -10] = -10
        data_np /= 10
        return data_np

    @staticmethod
    def normalize_002(data_np: np.ndarray) -> np.ndarray:
        """
        Following is the cp_scores normalization details:
            • [      0,    50 ] -> [  0.0,  0.2 ]
            • [     50,  2000 ] -> [  0.2,  0.4 ]
            • [   2000,  6000 ] -> [  0.4,  0.8 ]
            • [   6000, 10000 ] -> [  0.8,  1.0 ]
            • [    -50,     0 ] -> [ -0.2,  0.0 ]
            • [  -2000,   -50 ] -> [ -0.4, -0.2 ]
            • [  -6000, -2000 ] -> [ -0.8, -0.4 ]
            • [ -10000, -6000 ] -> [ -1.0, -0.8 ]

        :param data_np:
        :return: Normalized `np.ndarray`
        """
        data_np = copy.deepcopy(data_np)
        data_np *= 10000

        data_np = copy.deepcopy(data_np)
        for i, j in zip(ScoreNormalizer.input_range_2, ScoreNormalizer.output_range_2):
            ScoreNormalizer.__scale_limits(data_np, i[0], i[1], j[0], j[1])

        return data_np

    @staticmethod
    def denormalize_002(data_np: np.ndarray) -> np.ndarray:
        """
        Following is the cp_scores normalization details:
            • [  0.0,  0.2 ] -> [      0,    50 ]
            • [  0.2,  0.4 ] -> [     50,  2000 ]
            • [  0.4,  0.8 ] -> [   2000,  6000 ]
            • [  0.8,  1.0 ] -> [   6000, 10000 ]
            • [ -0.2,  0.0 ] -> [    -50,     0 ]
            • [ -0.4, -0.2 ] -> [  -2000,   -50 ]
            • [ -0.8, -0.4 ] -> [  -6000, -2000 ]
            • [ -1.0, -0.8 ] -> [ -10000, -6000 ]

        :param data_np:
        :return: Normalized `np.ndarray`
        """
        data_np = copy.deepcopy(data_np)

        # The following three lines were added, no other change to `normalize_002(...)`
        input_range, output_range = ScoreNormalizer.output_range_2, ScoreNormalizer.input_range_2
        input_range.reverse()
        output_range.reverse()

        for i, j in zip(input_range, output_range):
            ScoreNormalizer.__scale_limits(data_np, i[0], i[1], j[0], j[1])

        data_np /= 10000
        return data_np

    @staticmethod
    def normalize_003(data_np: np.ndarray) -> np.ndarray:
        """
        Following is the cp_scores normalization details:
            • [      0,    50 ] -> [  0.0,  0.3 ]
            • [     50,  2000 ] -> [  0.2,  0.4 ]
            • [   2000,  6000 ] -> [  0.4,  0.8 ]
            • [   6000, 10000 ] -> [  0.8,  1.0 ]
            • [    -50,     0 ] -> [ -0.2,  0.0 ]
            • [  -2000,   -50 ] -> [ -0.4, -0.2 ]
            • [  -6000, -2000 ] -> [ -0.8, -0.4 ]
            • [ -10000, -6000 ] -> [ -1.0, -0.8 ]

        :param data_np:
        :return: Normalized `np.ndarray`
        """
        data_np = copy.deepcopy(data_np)
        data_np *= 10000

        data_np = copy.deepcopy(data_np)
        for i, j in zip(ScoreNormalizer.input_range_3, ScoreNormalizer.output_range_3):
            ScoreNormalizer.__scale_limits(data_np, i[0], i[1], j[0], j[1])

        return data_np

    @staticmethod
    def denormalize_003(data_np: np.ndarray) -> np.ndarray:
        """
        Following is the cp_scores normalization details:
            • [  0.0,  0.2 ] -> [      0,    50 ]
            • [  0.2,  0.4 ] -> [     50,  2000 ]
            • [  0.4,  0.8 ] -> [   2000,  6000 ]
            • [  0.8,  1.0 ] -> [   6000, 10000 ]
            • [ -0.2,  0.0 ] -> [    -50,     0 ]
            • [ -0.4, -0.2 ] -> [  -2000,   -50 ]
            • [ -0.8, -0.4 ] -> [  -6000, -2000 ]
            • [ -1.0, -0.8 ] -> [ -10000, -6000 ]

        :param data_np:
        :return: Normalized `np.ndarray`
        """
        data_np = copy.deepcopy(data_np)

        # The following three lines were added, no other change to `normalize_002(...)`
        input_range, output_range = ScoreNormalizer.output_range_3, ScoreNormalizer.input_range_3
        input_range.reverse()
        output_range.reverse()

        for i, j in zip(input_range, output_range):
            ScoreNormalizer.__scale_limits(data_np, i[0], i[1], j[0], j[1])

        data_np /= 10000
        return data_np


########################################################################################################################
def generate_all_boards(depth_d: int):
    board_b = chess.Board()
    output_list = [[board_b, ], ]
    for i in tqdm(range(depth_d)):
        output_list.append(list())
        print(f"DEBUG: working on {i + 1}")
        with multiprocessing.Pool() as pool:
            temp_result = pool.map(func=PreprocessPGN.generate_boards_list, iterable=[j for j in output_list[-2]])
        output_list[-1] = [k for j in temp_result for k in j]
        print(f'DEBUG: work done, len(depth={i + 1}) = {len(output_list[-1])}')
        print(flush=True)
    return output_list


def boards_list_to_csv(file_name, boards_list, boards_list_depth):
    res_pd = pd.DataFrame(data=zip([i.fen() for i in boards_list], itertools.repeat(0)),
                          index=None,
                          columns=cs.COLUMNS)
    res_pd.to_csv(f"{file_name}_{boards_list_depth:06}.csv", index=False)


def generate_all_boards_to_csv(file_name: str, depth_d: int):
    res = generate_all_boards(depth_d)
    with multiprocessing.Pool() as pool:
        pool.starmap(boards_list_to_csv, [(file_name, res[i], i) for i in range(1, len(res))])
    return res


# with cs.ExecutionTime():
#     generate_all_boards_to_csv('boards', 5)


########################################################################################################################
def csv_score_generation(csv_folder_path: Union[str, Path], resume_file_name):
    global engine_sf
    engine_sf = step_01_engine.CustomEngine(src_path=None, hash_size_mb=16, depth=15, analyse_time=0.2)

    file_count = 1
    resume_file_count: int = cs.readpoint(resume_file_name, 1)[0]

    input_data = Path(csv_folder_path)
    input_files_list = sorted(list(input_data.glob("*.csv")))
    for i in tqdm(input_files_list, leave=None):
        if file_count < resume_file_count:
            print(f"DEBUG: skip {i} = {file_count}", file=sys.stderr)
            file_count += 1
            continue

        # print(f"DEBUG: processing file {i} = {file_count}")
        data = pd.read_csv(i)
        line_count = 2
        for j in tqdm(range(len(data)), leave=None):
            # data.loc[j][1] = engine_sf.evaluate(chess.Board(data.loc[j][0]))
            if not chess.Board(data.loc[j][0]).is_valid():
                print(f"\rERROR: board state not valid at line count = {line_count}")
                line_count += 1
                continue
            data.at[j, 'cp_score'] = engine_sf.evaluate_fen(data.loc[j][0])
            line_count += 1
            # print(f"\r\t{line_count}", end="", file=sys.stderr)

        data.to_csv(i, index=False)
        file_count += 1
        cs.savepoint(resume_file_name, file_count)
        print(f"DEBUG: successfully processed {i}", file=sys.stderr)


def pgn_to_csv_parallel(input_dir: str, output_dir: str):
    if output_dir is None:
        output_dir = input_dir

    with multiprocessing.Pool(processes=None, maxtasksperchild=1) as pool:
        pool.starmap(func=PreprocessPGN.pgn_to_csv,
                     iterable=[(i, output_dir, None, 3) for i in glob.glob(f"{Path(input_dir)}/*.pgn")],
                     chunksize=1)


# TODO: check the implementation
def pgn_to_csv_all_possible_state(input_path: str, output_path: str = None):
    if output_path is None:
        output_path = input_path
    for i in glob.glob(f"{Path(input_path)}/*.pgn"):
        pgn_obj = PreprocessPGN(pgn_file_path=i)
        res_pd = pd.DataFrame(data=None, index=None, columns=cs.COLUMNS)
        for j in pgn_obj.iterate_pgn():
            for k in PreprocessPGN.iterate_game(j):
                for l in PreprocessPGN.generate_boards(k):
                    res_pd.loc[len(res_pd)] = [l.fen(), None]
        res_pd.to_csv(f"{os.path.splitext(Path(i))[0]}.csv", index=False)


class FenToPkl:
    @staticmethod
    def __load_transform(board_encoder,
                         score_normalizer,
                         file_path: str,
                         print_execution_time: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        with cs.ExecutionTime(file=sys.stderr, print=print_execution_time):
            data = pd.read_csv(file_path, dtype={cs.COLUMNS[0]: str, cs.COLUMNS[1]: np.float32})
            data_x = data[cs.COLUMNS[0]].values
            data_y = data[cs.COLUMNS[1]].values
            # print(data.head())

            data_x_encoded: np.ndarray = board_encoder.encode_board_n_fen(data_x)
            # data_x_encoded = step_02.BoardEncoder.Encode778.encode_board_n_fen(data_x)

            if score_normalizer is not None:
                data_y_normalized: np.ndarray = score_normalizer(data_y)
            else:
                data_y_normalized: np.ndarray = data_y
            # data_y_normalized = step_02.ScoreNormalizer.normalize_002(data_y)

        del data, data_x, data_y
        return data_x_encoded, data_y_normalized

    @staticmethod
    def convert_fen_to_pkl_file(file_path: str, output_dir: str, move_dir: str,
                                board_encoder_str: str, score_normalizer_str: str,
                                suffix_to_append: str = "-be?????-sn???.pkl", ):
        data_x_encoded, data_y_normalized = FenToPkl.__load_transform(eval(board_encoder_str),
                                                                      eval(score_normalizer_str),
                                                                      file_path,
                                                                      print_execution_time=True)

        # Path(...).stem returns file name only without extension
        # compress=1, performs basic compression. compress=0 means no compression
        joblib.dump((data_x_encoded, data_y_normalized,),
                    filename=f"{Path(output_dir) / Path(file_path).stem}{suffix_to_append}",
                    compress=1)
        shutil.move(file_path, move_dir)
        del data_x_encoded, data_y_normalized

    @staticmethod
    def convert_fen_to_pkl_folder(input_dir: str, output_dir: str, move_dir: str,
                                  board_encoder: str, score_normalizer: str,
                                  suffix_to_append: str = "-be?????-sn???.pkl"):
        """
        Example:
            >>> FenToPkl.convert_fen_to_pkl_folder(
            ...     input_dir="../../aggregated output 03",
            ...     output_dir="../../aggregated output 03/be00778-sn002-pkl",
            ...     move_dir="../../aggregated output 03/03_csv",
            ...     board_encoder="BoardEncoder.Encode778",
            ...     score_normalizer="ScoreNormalizer.normalize_002",
            ...     suffix_to_append="-be00778-sn002.pkl"
            ... )
        :param input_dir:
        :param output_dir: 
        :param move_dir: 
        :param board_encoder:
        :param score_normalizer:
        :param suffix_to_append:
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(move_dir).mkdir(parents=True, exist_ok=True)
        if not Path(input_dir).exists():
            raise FileNotFoundError(f"Source path does not exists: '{input_dir}'")
        if not Path(output_dir).exists():
            raise FileNotFoundError(f"Destination path does not exists: '{output_dir}'")
        if not Path(move_dir).exists():
            raise FileNotFoundError(f"Move path does not exists: '{move_dir}'")

        with tqdm(sorted(glob.glob(f"{Path(input_dir)}/*.csv"))) as t:
            for ith_file in t:
                t.set_description(f"File: {Path(ith_file).name}")
                FenToPkl.convert_fen_to_pkl_file(ith_file, output_dir, move_dir, board_encoder, score_normalizer, suffix_to_append)


# pgn_to_csv_all_possible_state("/home/student/Desktop/fenil/35_Final Year Project/aggregated output 03/Test_kaufman")

########################################################################################################################
if __name__ == "__main__":
    from docopt import docopt

    doc_string = \
        """
        Usage: 
            step_02_preprocess.py generate_all_boards_to_csv --file_name=PATH --depth=N
            step_02_preprocess.py pgn_to_csv_parallel --input_dir=PATH [--output_dir=PATH]
            step_02_preprocess.py convert_fen_to_pkl_file --file_path=PATH --output_dir=PATH --move_dir=PATH --board_encoder=[BoardEncoder.Encode778] --score_normalizer=[ScoreNormalizer.normalize_001 | ScoreNormalizer.normalize_002 | ScoreNormalizer.normalize_003] --suffix_to_append=SUFFIX
            step_02_preprocess.py convert_fen_to_pkl_folder --input_dir=PATH --output_dir=PATH --move_dir=PATH --board_encoder=[BoardEncoder.Encode778] --score_normalizer=[ScoreNormalizer.normalize_001 | ScoreNormalizer.normalize_002 | ScoreNormalizer.normalize_003] --suffix_to_append=SUFFIX
            step_02_preprocess.py (-h | --help)
            step_02_preprocess.py --version
            
        Options:
            -h --help    show this
        """
    arguments = docopt(doc_string, argv=None, help=True, version=f"{cs.VERSION} - Preprocess", options_first=False)
    print(arguments)
    if arguments['generate_all_boards_to_csv']:
        generate_all_boards_to_csv(arguments['--file_name'], int(arguments['--depth']))
    elif arguments['pgn_to_csv_parallel']:
        pgn_to_csv_parallel(arguments['--input_dir'], arguments['--output_dir'])
    elif arguments['convert_fen_to_pkl_file']:
        FenToPkl.convert_fen_to_pkl_file(
            arguments['--file_path'],
            arguments['--output_dir'],
            arguments['--move_dir'],
            arguments['--board_encoder'],
            arguments['--score_normalizer'],
            arguments['--suffix_to_append'],
        )
    elif arguments['convert_fen_to_pkl_folder']:
        FenToPkl.convert_fen_to_pkl_folder(
            arguments['--input_dir'],
            arguments['--output_dir'],
            arguments['--move_dir'],
            arguments['--board_encoder'],
            arguments['--score_normalizer'],
            arguments['--suffix_to_append'],
        )
    else:
        print("ERROR: invalid command line arguments")

    r'''
    Example:
        python step_02_preprocess.py pgn_to_csv_parallel \
            --input_dir '/home/student/Desktop/KingBase2019-pgn' \
            --output_dir '/home/student/Desktop/KingBase2019-pgn/output'
    
        python step_02_preprocess.py convert_fen_to_pkl_file \
            --file_path '/home/student/Desktop/fenil_pc/aggregated output 03/03_csv/z__r_B50-B99__aa.csv' \
            --output_dir '/home/student/Desktop/fenil_pc/aggregated output 03/' \
            --move_dir '/home/student/Desktop/fenil_pc/aggregated output 03/03_csv_done' \
            --board_encoder 'BoardEncoder.Encode778' \
            --score_normalizer 'None' \
            --suffix_to_append '-be00778.pkl'

        python step_02_preprocess.py convert_fen_to_pkl_folder \
            --input_dir '/home/student/Desktop/fenil_pc/aggregated output 03/03_csv' \
            --output_dir '/home/student/Desktop/fenil_pc/aggregated output 03/be00778_new1' \
            --move_dir '/home/student/Desktop/fenil_pc/aggregated output 03/03_csv_done' \
            --board_encoder 'BoardEncoder.Encode778' \
            --score_normalizer 'None' \
            --suffix_to_append '-be00778.pkl'
    '''

    # POSSIBLE TASKS that can be performed
    # - Convert PGN to csv
    # - Generate score for csv
    # - CSV to PKL

    # pgn_to_csv_parallel(input_path="/home/student/Desktop/Fenil/Final Year Project/KingBase2019-pgn/")

    # pgn_obj = PreprocessPGN(pgn_file_path="KingBase2019-A80-A99.pgn")
    # print(pgn_obj.get_pgn_game_count())
    # pgn_obj.pgn_to_csv(output_path="./game_limited boards/")
