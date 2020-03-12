import abc
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
    @staticmethod
    def normalize_001(data_np: np.ndarray) -> np.ndarray:
        """
        All cp_scores between [0, 10] are scaled down to [0, 1]

        All cp_scores above 10 are scaled down to 1

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
        All cp_scores between [0, 50] are scaled down to [0, 0.2]

        All cp_scores above 10 are scaled down to 1

        NOTE: Results are not good

        :param data_np:
        :return: Normalized `np.ndarray`
        """
        data_np = copy.deepcopy(data_np)
        data_np *= 10000

        # [0.0, 0.2]
        data_np[np.logical_and(0 <= data_np, data_np <= 50)] /= (50 / 0.2)

        # [0.2, 0.4]
        data_np[np.logical_and(50 <= data_np, data_np <= 2000)] = (data_np[np.logical_and(50 <= data_np, data_np <= 2000)] - 50) / 1950 * 0.2 + 0.2

        # [0.4, 0.8]
        data_np[np.logical_and(2000 <= data_np, data_np <= 6000)] = (data_np[np.logical_and(2000 <= data_np, data_np <= 6000)] - 2000) / 4000 * 0.4 + 0.4

        # [0.8, 1.0]
        data_np[6000 <= data_np] = (data_np[6000 <= data_np] - 6000) / 4000 * 0.2 + 0.8

        # [-0.2, 0.0]
        data_np[np.logical_and(-50 <= data_np, data_np < 0)] /= (50 / 0.2)

        # [-0.4, -0.2]
        data_np[np.logical_and(-2000 <= data_np, data_np <= -50)] = (data_np[np.logical_and(-2000 <= data_np, data_np <= -50)] + 50) / 1950 * 0.2 - 0.2

        # [-0.8, -0.4]
        data_np[np.logical_and(-6000 <= data_np, data_np <= -2000)] = (data_np[np.logical_and(-6000 <= data_np, data_np <= -2000)] + 2000) / 4000 * 0.4 - 0.4

        # [-1.0, -0.8]
        data_np[data_np <= -6000] = (data_np[data_np <= -6000] + 6000) / 4000 * 0.2 - 0.8

        return data_np


#########################################################################################################################
def generate_all_boards(depth_d: int):
    board_b = chess.Board()
    output_list = [[board_b, ], ]
    for i in range(depth_d):
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


#########################################################################################################################
def csv_score_generation(csv_file_path: Union[str, Path], resume_file_name):
    global engine_sf
    engine_sf = step_01_engine.CustomEngine(src_path=None, hash_size_mb=8192, depth=15, analyse_time=0.2)

    file_count = 1
    resume_file_count: int = cs.readpoint(resume_file_name, 1)[0]

    input_data = Path(csv_file_path)
    input_files_list = sorted(list(input_data.glob("*.csv")))
    for i in input_files_list:
        if file_count < resume_file_count:
            print(f"DEBUG: skip {i} = {file_count}", file=sys.stderr)
            file_count += 1
            continue

        print(f"DEBUG: processing file {i} = {file_count}")
        data = pd.read_csv(i)
        line_count = 2
        for j in range(len(data)):
            # data.loc[j][1] = engine_sf.evaluate(chess.Board(data.loc[j][0]))
            if not chess.Board(data.loc[j][0]).is_valid():
                print(f"\rERROR: board state not valid at line count = {line_count}")
            data.at[j, 'cp_score'] = engine_sf.evaluate_fen(data.loc[j][0])
            line_count += 1
            print(f"\r\t{line_count}", end="", file=sys.stderr)
        print(f"\r", end="", file=sys.stderr)

        data.to_csv(i, index=False)
        file_count += 1
        cs.savepoint(resume_file_name, file_count)
        print(f"DEBUG: successfully processed {i}", file=sys.stderr)


def pgn_to_csv_parallel(kingbase_dir: str):
    your_host = '0.0.0.0'  # or use '0.0.0.0' if you have high enough privilege
    your_port = 60006  # the port to be used by the master node
    your_authkey = 'a1'  # this is the password clients use to connect to the master(i.e. the current node)
    your_chunksize = 1
    sh = Shell(has_input=False, record_output=True, record_errors=True, strip_empty=True)
    #################################

    print()
    cs.print_ip_port_auth(your_port=your_port, your_authkey=your_authkey)

    master = MasterNode(HOST=your_host, PORT=your_port, AUTHKEY=your_authkey, chunk_size=your_chunksize)
    master.start_master_server(if_join_as_worker=False)
    master.load_envir("""from step_02_preprocess import *""" +
                      """\ndef fun1(arg1):"""
                      """\n    pgn_obj = PreprocessPGN(pgn_file_path=arg1)""" +
                      """\n    return pgn_obj.get_pgn_game_count()"""
                      """\ndef fun2(arg2):"""
                      """\n    pgn_obj = PreprocessPGN(pgn_file_path=arg2)""" +
                      """\n    pgn_obj.pgn_to_csv(output_path="/home/student/Desktop/Fenil/Final Year Project/KingBase2019-pgn/data_out_pgn/", debug_flag=1)""",
                      from_file=False)
    master.register_target_function("fun2")  # CHANGE this as per requirement
    master.load_args([str(Path(kingbase_dir) / i) for i in sh.run(f"ls '{kingbase_dir}'").output(raw=False)])
    result = master.execute()

    joblib.dump(result, "pgn_game_processing.obj")


#########################################################################################################################
if __name__ == "__main__":
    pgn_to_csv_parallel(kingbase_dir="/home/student/Desktop/Fenil/Final Year Project/KingBase2019-pgn/")

    # pgn_obj = PreprocessPGN(pgn_file_path="KingBase2019-A80-A99.pgn")
    # print(pgn_obj.get_pgn_game_count())
    # pgn_obj.pgn_to_csv(output_path="./game_limited boards/")
