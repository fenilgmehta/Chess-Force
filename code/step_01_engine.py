import multiprocessing
import os
import random
import sys
import time
from pathlib import Path
from typing import Union

import chess.engine
import numpy as np


class CustomEngine:
    def __init__(self, src_path: Union[str, Path] = None,
                 cp_score_max: int = 8000,
                 mate_score_max: int = 10000,
                 mate_score_difference: int = 50,
                 cpu_cores: int = multiprocessing.cpu_count(),
                 hash_size_mb: int = 16,
                 depth: int = 20,
                 analyse_time: float = 1.0) -> None:
        self.MAX_THREADS_TO_USE = cpu_cores
        self.MAX_HASH_TABLE_SIZE = hash_size_mb  # size is in MB
        self.MAX_DEPTH = depth
        self.MAX_ANALYSE_TIME = analyse_time  # time is in seconds

        self.CP_SCORE_MAX = cp_score_max
        # MAX CP score witnessed = +7311
        self.MATE_SCORE_MAX = mate_score_max  # max score any board can get after check mate
        self.MATE_SCORE_DIFFERENCE = mate_score_difference  # minimum difference between CP value for boards with check mate possible in "n" and "n-1" moves

        if self.MAX_THREADS_TO_USE < 1:
            self.MAX_THREADS_TO_USE = 1
        if self.MAX_HASH_TABLE_SIZE < 16:
            self.MAX_HASH_TABLE_SIZE = 16
        if self.MAX_DEPTH < 1:
            self.MAX_DEPTH = 1
        if self.MAX_ANALYSE_TIME <= 0:
            self.MAX_ANALYSE_TIME = 0.1
        if (self.CP_SCORE_MAX is None) or (self.CP_SCORE_MAX <= 0):
            self.CP_SCORE_MAX = 8000
        if (self.MATE_SCORE_MAX is None) or (self.MATE_SCORE_MAX <= 0):
            self.MATE_SCORE_MAX = 10000
        if (self.MATE_SCORE_DIFFERENCE is None) or (self.MATE_SCORE_DIFFERENCE <= 0):
            self.MATE_SCORE_DIFFERENCE = 50

        if src_path is None:
            DEFAULT_ENGINE_PATH = Path(r"../chess_engines/Linux/stockfish_10_x64")
            if not DEFAULT_ENGINE_PATH.exists():
                DEFAULT_ENGINE_PATH = Path(r"chess_engines/Linux/stockfish_10_x64")
            if not DEFAULT_ENGINE_PATH.exists():
                print("WARNING: default path does not exists", file=sys.stderr)
            self.engine_path = DEFAULT_ENGINE_PATH
        else:
            self.engine_path = src_path

        self.engine_obj = self.__load()
        if self.engine_obj is None:
            raise Exception(f"FileNotFound: {self.engine_path}")

        self.__update_settings(hash_size=self.MAX_HASH_TABLE_SIZE, cpu_cores=self.MAX_THREADS_TO_USE)

    def __load(self) -> Union[chess.engine.SimpleEngine, None]:
        if not Path(self.engine_path).exists():
            print("ERROR: src_path does not exists", file=sys.stderr)

        src_path = str(self.engine_path)
        try:
            engine_obj = chess.engine.SimpleEngine.popen_uci(src_path)
        except PermissionError:
            print(f"WARNING: PermissionError occurred when loading the engine. Hence changing the permission to 777", file=sys.stderr)
            os.system(f"chmod 777 {src_path}")
            try:
                engine_obj = chess.engine.SimpleEngine.popen_uci(src_path)
            except PermissionError:
                print("ERROR: unable to change the permission of the engine.")
                return None
            except Exception as e:
                print(f"ERROR: unable to load engine, see the following error:\n\t{e}", file=sys.stderr)
                return None
        except Exception as e:
            print(f"ERROR: unable to load engine, see the following error:\n\t{e}", file=sys.stderr)
            return None
        return engine_obj

    def __update_settings(self, cpu_cores: int, hash_size: int):
        # https://python-chess.readthedocs.io/en/latest/engine.html#options
        # Even after changing the settings this statement will always print the same thing
        # print("", engine_sf.options["Hash"], "\n", engine_sf.options["Threads"])
        self.engine_obj.configure({"Threads": cpu_cores})
        self.engine_obj.configure({"Hash": hash_size})
        return

    # TODO: verify this function once
    def evaluate_board(self, board: chess.Board) -> int:
        # https://python-chess.readthedocs.io/en/latest/engine.html#analysing-and-evaluating-a-position
        score1 = self.engine_obj.analyse(
            board=board,
            limit=chess.engine.Limit(time=self.MAX_ANALYSE_TIME, depth=self.MAX_DEPTH),
            info=chess.engine.INFO_SCORE
        ).score

        if score1.white().is_mate():
            # white_sign = np.sign(info['score'].white().mate())
            # The above statement returns:
            # (1) +1 value if white mates black -> white win if optimal play
            # (2) -1 value if black mates white -> black win if optimal play
            return (self.MATE_SCORE_MAX * np.sign(score1.white().mate())) \
                   + (score1.white().score(mate_score=0) * self.MATE_SCORE_DIFFERENCE)
            # info['score'].white().score(mate_score=0)
            # The above statement return:
            # (1) +ve integer value if black mates white
            # (2) -ve integer value if white mates black

        int_score = int(str(score1.white()))
        if int_score > 0:
            return min(int_score, self.CP_SCORE_MAX)
        return max(int_score, -self.CP_SCORE_MAX)

    def evaluate_fen(self, board_fen: str):
        return self.evaluate_board(chess.Board(board_fen))

    def evaluate_normalized_board(self, board: chess.Board):
        return self.evaluate_board(board) / self.MATE_SCORE_MAX

    def evaluate_normalized_fen(self, board_fen: str):
        return self.evaluate_board(chess.Board(board_fen)) / self.MATE_SCORE_MAX

    def finish(self):
        self.engine_obj.close()


##################################################################################################
if __name__ == "__main__":
    engine_sf = None

    try:
        del engine_sf
    except:
        pass
    engine_sf = CustomEngine(src_path=None,
                             cp_score_max=8000,
                             mate_score_max=10000,
                             mate_score_difference=50,
                             cpu_cores=1,
                             hash_size_mb=16,
                             depth=20,
                             analyse_time=1.0)

    # board = chess.Board("r1bqkbnr/p1pp1ppp/1pn5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 2 4")
    BOARD_STR = ['rnb1k2r/1pppq1pp/4pn2/5p2/1pPP4/1Q2PN2/PP3PPP/RN2KB1R w KQkq - 2 8',
                 'r4N1k/5npp/3qp3/3p4/Pp3n2/1P2R3/3N1PPP/R4QK1 b - - 0 25',
                 '2r2r1k/pb4p1/3p1q1p/1Pn1p2P/8/3B1N2/PPQ2PP1/1K1RR3 w - - 0 19',
                 '8/8/4q1k1/4bp1p/7P/5QP1/6NK/8 b - - 5 69',
                 'r1b2rk1/1pp3pp/2np1b2/p3p2q/2PP1p2/PP2PNP1/1BQ2PBP/3R1RK1 w - - 0 16',
                 '5r1k/ppnqr1bp/2ppp1p1/8/2PP4/1P1Q1NP1/PB2RP1P/4R1K1 b - - 7 20',
                 'r1b2rk1/pp2qnpp/2p1p3/3pNp2/2PP4/3BP3/PPQ2PPP/R4RK1 w - - 1 14',
                 '8/8/8/5K2/5P2/1p4k1/1P6/8 w - - 1 76',
                 'r1bq1r1k/1pp3bp/p2n2p1/2NPpp2/8/1Q2B1P1/PP2PPBP/2RR2K1 b - - 3 16',
                 'r2q1rk1/pppn1bb1/3p1npp/4pp2/2PP3P/1PN1PN2/PB1Q1PP1/R3KB1R w KQ - 7 12']

    # ------------------------------------

    random.shuffle(BOARD_STR)
    start = time.time()
    for i in BOARD_STR: print(engine_sf.evaluate_fen(i))
    end = time.time()
    print(f"running time = {end - start}")
    print(f"board count = {len(BOARD_STR)}")

    # ------------------------------------

    board = chess.Board()

    start = time.time()
    for i in range(10): print(engine_sf.evaluate_board(board))
    end = time.time()
    print(f"running time = {end - start}")
    print(f"board count = {10}")

# default_param = {
#     "Write Debug Log": "false",
#     "Contempt": 0,
#     "Min Split Depth": 0,
#     "Threads": 1,
#     "Ponder": "false",
#     "Hash": 16,
#     "MultiPV": 1,
#     "Skill Level": 20,
#     "Move Overhead": 30,
#     "Minimum Thinking Time": 20,
#     "Slow Mover": 80,
#     "UCI_Chess960": "false",
# }
