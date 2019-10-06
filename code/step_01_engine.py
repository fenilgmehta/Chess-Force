from typing import Union
from pathlib import Path
import sys
import os
import multiprocessing
import pandas as pd
import chess.engine


class CustomEngine:
    def __init__(self, src_path: Union[str, Path] = None,
                 cpu_cores: int = multiprocessing.cpu_count(),
                 hash_size_mb: int = 4096,
                 depth: int = 15,
                 analyse_time: float = 0.2) -> None:
        self.MAX_THREADS_TO_USE = cpu_cores
        self.MAX_HASH_TABLE_SIZE = hash_size_mb  # size is in MB
        self.MAX_DEPTH = depth
        self.MAX_ANALYSE_TIME = analyse_time  # time is in seconds

        if self.MAX_THREADS_TO_USE < 1:
            self.MAX_THREADS_TO_USE = 1
        if self.MAX_HASH_TABLE_SIZE < 16:
            self.MAX_HASH_TABLE_SIZE = 16
        if self.MAX_DEPTH < 1:
            self.MAX_DEPTH = 1
        if self.MAX_ANALYSE_TIME <= 0:
            self.MAX_ANALYSE_TIME = 0.1

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
        # Even after changing the settings this statement will always print the same thing
        # print("", engine_sf.options["Hash"], "\n", engine_sf.options["Threads"])
        self.engine_obj.configure({"Threads": cpu_cores})
        self.engine_obj.configure({"Hash": hash_size})
        return

    def evaluate(self, board: chess.Board):
        # https://python-chess.readthedocs.io/en/v0.25.0/engine.html#analysing-and-evaluating-a-position
        info = self.engine_obj.analyse(board, chess.engine.Limit(time=self.MAX_ANALYSE_TIME, depth=self.MAX_DEPTH))
        # print(info["score"])
        return info["score"]

    # TODO
    def evaluate_normalized(self, board: chess.Board):
        return self.evaluate(board)

    def finish(self):
        self.engine_obj.close()


##################################################################################################
if __name__ == "__main__":
    engine_sf = CustomEngine()

    # board = chess.Board("r1bqkbnr/p1pp1ppp/1pn5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 2 4")
    board = chess.Board()
    import time

    start = time.time()
    print(engine_sf.evaluate(board))
    print(engine_sf.evaluate(board))
    print(engine_sf.evaluate(board))
    print(engine_sf.evaluate(board))
    print(engine_sf.evaluate(board))
    print(engine_sf.evaluate(board))
    print(engine_sf.evaluate(board))
    print(engine_sf.evaluate(board))
    print(engine_sf.evaluate(board))
    print(engine_sf.evaluate(board))
    end = time.time()
    print(end - start)

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

#
# from stockfish import Stockfish
#
# stockfish = Stockfish(str(DEFAULT_ENGINE_PATH), depth=21, param={"Hash": MAX_HASH_TABLE_SIZE, "Threads": MAX_THREADS_TO_USE})
# stockfish.set_fen_position(chess.Board().fen())
#
# stockfish.stockfish.stdin.write(f"go depth {stockfish.depth}\n")
# stockfish.stockfish.stdin.flush()
# while True:
#     text = stockfish.stockfish.stdout.readline().strip()
#     print(text)
#     # split_text = text.split(" ")
#     # if split_text[0] == "bestmove":
#     #     if split_text[1] == "(none)":
#     #         return None
#     #     self.info = last_text
#     #     return split_text[1]
#     # last_text = text
