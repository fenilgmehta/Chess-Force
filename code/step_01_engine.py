from typing import Union
from pathlib import Path
import chess.engine
import sys

DEFAULT_ENGINE_PATH = Path(r"../chess_engines/Linux/stockfish_10_x64")
if not DEFAULT_ENGINE_PATH.exists():
    DEFAULT_ENGINE_PATH = Path(r"chess_engines/Linux/stockfish_10_x64")
if not DEFAULT_ENGINE_PATH.exists():
    print("WARNING: default path does not exists", file=sys.stderr)


def load(src_path: Union[str, Path] = DEFAULT_ENGINE_PATH):
    if not Path(src_path).exists():
        print("ERROR: src_path does not exists", file=sys.stderr)
    return chess.engine.SimpleEngine.popen_uci(src_path)


# TODO
def update_settings(engine: chess.engine.SimpleEngine, hash_size: int, cpu_cores: int):
    return


# TODO
def evaluate(engine: chess.engine.SimpleEngine, board: chess.Board, time: int = None, depth: int = None):
    # https://python-chess.readthedocs.io/en/v0.25.0/engine.html#analysing-and-evaluating-a-position
    info = engine.analyse(board, chess.engine.Limit(time=time, depth=depth))
    print(info["score"])


# TODO
def evaluate_normalized(engine: chess.engine.SimpleEngine, board: chess.Board, time: int = None, depth: int = None):
    return evaluate(engine, board, time, depth)


def finish(engine: chess.engine.SimpleEngine):
    engine.close()
