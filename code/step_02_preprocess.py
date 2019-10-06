from typing import Union, IO, Iterable, List
from pathlib import Path
import os
import sys
import subprocess
import pandas as pd
import chess.pgn

import step_01_engine


def load_pgn(pgn_file_path: Union[str, Path]) -> Union[IO, None]:
    # https://python-chess.readthedocs.io/en/latest/pgn.html
    if not Path(pgn_file_path).exists():
        print(f"ERROR: {pgn_file_path} does not exist", file=sys.stderr)
        return None
    pgn = open(pgn_file_path, mode="r")
    print(f"DEBUG: pgn file {pgn_file_path} successfully loaded", file=sys.stderr)
    return pgn


def iterate_pgn(pgn_obj: chess.pgn) -> Iterable[chess.pgn.Game]:
    game = chess.pgn.read_game(pgn_obj)
    while game is not None:
        yield game
        try:
            game = chess.pgn.read_game(pgn_obj)
        except UnicodeDecodeError as e:
            print(f"WARNING: it seems pgn file has been completely read, UnicodeDecodeError occured:\n\t{e}", file=sys.stderr)
            break
    return


def iterate_game(nth_game) -> Iterable[chess.Board]:
    board = nth_game.board()
    for move in nth_game.mainline_moves():
        board.push(move)
        yield board
    return


def generate_boards(board: chess.Board) -> Iterable[chess.Board]:
    legal_move = board.legal_moves
    for move in legal_move:
        board.push(move)
        yield board
        board.pop()


# TODO
def encode_board_777(board) -> str:
    return ""


def preprocess_pgn(pgn_file_path: Union[str, Path], output_path: Union[str, Path], resume_file_name):
    def savepoint(game_count, file_count):
        # (last game written + 1), (next file number to be written)
        os.system(f"echo '{game_count},{file_count}' > '{resume_file_name}'")

    def readpoint() -> List[int]:
        if not Path(resume_file_name).exists():
            return [1, 1]
        return [int(i) for i in subprocess.getoutput(f"cat '{resume_file_name}'").split(",")]

    BATCH_SIZE: int = 10000
    INPUT_PGN_NAME: str = Path(pgn_file_path).name[:-4]
    COLUMNS = ["fen_board", "cp_score"]

    res_pd = pd.DataFrame(data=None, index=None, columns=COLUMNS)
    default_board = chess.Board()
    for i in default_board.legal_moves:
        default_board.push(i)
        res_pd.loc[len(res_pd)] = [default_board.board_fen(), None]
        default_board.pop()

    pgn_file = load_pgn(pgn_file_path)
    game_count = 1
    # file_count = 1
    resume_game_count, file_count = readpoint()

    for i in iterate_pgn(pgn_file):
        if game_count < resume_game_count:
            game_count += 1
            continue

        print(f"DEBUG: processing game_count = {game_count}", file=sys.stderr)
        board_count = 1
        for j in iterate_game(i):
            print(f"\r\t{board_count}", end="", file=sys.stderr)
            board_count += 1
            res_pd.loc[len(res_pd)] = [j.board_fen(), None]
            # for k in generate_boards(j):
            #     res_pd.loc[len(res_pd)] = [k.board_fen(), None]
        print("\r", end="", file=sys.stderr)

        game_count += 1
        if len(res_pd) > BATCH_SIZE:
            output_file = Path(output_path) / f"{INPUT_PGN_NAME}_{file_count:06}.csv"
            res_pd.to_csv(output_file, index=False)
            print(f"DEBUG: boards successfully written to file: {output_file}", file=sys.stderr)
            file_count += 1
            savepoint(game_count, file_count)

            res_pd = pd.DataFrame(data=None, index=None, columns=COLUMNS)


if __name__ == "__main__":
    preprocess_pgn(pgn_file_path="KingBase2019-A80-A99.pgn", output_path="./game_limited boards/", resume_file_name="z_game_num_limited.txt")
    # preprocess_pgn(pgn_file_path="KingBase2019-A80-A99.pgn", output_path="./game_all possible boards/", "z_game_num.txt")
