from typing import Union, IO, Iterable, List
from pathlib import Path
import os
import sys
import subprocess
import pandas as pd
import chess.pgn

import common_services as cs
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


#########################################################################################################################
engine_sf = None
MINI_BATCH_SIZE: int = 1000
BATCH_SIZE: int = 10000
COLUMNS = ["fen_board", "cp_score"]


def preprocess_pgn(pgn_file_path: Union[str, Path], output_path: Union[str, Path], resume_file_name):
    global engine_sf, BATCH_SIZE, COLUMNS

    if not Path(output_path).exists():
        Path(output_path).mkdir(parents=True)

    INPUT_PGN_NAME: str = Path(pgn_file_path).name[:-4]

    engine_sf = step_01_engine.CustomEngine(src_path=None, hash_size_mb=8192, depth=15, analyse_time=0.2)

    res_pd = pd.DataFrame(data=None, index=None, columns=COLUMNS)
    # Used this with "KingBase2019-A80-A99.pgn"
    default_board = chess.Board()
    for i in default_board.legal_moves:
        default_board.push(i)
        res_pd.loc[len(res_pd)] = [default_board.fen(), None]
        default_board.pop()

    pgn_file = load_pgn(pgn_file_path)
    game_count = 1
    # file_count = 1
    # (last game written + 1), (next file number to be written)
    resume_game_count, file_count = cs.readpoint(resume_file_name, 2)
    print(resume_game_count, file_count)
    if game_count < resume_game_count and (Path(output_path) / f"{INPUT_PGN_NAME}_{file_count:06}.csv").exists():
        res_pd = pd.read_csv(Path(output_path) / f"{INPUT_PGN_NAME}_{file_count:06}.csv")
        print(f"INFO: loading partially saved file, lines = {len(res_pd)}", file=sys.stderr)

    for i in iterate_pgn(pgn_file):
        if game_count < resume_game_count:
            print(f"DEBUG: skip {game_count}", file=sys.stderr)
            game_count += 1
            continue

        print(f"DEBUG: processing game_count = {game_count} ({len(res_pd)})", file=sys.stderr)
        board_count = 1
        for j in iterate_game(i):
            print(f"\r\t{board_count}", end="", file=sys.stderr)
            board_count += 1

            if not j.is_valid():
                print(f"*** WARNING: invalid board state in the game: {j.fen()}", file=sys.stderr)
                continue

            try:
                res_pd.loc[len(res_pd)] = [j.fen(), engine_sf.evaluate(j)]
                # for k in generate_boards(j):
                #     res_pd.loc[len(res_pd)] = [k.fen(), None]
            except Exception as e:
                print(f"ERROR: {e}")
                print(f"\tINFO: is_valid() = {j.is_valid()}, board.fen() = {j.fen()}")

        print("\r", end="", file=sys.stderr)

        game_count += 1
        if len(res_pd) > BATCH_SIZE or len(res_pd) > MINI_BATCH_SIZE * (max(0, len(res_pd) - 1) // MINI_BATCH_SIZE + 1):
            output_file = Path(output_path) / f"{INPUT_PGN_NAME}_{file_count:06}.csv"
            res_pd.to_csv(output_file, index=False)
            print(f"DEBUG: boards successfully written to file: {output_file}", file=sys.stderr)
            if len(res_pd) > BATCH_SIZE:
                file_count += 1
                cs.append_secondlast_line(resume_file_name, f"# {len(res_pd)} {game_count} {file_count}")
                res_pd = pd.DataFrame(data=None, index=None, columns=COLUMNS)
            cs.savepoint(resume_file_name, f"{game_count},{file_count}")

    if len(res_pd) > 0:
        output_file = Path(output_path) / f"{INPUT_PGN_NAME}_{file_count:06}.csv"
        res_pd.to_csv(output_file, index=False)
        print(f"\nDEBUG: boards successfully written to file: {output_file}", file=sys.stderr)
        file_count += 1
        cs.savepoint(resume_file_name, f"{game_count},{file_count}")


def preprocess_gen_score(csv_file_path: Union[str, Path], resume_file_name):
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
            data.at[j, 'cp_score'] = engine_sf.evaluate(chess.Board(data.loc[j][0]))
            line_count += 1
            print(f"\r\t{line_count}", end="", file=sys.stderr)
        print(f"\r", end="", file=sys.stderr)

        data.to_csv(i, index=False)
        file_count += 1
        cs.savepoint(resume_file_name, file_count)
        print(f"DEBUG: successfully processed {i}", file=sys.stderr)


if __name__ == "__main__":
    preprocess_pgn(pgn_file_path="KingBase2019-A80-A99.pgn", output_path="./game_limited boards/", resume_file_name="./game_limited boards/z_game_num_limited_str_game.txt")
    # preprocess_pgn(pgn_file_path="KingBase2019-A80-A99.pgn", output_path="./game_all possible boards/", "./game_all possible boards/z_game_num.txt")
    # preprocess_gen_score(csv_file_path="./game_limited boards/", resume_file_name="./game_limited boards/z_game_num_limited_cp_score.txt")
    # chess.Board().fen()
