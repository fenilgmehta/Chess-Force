import os
import random
from typing import Tuple

import numpy as np

import step_01_TicTacToe as step_01
from step_02_ffnn import FFNN_sklearn


def model_play_move_first_best(board_obj: step_01.TicTacToe, model_obj: FFNN_sklearn, to_debug=False) -> Tuple[step_01.TicTacToe, np.float]:
    best_moves = []
    best_score = -10
    for i, j in board_obj.get_legal_moves():
        board_obj.push(i, j)
        temp_score = model_obj.predict_score(board_obj)
        if to_debug:
            print(f"\n\nDEBUG: temp_score = {temp_score}, board = \n{board_obj}")
        best_moves.append((i, j, temp_score))
        # if temp_score > best_score:
        #     best_move = (i, j)
        #     best_score = temp_score
        board_obj.pop()
    best_moves.sort(key=lambda x: x[2], reverse=True)

    possible_moves = []
    for i in range(len(best_moves)):
        if best_moves[i][2] >= 0:
            possible_moves.append(best_moves[i])

    if len(possible_moves) == 0:
        print(f"INFO: no +ve score, playing: {best_moves[0]}")
        possible_moves.append(best_moves[0])

    print(f"INFO: possible moves = {best_moves}")
    print(f"INFO: moves to play = {possible_moves}")
    random.shuffle(possible_moves)
    board_obj.push(possible_moves[0][0], possible_moves[0][1])
    return possible_moves[0][2]


def model_play_move_first(board_obj: step_01.TicTacToe, model_obj: FFNN_sklearn, to_debug=False) -> Tuple[step_01.TicTacToe, np.float]:
    best_move = None
    best_score = -10
    for i, j in board_obj.get_legal_moves():
        board_obj.push(i, j)
        temp_score = model_obj.predict_score(board_obj)
        if to_debug:
            print(f"\n\nDEBUG: temp_score = {temp_score}, board = \n{board_obj}")
        if temp_score > best_score:
            best_move = (i, j)
            best_score = temp_score
        board_obj.pop()
    board_obj.push(best_move[0], best_move[1])
    return best_score


def model_play_move_second(board_obj: step_01.TicTacToe, model_obj: FFNN_sklearn, to_debug=False) -> Tuple[step_01.TicTacToe, np.float]:
    best_move = None
    best_score = 10
    for i, j in board_obj.get_legal_moves():
        board_obj.push(i, j)
        temp_score = model_obj.predict_score(board_obj)
        if to_debug:
            print(f"\n\nDEBUG: temp_score = {temp_score}, board = \n{board_obj}")
        if temp_score < best_score:
            best_move = (i, j)
            best_score = temp_score
        board_obj.pop()
    board_obj.push(best_move[0], best_move[1])
    return -best_score


def model_play_move(board_obj, model_obj, cpu_is_first: bool, to_debug=False):
    if cpu_is_first:
        return model_play_move_first_best(board_obj, model_obj, to_debug)
    else:
        return model_play_move_second(board_obj, model_obj, to_debug)


def play(cpu_is_first: bool, to_debug=False):
    ffnn_sk = FFNN_sklearn()
    ffnn_sk.load_model()
    ttt = step_01.TicTacToe(size_n=3, winner_len=3, max_depth=None, debug=0)

    if cpu_is_first:
        best_score = model_play_move(ttt, ffnn_sk, cpu_is_first, to_debug)
        print(f"\n\nDEBUG: board score = {best_score}")
        print(f"\n{ttt}")
    else:
        print(ttt)

    while not ttt.game_over:
        step_01.user_play(ttt)
        print(f"\n\n{ttt}")

        if ttt.game_over:
            continue

        best_score = model_play_move(ttt, ffnn_sk, cpu_is_first, to_debug)
        print(f"\n\nDEBUG: board score = {best_score}")
        print(f"\n{ttt}")
    else:
        print(f"\n\nGAME OVER: ", end="")
        if ttt.winner == 0:
            print(f"draw")
        else:
            if cpu_is_first:
                if ttt.winner == 1:
                    print(f"winner = CPU")
                else:
                    print(f"winner = PLAYER")
            else:
                if ttt.winner == 1:
                    print(f"winner = PLAYER")
                else:
                    print(f"winner = CPU")


if __name__ == "__main__":
    os.chdir("TicTacToe")
    first_move = input("Do you want to play first (y/n) ?") == 'y'
    play(not first_move, to_debug=False)
