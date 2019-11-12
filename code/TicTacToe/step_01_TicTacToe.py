import math
from typing import Dict, List, Tuple, Iterable

import numpy as np
import pandas as pd


class TicTacToe:
    SHAPE = ['x', 'o']

    def __init__(self, size_n: int = 3, winner_len: int = 3, max_depth: int = None, debug: int = 0):
        #  1 => x
        # -1 => 0

        self.SIZE: int = size_n
        self.SIZE_SQUARE: int = size_n ** 2
        self.MAX_SCORE: int = 2 * self.SIZE_SQUARE
        self.board: pd.DataFrame = pd.DataFrame(data=size_n * [size_n * [0, ]])

        self.WINNER_LEN: int = min(winner_len, self.SIZE)
        self.MAX_DEPTH: int = (self.SIZE_SQUARE + 1) if max_depth is None else max_depth
        self.DEBUG: int = debug

        self.turn: int = 1
        self.empty_cell_count: int = self.SIZE_SQUARE

        self._my_hash_table: Dict[Tuple[int], Tuple[int, int, int, int]] = dict()
        self._my_stack: List[Tuple[int, int]] = []

        self.game_over: bool = False
        self.winner: int = 0

    def reinitialize(self):
        self.empty_cell_count = 0
        for i in self.board.values.ravel():
            if i == 0:
                self.empty_cell_count += 1
        self.game_over = False
        self.winner = 0

    def __toggle_turn(self):
        self.turn *= (-1)

    def get_legal_moves(self) -> Iterable[Tuple[int, int]]:
        for i in range(self.SIZE):
            for j in range(self.SIZE):
                if self.is_legal_move(i, j):
                    yield (i, j,)

    def is_legal_move(self, i_new, j_new) -> bool:
        if (0 <= i_new < self.SIZE) and (0 <= j_new < self.SIZE):
            return bool(self.board.loc[i_new][j_new] == 0)
        return False

    def push(self, i_new, j_new):
        """
        Play next move at "i_new", "j_new" and change the turn

        :param i_new:
        :param j_new:
        """
        if self.game_over:
            raise Exception("ERROR: GameOver")
        if self.empty_cell_count == 0:
            raise Exception("ERROR: GameOver: Draw")

        if self.is_legal_move(i_new, j_new):
            self._my_stack.append((i_new, j_new,))
            self.board.loc[i_new][j_new] = self.turn
            self.__toggle_turn()
            self.empty_cell_count -= 1
            if self.check_winner(1):
                self.game_over = True
                self.winner = 1
            elif self.check_winner(-1):
                self.game_over = True
                self.winner = -1
            elif self.empty_cell_count == 0:
                self.game_over = True
                self.winner = 0
        else:
            raise Exception("ERROR: IllegalMove")

    def pop(self):
        self.game_over = False
        if len(self._my_stack) > 0:
            self.board.loc[self._my_stack[-1][0]][self._my_stack[-1][1]] = 0
            self._my_stack.pop()
            self.__toggle_turn()
            self.empty_cell_count += 1

    def __check_line_right(self, i_check, j_check, shape_to_check: int):
        if self.DEBUG >= 5:
            print(f"DEBUG: right = {i_check},{j_check}")

        if (j_check + self.WINNER_LEN) > self.SIZE:
            return False

        for j in range(self.WINNER_LEN):
            if shape_to_check != self.board.loc[i_check][j_check + j]:
                return False

        return True

    def __check_line_down(self, i_check, j_check, shape_to_check: int):
        if self.DEBUG >= 5:
            print(f"DEBUG: down = {i_check},{j_check}")

        if (i_check + self.WINNER_LEN) > self.SIZE:
            return False

        for i in range(self.WINNER_LEN):
            if shape_to_check != self.board.loc[i_check + i][j_check]:
                return False

        return True

    def __check_line_slant_right(self, i_check, j_check, shape_to_check: int):
        if self.DEBUG >= 5:
            print(f"DEBUG: slant right = {i_check},{j_check}")

        if (i_check + self.WINNER_LEN) > self.SIZE:
            return False
        if (j_check + self.WINNER_LEN) > self.SIZE:
            return False

        for k in range(self.WINNER_LEN):
            if shape_to_check != self.board.loc[i_check + k][j_check + k]:
                return False

        return True

    def __check_line_slant_left(self, i_check, j_check, shape_to_check: int):
        if self.DEBUG >= 5:
            print(f"DEBUG: slant left = {i_check},{j_check}")

        if (i_check + self.WINNER_LEN) > self.SIZE:
            return False
        if (j_check - self.WINNER_LEN) < -1:
            return False

        for k in range(self.WINNER_LEN):
            if shape_to_check != self.board.loc[i_check + k][j_check - k]:
                return False

        return True

    def check_winner(self, shape_to_check: int) -> bool:
        """
        Return a boolean stating whether "shape_to_check" (either 1 or -1) is winner or not
        :param shape_to_check:
        :return:
        """
        if self.DEBUG >= 5:
            print(f"\n\nDEBUG: called check_winner")
        for i in range(self.SIZE):
            for j in range(self.SIZE):
                result = self.__check_line_right(i, j, shape_to_check) or \
                         self.__check_line_down(i, j, shape_to_check) or \
                         self.__check_line_slant_right(i, j, shape_to_check) or \
                         self.__check_line_slant_left(i, j, shape_to_check)
                if result:
                    return True
        return False

    def __min_max(self, player_shape: int = 1,
                  opponent_shape: int = -1,
                  is_maximizer: bool = True,
                  depth_d: int = 1) -> Tuple[int, int, int, int]:
        """
        Returns integer denoting the Tic Tac Toe board score using min-max search

        :param player_shape:
        :param opponent_shape:
        :param is_maximizer:
        :param depth_d:
        :return:
        """
        if self.DEBUG >= 3:
            print(f"DEBUG: __min_max(...), player_shape = {player_shape}, opponent_shape = {opponent_shape}, is_maximizer = {is_maximizer}, depth_d = {depth_d}")

        if tuple(self.board.values.ravel()) in self._my_hash_table:
            return self._my_hash_table[tuple(self.board.values.ravel())]

        if self.check_winner(opponent_shape):
            if not is_maximizer:
                return self.MAX_SCORE - depth_d, -1, -1, player_shape
            else:
                return -(self.MAX_SCORE - depth_d), -1, -1, player_shape

        if depth_d > self.MAX_DEPTH:
            return 0, -1, -1, player_shape

        # DRAW situation
        if self.empty_cell_count == 0:
            return 0, -1, -1, -1

        i_res, j_res = -1, -1
        min_max_win_score = 0  # self.MAX_SCORE
        # if is_maximizer:
        #     min_max_win_score = -min_max_win_score

        for i in range(self.SIZE):
            for j in range(self.SIZE):
                if self.DEBUG >= 2 and depth_d == 1:
                    print(f"DEBUG: i = {i}, j = {j}")

                if self.is_legal_move(i, j):
                    self.push(i, j)

                    win_score = self.__min_max(opponent_shape, player_shape, not is_maximizer, depth_d + 1)[0]
                    if is_maximizer:
                        if win_score > min_max_win_score:
                            min_max_win_score = win_score
                            i_res, j_res = i, j
                    else:
                        if win_score < min_max_win_score:
                            min_max_win_score = win_score
                            i_res, j_res = i, j

                    # self._my_hash_table.setdefault(tuple(self.board.values.ravel()), (0 if abs(win_score / self.MAX_SCORE) == 1 else win_score, i_res, j_res,))
                    self._my_hash_table.setdefault(tuple(self.board.values.ravel()), (win_score, i_res, j_res, opponent_shape))
                    self.pop()

        if self.DEBUG >= 3:
            print(f"DEBUG: min_max_win_score = {min_max_win_score}, i_res = {i_res}, j_res = {j_res}")
        return min_max_win_score, i_res, j_res, player_shape

    def get_best_move(self, turn=None) -> Tuple[int, int]:
        if turn is None:
            turn = self.turn

        if self.DEBUG >= 1:
            print(f"DEBUG: get_best_move(...), turn = {turn}")

        if turn == 1:
            return self.__min_max(1, -1)[1:3]
        if turn == -1:
            return self.__min_max(-1, 1)[1:3]

    def encode(self):
        """
        Return a string to store in csv
        Can pre-process this to feed into the Neural Network
        f"{self.SIZE} {self.WINNER_LEN} {self.turn} {board values...}"

        :return:
        """
        result = [self.turn]
        result.extend(list(self.board.values.ravel() == 1))
        result.extend(list(self.board.values.ravel() == -1))
        return np.array(result).reshape(1, -1)

    def __str__(self):
        hyphen_len = math.ceil(math.log10(self.SIZE_SQUARE)) + 2
        line = "+"
        for i in range(self.SIZE):
            line += (hyphen_len * "-") + "+"

        result = "     "
        for i in range(self.SIZE):
            result += f"{i:{hyphen_len - 1}}  "
        result += "\n    " + line
        for i in range(self.SIZE):
            result += f"\n{i:3} |"
            for j in range(self.SIZE):
                if self.board.iloc[i][j] == 0:
                    result += f"{'':{hyphen_len - 1}} |"
                else:
                    result += f"{self.board.iloc[i][j] :{hyphen_len - 1}} |"
            result += f"\n    {line}"

        return result
        # return self.board.to_string()


def encode_for_model(encoded_str: str):
    arr: List[int] = [int(i) for i in encoded_str.split(" ")]
    board_one = [i == 1 for i in arr[3:]]
    board_two = [i == -1 for i in arr[3:]]

    result: List = [arr[0]]
    result.extend(board_one)
    result.extend(board_two)

    return result


def user_play(tic_tac_toe_obj: TicTacToe):
    while True:
        try:
            arr = [int(i) for i in input().split(" ")]
        except:
            print(f"WARNING: avoid adding extra spaces")
            continue
        if len(arr) != 2:
            print("WARNING: please give two space separated inputs only")
            continue
        i_temp, j_temp = arr
        if tic_tac_toe_obj.is_legal_move(i_temp, j_temp):
            tic_tac_toe_obj.push(i_temp, j_temp)
            return
        print(f"WARNING: please give legal moves")


COLUMN_NAMES = ["board", "score"]

if __name__ == "__main__":
    t = TicTacToe(size_n=3, winner_len=3, max_depth=None, debug=2)
    t.get_best_move()

    csv_output = pd.DataFrame(data=len(t._my_hash_table) * [[None, None]])
    index_i = 0
    for i, j in t._my_hash_table.items():
        turn_val = (sum(i) == 0)
        board_one = [kk == 1 for kk in i]
        board_two = [kk == -1 for kk in i]
        col1 = f"{int(turn_val)}"
        for kk in board_one: col1 += f" {int(kk)}"
        for kk in board_two: col1 += f" {int(kk)}"
        csv_output.loc[index_i][0] = col1
        csv_output.loc[index_i][1] = j[0]  # NOTE: change to "j" if required
        # if col1.__contains__("1 0 0 0 1 0 0 1 0 0 0 1 0 0 1 1 0 0"):
        #     print(i, turn_val, col1)
        index_i += 1

    csv_output.columns = ["board", "score"]
    csv_output.to_csv(f"TicTacToe_dataset_{len(t._my_hash_table)}.csv", index=False)

    # while not t.game_over:
    #     i_best, j_best = t.get_best_move()
    #     t.push(i_best, j_best)
    #
    #     if t.game_over:
    #         continue
    #     user_play(t)
    # else:
    #     print(f"\n\nGAME OVER: ", end="")
    #     if t.winner == 0:
    #         print(f"draw")
    #     else:
    #         print(f"winner = {t.winner}")
