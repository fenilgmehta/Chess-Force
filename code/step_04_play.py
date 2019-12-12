import collections.abc
import copy
import operator
import time
from typing import Tuple, List, Union, Any

import chess
# import chess.pgn
import numpy as np
from termcolor import colored

import common_services as cs
import step_01_engine as step_01
import step_02_preprocess as step_02
import step_03a_ffnn as step_03


#########################################################################################################################
# This is for prediction
class ChessPredict:
    def __init__(self,
                 function_to_predict_1_board=None, function_to_predict_n_board=None, to_maximize=True,
                 predict_move_1=None):
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
        self.to_maximize = to_maximize

        self.__initial_score = float(((-1) ** to_maximize) * (2 ** 30))
        self.__compare = None
        if self.to_maximize:
            self.__compare = operator.gt
        else:
            self.__compare = operator.lt

    # NOTE: don't use this as it will be slow
    def predict_score_1(self, board: chess.Board) -> np.float:
        """
        Returns the centi-pawn score for `board` it is given as the parameter.

        :return: np.float
        """
        return self.function_to_predict_1(board)

    # NOTE: don't use this as it will be slow
    def predict_best_move_1(self, board: chess.Board) -> Tuple[chess.Move, np.float]:
        """

        :return: Tuple[chess.Move, np.float]
        """
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

    def predict_score_n(self, boards: Union[List, Tuple, np.array]) -> np.array:
        """
        Returns centi-pawn score for all the boards passed to it.

        :return: np.array
        """
        return self.function_to_predict_n(boards)

    def predict_best_move_n(self, board: chess.Board) -> Tuple[chess.Move, np.float]:
        """
        Generate all possible boards at next level from `board`.
        Predict score of all boards.

        :param board:
        :return: Tuple[chess.Move, np.float]
        """
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


#########################################################################################################################
# This if for CLI game interface

class ChessPlayCLI:
    def __init__(self,
                 player1_name: str = "Player 1",
                 player2_name: str = "Player 2",
                 player1_chess_predict: ChessPredict = None,
                 player2_chess_predict: ChessPredict = None,
                 cpu_to_play_first: bool = True):
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.player1_chess_predict = player1_chess_predict
        self.player2_chess_predict = player2_chess_predict
        self.cpu_to_play_first = cpu_to_play_first

    def __cpu_obj_playable(self, pi_chess_predict, is_player1: bool):
        return (pi_chess_predict is not None) and self.cpu_to_play_first == is_player1

    @staticmethod
    def __pretty_board(board: chess.Board) -> str:
        res_out = board.__str__()
        res_out = res_out.split("\n")
        for i in range(len(res_out)):
            res_out[i] = f"[{8 - i}] " + res_out[i]
        res_out.insert(0, f"   [a|b|c|d|e|f|g|h]")
        res_out.insert(0, f"\n\nBoard state number = {board.fullmove_number}")
        return "\n".join(res_out)

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
        print(ChessPlayCLI.__pretty_board(board_play), end="\n\n")
        # board_play.is_repetition()
        # board_play.can_claim_threefold_repetition()
        # board_play.is_fivefold_repetition()
        while not (board_play.is_checkmate() or board_play.is_stalemate() or board_play.is_insufficient_material() or board_play.can_claim_threefold_repetition()):
            print(f"DEBUG: board_play.fen() = {board_play.fen()}")
            print(f"DEBUG: Legal moves = {list(map(str, list(board_play.legal_moves)))}")
            if current_player1_turn:
                # player 1 plays
                if self.player1_chess_predict is None:
                    move_selected = self.user_play(board_play=board_play, is_player1=current_player1_turn)
                else:
                    move_selected, move_score = self.player1_chess_predict.predict_best_move_n(board=board_play)
                    print(f"DEBUG: [{self.player1_name}] AI's move = {move_selected}")
                    print(f"DEBUG: [{self.player1_name}] AI's move_score = {move_score}")
            else:
                # player 2 plays
                if self.player2_chess_predict is None:
                    move_selected = self.user_play(board_play=board_play, is_player1=current_player1_turn)
                else:
                    move_selected, move_score = self.player2_chess_predict.predict_best_move_n(board=board_play)
                    print(f"DEBUG: [{self.player2_name}] AI's move = {move_selected}")
                    print(f"DEBUG: [{self.player2_name}] AI's move_score = {move_score}")
            board_play.push(move_selected)
            print("\n" + ChessPlayCLI.__pretty_board(board_play), end="\n\n")
            current_player1_turn ^= True

        print("\n\n")
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
        return


#########################################################################################################################
# This is for GUI game interface
# REFER: https://github.com/PySimpleGUI/PySimpleGUI
class ChessPlay:
    CHESS_PATH = '.'  # path to the chess pieces

    BLANK = 0  # piece names
    PAWNB = 1
    KNIGHTB = 2
    BISHOPB = 3
    ROOKB = 4
    KINGB = 5
    QUEENB = 6
    PAWNW = 7
    KNIGHTW = 8
    BISHOPW = 9
    ROOKW = 10
    KINGW = 11
    QUEENW = 12

    initial_board = [[ROOKB, KNIGHTB, BISHOPB, QUEENB, KINGB, BISHOPB, KNIGHTB, ROOKB],
                     [PAWNB, ] * 8,
                     [BLANK, ] * 8,
                     [BLANK, ] * 8,
                     [BLANK, ] * 8,
                     [BLANK, ] * 8,
                     [PAWNW, ] * 8,
                     [ROOKW, KNIGHTW, BISHOPW, QUEENW, KINGW, BISHOPW, KNIGHTW, ROOKW]]

    blank = os.path.join(CHESS_PATH, 'blank.png')
    bishopB = os.path.join(CHESS_PATH, 'nbishopb.png')
    bishopW = os.path.join(CHESS_PATH, 'nbishopw.png')
    pawnB = os.path.join(CHESS_PATH, 'npawnb.png')
    pawnW = os.path.join(CHESS_PATH, 'npawnw.png')
    knightB = os.path.join(CHESS_PATH, 'nknightb.png')
    knightW = os.path.join(CHESS_PATH, 'nknightw.png')
    rookB = os.path.join(CHESS_PATH, 'nrookb.png')
    rookW = os.path.join(CHESS_PATH, 'nrookw.png')
    queenB = os.path.join(CHESS_PATH, 'nqueenb.png')
    queenW = os.path.join(CHESS_PATH, 'nqueenw.png')
    kingB = os.path.join(CHESS_PATH, 'nkingb.png')
    kingW = os.path.join(CHESS_PATH, 'nkingw.png')

    images = {BISHOPB: bishopB, BISHOPW: bishopW, PAWNB: pawnB, PAWNW: pawnW, KNIGHTB: knightB, KNIGHTW: knightW,
              ROOKB: rookB, ROOKW: rookW, KINGB: kingB, KINGW: kingW, QUEENB: queenB, QUEENW: queenW, BLANK: blank}

    def open_pgn_file(self, filename):
        pgn = open(filename)
        first_game = chess.pgn.read_game(pgn)
        moves = [move for move in first_game.main_line()]
        return moves

    def render_square(self, image, key, location):
        if (location[0] + location[1]) % 2:
            color = '#B58863'
        else:
            color = '#F0D9B5'
        return sg.RButton('', image_filename=image, size=(1, 1), button_color=('white', color), pad=(0, 0), key=key)

    def redraw_board(self, window, board):
        for i in range(8):
            for j in range(8):
                color = '#B58863' if (i + j) % 2 else '#F0D9B5'
                piece_image = self.images[board[i][j]]
                elem = window.FindElement(key=(i, j))
                elem.Update(button_color=('white', color),
                            image_filename=piece_image, )

    def PlayGame(self):
        menu_def = [['&File', ['&Open PGN File', 'E&xit']],
                    ['&Help', '&About...'], ]

        # sg.SetOptions(margins=(0,0))
        sg.ChangeLookAndFeel('GreenTan')
        # create initial board setup
        psg_board = copy.deepcopy(self.initial_board)
        # the main board display layout
        board_layout = [[sg.T('     ')] + [sg.T('{}'.format(a), pad=((23, 27), 0), font='Any 13') for a in 'abcdefgh']]
        # loop though board and create buttons with images
        for i in range(8):
            row = [sg.T(str(8 - i) + '   ', font='Any 13')]
            for j in range(8):
                piece_image = self.images[psg_board[i][j]]
                row.append(self.render_square(piece_image, key=(i, j), location=(i, j)))
            row.append(sg.T(str(8 - i) + '   ', font='Any 13'))
            board_layout.append(row)
        # add the labels across bottom of board
        board_layout.append([sg.T('     ')] + [sg.T('{}'.format(a), pad=((23, 27), 0), font='Any 13') for a in 'abcdefgh'])

        # setup the controls on the right side of screen
        openings = (
            'Any', 'Defense', 'Attack', 'Trap', 'Gambit', 'Counter', 'Sicillian', 'English', 'French', 'Queen\'s openings',
            'King\'s Openings', 'Indian Openings')

        board_controls = [[sg.RButton('New Game', key='New Game'), sg.RButton('Draw')],
                          [sg.RButton('Resign Game'), sg.RButton('Set FEN')],
                          [sg.RButton('Player Odds'), sg.RButton('Training')],
                          [sg.Drop(openings), sg.Text('Opening/Style')],
                          [sg.CBox('Play As White', key='_white_')],
                          [sg.Drop([2, 3, 4, 5, 6, 7, 8, 9, 10], size=(3, 1), key='_level_'), sg.Text('Difficulty Level')],
                          [sg.Text('Move List')],
                          [sg.Multiline([], do_not_clear=True, autoscroll=True, size=(15, 10), key='_movelist_')],
                          ]

        # layouts for the tabs
        controls_layout = [[sg.Text('Performance Parameters', font='_ 20')],
                           [sg.T('Put stuff like AI engine tuning parms on this tab')]]

        statistics_layout = [[sg.Text('Statistics', font=('_ 20'))],
                             [sg.T('Game statistics go here?')]]

        board_tab = [[sg.Column(board_layout)]]

        # the main window layout
        layout = [[sg.Menu(menu_def, tearoff=False)],
                  [sg.TabGroup([[sg.Tab('Board', board_tab),
                                 sg.Tab('Controls', controls_layout),
                                 sg.Tab('Statistics', statistics_layout)]], title_color='red'),
                   sg.Column(board_controls)],
                  [sg.Text('Click anywhere on board for next move', font='_ 14')]]

        window = sg.Window('Chess',
                           default_button_element_size=(12, 1),
                           auto_size_buttons=False,
                           icon='kingb.ico').Layout(layout)

        filename = sg.PopupGetFile('\n'.join(('To begin, set location of AI EXE file',
                                              'If you have not done so already, download the engine',
                                              'Download the StockFish Chess engine at: https://stockfishchess.org/download/')),
                                   file_types=(('Chess AI Engine EXE File', '*.exe'),))
        if filename is None:
            sys.exit()
        engine = chess.uci.popen_engine(filename)
        engine.uci()
        info_handler = chess.uci.InfoHandler()
        engine.info_handlers.append(info_handler)

        board = chess.Board()
        move_count = 1
        move_state = move_from = move_to = 0
        # ---===--- Loop taking in user input --- #
        while not board.is_game_over():

            if board.turn == chess.WHITE:
                engine.position(board)

                # human_player(board)
                move_state = 0
                while True:
                    button, value = window.Read()
                    if button in (None, 'Exit'):
                        exit()
                    if button == 'New Game':
                        sg.Popup('You have to restart the program to start a new game... sorry....')
                        break
                        psg_board = copy.deepcopy(initial_board)
                        redraw_board(window, psg_board)
                        move_state = 0
                        break
                    level = value['_level_']
                    if type(button) is tuple:
                        if move_state == 0:
                            move_from = button
                            row, col = move_from
                            piece = psg_board[row][col]  # get the move-from piece
                            button_square = window.FindElement(key=(row, col))
                            button_square.Update(button_color=('white', 'red'))
                            move_state = 1
                        elif move_state == 1:
                            move_to = button
                            row, col = move_to
                            if move_to == move_from:  # cancelled move
                                color = '#B58863' if (row + col) % 2 else '#F0D9B5'
                                button_square.Update(button_color=('white', color))
                                move_state = 0
                                continue

                            picked_move = '{}{}{}{}'.format('abcdefgh'[move_from[1]], 8 - move_from[0],
                                                            'abcdefgh'[move_to[1]], 8 - move_to[0])

                            if picked_move in [str(move) for move in board.legal_moves]:
                                board.push(chess.Move.from_uci(picked_move))
                            else:
                                print('Illegal move')
                                move_state = 0
                                color = '#B58863' if (move_from[0] + move_from[1]) % 2 else '#F0D9B5'
                                button_square.Update(button_color=('white', color))
                                continue

                            psg_board[move_from[0]][move_from[1]] = BLANK  # place blank where piece was
                            psg_board[row][col] = piece  # place piece in the move-to square
                            redraw_board(window, psg_board)
                            move_count += 1

                            window.FindElement('_movelist_').Update(picked_move + '\n', append=True)

                            break
            else:
                engine.position(board)
                best_move = engine.go(searchmoves=board.legal_moves, depth=level, movetime=(level * 100)).bestmove
                move_str = str(best_move)
                from_col = ord(move_str[0]) - ord('a')
                from_row = 8 - int(move_str[1])
                to_col = ord(move_str[2]) - ord('a')
                to_row = 8 - int(move_str[3])

                window.FindElement('_movelist_').Update(move_str + '\n', append=True)

                piece = psg_board[from_row][from_col]
                psg_board[from_row][from_col] = BLANK
                psg_board[to_row][to_col] = piece
                redraw_board(window, psg_board)

                board.push(best_move)
                move_count += 1
        sg.Popup('Game over!', 'Thank you for playing')

    # Download the StockFish Chess engine at: https://stockfishchess.org/download/
    # engine = chess.uci.popen_engine(r'E:\DownloadsE\stockfish-9-win\Windows\stockfish_9_x64.exe')
    # engine.uci()
    # info_handler = chess.uci.InfoHandler()
    # engine.info_handlers.append(info_handler)
    # level = 2
    # PlayGame()


#########################################################################################################################
if __name__ == "__main__":
    # play_chess()

    # a = ChessPlay()
    # a.PlayGame()
    ffnn_keras = step_03.FFNNKeras(step_03.KerasModels.model_001, step_02.BoardEncoder.Encode778)
    ffnn_keras.c_load_weights("ffnn_keras_v004_000010_weights.h5")
    chess_predict_1 = ChessPredict(ffnn_keras.c_predict_board_1, ffnn_keras.c_predict_board_n, to_maximize=True)
    chess_predict_2 = ChessPredict(ffnn_keras.c_predict_board_1, ffnn_keras.c_predict_board_n, to_maximize=True)
    ChessPlayCLI(player1_name="A",
                 player2_name="B",
                 player1_chess_predict=chess_predict_1,
                 player2_chess_predict=chess_predict_2,
                 cpu_to_play_first=True).play()
