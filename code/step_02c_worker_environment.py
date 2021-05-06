# IMPORTANT
import math
import multiprocessing
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Union, Tuple

import chess
import fire

import step_01_engine

########################################################################################################################
# IMPORTANT SETTINGS
STOCK_FISH = {'src_path': './stockfish_10_x64', 'cpu_cores': 1, 'hash_size_mb': 16, 'depth': 20, 'analyse_time': 2.0}

random_obj = random.Random(f"{os.getpid()}_{time.time()}")
time.sleep(random_obj.random() * 5)  # CHANGE REQUIRED: NOTE: use 5 if less clients are connected and chunk size is small


########################################################################################################################
def board_eval_list(boards_fen_list) -> List[float]:
    global STOCK_FISH
    # print(f"DEBUG: board_eval_list: inside")
    # print(f"DEBUG: {type(boards_fen_list)}, len={len(boards_fen_list)}")
    engine_sf1 = step_01_engine.CustomEngine(src_path=STOCK_FISH['src_path'],
                                             cpu_cores=STOCK_FISH['cpu_cores'],
                                             hash_size_mb=STOCK_FISH['hash_size_mb'],
                                             depth=STOCK_FISH['depth'],
                                             analyse_time=STOCK_FISH['analyse_time'])
    result = []
    for ith_board_fen in boards_fen_list:
        try:
            chess.Board(ith_board_fen)
        except:
            result.append(float('inf'))
            continue
        result.append(engine_sf1.evaluate_normalized_fen(ith_board_fen))
    engine_sf1.close()
    return result


def board_map_list(boards_fen_list, process_count):
    # print(f"INFO: board_map_list: inside")
    if isinstance(boards_fen_list, (list, tuple,)):
        # print(f"INFO: {len(boards_fen_list)}, {type(boards_fen_list)}, {type(process_count)}")

        if not isinstance(process_count, int):
            process_count = multiprocessing.cpu_count()
        if process_count >= 1:
            process_count = min(process_count, len(boards_fen_list), multiprocessing.cpu_count())
        elif process_count <= 0:
            process_count = multiprocessing.cpu_count() + process_count

        block_size = int(math.ceil(len(boards_fen_list) / process_count))
        args1_jobs = [boards_fen_list[i:i + block_size] for i in range(0, len(boards_fen_list), block_size)]

        # print(f"DEBUG: len(boards_fen_list) = {len(boards_fen_list)}, "
        #       f"process_count={process_count}, block_size = {block_size}, "
        #       f"len(args1_jobs) = {len(args1_jobs)}, args1_jobs = {args1_jobs}, ")

        with multiprocessing.Pool(processes=process_count) as pool:
            result: List[List[float]] = pool.map(func=board_eval_list, iterable=args1_jobs)
        return [ith_val
                for result_i in result
                for ith_val in result_i]
    else:
        print("WARNING: board_map_list: boards_fen_list is not an instance of [list, tuple, ]")
        print("WARNING: board_map_list: boards_fen_list is not an instance of [list, tuple, ]", file=sys.stderr)
        return [-111]


def board_mapper(args1):
    # print(args1[0], end="\n\n")
    # print(args1[1], end="\n\n")
    # print(type(args1[0]))
    # print(type(args1[1]))
    # if len(args1) > 2:
    #     print(args1[2])

    boards_fen_list: Tuple[str] = args1[0]
    process_count: Union[int, float, None] = args1[1]
    result: List[float] = board_map_list(boards_fen_list=boards_fen_list, process_count=process_count)
    print(dict(zip(boards_fen_list, result)))


########################################################################################################################
# FUNCTIONS to remotely patch/update the client side programs and restart the client program

def random_sleep(factor: float = 1.0):
    # print("random_sleep =", os.getpid(), (os.getpid() % 100)/10)
    time.sleep(factor)
    time.sleep(factor * ((os.getpid() % 100) / 10))


def random_select_process(pid, file_name):
    # MAX delay due to next 4 lines = 11 seconds
    os.system('rm -f ' + file_name)
    random_sleep(factor=1.0)
    os.system('echo ' + pid + ' >> ' + file_name)
    random_sleep(factor=0.1)

    # print(os.getcwd())
    if not Path(file_name).exists():
        print("FILE NOT FOUND: " + pid)
        time.sleep(secs=1.0)
    if not Path(file_name).exists():
        return False

    # MAX delay due to next 5 lines = 35 seconds
    with open(file_name) as f:
        first_line = f.readline()
    if first_line.strip() != pid:
        return False

    return True


########################################################################################################################
# NOTE: stop the server after +15 seconds depending on the command being executed
# WARNING: be careful
def fun_poweroff(args1):
    # WARNING: immediate shutdown
    # os.system("echo 'student' | sudo -S init 0")

    # WARNING: shutdown the computer after 1 minute
    # os.system("echo 'student' | sudo -S shutdown --poweroff +1 'Time to stop the processing'")

    # WARNING: cancel the scheduled shutdown
    # os.system("echo 'student' | sudo -S shutdown -c")

    # os.system(r"ps -e | grep python | awk '{print $1}' | xargs -I{} kill -9 {}")
    time.sleep(30)


########################################################################################################################


if __name__ == "__main__":
    # import sys
    # print(sys.argv[0], file=sys.stderr)
    # print(sys.argv[1], file=sys.stderr)
    # print(sys.argv[2], file=sys.stderr)
    # print(type(sys.argv[2]), file=sys.stderr)
    # print(len(sys.argv), file=sys.stderr)
    # print(type(eval(sys.argv[1])), file=sys.stderr)
    # os.kill(os.getpid(), -9)

    fire.Fire()
