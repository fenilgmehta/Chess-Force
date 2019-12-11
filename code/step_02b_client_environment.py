# IMPORTANT
import math
import multiprocessing
import os
import random
import time
from pathlib import Path

import chess

import common_services as cs
import step_01_engine

#########################################################################################################################
# IMPORTANT SETTINGS
STOCK_FISH = {'src_path': None, 'cpu_cores': 1, 'hash_size_mb': 16, 'depth': 20, 'analyse_time': 1.0}

random_obj = random.Random(f"{os.getpid()}_{time.time()}")
time.sleep(random_obj.random() * 5)

#########################################################################################################################
engine_sf = step_01_engine.CustomEngine(src_path=STOCK_FISH['src_path'],
                                        cpu_cores=STOCK_FISH['cpu_cores'],
                                        hash_size_mb=STOCK_FISH['hash_size_mb'],
                                        depth=STOCK_FISH['depth'],
                                        analyse_time=STOCK_FISH['analyse_time'])


def fun_board_eval_obj(args1):
    global engine_sf
    try:
        chess.Board(args1)
    except:
        return -100000000
    return engine_sf.evaluate_normalized_fen(args1)


#########################################################################################################################
def board_eval_list(boards_fen_list):
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
        # print(f"DEBGUp: {ith_board_fen}")
        try:
            chess.Board(ith_board_fen)
        except:
            result.append(-100000000)
            continue
        # with cs.ExecutionTime():
        result.append(engine_sf1.evaluate_normalized_fen(ith_board_fen))
    # print(f"PID[{os.getpid()}] = {result}")
    return result


def board_map_list(boards_fen_list, process_count):
    # print(f"INFO: board_map_list: inside")
    if isinstance(boards_fen_list, (list, tuple,)):
        # print(f"INFO: {len(boards_fen_list)}, {type(boards_fen_list)}, {type(process_count)}")

        if not isinstance(process_count, int):
            process_count = multiprocessing.cpu_count()
        process_count = min(process_count, len(boards_fen_list), multiprocessing.cpu_count())

        block_size = int(math.ceil(len(boards_fen_list) / process_count))
        args1_jobs = [boards_fen_list[i:i + block_size] for i in range(0, len(boards_fen_list), block_size)]

        # print(f"DEBUG: len(boards_fen_list) = {len(boards_fen_list)}, process_count={process_count}, block_size = {block_size}, len(args1_jobs) = {len(args1_jobs)}, args1_jobs = {args1_jobs}, ")

        with multiprocessing.Pool(processes=process_count) as pool:
            result = pool.map(func=board_eval_list, iterable=args1_jobs)
        return [ith_val
                for result_i in result
                for ith_val in result_i]
    else:
        print("WARNING: board_map_list: boards_fen_list is not an instance of [list, tuple, ]")
        return -111


def fun_board_mapper(args1):
    # print(f"INFO: fun_board_mapper: inside")
    # print(f"DEBUG: 87: {len(args1)}")
    # for i in range(len(args1)): print(f"DBEUG: 89[{i}]: {args1[i]}")

    boards_fen_list = args1[0]
    process_count = args1[1]
    return board_map_list(boards_fen_list=boards_fen_list, process_count=process_count)


#########################################################################################################################
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


SHELL_COMMAND_RESTART_CLIENT_PROGRAM = '''
ps -e | grep python | awk '{print $1}' | xargs -I{} kill -9 {}
cd '/home/student/1_FinalYearProject_client/Chess-Kesari/code'
echo $PATH >> remote_command.txt
sleep 30
nohup python step_02b_preprocess_client.py & tail -f nohup.out  # the tail command causes the output to be print to the terminal screen
'''


# NOTE: stop the server `after +15 seconds` OR `as soon as you get a message saying no client connected`
def fun_restart_client_program(args1):
    global SHELL_COMMAND_RESTART_CLIENT_PROGRAM
    pid = str(os.getpid())
    file_name = 'remote_command.txt'

    if random_select_process(pid, file_name):
        print("[WARNING] SERVER command being run by pid = " + str(os.getpid()))
        os.system(SHELL_COMMAND_RESTART_CLIENT_PROGRAM)

    # halt the execution of other processes for some time
    time.sleep(60)


PATCH_FILES = [
    r"""/home/student/1_FinalYearProject_client/Chess-Kesari/code/minimalcluster/worker_node.py""",
]
SHELL_COMMAND_PATCHES = [
    rf''' sed -i 's/if len(procs) > nprocs and active_processes != nprocs:/if (not bool_changed) and len(procs) > nprocs and active_processes != nprocs:/' '{PATCH_FILES[0]}' ''',
    rf''' sed -i 's/job_id, job_detail = job_q.get_nowait()/job_id, job_detail = job_q.get(block=True, timeout=2.94)  # NEW: to test: 20191209T2203/' '{PATCH_FILES[0]}' ''',
]


# NOTE: stop the server `after +15 seconds` OR `as soon as you get a message saying no client connected`
def fun_patch_client_side(args1):
    global SHELL_COMMAND_PATCHES, SHELL_COMMAND_RESTART_CLIENT_PROGRAM
    pid = str(os.getpid())
    file_name = 'remote_command.txt'

    if random_select_process(pid, file_name):
        print("[WARNING] SERVER command being run by pid = " + str(os.getpid()))
        os.system(SHELL_COMMAND_PATCHES[0])
        os.system(SHELL_COMMAND_RESTART_CLIENT_PROGRAM)

    # halt the execution of other processes for some time
    time.sleep(60)


#########################################################################################################################
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

#########################################################################################################################

# globals().update(locals())
# while True:
#     try:
#         job_id, job_detail = job_q.get_nowait()
#         print_debug(f"DEBUG: [{str(os.getppid()):5} -> {str(os.getpid()):5}] to work on: {job_id}", level=3)
#         # history_q.put({job_id: hostname})
#         history_d[job_id] = hostname
#         outdict = {n: globals()[fun](n) for n in job_detail}
#         result_q.put((job_id, outdict))
#         print_debug(f"DEBUG: [{str(os.getppid()):5} -> {str(os.getpid()):5}] work finished: {job_id}", level=3)
#     except EOFError:
#         print_debug(f"\n[ERROR] [{str(os.getppid()):5} -> {str(os.getpid()):5}] Connection closed by the server. STOPPING the spawned processes...", level=1)
#         break
#     except Queue.Empty:
#         print_debug(f"\nDEBUG: [{str(os.getppid()):5} -> {str(os.getpid()):5}] Queue.Empty: returning from single_worker(...)", level=3)
#         break
#     except Exception as e:
#         # Send the Unexpected error to master node. This will not stop the master node, it will continue to execute
#         error_q.put("Worker Node '{}': ".format(hostname) + "; ".join([repr(e) for e in sys.exc_info()]))
#         error_q.put(f"{job_id},{len(job_detail)}")
#         print_debug(f"\nDEBUG: [{str(os.getppid()):5} -> {str(os.getpid()):5}] Unknown error, returning from single_worker(...)\n\t{e}", level=3)
#         break
#
# os.kill(os.getpid(), 9)
# # exit(0)  # This does not stop the spawned process on remote worker node
# # sys.exit()  # This does not stop the spawned process on remote worker node
# print_debug(f"DEBUG: using return: STOPPING single_worker(...) {os.getpid()}", level=3)
# return
