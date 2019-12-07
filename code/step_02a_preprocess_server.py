# SERVER
# REFER: https://github.com/XD-DENG/minimalcluster-py
import gc
import os
import pickle
import time
from multiprocessing import Process, Queue
from os.path import isfile, join
from pathlib import Path

import pandas as pd
from minimalcluster import MasterNode

import common_services as cs


###########################################################################################################


def write_dict_to_dataframe(shared_result_q):
    while True:
        q_obj = shared_result_q.get()
        # print(f"DEBUG: got an object in shared queue")
        if type(q_obj) is str:
            print(f"DEBUG: received a str in queue, exiting write_dict_to_dataframe(...)")
            os.kill(os.getpid(), 9)
            return

        # with cs.ExecutionTime(name="Write dictionary to CSV"):
        #     print(f"DEBUG: received a dictionary in queue, working...")
        #     dict_obj, out_file_name = q_obj
        #     data_out = pd.DataFrame(data=len(dict_obj) * [[None, None]], columns=cs.COLUMNS)
        #     k = 0
        #     for i, j in dict_obj.items():
        #         data_out.loc[k] = [i, j]
        #         k += 1
        #     data_out.to_csv(out_file_name, index=False)
        #     print(f"SUCCESSFULLY written: '{out_file_name}'")
        # del k, data_out

        with cs.ExecutionTime(name="Write dictionary with pickle"):
            print(f"DEBUG: received a dictionary in queue, working...")
            dict_obj, out_file_name = q_obj
            pickle.dump(dict_obj, open(f"{out_file_name}.obj", 'wb+'))
            print(f"SUCCESSFULLY written: '{out_file_name}.obj'")
        del dict_obj


def ring_beep():
    while True:
        print("\a")
        time.sleep(0.5)


###########################################################################################################

# IMPORTANT SETTINGS
# input_file = 'Chess_preprocessed_data_KingBase2019-A00-A39_000001.csv'
INPUT_FOLDER = "../../../data_in_combined"
OUTPUT_FOLDER = "../../../data_out_combined"
STOCK_FISH = {'src_path': None, 'cpu_cores': 1, 'hash_size_mb': 16, 'depth': 20, 'analyse_time': 1.0}

your_host = '0.0.0.0'  # or use '0.0.0.0' if you have high enough privilege
your_port = 60005  # the port to be used by the master node
your_authkey = 'a1'  # this is the password clients use to connect to the master(i.e. the current node)
your_chunksize = 100
APPROX_MAX_JOB_TIME = your_chunksize * 0.9
WORKER_THRESHOLD_TO_RESTART = 400
#################################

print()
cs.print_ip_port_auth(your_port=your_port, your_authkey=your_authkey)

master = MasterNode(HOST=your_host, PORT=your_port, AUTHKEY=your_authkey, chunk_size=your_chunksize, debug_level=3)
master.start_master_server(if_join_as_worker=False)
# master.stop_as_worker()
# master.join_as_worker()

input("\n\nPress ENTER to start task delegation :)\n")

###########################################################################################################

# import os
# os.environ["PYTHONPATH"] = "/home/student/Desktop/fenil/Chess-Kesari-master/code/"

shell_command1 = r'''
"""
sed -i 's#if len(procs) > nprocs and active_processes != nprocs:#if (not bool_changed) and len(procs) > nprocs and active_processes != nprocs:#' '/home/student/1_FinalYearProject_client/Chess-Kesari/code/minimalcluster/worker_node.py'

ps -e | grep python | awk '{print $1}' | xargs -I{} kill -9 {}
cd '/home/student/1_FinalYearProject_client/Chess-Kesari/code'
echo $PATH >> remote_command.txt
sleep 30
nohup python step_02b_preprocess_client.py &
"""
'''

shell_command2 = r'''
"""
sed -i 's#1_FinalYearProject_client/Chess-Kesari/code#/home/student/1_FinalYearProject_client/Chess-Kesari/code#'
echo 'student' | sudo chmod +x /home/student/1client.sh
echo 'student' | sudo cp /home/student/1client.sh /etc/init.d
echo 'student' | sudo update-rc.d 1client.sh defaults
"""
'''

shell_command3 = r'''
"""
ps -e | grep python | awk '{print $1}' | xargs -I{} kill -9 {}
"""
'''

envir_statement = f'''
# IMPORTANT
import os
import time
from pathlib import Path

import chess
import step_01_engine

engine_sf = step_01_engine.CustomEngine(src_path={STOCK_FISH['src_path']}, cpu_cores={STOCK_FISH['cpu_cores']}, hash_size_mb={STOCK_FISH['hash_size_mb']}, depth={STOCK_FISH['depth']}, analyse_time={STOCK_FISH['analyse_time']})

def fun1(args1):
    try:
        chess.Board(args1)
    except:
        return -100000000
    return engine_sf.evaluate_normalized_fen(args1)

def random_sleep(factor = 1):
    # print("random_sleep =", os.getpid(), (os.getpid() % 100)/10)
    time.sleep(factor)
    time.sleep(factor * ((os.getpid() % 100)/10))

# NOTE: stop the server after +15 seconds OR you get a message saying no client connected
def fun2(args1):
    pid = str(os.getpid())
    file_name = 'remote_command.txt'

    # MAX delay due to next 4 lines = 11 seconds
    os.system('rm -f ' + file_name)
    random_sleep(1)
    os.system('echo ' + pid + ' >> ' + file_name)
    random_sleep(0.1)

    # print(os.getcwd())
    if not Path(file_name).exists():
        print("FILE NOT FOUND: " + pid)
        time.sleep(20)

    # MAX delay due to next 5 lines = 35 seconds
    with open(file_name) as f:
        first_line = f.readline()
    if first_line.strip() == pid:
        print("[WARNING] SERVER command being run by pid = " + str(os.getpid()))
        os.system({shell_command1})

    # halt the execution of other processes for some time
    time.sleep(60)

# NOTE: stop the server after +15 seconds
# WARNING: be careful
def fun_poweroff(args1):
    # WARNING: immediate shutdown
    os.system("echo 'student' | sudo -S init 0")

    # WARNING: shutdown the computer after 1 minute
    # os.system("echo 'student' | sudo -S shutdown --poweroff +1 'Time to stop the processing'")

    # WARNING: cancel the scheduled shutdown
    # os.system("echo 'student' | sudo -S shutdown -c")

    os.system({shell_command3})

    time.sleep(30)

'''

shared_result_q = Queue()
dict_process = Process(target=write_dict_to_dataframe, args=(shared_result_q,), daemon=True)
dict_process.start()
# daemon=False, this means that this process will not exit even when the parent process has exited or is killed
# write_dict_to_dataframe(result, output_file_path)

# os.system("rm -r data_out/")  # TODO: remove this - it is DANGER
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
if not Path(INPUT_FOLDER).exists():
    print(f"DEBUG: Input folder does not exists: '{INPUT_FOLDER}'")
    exit(1)
input_files = [tempf for tempf in os.listdir(INPUT_FOLDER) if isfile(join(INPUT_FOLDER, tempf))]
for i in sorted(input_files):
    print()
    input_file_name = Path(i).name
    input_file_path = f'{INPUT_FOLDER}/{i}'
    output_file_name = f'out_{input_file_name}'
    output_file_path = f'{OUTPUT_FOLDER}/{output_file_name}'
    if Path(output_file_path).exists() or Path(f"{output_file_path}.obj").exists():
        print(f"SKIPPING: {input_file_name}, '{output_file_name}' already exists")
        continue
    # while len(master.list_workers()) == 0:
    #     time.sleep(0.5)

    sorted_worker = sorted(
        master.list_workers(approx_max_workers=1000),
        key=lambda i1: (len(i1[0]) - i1[0].rfind("/"), i1[0][i1[0].rfind("/"):],)
    )
    print(f"\n\nConnected workers count = {len(sorted_worker)}")

    if len(sorted_worker) > WORKER_THRESHOLD_TO_RESTART:
        print(f"DEBUG: WARNING: worker count has increased above {WORKER_THRESHOLD_TO_RESTART}")
        print(f"\tRESTART required the master")
        Process(target=ring_beep, daemon=False).start()

    print(f"Connected workers:\n")  # this print the list of connected Worker nodes
    for i_sw in sorted_worker:
        print(f"\t{i_sw},")

    time.sleep(0.91)
    print(f"WORKING ON: {input_file_name} -> {output_file_name}")
    data_in = list(set(pd.read_csv(input_file_path)[cs.COLUMNS[0]]))  # cs.COLUMNS[0] == 'fen_board'
    print(f"DEBUG: Input len = {len(data_in)}")

    with cs.ExecutionTime():
        master.load_envir(envir_statement, from_file=False)
        master.register_target_function("fun1")
        master.load_args(data_in)
        result = master.execute(approx_max_job_time=APPROX_MAX_JOB_TIME)

    print(f"DEBUG: Result len = {len(result.keys())}")
    shared_result_q.put((result, output_file_path,))

    del data_in
    gc.collect()
    # print(f"1 second pause")
    # time.sleep(1)

shared_result_q.put(".")
dict_process.join()
master.shutdown()
print(f"\n\nProcessing successfully complete :)\n\nPress Ctrl+C to exit\n")

###########################################################################################################

# STATS

##### 1 core
# 1 client  - 1 instance       =  96 sec
# 4 clients - 1 instance       =  50 sec
# 1 client  - 4 instances      =  50 sec

##### 4 core
# 1 client  - 1 instance       =  96 sec
# 1 client  - 4 instances      =  52 sec
# 4 client  - 1 instance       =  51 sec
