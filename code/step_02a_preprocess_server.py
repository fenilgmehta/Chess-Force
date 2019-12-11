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


#########################################################################################################################


def ring_beep():
    while True:
        print("\a")
        time.sleep(0.5)


#########################################################################################################################

# IMPORTANT SETTINGS
# input_file = 'Chess_preprocessed_data_KingBase2019-A00-A39_000001.csv'
INPUT_FOLDER = "../../../data_in_combined"
OUTPUT_FOLDER = "../../../data_out_combined"

your_host = '0.0.0.0'  # or use '0.0.0.0' if you have high enough privilege
your_port = 60005  # the port to be used by the master node
your_authkey = 'a1'  # this is the password clients use to connect to the master(i.e. the current node)
your_chunksize = 30  # chunks in which the worker node should get the work to execute

APPROX_MAX_JOB_TIME = your_chunksize * 0.9  # max time for which the master node should wait before pushing the job back to the queue as it did not get a reply in expected time
WORKER_THRESHOLD_TO_RESTART = 400           # If number of workers go above this number, ring beep sound repeatedly
CLIENT_SIDE_WORKERS_PROCESSES = 128         # The max number of parallel processes to be used on client for computation

#################################
print()
cs.print_ip_port_auth(your_port=your_port, your_authkey=your_authkey)

master = MasterNode(HOST=your_host, PORT=your_port, AUTHKEY=your_authkey, chunk_size=your_chunksize, debug_level=3)
master.start_master_server(if_join_as_worker=False)
# master.stop_as_worker()
# master.join_as_worker()

input("\n\nPress ENTER to start task delegation :)\n")

#########################################################################################################################

# daemon=False, this means that this process will not exit even when the parent process has exited or is killed
# write_dict_to_dataframe(result, output_file_path)

# os.system("rm -r data_out/")  # TODO: remove this - it is DANGER
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
if not Path(INPUT_FOLDER).exists():
    print(f"DEBUG: Input folder does not exists: '{INPUT_FOLDER}'")
    exit(1)
input_files = [tempf for tempf in os.listdir(INPUT_FOLDER) if isfile(join(INPUT_FOLDER, tempf))]
for i in sorted(input_files, reverse=False):
    print()
    input_file_name = Path(i).name
    input_file_path = f'{INPUT_FOLDER}/{i}'
    output_file_name = f'out_{input_file_name}'
    output_file_path = f'{OUTPUT_FOLDER}/{output_file_name}'
    if Path(output_file_path).exists():
        print(f"SKIPPING: {input_file_name}, '{output_file_name}' already exists")
        continue
    elif Path(f"{output_file_path}.obj").exists():
        print(f"SKIPPING: {input_file_name}, '{output_file_name}'.obj already exists")
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
        master.load_envir("step_02b_client_environment.py", from_file=True)
        master.register_target_function("fun_board_eval_obj")
        # data_in_new = [(tuple(data_in[i: i + your_chunksize]), CLIENT_SIDE_WORKERS_PROCESSES,) for i in range(0, len(data_in), your_chunksize)]
        master.load_args(data_in)
        result = master.execute(approx_max_job_time=APPROX_MAX_JOB_TIME)

    print(f"DEBUG: Result len = {len(result.keys())}")
    pd.DataFrame(data=list(result.items()), columns=cs.COLUMNS).to_csv(output_file_path, index=False)

    del data_in
    gc.collect()
    # print(f"1 second pause")
    # time.sleep(1)

master.shutdown()
print(f"\n\nProcessing successfully complete :)\n\nPress Ctrl+C to exit\n")

#########################################################################################################################

# STATS

##### 1 core
# 1 client  - 1 instance       =  96 sec
# 4 clients - 1 instance       =  50 sec
# 1 client  - 4 instances      =  50 sec

##### 4 core
# 1 client  - 1 instance       =  96 sec
# 1 client  - 4 instances      =  52 sec
# 4 client  - 1 instance       =  51 sec
