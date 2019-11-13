# SERVER
# REFER: https://github.com/XD-DENG/minimalcluster-py
import os
from os.path import isfile, join
from pathlib import Path

import pandas as pd
from minimalcluster import MasterNode

import common_services as cs


###########################################################################################################


def write_dict_to_dataframe(dict_obj, out_file_name):
    import common_services as cs
    data_out = pd.DataFrame(data=len(dict_obj) * [[None, None]], columns=cs.COLUMNS)
    k = 0
    for i, j in dict_obj.items():
        data_out.loc[k] = [i, j]
        k += 1
    data_out.to_csv(f'{out_file_name}', index=False)


###########################################################################################################

# IMPORTANT SETTINGS
# input_file = 'Chess_preprocessed_data_KingBase2019-A00-A39_000001.csv'
INPUT_FOLDER = "./data_in"
OUTPUT_FOLDER = "./data_out"
STOCK_FISH = {'src_path': None, 'cpu_cores': 1, 'hash_size_mb': 16, 'depth': 20, 'analyse_time': 1.0}

your_host = '0.0.0.0'  # or use '0.0.0.0' if you have high enough privilege
your_port = 60005  # the port to be used by the master node
your_authkey = 'a1'  # this is the password clients use to connect to the master(i.e. the current node)
your_chunksize = 10
#################################

print()
cs.print_ip_port_auth(your_port=your_port, your_authkey=your_authkey)

master = MasterNode(HOST=your_host, PORT=your_port, AUTHKEY=your_authkey, chunksize=your_chunksize)
master.start_master_server(if_join_as_worker=False)
# master.stop_as_worker()
# master.join_as_worker()

input("\n\nPress ENTER to start task delegation :)\n")

###########################################################################################################

# import os
# os.environ["PYTHONPATH"] = "/home/student/Desktop/fenil/Chess-Kesari-master/code/"

envir_statement = f'''
# IMPORTANT
import step_01_engine
engine_sf = step_01_engine.CustomEngine(src_path={STOCK_FISH['src_path']}, cpu_cores={STOCK_FISH['cpu_cores']}, hash_size_mb={STOCK_FISH['hash_size_mb']}, depth={STOCK_FISH['depth']}, analyse_time={STOCK_FISH['analyse_time']})

def fun1(args1):
    return engine_sf.evaluate_normalized_fen(args1)
'''

os.system("rm -r data_out/")  # TODO: remove this - it is DANGER
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
input_files = [tempf for tempf in os.listdir(INPUT_FOLDER) if isfile(join(INPUT_FOLDER, tempf))]
for i in input_files:
    print()
    input_file_name = Path(i).name
    input_file_path = f'{INPUT_FOLDER}/{i}'
    output_file_name = f'out_{input_file_name}'
    output_file_path = f'{OUTPUT_FOLDER}/{output_file_name}'
    if Path(output_file_path).exists():
        print(f"SKIPPING: {input_file_name}, '{output_file_name}' already exists")
        continue
    # while len(master.list_workers()) == 0:
    #     time.sleep(0.5)
    print(f"\n\nConnected workers:\n{master.list_workers()}\n\n")  # this print the list of connected Worker nodes
    print(f"WORKING ON: {input_file_name} -> {output_file_name}")
    data_in = list(set(pd.read_csv(input_file_path)[cs.COLUMNS[0]]))  # cs.COLUMNS[0] == 'fen_board'

    master.load_envir(envir_statement, from_file=False)
    master.register_target_function("fun1")
    master.load_args(data_in)
    result = master.execute()

    write_dict_to_dataframe(result, output_file_path)
    print(f"SUCCESSFULLY written: '{output_file_name}'")
    # print(f"1 second pause")
    # time.sleep(1)

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
