# SERVER
import gc
import glob
import logging
import os
import signal
import time
from multiprocessing import Process
from pathlib import Path
from typing import List, Union

import dispy
import dispy.httpd
import pandas as pd
from tqdm.autonotebook import tqdm

import common_services as cs


def ring_beep(time_seconds: int = -1):
    while time_seconds != 0:
        print("\a", end="", flush=True)
        time.sleep(0.5)
        print("\a", end="", flush=True)
        time.sleep(0.5)
        time_seconds -= 1


class UserSettings:
    def __init__(
            self, input_dir: str,
            output_dir: str,
            host_ip: str,
            host_port: int,
            host_authkey: str,
            worker_processes: int,
            jobs_per_process: Union[str, None],
            chunksize: Union[str, None] = None
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.host_ip = host_ip
        self.host_port = host_port
        self.host_authkey = host_authkey
        self.worker_processes = worker_processes
        if jobs_per_process is not None:
            self.jobs_per_process = int(jobs_per_process)
            self.chunksize = self.jobs_per_process * abs(self.worker_processes)
        else:
            self.chunksize = int(chunksize)


def cluster_generate_scores(user_settings: UserSettings):
    cluster = dispy.JobCluster('step_02c_worker_environment.py',
                               ip_addr=[user_settings.host_ip],
                               depends=['../chess_engines/Linux/stockfish_10_x64', 'step_01_engine.py', ],
                               loglevel=logging.DEBUG)
    http_server = dispy.httpd.DispyHTTPServer(cluster, host=user_settings.host_ip, port=45623, poll_sec=1)

    Path(user_settings.output_dir).mkdir(parents=True, exist_ok=True)
    if not Path(user_settings.input_dir).exists():
        print(f"DEBUG: input_dir does not exists: '{user_settings.input_dir}'")
        exit(1)
    if not Path(user_settings.output_dir).exists():
        print(f"DEBUG: output_dir does not exists: '{user_settings.output_dir}'")
        exit(1)

    # input_files = [tempf for tempf in os.listdir(user_settings.input_dir) if isfile(join(user_settings.input_dir, tempf))]
    input_files = [Path(tempf).name for tempf in glob.glob(f"{Path(user_settings.input_dir)}/*.csv")]
    for i in tqdm(sorted(input_files, reverse=False)):
        print()
        input_file_name = Path(i).name
        input_file_path = f'{user_settings.input_dir}/{i}'
        output_file_name = f'out_{input_file_name}'
        output_file_path = f'{user_settings.output_dir}/{output_file_name}'

        if Path(output_file_path).exists():
            print(f"SKIPPING: {input_file_name}, '{output_file_name}' already exists")
            continue

        cluster.print_status()  # shows which nodes executed how many jobs etc.

        time.sleep(0.91)
        print(f"WORKING ON: {input_file_name} -> {output_file_name}")
        # NOTE: it is important to have ONLY unique elements in data_in
        # to avoid duplicate computations and save time
        data_in = list(set(pd.read_csv(input_file_path)[cs.COLUMNS[0]]))  # cs.COLUMNS[0] == 'fen_board'
        print(f"DEBUG: Input len = {len(data_in)}")

        data_in_new = [
            (
                tuple(data_in[j: j + user_settings.chunksize]),
                user_settings.worker_processes,
            )
            for j in range(0, len(data_in), user_settings.chunksize)
        ]

        print("Submitting jobs:")
        jobs: List[dispy.DispyJob] = []
        for j in tqdm(data_in_new):
            job = cluster.submit('board_mapper', j)
            jobs.append(job)

        print("Waiting for job results:")
        result = {}
        for job in tqdm(jobs):
            res = job()  # waits for job to finish and returns results
            partial_result: dict = eval(job.stdout.decode())
            print(f"sys.stderr: {job.id}: {job.stderr.decode()}")
            # host, n = job()  # waits for job to finish and returns results
            # other fields of 'job' that may be useful:
            # job.stdout, job.stderr, job.exception, job.ip_addr, job.end_time
            result.update(partial_result)

        print(f"DEBUG: Result len = {len(result.keys())}")
        pd.DataFrame(data=list(result.items()), columns=cs.COLUMNS).to_csv(output_file_path, index=False)
        print(f"SUCCESSFULLY written {output_file_path}")

        gc.collect()
        time.sleep(2)
        # print(f"1 second pause")

    cluster.print_status()  # Shows which nodes executed how many jobs etc.
    cluster.close(timeout=None, terminate=False)  # Close the cluster (jobs can no longer be submitted to it).
    http_server.shutdown()  # This waits until browser gets all updates

    # ring_beep(60)
    bell_process = Process(target=ring_beep, daemon=False)
    bell_process.start()
    print(f"\n\nProcessing successfully complete :)\n\nPress ENTER to exit\n")
    input()
    os.kill(bell_process.pid, signal.SIGTERM)


if __name__ == "__main__":
    from docopt import docopt

    doc_string = '''
    Usage:
        step_02a_preprocess_server_dispy.py --input_dir=PATH --output_dir=PATH --host_ip=IP [--host_port=PORT] [--host_authkey=KEYPASS] --worker_processes=N (--jobs_per_process=N | --chunksize=N)
        step_02a_preprocess_server_dispy.py (-h | --help)
        step_02a_preprocess_server_dispy.py --version

    Options:
        --input_dir=PATH        Path to the directory having csv files
        --output_dir=PATH       Path to the directory where results shall be saved
        --host_ip=IP            IP address of the server
        --host_port=PORT        Port number to be used by the server [default: 61590]
        --host_authkey=KEEPASS  String to be used to allow only authenticate worker processes to the server
        --worker_processes=N    Number of processes to be used on each worker (if +ve then `N` CPU's are used, if -ve then `MAX_CPU - N` CPU's are used) [default: -1]
        --jobs_per_process=N    Number of jobs to be given to one process on the worker. NOTE: if `worker_processes` is -1, then only `jobs_per_process` will be taken as chunksize, else `worker_processes*jobs_per_process` is chunksize
        --chunksize=N           Number of jobs to be given to one worker
        -h --help               Show this
        --version               Show version
    '''
    arguments = docopt(doc_string, argv=None, help=True, version=f"{cs.VERSION} - Generate scores", options_first=False)
    # arguments = eval("""{'--chunksize': None,
    #  '--help': False,
    #  '--host_authkey': '',
    #  '--host_ip': '192.168.0.102',
    #  '--host_port': '45645',
    #  '--input_dir': '/home/student/Desktop/fenil_pc/aggregated output 03/Test_kaufman',
    #  '--jobs_per_process': '30',
    #  '--output_dir': '/home/student/Desktop/fenil_pc/aggregated output 03/Test_kaufman/output',
    #  '--worker_processes': '-1'}
    # """)
    """
    python step_02a_preprocess_server_dispy.py \
        --input_dir="/home/student/Desktop/fenil_pc/aggregated output 03/Test_kaufman" \
        --output_dir="/home/student/Desktop/fenil_pc/aggregated output 03/Test_kaufman/output" \
        --host_ip="192.168.0.102" \
        --worker_processes="-1" \
        --jobs_per_process=30
    """

    print("\n\n", arguments, "\n\n", sep="")
    cluster_generate_scores(
        UserSettings(
            arguments['--input_dir'],
            arguments['--output_dir'],
            arguments['--host_ip'],
            int(arguments['--host_port']),
            arguments['--host_authkey'],
            int(arguments['--worker_processes']),
            arguments['--jobs_per_process'],
            arguments['--chunksize']
        )
    )

# pip install dispy
# dispynode.py --debug --zombie_interval=1 --clean --cpus=1  # zombie interval is 1 minute
