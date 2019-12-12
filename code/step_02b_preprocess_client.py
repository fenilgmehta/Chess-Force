# CLIENT
import multiprocessing
import time

from minimalcluster import WorkerNode, set_debug_level

import common_services as cs

set_debug_level(3)
your_host, your_port, your_authkey = cs.read_ip_port_auth('step_02_preprocess_server_ip.txt')
your_port: int = int(your_port)
# your_host: str = '192.168.0.104'
# your_port: int = 60005
# your_authkey: str = 'a1'
print(f"Master node IP, Port, AuthKey = '{your_host}', '{your_port}', '{your_authkey}'")
# N_processors_to_use = multiprocessing.cpu_count()
N_processors_to_use = 1024  # UPDATED: 20191209T2203

while True:
    try:
        print()
        worker = WorkerNode(your_host, your_port, your_authkey, nprocs=N_processors_to_use, debug_level=3, limit_nprocs_to_cpucores=True)
        worker.join_cluster(approx_max_job_time=30)
    except Exception as e:
        print(f"\n\nERROR: {e}")
    finally:
        time.sleep(1)
