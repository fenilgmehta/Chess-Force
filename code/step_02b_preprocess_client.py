# CLIENT
import multiprocessing
import time

from minimalcluster import WorkerNode

import common_services as cs

your_host, your_port, your_authkey = cs.read_ip_port_auth('step_02_preprocess_server_ip.txt')
your_port: int = int(your_port)
# your_host: str = '192.168.0.105'
# your_port: int = 60003
# your_authkey: str = 'a1'
print(f"Master node IP, Port, AuthKey = '{your_host}', '{your_port}', '{your_authkey}'")
N_processors_to_use = multiprocessing.cpu_count()

while True:
    try:
        print()
        worker = WorkerNode(your_host, your_port, your_authkey, nprocs=N_processors_to_use)
        worker.join_cluster()
    except Exception as e:
        print(f"\n\nERROR: {e}")
    finally:
        time.sleep(1)
