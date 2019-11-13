import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import List, Union

COLUMNS = ["fen_board", "cp_score"]


###########################################################################################################

def append_secondlast_line(resume_file_name, msg):
    # (last game written + 1), (next file number to be written)
    if not Path(resume_file_name).exists():
        savepoint(resume_file_name, msg)
    os.system(f"""
        last_line=`tail -n1 '{resume_file_name}'`
        sed --in-place '$d' '{resume_file_name}'
        echo '{msg}' >> '{resume_file_name}'
        echo $last_line >> '{resume_file_name}'
    """)


def savepoint(resume_file_name, msg):
    if not Path(resume_file_name).exists():
        print(f"DEBUG: file created: '{resume_file_name}'", file=sys.stderr)
        os.system(f"echo '{msg}' >> '{resume_file_name}'")
    else:
        os.system(f"sed --in-place '$d' '{resume_file_name}' ; echo '{msg}' >> '{resume_file_name}'")


def readpoint(resume_file_name, variable_length) -> Union[List[int], int]:
    if not Path(resume_file_name).exists():
        print(f"DEBUG: file does not exists: '{resume_file_name}'"
              f"\n\treturning default value, i.e. array of 1's of size {variable_length}", file=sys.stderr)
        return variable_length * [1, ]

    result = [int(i) for i in subprocess.getoutput(f"tail -n1 '{resume_file_name}'").split(",")]

    if len(result) > variable_length:
        print(f"WARNING: expected values = {variable_length}, values read = {len(result)}"
              f"\n\tReturning only first '{variable_length}' values of {result}")
    elif len(result) < variable_length:
        print(f"WARNING: expected values = {variable_length}, values read = {len(result)}"
              f"\n\tReturning only first '{len(result)}' values of {result}")

    return result[:variable_length]


###########################################################################################################

def print_ip_port_auth(file_to_write: Union[str, Path] = 'step_02_preprocess_server_ip.txt', your_port: int = None, your_authkey: str = None) -> None:
    if your_port is None or not (isinstance(your_port, int)) or not (1000 <= your_port <= 65535):
        raise ValueError("ERROR: your_port should be int -> [1000-65535]")
    if your_authkey is None or not (isinstance(your_authkey, str)):
        raise ValueError("ERROR: your_port should be any str object")

    # Python Program to Get IP Address
    hostname = socket.gethostname()
    # IPAddr = socket.gethostbyname(hostname)
    IPAddr = [l for l in ([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2]
                           if not ip.startswith("127.")][:1],
                          [[(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) if l][0][0]
    print(f"Your Computer Name is = '{hostname}'")
    print(f"Your Computer IP Address is = '{IPAddr}'")
    os.system(f"echo '{IPAddr},{your_port},{your_authkey}' > '{str(file_to_write)}'")


def read_ip_port_auth(file_to_read) -> List[str]:
    file1 = open(file_to_read, "r")
    return file1.readline().strip('\n').strip().split(",")

###########################################################################################################
