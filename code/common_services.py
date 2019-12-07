import gc
import os
import socket
import subprocess
import sys
import timeit
from pathlib import Path
from typing import List, Union

from shell import Shell

COLUMNS = ["fen_board", "cp_score"]


###########################################################################################################

# Thanks to StackOverFlow
# REFER QUESTION : https://stackoverflow.com/questions/7370801/measure-time-elapsed-in-python/41408510#41408510
# REFER ANSWER   : Shital Shah (https://stackoverflow.com/users/207661/shital-shah)

class ExecutionTime:
    def __init__(self, name="(block)", seconds_precision=1, file=sys.stdout, no_print=False, disable_gc=False):
        """
        Simple class to measure execution time of any block of code quickly.

        Usage:

        with ExecutionTime():
            something_to_work()

        Output:
        [ExecutionTime] "name" = hh:mm:ss.milliseconds

        :param name: any str which is to be printed along with execution time
        :param seconds_precision: the precision to which milliseconds should be printed
        :param file: place to print the execution time
        :param no_print: whether to print the execution time or not
        :param disable_gc: disable garbage collector or not
        """
        assert seconds_precision >= 0, "`seconds_precision` should be a positive integer"

        self.name = name
        self.seconds_precision = seconds_precision
        self.file = file
        self.no_print = no_print
        self.disable_gc = disable_gc

        self.__gc_old = None
        self.__start_time = None
        self.elapsed = 0.0
        self.elapsed_str = 'None'

    def __enter__(self):
        if self.disable_gc:
            self.__gc_old = gc.isenabled()
            gc.disable()
        self.__start_time = timeit.default_timer()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = timeit.default_timer() - self.__start_time
        seconds = self.elapsed % 60
        minutes = int(self.elapsed // 60) % 60
        hours = int(self.elapsed // 3600)
        self.elapsed_str = f"{hours:02}:{minutes:02}:{seconds:0{2 + 1 + self.seconds_precision}.{self.seconds_precision}f}"

        if self.disable_gc and self.__gc_old:
            gc.enable()
        if not self.no_print:
            print('[ExecutionTime] "{}" = {}'.format(self.name, self.elapsed_str), file=self.file)
        return False  # re-raise any exceptions


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


def read_last_line(resume_file_name) -> str:
    if not Path(resume_file_name).exists():
        print(f"DEBUG: file does not exists: '{resume_file_name}'"
              f"\n\treturning default value, i.e. array of 1's of size {variable_length}", file=sys.stderr)
        return ''

    sh = Shell(has_input=False, record_output=True, record_errors=True, strip_empty=True)

    return sh.run(f"tail -n 1 '{resume_file_name}'").output(raw=True).strip('\n').strip()


###########################################################################################################

def get_network_ip() -> str:
    """
    Returns the network IP address of the client machine.
    If not connected to the network, will return loop-back IP address
    :return: str
    """

    try:
        return [
            l for l in (
                [ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")][:1],
                [[(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]
            ) if l
        ][0][0]
    except OSError as e:
        print(f"OSError: {e}")
        print(f"WARNING: returning loop-back IP address: 127.0.0.1")
        return "127.0.0.1"


def print_ip_port_auth(file_to_write: Union[str, Path] = 'step_02_preprocess_server_ip.txt', your_port: int = None, your_authkey: str = None) -> None:
    if your_port is None or not (isinstance(your_port, int)) or not (1000 <= your_port <= 65535):
        raise ValueError("ERROR: your_port should be int -> [1000-65535]")
    if your_authkey is None or not (isinstance(your_authkey, str)):
        raise ValueError("ERROR: your_port should be any str object")

    # Python Program to Get IP Address
    hostname = socket.gethostname()
    IPAddr = get_network_ip()
    # IPAddr = socket.gethostbyname(hostname)
    print(f"Your Computer Name is = '{hostname}'")
    print(f"Your Computer IP Address is = '{IPAddr}'")
    os.system(f"echo '{IPAddr},{your_port},{your_authkey}' > '{str(file_to_write)}'")


def read_ip_port_auth(file_to_read: Union[str, Path] = 'step_02_preprocess_server_ip.txt') -> List[str]:
    if not Path(file_to_read).exists():
        raise FileNotFoundError(file_to_read)

    file1 = open(str(file_to_read), "r")
    return file1.readline().strip('\n').strip().split(",")

###########################################################################################################
