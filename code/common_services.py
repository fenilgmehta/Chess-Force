import os
import sys
import subprocess
from pathlib import Path
from typing import List, Union


def append_secondlast_line(resume_file_name, msg):
    # (last game written + 1), (next file number to be written)
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
