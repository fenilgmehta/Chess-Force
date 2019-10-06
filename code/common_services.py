import os
import subprocess
from pathlib import Path
from typing import List, Union


def append_secondlast_line(resume_file_name, msg):
    # (last game written + 1), (next file number to be written)
    os.system(f"""
last_line=`tail -n1 '{resume_file_name}'`
sed '$d' '{resume_file_name}'
echo '{msg}' >> '{resume_file_name}'
echo $last_line >> '{resume_file_name}'
""")


def savepoint(resume_file_name, msg):
    os.system(f"sed '$d' '{resume_file_name}' ; echo '{msg}' >> '{resume_file_name}'")


def readpoint(resume_file_name, variable_length) -> Union[List[int], int]:
    if not Path(resume_file_name).exists():
        return variable_length * [1, ]
    return [int(i) for i in subprocess.getoutput(f"tail -n1 '{resume_file_name}'").split(",")][:variable_length]
