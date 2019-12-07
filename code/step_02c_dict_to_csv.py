import glob
import os
import pickle
import time

import pandas as pd

import common_services as cs

OUTPUT_FOLDER = "../../../data_out_combined"
bool_empty_in_past = False
while True:
    obj_files_list = glob.glob(f"{OUTPUT_FOLDER}/*.obj")
    if len(obj_files_list) == 0:
        if bool_empty_in_past:
            print(".", end="", flush=True)
        else:
            print("\nLooking for *.obj files.", end="")
            bool_empty_in_past = True
    else:
        bool_empty_in_past = False
        print()

    for i_file in obj_files_list:
        with cs.ExecutionTime(name="Write dictionary to CSV"):
            print(f"\nDEBUG: found a dictionary in OUTPUT_FOLDER, working...")
            print(f"val i_file = {i_file}")
            dict_obj = pickle.load(open(i_file, 'rb'))
            data_out = pd.DataFrame(data=len(dict_obj) * [[None, None]], columns=cs.COLUMNS)
            k = 0
            for i_key, j_val in dict_obj.items():
                data_out.loc[k] = [i_key, j_val]
                k += 1
            data_out.to_csv(i_file[:-4], index=False)
            print(f"SUCCESSFULLY written: '{i_file[:-4]}'")
            print(f"removing i_file = {i_file}")
            print()
            os.system(f"rm -f {i_file}")
    time.sleep(1)
