# Progress Report


### 4 October 2019
1. Came across a library called minimalcluster which can be used to perform distributed computing over a cluster of computers over a LAN network.
    - Limitations - chess.Board object counld not be passed over the network


### 13 November 2019
1. Found a BUG in minimalcluster
    - Details: processes spawned on remote clients did not stop even after executing return or exit(0)
    - Root cause: -- not known --
    - Solution: use os.kill(os.getpid(), 9) in place of return or exit(0)
2. ADD debug statements to the library and two new methods to help the user decide the debuging level
3. UPDATE minimalcluster worker_node.py to unblock the condition that maximum processes that can be created is less than or equal to available CPU cores
    - After update: user can change the flag when instatiating the worker node to unblock the condition.
    - Use case: if a network operation or a memory operation is to be performed, then processes have to wait for long time. This can be solved by created processes more than the available CPU cores
4. UPDATE minimalcluster to raise Exception if any problem occures instead of exiting the program abruptly.
5. UPDATE minimalcluster worker_node.py hostname to include IP address along with hostname.


### 15 November 2019 - from 8:30 AM to 5:30 PM
1. First day of distributed computing work
2. Installed basic requirements on 17+18+6 computers
3. Manually goto all computers and note down the IP address.
4. Wrote shell scripts to work with the IP addresses, send data to all computers and execute the client program(*step_02b_preprocess_client.py*)
5. Take backup of the processed data.
6. PROCESSED
    - 70 of 83 `.csv` files of KingBase2019-A00-A39.pgn - each file contains atleat 10000 chess boards


### 18 November 2019 - from 8:30 AM to 5:30 PM
1. Second day of distributed computing work
2. Manually change the IP address to static IP address.
3. Shell script and ".tar" files created to expedite the requirements installation, client program extraction, environment cleanup(removing the installation files) and starting of the *step_02b_preprocess_client.py* program on each computer.
    - pyreq.tar  : tar file with miniconda3 and all the required Python packages
    - pyreq.sh   : shell scripts to install miniconda3 and the required Python packages and perform cleanup of unneeded files
    - chess.tar  : tar file with source code of the project
    - chess.sh   : shell scripts to extract the project and perform cleanup of unneeded files
    - 1client.sh : execute minimalcluster client program
4. BUG found in minimalcluster library when number of clients increased above 40.
    - Details: The client part of the library used to check the network shared queue every 0.01 second to find whether it is empty or not. If not empty, then it will append its hostname to the list. This caused infinite loop as the master part of the library used to take more time that 0.01 to collect the queue elements.
    - Root cause: the method workers_list() was responsible for it. It was a library bug.
    - Solution: We added a time.sleep(1.01) at the client part of the library, so that the master process has enough time to collect all the queue elements. This reduced the network load as well
5. BUG found in minimalcluster library with increasing number of clients.
    - Details: The data distributed amongst the clients as either increasing or decreasing every time. This resulted in partial processing of the input given.
    - Root cause: network latency caused this. Multiple clients were working on the same data/task/job and when the results were returned, the count was added in a variable even if the data is already returned by another client.
    - Solution: Updated the library to maintain a local dictionary with job_id of all the tasks to be performed/not yet completed. Every time a result is returned by the client, the job_id of the task is checked in the dictionary and if it returned for the first time by any client, then it is removed from the dictionary and result length is added to the counter variable.
6. Unable to take backup of the processed data as we got late. Hence file processing progress not known.


### 19 November 2019 - from 9:00 AM to 4:45 PM
1. Third day of distributed computing work
2. Setup more computers to act as client
3. All the processed data got deleted. We were unable to recove it.
    - Reason: -- not known --
4. Take lunch break
5. Restore data from day 1 (15 November 2019)
6. WRITE shell script and Python code to combine multiple *.csv* files into one to reduce the latency of switching between the processing of two files
7. UPDATE the server program to write the processed data returned on a new process and start processing of next *.csv* file in parallel
    - Reason: it took approximately 2:30 minutes to convert a dictionary with `~45,000` entries to a pandas.DataFrame and save it to a csv.
    - Results: latency to switch from one *.csv* to another took less than 10 seconds.
8. Will resume after end-semester exams which start from 26 November 2019.
9. PROCESSED - resume work from 15th November 2019
    - 13 of 83 `.csv` files of `KingBase2019-A00-A39.pgn` - each file contains atleat 10000 chess boards
    - 14 of 85 `.csv` files of `KingBase2019-A00-A39.pgn` - combined files where each file is concatenation of 5 files with each file having atleat 10000 chess boards


### 4 December 2019 - from 9:00 AM to 3:45 PM
1. First day after the 4th paper of the end-semester exam.
2. Freshly installed the client program on all the computers.
3. UPDATE 2chess.sh implementation
    - Earlier, we had to run pyreq.sh, chess.sh and 1client.sh to get the client connected to the server for the first time.
    - Now, just run 2chess.sh and it will install all the python requirements, extract the source code and create the 1client.sh file in home directory(i.e. `/home/student` ) of the user `student` and run 1client.sh
4. BUG found in worker_node.py
    - Problem: The created processes do not exit with exit() and sys.exit()
    - Effect: The client only works for the first task. After that, it would be idle. We have to manually restart them to get them working for the next task.
    - Solution:  os.kill(os.getpid()) stops the running process.
    - This was fixed in the evening. Hence it could only be in effect on the next day.
5. PROCESSED - resume work from 19th November 2019
    - 46 of 85 `.csv` files of `KingBase2019-A00-A39.pgn` - combined files where each file is concatenation of 5 files with each file having atleat 10000 chess boards
6. Make changes in worker_node.py to remove code duplication.
    - Used finally clause of try...catch for exiting any block of code.
7. UPDATE master_node.py
    - Add lots of debug statements
    - Fix launch master as worker node feature as old parameters were being used.

### 5 December 2019 - from 9:00 AM to 4:00 PM
1. Try testing the worker_node.py with a few computers only
2. BUG found in worker_node.py
    - Problem: the finally clause was being executed after each try block however it was to be executed only 
3. BUG found in worker_node.py
    - Problem: there were becomming idle
    - Cause: Python and network internal problem, the process kept waiting for network data.
    - Soluction: A smart algorithm was written which would take `approx_max_job_time` as its input and if processes more than `nproc` are spawned, then it will wait for `approx_max_job_time` to see if there is change in status of any alive process. If yes, reset the time counter. If no, kill all the spawned processes and reset the connection with the master.
4. As there are a lot of changes, we had to replace old source code files with updated ones.
5. WRITE shell script to delete old source code files and extract new source code
    - 3chess.sh
6. UPDATE master_node.py
    - `in execute(...)` method now takes a parameter called `approx_max_job_time` which tells master how much time it should wait before putting any job back into the job queue.
    - `list_workers(...)` method takes a parameter called `approx_max_workers` which tells the master how many max workers can be connected to it. If there are more in the queue, then it will skip them and empty the queue.
    - if any worker node returns with an error in error_q, then remove that job_id from pending jobs queue, i.e. skip that job_id job
7. UPDATE step_02a_preprocess_server.py:
    - the registered function is updated to return -100000000 if the string FEN board is not convertable to chess.Board object
7. PROCESSED - resume work from 4th December 2019
    - 25 of 85 `.csv` files of `KingBase2019-A00-A39.pgn` - combined files where each file is concatenation of 5 files with each file having atleat 10000 chess boards
    - 40 of 63 `.csv` files of `KingBase2019-A40-A79.pgn` - combined files where each file is concatenation of 5 files with each file having atleat 10000 chess boards


### 6 December 2019 - from 9:00 AM to 3:30 PM
1. UPDATE `step_02a_preprocess_server.py`
    - Add function to the environment string which allows us to shutdown the connected clients by executing a command on all at once.
    - Add function to randomly wait for some time and append its `PID` to a file. If its `PID` is first in the file, then execute the shell command, otherwise sleep. This allows us to update source code on client computers if they are connected to the master node and restart step_02b_preprocess_client.py
2. We had approximately 100 clients connected to the server.
3. UPDATE `step_02a_preprocess_server.py` to write the result dictionary directly using pickle as an object file `.obj`
    - Advantage: the results are written to the disk in just one second.
4. WRITE `step_02c_dict_to_csv.py` to constantly look for `.obj` and as soon as it if found, process and convert it to a `.csv` file
    - Advantage: this allows us to restart the master node of the cluster without having to wait for the last calculated results to be written to the disk as a `.csv` file.
5. UPDATE `step_02a_preprocess_server.py` to beep if `list_workers(...)` returns more than 500 workers indicating problem in cluster working which impacted the performance.
6. PROCESSED - resume work from 5th December 2019
    - 23 of 63 `.csv` files of `KingBase2019-A40-A79.pgn` - combined files where each file is concatenation of 5 files with each file having atleat 10000 chess boards
    - 12 of 12 `.csv` files of `KingBase2019-A80-A99.pgn` - same as above
    - 47 of 66 `.csv` files of `KingBase2019-B00-B19.pgn` - same as above
7. Will resume after end-semester exams which ends on 9th December 2019
8. ADD class `ChessPlayCLI` to `step_04_play.py` which can be used to play chess via command line interface
`
    - class ChessPredict
        * __init__(...)
        * predict_score_1(...)
        * predict_best_move_1(...)
        * predict_score_n(...)
        * predict_best_move_n(...)
    - class ChessPlayCLI
        * __init__(...)
        * __cpu_obj_playable(...)
        * __get_user_input(...)
        * __pretty_board(...)
        * play()
`



### 7 December 2019
1. UPDATE encode_board method to include check mate status as well. Thus leading to 778 features/attributes
2. FINISH encode_board_1_778 implementation.
3. UPDATE `step_02_preprocess.py` to have class called `BoardEncoder` with the following methods:
    - is_check(...)
    - is_checkmate(...)
    - encode_board_1_778(...)
    - encode_board_n_778(...)     : uses all available CPU cores
    - encode_board_1_778_fen(...)
    - encode_board_n_778_fen(...) : uses all available CPU cores
4. UPDATE `step_03a_ffnn.py` and finish the implementation of FFNNKeras
`
    - class FFNNKeras
        * __init__(...)
        * __generate_model(...)
        * c_save_model(...)
        * c_save_weights(...)
        * c_load_model(...)
        * c_load_weights(...)
        * c_train_model(...)
        * c_evaluate_model(...)
        * c_predict(...)
        * c_predict_board_1(...)
        * c_predict_board_n(...)
        * c_predict_fen_1(...)
        * c_predict_fen_n(...)
    - class FFNNTorch
        * __init__(...)
        * c_save_model(...)
        * c_load_model(...)
        * c_train_model(...)
        * c_predict(...)
        * predict_board(...)
        * predict_fen(...)
`

5. Saved the first model trained with 45258 chess boards from the file `out_combined_KingBase2019-B00-B19_000000.csv`
    - Learning rate = 0.001, optimizer = 'adam'
    - Results after 10 epochs:
        * loss: 0.0753
        * mae: 0.0992
        * val_loss: 0.2326 : validation Loss
        * val_mae: 0.2112  : validation Mean Absolute Error


### 10 December 2019
5. PROCESSED - resume work from 6th December 2019
