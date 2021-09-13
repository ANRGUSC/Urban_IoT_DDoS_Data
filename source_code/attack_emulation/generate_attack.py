import math
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import random
from multiprocessing import Pool
from itertools import product
sys.path.append("../")
import project_config as CONFIG


def prepare_output_directory(output_path):
    """Prepare the output directory by deleting the old files and create an empty directory.

    Keyword arguments:
    output_path -- path to the output directory
    """
    dir_name = str(os.path.dirname(output_path))
    os.system("rm -rf " + dir_name)
    os.system("mkdir -p " + dir_name)


def load_dataset(path):
    """Load the dataset and change the type of the "TIME" column to datetime.

    Keyword arguments:
    path -- path to the dataset
    """
    data = pd.read_csv(path)
    data["TIME"] = pd.to_datetime(data["TIME"])
    return data


def generate_attack_train(benign_data, attack_begin_date, attack_end_date, attacked_ratio_nodes, attack_duration,
                          attack_start_times, output_path, data_type):
    """Create attack in the benign dataset for the given features based on the data type.

    Keyword arguments:
    benign_data -- benign dataset to be used for attacking
    attack_begin_date -- the begin date of the attack
    attack_end_date -- the end date of the attack
    attacked_ratio_nodes -- the ratio of the nodes in the benign dataset to be attacked.
    attack_duration -- the duration of the attack
    attack_start_times -- the start times of the attacks withing the attack_begin_date and attack_end_date
    output_path -- the output path for storing the attacked dataset
    data_type -- could be "train" or "test". For the test data_type, we select the attacked nodes in the way that
                higher ratio attacked nodes contain the lower ratio attacked nodes.
    """

    if data_type == "train":
        random.seed()
    else:
        random.seed(1)
    data_selected_nodes = benign_data.loc[(benign_data["TIME"] >= attack_begin_date) &
                                          (benign_data["TIME"] < attack_end_date)]
    nodes = list(data_selected_nodes["NODE"].unique())
    num_attacked_nodes = math.ceil(len(nodes)*attacked_ratio_nodes)

    # ignore the attack start times which are included in the attack interval of the previous attack
    i = 0
    while i < len(attack_start_times)-1:
        if attack_start_times[i+1] <= attack_start_times[i] + attack_duration:
            attack_start_times.pop(i+1)
            continue
        i += 1

    for attack_start_time in attack_start_times:
        attacked_nodes = list(random.sample(nodes, k=num_attacked_nodes))
        attack_finish_time = attack_start_time + attack_duration

        benign_data.loc[(benign_data["NODE"].isin(attacked_nodes)) &
                        (benign_data["TIME"] >= attack_start_time) &
                        (benign_data["TIME"] < attack_finish_time), "ACTIVE"] = 1
        benign_data.loc[(benign_data["NODE"].isin(attacked_nodes)) &
                        (benign_data["TIME"] >= attack_start_time) &
                        (benign_data["TIME"] < attack_finish_time), "ATTACKED"] = 1

    benign_data["BEGIN_DATE"] = attack_begin_date
    benign_data["END_DATE"] = attack_end_date
    benign_data["NUM_NODES"] = len(nodes)
    benign_data["ATTACK_RATIO"] = attacked_ratio_nodes
    benign_data["ATTACK_DURATION"] = attack_duration

    output_path +=  "attacked_data_" + data_type + '_' + str(attack_begin_date) + '_' + str(attack_end_date) +\
                    "_ratio_" + str(attacked_ratio_nodes) + "_duration_" + str(attack_duration) + ".csv"
    benign_data.to_csv(output_path, index=False)


def main_generate_attack(benign_dataset_path, data_type, num_train_days, num_test_days):
    """The main function to be used for calling generate_attack_train function

    Keyword arguments:
    benign_dataset_path -- the path to the benign dataset to be used for attacking
    data_type -- could be "train" or "test". For the test data_type, we select the attacked nodes in the way that
                higher ratio attacked nodes contain the lower ratio attacked nodes.
    num_train_days -- the number of days to be used for creating the attacked dataset for training purposes
    num_test_days -- the number of days to be used for creating the attacked dataset for testing purposes
    """

    benign_data = load_dataset(benign_dataset_path)

    # set the begin and end date of the dataset to be attacked
    if data_type == "train":
        attack_begin_date = benign_data.loc[0, "TIME"] + timedelta(days=2)
        attack_end_date = benign_data.loc[0, "TIME"] + timedelta(days=2+num_train_days)
        output_path = CONFIG.OUTPUT_DIRECTORY + "attack_emulation/Output/attacked_data/train/"
    else:
        attack_begin_date = benign_data.loc[0, "TIME"] + timedelta(days=2+num_train_days+2)
        attack_end_date = benign_data.loc[0, "TIME"] + timedelta(days=2+num_train_days+2+num_test_days)
        output_path = CONFIG.OUTPUT_DIRECTORY + "attack_emulation/Output/attacked_data/test/"

    prepare_output_directory(output_path)
    # set the begin and end date of the dataset to be stored for generating features in generate_training_data.py
    # here we choose begin_date - 2days because in the features we have an occupancy average of 48 hours
    slice_benign_data_start = attack_begin_date-timedelta(days=2)
    slice_benign_data_end = attack_end_date

    benign_data = benign_data.loc[(benign_data["TIME"] >= slice_benign_data_start) &
                                  (benign_data["TIME"] < slice_benign_data_end)]
    benign_data_save = benign_data.copy()
    nodes = list(benign_data_save["NODE"].unique())
    benign_data_save["BEGIN_DATE"] = attack_begin_date
    benign_data_save["END_DATE"] = attack_end_date
    benign_data_save["NUM_NODES"] = len(nodes)
    benign_data_save["ATTACK_RATIO"] = 0.0
    benign_data_save["ATTACK_DURATION"] = timedelta(hours=0)

    output_path_benign = output_path +  "attacked_data_" + data_type + '_' + str(attack_begin_date) + '_' +\
                         str(attack_end_date) + "_ratio_0_duration_0.csv"
    benign_data_save.to_csv(output_path_benign, index=False)
    # set the ratio, duration, and start time of the attack
    # in the case that start time of an attack be included in the interval of the previous attack, it will be ignored
    attacked_ratio_nodes = [1]
    attack_duration = [timedelta(hours=1), timedelta(hours=2), timedelta(hours=4), timedelta(hours=8), timedelta(hours=16)]
    attack_start_times = []
    for attack_day in range(num_train_days):
        attack_start_times.append(attack_begin_date + timedelta(days=attack_day, hours=2))

    p = Pool()
    p.starmap(generate_attack_train, product([benign_data], [attack_begin_date], [attack_end_date],
                                             attacked_ratio_nodes, attack_duration, [attack_start_times], [output_path],
                                             [data_type]))
    p.close()
    p.join()


if __name__ == "__main__":
    benign_dataset_path = CONFIG.OUTPUT_DIRECTORY + "clean_dataset/Output/benign_data/benign_data_2021-01-02 00:00:00_2021-02-01 23:59:58_time_step_30_num_ids_20.csv"
    num_train_days = 7
    num_test_days = 7

    main_generate_attack(benign_dataset_path, "train", num_train_days, num_test_days)
    print("generate train data done.")
    main_generate_attack(benign_dataset_path, "test", num_train_days, num_test_days)
    print("generate test data done.")