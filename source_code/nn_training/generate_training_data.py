import glob
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from itertools import product
from multiprocessing import Pool, Manager
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


def generate_features_time_slot(training_data_rows, data, time, nodes, durations, training_data):
    """Create the training data by generate the features for a given timeslot in the dataset

    Keyword arguments:
    training_data_rows -- the shared memory between cores for storing the training data rows
    data -- the dataset to be used for generating the features
    time -- the time for generating features
    nodes -- the selected nodes for generating features
    durations -- the average time durations for generating the features
    training_data -- the training data template to be used for adding the generated features to the main output file
    """
    row = {}
    row["BEGIN_DATE"] = data.loc[0, "BEGIN_DATE"]
    row["END_DATE"] = data.loc[0, "END_DATE"]
    row["NUM_NODES"] = data.loc[0, "NUM_NODES"]
    row["ATTACK_RATIO"] = data.loc[0, "ATTACK_RATIO"]
    row["ATTACK_DURATION"] = data.loc[0, "ATTACK_DURATION"]
    row["TIME"] = time

    for node in nodes:
        for duration, duration_name in durations.items():
            occupied = data.loc[(data["NODE"] == node) &
                         (data["TIME"] <= time) &
                         (data["TIME"] >= time-duration), "ACTIVE"].mean()
            row[str(duration_name)] = occupied

        row["NODE"] = node
        row["ATTACKED"] = data.loc[(data["NODE"] == node) & (data["TIME"] == time), "ATTACKED"].values[0]
        training_data = training_data.append(row, ignore_index=True)

    training_data_rows.append(training_data)


def generate_features(attacked_dataset_path, output_path, data_type, num_train_days, num_test_days):
    """Create the training data by generate the features for a given attacked dataset

    Keyword arguments:
    attacked_dataset_path -- the path to the attacked dataset to be used for generating the features
    output_path -- the output path for storing the training data
    data_type -- could be "train" or "test". For the test data_type, we select the attacked nodes in the way that
                higher ratio attacked nodes contain the lower ratio attacked nodes.
    num_train_days -- the number of days to be used for creating the training dataset
    num_test_days -- the number of days to be used for creating the testing dataset
    """

    attacked_data = load_dataset(attacked_dataset_path)
    if data_type == "train":
        begin_date = attacked_data.loc[0, "TIME"] + timedelta(days=2)
        end_date = attacked_data.loc[0, "TIME"] + timedelta(days=2+num_train_days)
    else:
        begin_date = attacked_data.loc[0, "TIME"] + timedelta(days=2)
        end_date = attacked_data.loc[0, "TIME"] + timedelta(days=2+num_test_days)


    times = list(set(attacked_data.loc[ (attacked_data["TIME"] >= begin_date) &
                                        (attacked_data["TIME"] < end_date), "TIME"]))
    nodes = list(attacked_data["NODE"].unique())

    durations = {timedelta(minutes=1): "1min",
                 timedelta(minutes=10): "10min", timedelta(minutes=30): "30min", timedelta(hours=1): "1hr",
                 timedelta(hours=2): "2hr", timedelta(hours=4): "4hr", timedelta(hours=8): "8hr",
                 timedelta(hours=16): "16hr", timedelta(hours=24): "24hr", timedelta(hours=30): "30hr",
                 timedelta(hours=36): "36hr", timedelta(hours=42): "42hr", timedelta(hours=48): "48hr"}

    template_column = ["BEGIN_DATE", "END_DATE", "NUM_NODES", "ATTACK_RATIO", "ATTACK_DURATION", "NODE", "TIME"]

    for duration, duration_name in durations.items():
        template_column.append(str(duration_name))
    template_column.append("ATTACKED")
    training_data = pd.DataFrame(columns=template_column)

    manager = Manager()
    training_data_rows = manager.list([training_data])

    p = Pool()
    p.starmap(generate_features_time_slot, product([training_data_rows], [attacked_data], times, [nodes], [durations],
                                                   [training_data]))
    p.close()
    p.join()


    training_data = pd.concat(training_data_rows, ignore_index=True)
    training_data = training_data.sort_values(by=["TIME"])
    training_data.to_csv(output_path, index=False)


def combine_data(input_path, output_path):
    """Combine the csv files in the input_path directory and output the combined one to the output_path.

    Keyword arguments:
    input_path -- The path to the input directory.
    output_path -- The path to the output_directory for storing the combined data.
    """
    training_data = pd.DataFrame()
    for fname in glob.glob(input_path):
        print(fname)
        temp = pd.read_csv(fname)
        training_data = training_data.append(temp)
    training_data.to_csv(output_path, index=False)


def main_generate_features(data_type, num_train_days, num_test_days):
    """The main function to be used for calling  generate_features function.

    Keyword arguments:
    data_type -- could be "train" or "test". For the test data_type, we select the attacked nodes in the way that
                higher ratio attacked nodes contain the lower ratio attacked nodes.
    num_train_days -- the number of days to be used for creating the training dataset
    num_test_days -- the number of days to be used for creating the testing dataset
    """
    # path = "Output/attacked_data/train/*.csv"

    if data_type == "train":
        path = CONFIG.OUTPUT_DIRECTORY + "attack_emulation/Output/attacked_data/train/*.csv"
    else:
        path = CONFIG.OUTPUT_DIRECTORY + "attack_emulation/Output/attacked_data/test/*.csv"

    output_directory = CONFIG.OUTPUT_DIRECTORY + "nn_training/Output/" + data_type + "_data/"
    prepare_output_directory(output_directory)
    for attacked_dataset_path in glob.glob(path):
        print(attacked_dataset_path)
        fname = list(attacked_dataset_path.split('/'))[-1]
        output_path = output_directory + data_type + "_data_" + fname[20:]
        generate_features(attacked_dataset_path, output_path, data_type, num_train_days, num_test_days)


def main_combine_data(data_type):
    """The main function to be used for calling  combine_data function.

    Keyword arguments:
    data_type -- could be "train" or "test". For the test data_type, we select the attacked nodes in the way that
                higher ratio attacked nodes contain the lower ratio attacked nodes.
    """
    if data_type == "train":
        input_path = CONFIG.OUTPUT_DIRECTORY + "nn_training/Output/train_data/*.csv"
        output_path = CONFIG.OUTPUT_DIRECTORY + "nn_training/Output/train_data/train_data.csv"
    else:
        input_path = CONFIG.OUTPUT_DIRECTORY + "nn_training/Output/test_data/*.csv"
        output_path = CONFIG.OUTPUT_DIRECTORY + "nn_training/Output/test_data/test_data.csv"

    combine_data(input_path, output_path)


if __name__ == "__main__":
    """The main function of the file for generating the training data

    Keyword arguments:
    num_train_days -- number of days for attacking the dataset for training
    num_test_days -- number of days for attacking the dataset for testing
    """
    num_train_days = 1
    num_test_days = 1
    main_generate_features("train", num_train_days, num_test_days)
    main_combine_data("train")
    main_generate_features("test", num_train_days, num_test_days)
    main_combine_data("test")
