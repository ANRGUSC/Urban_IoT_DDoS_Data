import math
import sys
import os
import statistics
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from ast import literal_eval
import numpy as np
from datetime import datetime, timedelta
import random
from multiprocessing import Pool, Manager
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


def generate_active_time_mean_all(data, time_step, active_mean_time, active_mean_time_rows):
    """ Generate the data of the mean active time of the nodes

    Keyword arguments:
    data -- the main dataset
    time_step -- the time-step between the rows of the main dataset
    active_mean_time -- the dataframe template for storing the active mean time information
    active_mean_time_rows -- the shared variable among all processors for storing the mean active time information
    """
    data = data.sort_values(by=["TIME"])
    data["TIME_2"] = data["TIME"].dt.time
    data["TIME_2"] = pd.to_datetime(data["TIME_2"].astype(str))
    node = data["NODE"][0]
    date = data["TIME_2"][0]
    date_8_am = datetime(date.year, date.month, date.day, 8, 0, 0)
    date_8_pm = datetime(date.year, date.month, date.day, 20, 0, 0)
    print("node: ", node)

    day_active_list = []
    day_not_active_list = []
    night_active_list = []
    night_not_active_list = []

    day_sum_active = 0
    day_sum_not_active = 0
    night_sum_active = 0
    night_sum_not_active = 0
    day_before = None
    night_before = None

    for index, row in data.iterrows():
        if row["TIME_2"] == date_8_am or row["TIME_2"] == date_8_pm:
            day_before = None
            night_before = None

        if row["TIME_2"] >= date_8_am and row["TIME_2"] < date_8_pm:
            if day_before is None:
                day_before = row["ACTIVE"]
            if night_sum_active != 0:
                night_active_list.append(night_sum_active * time_step / 60)
                night_sum_active = 0
            if night_sum_not_active != 0:
                night_not_active_list.append(night_sum_not_active * time_step / 60)
                night_sum_not_active = 0

            if row["ACTIVE"] == 1 and day_before == 1:
                day_sum_active += 1
            elif row["ACTIVE"] == 1 and day_before == 0:
                day_not_active_list.append(day_sum_not_active * time_step / 60)
                day_sum_not_active = 0
            elif row["ACTIVE"] == 0 and day_before == 0:
                day_sum_not_active += 1
            elif row["ACTIVE"] == 0 and day_before == 1:
                day_active_list.append(day_sum_active * time_step / 60)
                day_sum_active = 0
            day_before = row["ACTIVE"]
        else:
            if night_before is None:
                night_before = row["ACTIVE"]
            if day_sum_active != 0:
                day_active_list.append(day_sum_active * time_step / 60)
                day_sum_active = 0
            if day_sum_not_active != 0:
                day_not_active_list.append(day_sum_not_active * time_step / 60)
                day_sum_not_active = 0

            if row["ACTIVE"] == 1 and night_before == 1:
                night_sum_active += 1
            elif row["ACTIVE"] == 1 and night_before == 0:
                night_not_active_list.append(night_sum_not_active * time_step / 60)
                night_sum_not_active = 0
            elif row["ACTIVE"] == 0 and night_before == 0:
                night_sum_not_active += 1
            elif row["ACTIVE"] == 0 and night_before == 1:
                night_active_list.append(night_sum_active * time_step / 60)
                night_sum_active = 0
            night_before = row["ACTIVE"]

    if day_sum_active != 0:
        day_active_list.append(day_sum_active * time_step / 60)
    if day_sum_not_active != 0:
        day_not_active_list.append(day_sum_not_active * time_step / 60)
    if night_sum_active != 0:
        night_active_list.append(night_sum_active * time_step / 60)
    if night_sum_not_active != 0:
        night_not_active_list.append(night_sum_not_active * time_step / 60)

    if len(day_active_list) == 0:
        day_active_list.append(0)
    if len(day_not_active_list) == 0:
        day_not_active_list.append(0)
    if len(night_active_list) == 0:
        night_active_list.append(0)
    if len(night_not_active_list) == 0:
        night_not_active_list.append(0)

    active_mean_time = active_mean_time.append({"NODE": node, "TIME": "DAY",
                                                "ACTIVE_MEAN": statistics.mean(day_active_list),
                                                "NOT_ACTIVE_MEAN": statistics.mean(day_not_active_list),
                                                "ACTIVE": day_active_list, "NOT_ACTIVE": day_not_active_list},
                                               ignore_index= True)
    active_mean_time = active_mean_time.append({"NODE": node, "TIME": "NIGHT",
                                                "ACTIVE_MEAN": statistics.mean(night_active_list),
                                                "NOT_ACTIVE_MEAN": statistics.mean(night_not_active_list),
                                                "ACTIVE": night_active_list, "NOT_ACTIVE": night_not_active_list},
                                               ignore_index= True)
    active_mean_time_rows.append(active_mean_time)


def plot_active_time_mean_per_entry(data, output_path):
    """ Plot the mean active time of the nodes according to the entries in the dataset

    Keyword arguments:
    data -- the data of the mean active time of the nodes
    output_path -- output path for storing the plots
    """
    temp = data.loc[data["TIME"] == "DAY"].ACTIVE.sum()
    temp = list(filter(lambda x: x != 0, temp))
    plt.clf()
    plt.hist(temp, bins=50, label="Average= {:.2f}".format(statistics.mean(temp)))
    plt.yscale('log')
    #plt.title("Day Active Mean Time")
    plt.xlabel("Active Mean Time (Minutes)")
    plt.ylabel("Number of Entries")
    plt.legend()
    output_path_day = output_path + "day_active_mean_time.png"
    plt.savefig(output_path_day)

    temp = data.loc[data["TIME"] == "DAY"].NOT_ACTIVE.sum()
    temp = list(filter(lambda x: x != 0, temp))
    plt.clf()
    plt.hist(temp, bins=50, label="Average= {:.2f}".format(statistics.mean(temp)))
    plt.yscale('log')
    #plt.title("Day Inactive Mean Time")
    plt.xlabel("Inactive Mean Time (Minutes)")
    plt.ylabel("Number of Entries")
    plt.legend()
    output_path_day = output_path + "day_not_active_mean_time.png"
    plt.savefig(output_path_day)

    temp = data.loc[data["TIME"] == "NIGHT"].ACTIVE.sum()
    temp = list(filter(lambda x: x != 0, temp))
    plt.clf()
    plt.hist(temp, bins=50, label="Average= {:.2f}".format(statistics.mean(temp)))
    plt.yscale('log')
    #plt.title("Night Active Mean Time")
    plt.xlabel("Active Mean Time (Minutes)")
    plt.ylabel("Number of Entries")
    plt.legend()
    output_path_night = output_path + "night_active_mean_time.png"
    plt.savefig(output_path_night)

    temp = data.loc[data["TIME"] == "NIGHT"].NOT_ACTIVE.sum()
    temp = list(filter(lambda x: x != 0, temp))
    plt.clf()
    plt.hist(temp, bins=50, label="Average= {:.2f}".format(statistics.mean(temp)))
    plt.yscale('log')
    #plt.title("Night Inactive Mean Time")
    plt.xlabel("Inactive Mean Time (Minutes)")
    plt.ylabel("Number of Entries")
    plt.legend()
    output_path_night = output_path + "night_not_active_mean_time.png"
    plt.savefig(output_path_night)


def plot_active_time_mean_per_node(data, output_path):
    """ Plot the mean active time of the nodes according to the nodes in the dataset

    Keyword arguments:
    data -- the data of the mean active time of the nodes
    output_path -- output path for storing the plots
    """
    nodes = list(data["NODE"].unique())

    plot_data = pd.DataFrame()
    for node in nodes:
        temp = data.loc[(data["TIME"] == "DAY") & (data["NODE"] == node)].ACTIVE.sum()
        temp = list(filter(lambda x: x != 0, temp))
        if len(temp) == 0:
            temp = [0]
        active_mean = statistics.mean(temp)

        temp = data.loc[(data["TIME"] == "DAY") & (data["NODE"] == node)].NOT_ACTIVE.sum()
        temp = list(filter(lambda x: x != 0, temp))
        if len(temp) == 0:
            temp = [0]
        not_active_mean = statistics.mean(temp)
        plot_data = plot_data.append({"NODE": node, "TIME": "DAY", "ACTIVE_MEAN": active_mean,
                                      "NOT_ACTIVE_MEAN": not_active_mean}, ignore_index=True)

        temp = data.loc[(data["TIME"] == "NIGHT") & (data["NODE"] == node)].ACTIVE.sum()
        temp = list(filter(lambda x: x != 0, temp))
        if len(temp) == 0:
            temp = [0]
        active_mean = statistics.mean(temp)

        temp = data.loc[(data["TIME"] == "NIGHT") & (data["NODE"] == node)].NOT_ACTIVE.sum()
        temp = list(filter(lambda x: x != 0, temp))
        if len(temp) == 0:
            temp = [0]
        not_active_mean = statistics.mean(temp)
        plot_data = plot_data.append({"NODE": node, "TIME": "NIGHT", "ACTIVE_MEAN": active_mean,
                                      "NOT_ACTIVE_MEAN": not_active_mean}, ignore_index=True)


    temp = list(plot_data.loc[plot_data["TIME"] == "DAY"]["ACTIVE_MEAN"].values)
    plt.clf()
    plt.hist(temp, bins=50, label="Average= {:.2f}".format(statistics.mean(temp)))
    #plt.yscale('log')
    #plt.title("Day Active Mean Time")
    plt.xlabel("Active Mean Time (Minutes)")
    plt.ylabel("Number of Nodes")
    plt.legend()
    output_path_day = output_path + "day_active_mean_time.png"
    plt.savefig(output_path_day)

    temp = list(plot_data.loc[plot_data["TIME"] == "DAY"]["NOT_ACTIVE_MEAN"].values)
    plt.clf()
    plt.hist(temp, bins=50, label="Average= {:.2f}".format(statistics.mean(temp)))
    #plt.yscale('log')
    #plt.title("Day Inactive Mean Time")
    plt.xlabel("Inactive Mean Time (Minutes)")
    plt.ylabel("Number of Nodes")
    plt.legend()
    output_path_day = output_path + "day_not_active_mean_time.png"
    plt.savefig(output_path_day)

    temp = list(plot_data.loc[plot_data["TIME"] == "NIGHT"]["ACTIVE_MEAN"].values)
    plt.clf()
    plt.hist(temp, bins=50, label="Average= {:.2f}".format(statistics.mean(temp)))
    #plt.yscale('log')
    #plt.title("Night Active Mean Time")
    plt.xlabel("Active Mean Time (Minutes)")
    plt.ylabel("Number of Nodes")
    plt.legend()
    output_path_night = output_path + "night_active_mean_time.png"
    plt.savefig(output_path_night)

    temp = list(plot_data.loc[plot_data["TIME"] == "NIGHT"]["NOT_ACTIVE_MEAN"].values)
    plt.clf()
    plt.hist(temp, bins=50, label="Average= {:.2f}".format(statistics.mean(temp)))
    #plt.yscale('log')
    #plt.title("Night Inactive Mean Time")
    plt.xlabel("Inactive Mean Time (Minutes)")
    plt.ylabel("Number of Nodes")
    plt.legend()
    output_path_night = output_path + "night_not_active_mean_time.png"
    plt.savefig(output_path_night)


def main_generate_active_time_all():
    """ The main function for generating the data of mean active time of the nodes
    """
    benign_dataset_path = CONFIG.OUTPUT_DIRECTORY + "clean_dataset/Output/benign_data/benign_data_2021-01-02 00:00:00_2021-02-01 23:59:58_time_step_30_num_ids_20.csv"
    benign_data = load_dataset(benign_dataset_path)
    print("Benign data loaded: ", datetime.now())

    benign_data = benign_data.sort_values(by=["NODE"]).reset_index(drop=True)
    print("benign_data sorted: ", datetime.now())

    dataset_list = []
    time_step = 30
    start_index = 0
    index_step = (int)(31*24*60*(60/time_step))
    while start_index < len(benign_data):
        end_index = start_index + index_step
        selected_data = benign_data[start_index:end_index].reset_index(drop=True)

        dataset_list.extend([selected_data])
        start_index += index_step


    print("Dataset_list created: ", datetime.now())
    del benign_data
    print("benign data deleted: ", datetime.now())
    active_mean_time = pd.DataFrame(columns=["NODE", "DATE", "TIME", "ACTIVE", "NOT_ACTIVE"])

    manager = Manager()
    active_mean_time_rows = manager.list([active_mean_time])
    print("active_mean_time_rows added to manager: ", datetime.now())

    output_path = CONFIG.OUTPUT_DIRECTORY + "stats/Output/nodes_active_mean_time/data/"
    prepare_output_directory(output_path)
    print("Output directory prepared: ", datetime.now())
    print("Start preparing processors pool: ", datetime.now())

    p = Pool()
    p.starmap(generate_active_time_mean_all, product(dataset_list, [time_step], [active_mean_time],
                                                                   [active_mean_time_rows]))
    p.close()
    p.join()

    active_mean_time = pd.concat(active_mean_time_rows, ignore_index=True)
    active_mean_time = active_mean_time.sort_values(by=["NODE", "DATE"])
    output_path += "active_mean_time_all.csv"
    active_mean_time.to_csv(output_path, index=False)


def main_plot_active_time_mean_per_entries():
    """ The main function for plotting the mean active time of the nodes according to the entries in the dataset
    """
    dataset_path = CONFIG.OUTPUT_DIRECTORY + "stats/Output/nodes_active_mean_time/data/active_mean_time_all.csv"
    data = pd.read_csv(dataset_path, converters={"ACTIVE": literal_eval, "NOT_ACTIVE": literal_eval})

    output_path = CONFIG.OUTPUT_DIRECTORY + "stats/Output/nodes_active_mean_time/plot_per_entry/"
    prepare_output_directory(output_path)

    plot_active_time_mean_per_entry(data, output_path)


def main_plot_active_time_mean_per_nodes():
    """ The main function for plotting the mean active time of the nodes according to the nodes
    """
    dataset_path = CONFIG.OUTPUT_DIRECTORY + "stats/Output/nodes_active_mean_time/data/active_mean_time_all.csv"
    data = pd.read_csv(dataset_path, converters={"ACTIVE": literal_eval, "NOT_ACTIVE": literal_eval})

    output_path = CONFIG.OUTPUT_DIRECTORY + "stats/Output/nodes_active_mean_time/plot_per_node/"
    prepare_output_directory(output_path)

    plot_active_time_mean_per_node(data, output_path)


if __name__ == "__main__":
    main_generate_active_time_all()
    main_plot_active_time_mean_per_entries()
    main_plot_active_time_mean_per_nodes()