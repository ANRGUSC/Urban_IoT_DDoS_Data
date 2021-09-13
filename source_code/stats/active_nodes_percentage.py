import math
import sys
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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


def active_nodes_percentage_per_day(data, date, output_path):
    data = data.loc[(data["TIME"] >= date) & (data["TIME"] < (date+timedelta(hours=24)))]
    temp = data.groupby(["TIME"]).mean().reset_index()
    temp = temp[["TIME", "ACTIVE"]]
    temp = temp.sort_values(by=["TIME"])

    times = list(temp["TIME"].values)

    plt.clf()

    fig, ax = plt.subplots()
    ax.plot(times, temp["ACTIVE"], label="Active")
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    myFmt = mdates.DateFormatter('%H')
    ax.xaxis.set_major_formatter(myFmt)
    #ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Activity Percentage")

    output_path += "active_nodes_percentage_" + str(date)[0:10] + ".png"
    fig.savefig(output_path)


def active_nodes_percentage_all(data, output_path):
    data = data.sort_values(by=["TIME"])
    data["TIME_2"] = data["TIME"].dt.time
    data["TIME_2"] = pd.to_datetime(data["TIME_2"].astype(str))

    temp = data.groupby(["TIME_2"]).mean().reset_index()
    temp = temp[["TIME_2", "ACTIVE"]]
    temp = temp.sort_values(by=["TIME_2"])
    times = list(temp["TIME_2"].values)

    plt.clf()

    fig, ax = plt.subplots()
    ax.plot(times, temp["ACTIVE"], label="ACTIVE")
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    myFmt = mdates.DateFormatter('%H')
    ax.xaxis.set_major_formatter(myFmt)
    #ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Activity Percentage")

    output_path += "active_nodes_percentage_all.png"
    fig.savefig(output_path)


def main_active_nodes_percentage_per_day():
    benign_dataset_path = CONFIG.OUTPUT_DIRECTORY + "clean_dataset/Output/benign_data/benign_data_2021-01-02 00:00:00_2021-02-01 23:59:58_time_step_120_num_ids_3.csv"
    benign_data = load_dataset(benign_dataset_path)
    benign_data["TIME"] = pd.to_datetime(benign_data["TIME"])

    begin_date = benign_data.loc[0, "TIME"]
    begin_date = datetime(begin_date.year, begin_date.month, begin_date.day+1, 0, 0, 0)
    end_date = benign_data.loc[len(benign_data)-1, "TIME"]

    dates = []
    date = begin_date
    while date < end_date:
        dates.append(date)
        date += timedelta(hours=24)

    output_path = CONFIG.OUTPUT_DIRECTORY + "stats/Output/active_nodes_percentage/per_day/"
    prepare_output_directory(output_path)

    for date in dates:
        active_nodes_percentage_per_day(benign_data, date, output_path)


def main_active_nodes_percentage_all():
    benign_dataset_path = CONFIG.OUTPUT_DIRECTORY + "clean_dataset/Output/benign_data/benign_data_2021-01-02 00:00:00_2021-02-01 23:59:58_time_step_30_num_ids_20.csv"
    benign_data = load_dataset(benign_dataset_path)
    benign_data["TIME"] = pd.to_datetime(benign_data["TIME"])

    output_path = CONFIG.OUTPUT_DIRECTORY + "stats/Output/active_nodes_percentage/all/"
    prepare_output_directory(output_path)
    active_nodes_percentage_all(benign_data, output_path)


if __name__ == "__main__":
    main_active_nodes_percentage_per_day()
    main_active_nodes_percentage_all()