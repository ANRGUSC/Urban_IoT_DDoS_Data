import math
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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


def generate_correlation_data(data1, data2, plot_data, plot_data_rows):
    """ Generate the data of correlation information of the nodes vs their distance

    Keyword arguments:
    data1 -- the dataset of the first node
    data2 -- the dataset of the second node
    plot_data -- the dataframe template for storing the correlation
    plot_data_rows -- the shared variable for storing the correlation data among all processsors
    """
    node1 = data1["NODE"][0]
    node2 = data2["NODE"][0]
    print("Node 1: ", node1, " - Node 2: ", node2)

    dist = math.sqrt((data1["LAT"][0] - data2["LAT"][0]) ** 2 + (data1["LNG"][0] - data2["LNG"][0]) ** 2)
    correlation = data1["ACTIVE"].corr(data2["ACTIVE"])

    plot_data = plot_data.append({"NODE_1": node1, "NODE_2": node2, "DISTANCE": dist,
                                  "CORRELATION": correlation}, ignore_index=True)
    plot_data_rows.append(plot_data)


def plot_correlation_data(data, output_path):
    """ Plot the correlation of the nodes vs their distance

    Keyword arguments:
    data -- the dataset of the pairwise correlation information of the nodes
    output_path -- output path for storing the plots
    """
    plt.clf()
    plt.scatter(data["DISTANCE"], data["CORRELATION"])
    plt.xlabel("Distance")
    plt.ylabel("Correlation")
    #plt.title("Pearson's Correlation")

    path = output_path + "pearson_correlation.png"
    plt.savefig(path)


def plot_average_correlation_data(data, output_path):
    """ Plot the mean correlation of the nodes vs their distance

    Keyword arguments:
    data -- the dataset of the pairwise correlation information of the nodes
    output_path -- output path for storing the plots
    """
    data = data.sort_values(by=["DISTANCE"]).reset_index(drop=True)
    min_dist = data["DISTANCE"][0]
    max_dist = data["DISTANCE"][data.shape[0]-1]
    dist_step = (max_dist - min_dist) / 3
    distances = np.arange(min_dist, max_dist, dist_step)

    plot_dists = []
    plot_correlations = []

    i = 0
    while i < len(distances)-1:
        print("index: ", i)
        temp = data.loc[(data["DISTANCE"] >= distances[i]) &
                        (data["DISTANCE"] < distances[i+1])]
        if temp is not None:
            plot_dists.append(distances[i])
            plot_correlations.append(temp["CORRELATION"].mean())
        i += 1

    plt.clf()
    plt.plot(plot_dists, plot_correlations)
    plt.xlabel("Distance")
    plt.ylabel("Mean Correlation")
    #plt.title("Mean Correlation")

    path = output_path + "mean_pearson_correlation.png"
    plt.savefig(path)
    plt.show()


def main_generate_correlation_data():
    """ The main function for generating the correlation of the nodes vs their distance
    """
    benign_dataset_path = CONFIG.OUTPUT_DIRECTORY + "clean_dataset/Output/benign_data/benign_data_2021-01-02 00:00:00_2021-02-01 23:59:58_time_step_30_num_ids_20.csv"
    benign_data = load_dataset(benign_dataset_path)

    begin_date = benign_data.loc[0, "TIME"]
    begin_date = datetime(begin_date.year, begin_date.month, begin_date.day+1, 0, 0, 0)
    benign_data = benign_data.loc[benign_data["TIME"] >= begin_date]
    benign_data = benign_data.sort_values(by=["NODE"]).reset_index(drop=True)
    print("benign_data sorted: ", datetime.now())

    data_list = []
    time_step = 30
    start_index = 0
    index_step = (int)(31*24*60*(60/time_step))
    while start_index < len(benign_data):
        end_index = start_index + index_step
        selected_data = benign_data[start_index:end_index].sort_values(by=["TIME"]).reset_index(drop=True)

        data_list.extend([selected_data])
        start_index += index_step

    del benign_data

    plot_data = pd.DataFrame(columns=["NODE_1", "NODE_2", "DISTANCE", "CORRELATION"])
    manager = Manager()
    plot_data_rows = manager.list([plot_data])

    p = Pool()
    p.starmap(generate_correlation_data, product(data_list, data_list, [plot_data], [plot_data_rows]))
    p.close()
    p.join()

    plot_data = pd.concat(plot_data_rows, ignore_index=True)
    plot_data = plot_data.sort_values(by=["NODE_1", "NODE_2"])

    output_path = CONFIG.OUTPUT_DIRECTORY + "stats/Output/correlation/correlation.csv"
    prepare_output_directory(output_path)
    plot_data.to_csv(output_path, index=False)


def main_plot_correlation():
    """ The main function for plotting the correlation of the nodes vs their distance
    """
    dataset_path = CONFIG.OUTPUT_DIRECTORY + "stats/Output/correlation/correlation.csv"
    plot_data = pd.read_csv(dataset_path)

    print(len(list(plot_data["NODE_1"].unique())))

    output_path = CONFIG.OUTPUT_DIRECTORY + "stats/Output/correlation/plot/"
    prepare_output_directory(output_path)

    plot_correlation_data(plot_data, output_path)
    plot_average_correlation_data(plot_data, output_path)


if __name__ == "__main__":
    main_generate_correlation_data()
    main_plot_correlation()
