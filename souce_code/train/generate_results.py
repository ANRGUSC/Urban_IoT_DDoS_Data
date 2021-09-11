import sys
import glob
import pylab as pl
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, classification_report, precision_recall_fscore_support
import random
import os
from multiprocessing import Pool
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pickle import load


sys.path.append("../")
import project_config as CONFIG


def prepare_output_directory(output_path):
    dir_name = str(os.path.dirname(output_path))
    os.system("rm -rf " + dir_name)
    os.system("mkdir -p " + dir_name)


def load_dataset(path):
    data = pd.read_csv(path)
    data["TIME"] = pd.to_datetime(data["TIME"])
    return data


def get_input_target(data, num_devices, scaler):
    temp = data.drop(columns=["TIME", "NODE", "BEGIN_DATE", "END_DATE", "NUM_NODES", "ATTACK_RATIO", "ATTACK_DURATION"])
    X = temp.iloc[:,0:-num_devices]
    y = temp.iloc[:,-num_devices:]
    y = y.astype(int)
    X = scaler.transform(X)
    return X, y


def load_model_weights(model_path_input, node, metric, mode):
    saved_model_path = model_path_input + str(node) + '/'
    scaler_path = saved_model_path + "scaler.pkl"
    model_path = saved_model_path + "final_model/"
    logs_path = saved_model_path + "logs/logs.csv"
    logs = pd.read_csv(logs_path)
    metrics = list(logs.columns)
    metrics.remove("epoch")

    if mode == "max":
        logs = logs.sort_values(by=[metric], ascending=False).reset_index(drop=True)
    elif mode == "min":
        logs = logs.sort_values(by=[metric]).reset_index(drop=True)

    epoch = str((int)(logs["epoch"][0]) + 1).zfill(4)
    checkpoint_path = saved_model_path + "checkpoints/all/weights-" + epoch

    model = tf.keras.models.load_model(model_path)
    model.load_weights(checkpoint_path)
    model.summary()
    scaler = load(open(scaler_path, 'rb'))

    return model, scaler


def generate_general_report(train_dataset, test_dataset, model_path_input, metric, mode, output_path):
    seed = 1
    tf.random.set_seed(seed)
    random.seed(seed)

    data = pd.DataFrame()

    nodes = list(train_dataset["NODE"].unique())
    for index, node in enumerate(nodes):
        print("node_index: ", index)

        model, scaler = load_model_weights(model_path_input, node, metric, mode)
        train_dataset_node = train_dataset.loc[train_dataset["NODE"] == node]
        test_dataset_node = test_dataset.loc[test_dataset["NODE"] == node]

        num_devices = 1
        X_train, y_train = get_input_target(train_dataset_node, num_devices, scaler)
        X_test, y_test = get_input_target(test_dataset_node, num_devices, scaler)

        row = {"node": node}
        evaluate_train = model.evaluate(X_train, y_train, verbose=1, return_dict=True, use_multiprocessing=True)
        evaluate_test = model.evaluate(X_test, y_test, verbose=1, return_dict=True, use_multiprocessing=True)
        row.update(evaluate_train)
        for key, value in evaluate_test.items():
            row["val_" + key] = value

        data = data.append(row, ignore_index=True)

    data = data.append(data.mean(axis=0), ignore_index=True)
    output_path += "general_report_" + metric + '_' + mode + ".csv"
    #prepare_output_directory(output_path)
    dir_name = str(os.path.dirname(output_path))
    os.system("mkdir -p " + dir_name)
    data.to_csv(output_path, index=False)
    print(data)


def generate_attack_prediction_vs_time(model_path_input, train_dataset, test_dataset, metric, mode, output_path):

    seed = 1
    tf.random.set_seed(seed)
    random.seed(seed)

    train_result = pd.DataFrame()
    test_result = pd.DataFrame()

    train_dataset = train_dataset.sort_values(by=["TIME"])
    test_dataset = test_dataset.sort_values(by=["TIME"])

    nodes = list(train_dataset["NODE"].unique())
    for index, node in enumerate(nodes):
        model, scaler = load_model_weights(model_path_input, node, metric, mode)

        num_devices = 1
        train_dataset_node = train_dataset.loc[train_dataset["NODE"] == node]
        X_train, y_train = get_input_target(train_dataset_node, num_devices, scaler)
        y_pred_train = np.rint(model.predict(X_train)).astype(int)
        y_train = np.rint(y_train)

        train_dataset_2 = train_dataset_node.copy()
        train_dataset_2["TRUE"] = y_train["ATTACKED"]
        train_dataset_2["PRED"] = y_pred_train
        train_dataset_2["TP"] = train_dataset_2["ATTACKED"] & train_dataset_2["PRED"]
        train_dataset_2["FP"] = train_dataset_2["TP"] ^ train_dataset_2["PRED"]
        train_result = train_result.append(train_dataset_2, ignore_index=True)

        test_dataset_node = test_dataset.loc[test_dataset["NODE"] == node]
        X_test, y_test = get_input_target(test_dataset_node, num_devices, scaler)
        y_pred_test = np.rint(model.predict(X_test)).astype(int)
        y_test = np.rint(y_test)
        test_dataset_2 = test_dataset_node.copy()
        test_dataset_2["TRUE"] = y_test["ATTACKED"]
        test_dataset_2["PRED"] = y_pred_test
        test_dataset_2["TP"] = test_dataset_2["ATTACKED"] & test_dataset_2["PRED"]
        test_dataset_2["FP"] = test_dataset_2["TP"] ^ test_dataset_2["PRED"]
        test_result = test_result.append(test_dataset_2, ignore_index=True)

    #prepare_output_directory(output_path)
    os.system("mkdir -p " + output_path)
    train_result_output_path = output_path + "train_result_" + metric + '_' + mode + ".csv"
    train_result.to_csv(train_result_output_path, index=False)

    test_result_output_path = output_path + "test_result_" + metric + '_' + mode + ".csv"
    test_result.to_csv(test_result_output_path, index=False)


def plot_attack_prediction_vs_time(train_result_path, test_result_path, train_output_path, test_output_path):

    train_result = load_dataset(train_result_path)
    attack_ratios = list(train_result["ATTACK_RATIO"].unique())
    attack_durations = list(train_result["ATTACK_DURATION"].unique())
    prepare_output_directory(train_output_path)

    for attack_ratio in attack_ratios:
        for attack_duration in attack_durations:
            plot_data = train_result.loc[(train_result["ATTACK_RATIO"] == attack_ratio) &
                                         (train_result["ATTACK_DURATION"] == attack_duration)]
            if plot_data.shape[0] == 0:
                continue
            plot_data = plot_data.groupby(["TIME"]).mean().reset_index()
            plot_data = plot_data.sort_values(by=["TIME"])
            plt.clf()
            fig, ax = plt.subplots()
            ax.plot(plot_data["TIME"], plot_data["TRUE"], label="True")
            ax.plot(plot_data["TIME"], plot_data["TP"], label="TP")
            ax.plot(plot_data["TIME"], plot_data["FP"], label="FP")
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
            myFmt = mdates.DateFormatter('%H')
            ax.xaxis.set_major_formatter(myFmt)
            ax.legend()
            ax.set_xlabel("Time")
            ax.set_ylabel("Attack")
            ax.set_title("Attack_Ratio= " + str(attack_ratio) + " - Duration: " + str(attack_duration))

            output_path = train_output_path + "train_attack_prediction_vs_time_" + str(attack_duration) + \
                                '_attackRatio_' + str(attack_ratio) + '_duration_' +\
                                str(attack_duration) + '.png'
            fig.savefig(output_path)


    test_result = load_dataset(test_result_path)
    attack_ratios = list(test_result["ATTACK_RATIO"].unique())
    attack_durations = list(test_result["ATTACK_DURATION"].unique())
    prepare_output_directory(test_output_path)

    for attack_ratio in attack_ratios:
        for attack_duration in attack_durations:
            plot_data = test_result.loc[(test_result["ATTACK_RATIO"] == attack_ratio) &
                                         (test_result["ATTACK_DURATION"] == attack_duration)]
            if plot_data.shape[0] == 0:
                continue
            plot_data = plot_data.groupby(["TIME"]).mean().reset_index()
            plot_data = plot_data.sort_values(by=["TIME"])
            plt.clf()
            fig, ax = plt.subplots()
            ax.plot(plot_data["TIME"], plot_data["TRUE"], label="True")
            ax.plot(plot_data["TIME"], plot_data["TP"], label="TP")
            ax.plot(plot_data["TIME"], plot_data["FP"], label="FP")
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
            myFmt = mdates.DateFormatter('%H')
            ax.xaxis.set_major_formatter(myFmt)
            ax.legend()
            ax.set_xlabel("Time")
            ax.set_ylabel("Attack")
            ax.set_title("Attack_Ratio= " + str(attack_ratio) + " - Duration: " + str(attack_duration))

            output_path = test_output_path + "test_attack_prediction_vs_time_" + str(attack_duration) + \
                                 '_attackRatio_' + str(attack_ratio) + '_duration_' + \
                                 str(attack_duration) + '.png'
            fig.savefig(output_path)


def main_general_report(metric, mode):
    train_dataset_path = CONFIG.OUTPUT_DIRECTORY + "pre_process/Output/train_data/train_data.csv"
    test_dataset_path = CONFIG.OUTPUT_DIRECTORY + "pre_process/Output/test_data/test_data.csv"
    train_dataset = load_dataset(train_dataset_path)
    test_dataset = load_dataset(test_dataset_path)
    model_path = CONFIG.OUTPUT_DIRECTORY + "train/Output/saved_model/"

    output_path = CONFIG.OUTPUT_DIRECTORY + "train/Output/report/"

    generate_general_report(train_dataset, test_dataset, model_path, metric, mode, output_path)


def main_generate_attack_prediction_vs_time(metric, mode):

    train_dataset_path = CONFIG.OUTPUT_DIRECTORY + "pre_process/Output/train_data/train_data.csv"
    test_dataset_path = CONFIG.OUTPUT_DIRECTORY + "pre_process/Output/test_data/test_data.csv"
    train_dataset = load_dataset(train_dataset_path)
    test_dataset = load_dataset(test_dataset_path)
    model_path_input = CONFIG.OUTPUT_DIRECTORY + "train/Output/saved_model/"
    output_path = CONFIG.OUTPUT_DIRECTORY + "train/Output/attack_prediction_vs_time/data/"

    generate_attack_prediction_vs_time(model_path_input, train_dataset, test_dataset, metric, mode, output_path)


def main_plot_attack_prediction_vs_time():
    train_output_path = CONFIG.OUTPUT_DIRECTORY + "train/Output/attack_prediction_vs_time/plot/train/"
    test_output_path = CONFIG.OUTPUT_DIRECTORY + "train/Output/attack_prediction_vs_time/plot/test/"
    prepare_output_directory(train_output_path)
    prepare_output_directory(test_output_path)

    train_result_path = CONFIG.OUTPUT_DIRECTORY + "train/Output/attack_prediction_vs_time/data/train_result_" + metric + '_' + mode + ".csv"
    test_result_path = CONFIG.OUTPUT_DIRECTORY + "train/Output/attack_prediction_vs_time/data/test_result_" + metric + '_' + mode + ".csv"

    plot_attack_prediction_vs_time(train_result_path, test_result_path, train_output_path, test_output_path)


if __name__ == "__main__":
    seed = 1
    tf.random.set_seed(seed)
    random.seed(seed)

    metric = "binary_accuracy"
    mode = "max"

    main_general_report(metric, mode)

    main_generate_attack_prediction_vs_time(metric, mode)
    main_plot_attack_prediction_vs_time()
