import sys
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, classification_report
from sklearn.utils import resample
import random
import os
from pickle import dump
import matplotlib.pyplot as plt
import glob
from datetime import datetime

sys.path.append("../")
import project_config as CONFIG


def prepare_output_directory(output_path):
    dir_name = str(os.path.dirname(output_path))
    os.system("rm -rf " + dir_name)
    os.system("mkdir -p " + dir_name)


def load_dataset(path):
    data = pd.read_csv(path)
    return data


def get_train_dataset_input_output(data, num_devices, scaler_save_path):
    temp = data.drop(columns=["TIME", "NODE", "BEGIN_DATE", "END_DATE", "NUM_NODES", "ATTACK_RATIO", "ATTACK_DURATION"])
    X = temp.iloc[:,0:-num_devices]
    y = temp.iloc[:,-num_devices:]
    X = np.asarray(X).astype(np.float)
    y = np.asarray(y).astype(np.float)
    print(X.shape)
    print(y.shape)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    dump(scaler, open(scaler_save_path, 'wb'))

    return X, y, scaler


def get_test_dataset_input_output(data, num_devices, scaler):
    temp = data.drop(columns=["TIME", "NODE", "BEGIN_DATE", "END_DATE", "NUM_NODES", "ATTACK_RATIO", "ATTACK_DURATION"])
    X = temp.iloc[:,0:-num_devices]
    y = temp.iloc[:,-num_devices:]
    X = np.asarray(X).astype(np.float)
    y = np.asarray(y).astype(np.float)
    print(X.shape)
    print(y.shape)

    X = scaler.transform(X)

    return X, y


def create_nn_model(input_shape, output_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(8, input_shape=(input_shape,), activation='relu'))
    tf.keras.layers.BatchNormalization()
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(output_shape, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.BinaryAccuracy(),
                           tf.keras.metrics.BinaryCrossentropy(), tf.keras.metrics.Recall(),
                           tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(),
                           tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()])
    model.summary()
    return model


def setup_callbacks(saved_model_path):
    checkpoint_path = saved_model_path + "checkpoints/all/weights-{epoch:04d}"
    prepare_output_directory(checkpoint_path)
    cp = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True)

    log_path = saved_model_path + "logs/logs.csv"
    prepare_output_directory(log_path)
    csv_logger = tf.keras.callbacks.CSVLogger(log_path, separator=',', append=False)

    callbacks = [cp, csv_logger]
    return callbacks


def plot_logs(logs_path, output_path):
    logs = pd.read_csv(logs_path)
    metrics = logs.columns.values
    new_metrics = {}
    for metric in metrics:
        if metric[-2] == '_':
            new_metrics[metric] = metric[:-2]
        elif metric[-3] == '_':
            new_metrics[metric] = metric[:-3]

    logs = logs.rename(new_metrics, axis="columns")
    metrics = logs.columns.values

    for metric in metrics:
        if metric == "epoch" or "val" in metric:
            continue
        plt.clf()
        plt.plot(logs["epoch"], logs[metric], label="Train")
        plt.plot(logs["epoch"], logs["val_"+metric], label="Test")
        plt.xlabel("Epoch Number")
        plt.ylabel(metric)
        plt.title(metric + " vs epoch")
        plt.legend()
        plt.savefig(output_path + metric + ".png")


def main_plot_logs():
    all_saved_models_path = CONFIG.OUTPUT_DIRECTORY + "train/Output/saved_model/*"
    for directory in glob.glob(all_saved_models_path):
        print(directory)
        logs_path = directory + "/logs/logs.csv"
        output_path = directory + "/logs/pics/"
        prepare_output_directory(output_path)
        plot_logs(logs_path, output_path)


def main_train_model():
    seed = 1
    tf.random.set_seed(seed)
    random.seed(seed)

    train_dataset_path = CONFIG.OUTPUT_DIRECTORY + "pre_process/Output/train_data/train_data.csv"
    test_dataset_path = CONFIG.OUTPUT_DIRECTORY + "pre_process/Output/test_data/test_data.csv"
    train_dataset_all = load_dataset(train_dataset_path)
    test_dataset_all = load_dataset(test_dataset_path)
    model_output_path = CONFIG.OUTPUT_DIRECTORY + "train/Output/saved_model/"
    prepare_output_directory(model_output_path)

    initial_model_path = CONFIG.OUTPUT_DIRECTORY + "train/Output/initial_model/"
    prepare_output_directory(initial_model_path)

    num_devices = 1
    initial_scaler_save_path = initial_model_path + "scaler.pkl"
    X_train, y_train, scaler = get_train_dataset_input_output(train_dataset_all, num_devices, initial_scaler_save_path)
    model = create_nn_model(X_train.shape[1], y_train.shape[1])
    model.save(initial_model_path)
    #model = tf.keras.models.load_model(initial_model_path)

    nodes = list(train_dataset_all["NODE"].unique())

    for node_index, node in enumerate(nodes):
        scaler_save_path = model_output_path + str(node) + "/scaler.pkl"
        prepare_output_directory(scaler_save_path)

        saved_model_path = model_output_path + str(node) + '/'
        prepare_output_directory(saved_model_path)

        callbacks_list = setup_callbacks(saved_model_path)

        train_dataset = train_dataset_all.loc[train_dataset_all["NODE"] == node]
        test_dataset = test_dataset_all.loc[test_dataset_all["NODE"] == node]

        print(train_dataset["ATTACKED"].value_counts())
        print(test_dataset["ATTACKED"].value_counts())
        attacked_data = train_dataset.loc[train_dataset["ATTACKED"] == 1]
        not_attacked_data = train_dataset.loc[train_dataset["ATTACKED"] == 0]
        attacked_data = resample(attacked_data, replace=True, n_samples=not_attacked_data.shape[0], random_state=10)
        train_dataset = pd.concat([attacked_data, not_attacked_data])
        print(train_dataset["ATTACKED"].value_counts())

        num_devices = 1
        X_train, y_train, scaler = get_train_dataset_input_output(train_dataset, num_devices, scaler_save_path)
        X_test, y_test = get_test_dataset_input_output(test_dataset, num_devices, scaler)

        model = tf.keras.models.load_model(initial_model_path)

        epochs = 2
        batch_size = 32

        model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test), epochs=epochs,
                            verbose=1, callbacks=callbacks_list)

        model.save(saved_model_path + "final_model")


if __name__ == "__main__":
    main_train_model()
    main_plot_logs()