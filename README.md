# Urban_IoT_Data

This repository presents the source code for analyzing an Urban IoT Activity Dataset. It helps with generating attacks on the datasets and a neural network model for detecting them.


## Instructions for running the codes

The requirements.txt file contains the modules needed to run these scripts and can be installed by running the following command in the terminal:
* pip install -r requirements.txt

## Project config file

The project config file can be found in [/source_code](https://github.com/ANRGUSC/Urban_IoT_Data/tree/main/source_code). The path to the output directory can be set in this file.

## Pre-processing the dataset

Before running any code, the original dataset need to be unzip in the [/dataset directory](https://github.com/ANRGUSC/Urban_IoT_Data/tree/main/dataset). Three python scripts can be found in [/source_code/pre_process](https://github.com/ANRGUSC/Urban_IoT_Data/tree/main/source_code/pre_process) folder for pre-processing the raw dataset. 

### pre_process.py

This file genrates the bening_dataset which has N nodes and each node has time entries starting from the beginning to the end of original dataset with a step of time_step.

Input:
- Input Dataset
- Number of nodes
- Timestep

Output:
- Benign dataset


### generate_attack.py

This script genrates the attacked dataset by considering the ratio of the nodes that are under attack, the attack duration, and also the attack start dates.

Input:
- Bening dataset
- Number of attack days
- Attack ratio
- Attack duration
- Attack start dates

Output:
- Attacked dataset

### generate_training_data.py

This script generates the training data by considering the different time windows for averaging the occupancies on the nodes.

Input:
- Attacked dataset
- Averaging time windows

Output:
- Training dataset


## Training the dataset

Given the generated training datset in the pre-processing procedure, we provide a feed-forward neural netwrok model to train on the data and detect the attacked nodes and times. Two python scripts can be found in [/source_code/train](https://github.com/ANRGUSC/Urban_IoT_Data/tree/main/source_code/train) folder for training the dataset and generating the results. 


### train_nn.py

This script create a feed-forward neural network to train on the training dataset for detecting the attackers. The scrip save the final model and also the epochs logs and weights.

Input:
- Training dataset
- Number of epochs

Output:
- Trained neural network model with epochs' logs and weights


### generate_results.py

This script provides analysis like, accuracy, loss, confusion matrix, etc. based on the trained model. Furthermore, it plots that true positive, false positive, and true attacks versus time.

Input:
- Training dataset
- Trained neural network model

Output:
- General analysis on the training like accuracy, loss, confusion matrix, etc.
- Plots of true positive, false positive, and true attacks versus time for different attack ratios and durations



