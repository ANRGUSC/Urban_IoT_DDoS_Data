# Urban_IoT_DDoS_Data

This repository presents the source code for analyzing an Urban IoT Activity Dataset. It helps with generating attacks on the datasets and a neural network model for detecting them.


## Instructions for running the codes

The requirements.txt file contains the modules needed to run these scripts and can be installed by running the following command in the terminal:
* pip install -r requirements.txt

## Project config file

The project config file can be found in [/source_code](https://github.com/ANRGUSC/Urban_IoT_Data/tree/main/source_code). The path to the output directory can be set in this file.

## Cleaning the dataset

Before running any code, the original dataset need to be unzip in the [/dataset directory](https://github.com/ANRGUSC/Urban_IoT_Data/tree/main/dataset). One python scripts can be found in [/source_code/clean_dataset](https://github.com/ANRGUSC/Urban_IoT_Data/tree/main/source_code/clean_dataset) folder for pre-processing the original dataset. 

### clean_dataset.py

This file genrates the bening_dataset which has N nodes and each node has time entries starting from the beginning to the end of original dataset with a step of time_step.

Input:
- Input Dataset
- Number of nodes
- Timestep

Output:
- Benign dataset


## Dataset Statistics

Three python scripts can be found in [/source_code/stats](https://github.com/ANRGUSC/Urban_IoT_Data/tree/main/source_code/stats) folder for generating the statistics of the benign dataset.


### active_nodes_percentage.py

This script generates the plots of active nodes percentage vs time.

Input:
- Benign dataset

Output:
- Plots of active nodes percentage vs time.

### correlation.py

This script generates the plot of Pearson correlation of nodes vs their distance.

Input:
- Benign dataset

Output:
- Plot of Pearson correlation of nodes vs their distance

### nodes_active_mean_time.py

This script generates the histograms of nodes active and inactive mean time vs time of the day.

Input:
- Benign dataset

Output:
- Plot histograms of nodes active and inactive mean time vs time of the day


## Attack emulation

One python scripts can be found in [/source_code/attack_emulation](https://github.com/ANRGUSC/Urban_IoT_Data/tree/main/source_code/attack_emulation) folder for generating DDoS attack on the original dataset. 


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


## Training neural network

Three python scripts can be found in [/source_code/nn_training](https://github.com/ANRGUSC/Urban_IoT_Data/tree/main/source_code/nn_training) folder for generating the labeled training and testing dataset, train a feed-forward neural network, and generating the results of training.


### generate_training_data.py

This script generates the training data by considering the different time windows for averaging the occupancies on the nodes.

Input:
- Attacked dataset
- Averaging time windows

Output:
- Training dataset


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


## Acknowledgement

   This material is based upon work supported in part by Defense Advanced Research Projects Agency (DARPA) under Contract No. HR001120C0160 for the Open, Programmable, Secure 5G (OPS-5G) program. Any views, opinions, and/or findings expressed are those of the author(s) and should not be interpreted as representing the official views or policies of the Department of Defense or the U.S. Government. 



