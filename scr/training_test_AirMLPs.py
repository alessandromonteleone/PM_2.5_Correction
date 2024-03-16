import torch
import AirMLP.deep.model as models
import AirMLP.deep.training as training
import pandas as pd
from AirMLP.tool.preprocessing import DataCollection
from AirMLP.tool.create_dataset import creation
from AirMLP.tool.create_dataset import creation_v2
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.utils.data as data
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import math

import os

from AirMLP.deep.training import training_mlp
import statistics

import statistics


def plotter(model, tr_loss, ts_loss, r2test, folder, config):
    plt.clf()

    r2_value = statistics.median(r2test[-7:]) if len(r2test) >= 7 else 'Not enough data'

    plt.title(f"{config}\nr2 median value: {r2_value}")
    plt.plot(tr_loss, '-g', label="Train loss,MSE")
    plt.plot(ts_loss, '-b', label="Test loss,MSE")
    leg = plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.savefig(folder + "loss.png")

    plt.clf()
    plt.title(f"{config}\nr2 median value: {r2_value}")
    plt.plot(r2test, '-g', label="R2 test")
    plt.xlabel("Epoch")
    plt.ylabel("R2 Value")
    leg = plt.legend(loc='lower right')
    plt.savefig(folder + "r2.png")

    plt.clf()


# Config
PROB = 1
NUM_EPOCHS_ = 200
device = torch.device("cpu")

# Hyperparameters
NUM_RECORD = [20]
BATCH_SIZE = [64]
NUM_HIDDEN = [1500]

#train_file = '../MitH/dataset/corrected_data/train.csv'
#test_file = '../MitH/dataset/corrected_data/test.csv'

#train_file = '../MitH/dataset/preprocessed_data/train.csv'
#test_file = '../MitH/dataset/preprocessed_data/test.csv'

train_file = '../MitH/dataset/raw_data/train.csv'
test_file = '../MitH/dataset/raw_data/test.csv'


# Data Loader
#collection = DataCollection(drop_null=True)
#gt = collection.get_gt()

X = torch.tensor([]).to(device)
y = torch.tensor([]).to(device)

# modello
# device

num_features = 9 #features di input

num_trains = 1
res_trainings = list()
try:
    for record in NUM_RECORD:
        # Assuming tmp is your dataset
        X = torch.tensor([]).to(device)
        y = torch.tensor([]).to(device)

        # Load data using creation_v2
        (X_train, y_train), _ = creation_v2(train_file, test_file, lookback=record)
        print('X_train', X_train.shape)
        print('y_train', y_train.shape)
        # Flatten and concatenate the tensors
        X = torch.cat([X.clone(), X_train.flatten(-2)])
        y = torch.cat([y.clone(), y_train.flatten(-2)[:, 0]])

        print('X', X.shape)
        print('y', y.shape)

        for batch_s in BATCH_SIZE:
            for hidden in NUM_HIDDEN:

                model = models.AirMLP_7(num_fin=record * num_features, num_hidden=hidden).to(device)
                model_name = "AirMLP_7"

                config = f"record: {record} total_dim: {record * num_features}, batch_size: {batch_s}, hidden: {hidden}, model: {model}"
                config_short = f"record: {record} total_dim: {record * num_features}, batch_size: {batch_s}, hidden: {hidden}, model: {model_name}"
                print(f"Combination number: {num_trains}")
                print(config)
                res_tmp = training_mlp(X, y.view(-1, 1), model, batch_s, NUM_EPOCHS_, device)

                dir_new = rf"./results/trainings_{num_trains:03d}/"
                if not os.path.exists(dir_new):
                    os.makedirs(dir_new)
                num_trains += 1
                plotter(model, res_tmp[0], res_tmp[1], res_tmp[2], dir_new, config_short)

                torch.save(model,dir_new+"weights.pth")
                with open(dir_new + "config.txt", "w") as f:
                    f.write(config)
                with open(dir_new + "epoch_value.txt", "w") as f:
                    f.write("=========================")
                    f.write(str(res_tmp[0]))
                    f.write("\n=========================")
                    f.write(str(res_tmp[1]))
                    f.write("\n=========================")
                    f.write(str(res_tmp[2]))
                    f.write("\n=========================")
        #os.system(f"zip -r result_{num_trains}.zip ./results/")

except KeyboardInterrupt:
    #os.system(f"zip -r result_{num_trains}.zip ./results/")
    # Copiare i risultati senza comprimerli in caso di interruzione
    for foldername, subfolders, filenames in os.walk("./results/"):
        for filename in filenames:
            src_filepath = os.path.join(foldername, filename)
            dest_filepath = os.path.join(f"result_{num_trains}", filename)
            shutil.copy2(src_filepath, dest_filepath)