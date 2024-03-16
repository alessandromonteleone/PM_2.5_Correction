import torch
import numpy as np
import torch.nn as nn

import torch.optim as optim
import torch.utils.data as data
import torch
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split

import torch.optim as optim
import torch.utils.data as data
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import math



def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)

    return r2, mae, mape, rmse, mse


def training_lstm(model, device, dataset, num_epochs=200,batch=512,lr=0.0001,lookback=20):

    '''
    Training function for AirModel. Put in a different file for ease
    '''

    # dataset must be: ((X_train,y_train),(X_test,y_test))
    train, test = dataset
    X_train, y_train = train
    X_test, y_test = test
    

    # Training
    
    loss_fn = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=batch)



    n_epochs = num_epochs
    for epoch in range(n_epochs):
        model.train()
        i=0
        for X_batch, y_batch in loader:
            i+=1
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        if epoch % 10 != 0:
            continue
        model.eval()
        with torch.no_grad():
            train_rmse = torch.sqrt(loss_fn(model(X_train), y_train))
            test_rmse = torch.sqrt(loss_fn(model(X_test), y_test))
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))


def training_mlp(X,y, model_, BATCH_SIZE, NUM_EPOCHS, device):
    # Training
    LR=5e-5
    #loss_fn = nn.MSELoss(reduction='mean').to(device)
    loss_fn = nn.L1Loss().to(device)
    optimizer = optim.RAdam(model_.parameters(), lr=LR)
    loss_eval = nn.L1Loss().to(device)
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=BATCH_SIZE)
    loader_test = data.DataLoader(data.TensorDataset(X_test, y_test), shuffle=False, batch_size=BATCH_SIZE)

    tr_loss = list()
    ts_loss = list()
    r2_test_list = list()
    mae_test_list = list()
    mape_test_list = list()
    rmse_test_list = list()
    mse_test_list = list()
    
    for epoch in range(NUM_EPOCHS):
        
        model_.train()
        for X_batch, y_batch in loader:
            
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()

            y_pred = model_(X_batch)
            #print(y_pred.shape)
            loss = loss_fn(y_pred, y_batch)
            #print(y_batch.shape, y_pred.shape)
            
            loss.backward()

            optimizer.step()
        
        model_.eval()
        
        pred_train = model_(X_train)
        pred_test = model_(X_test)
        tr = loss_eval(pred_train,y_train)
        ts = loss_eval(pred_test,y_test)

        tr_loss.append(tr.cpu().detach().numpy().item())
        ts_loss.append(ts.cpu().detach().numpy().item())


        # Calculate additional metrics for the test set
        r2_test, mae_test, mape_test, rmse_test, mse_test = calculate_metrics(
            y_test.cpu().detach().numpy(), pred_test.cpu().detach().numpy()
        )

        r2_test_list.append(r2_test)
        mae_test_list.append(mae_test)
        mape_test_list.append(mape_test)
        rmse_test_list.append(rmse_test)
        mse_test_list.append(mse_test)


        # Validation
        if epoch % 1 != 0:
            continue
        
        print(f"Epoch {epoch:3d} train mse.: {tr:.3f} test mse.: {ts:.3f}")

        print(f"On test -- R-2 : {r2_test:.3f}   MAE: {mae_test:.3f}, MAPE: {mape_test:.3f}%, RMSE: {rmse_test:.3f}, MSE: {mse_test:.3f}")

    model_.eval()
    return (tr_loss, ts_loss, r2_test_list)