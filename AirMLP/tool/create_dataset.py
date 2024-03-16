import torch
import pandas as pd
def create_dataset(dataset, lookback):
        batch = len(dataset) - lookback
        
        X=torch.zeros((batch,lookback,dataset.shape[1]-1))
        y=torch.zeros((batch,lookback,1))
        for i in range(len(dataset)-lookback):
            feature = dataset[i:i+lookback,:-1]
            target = dataset[i+lookback,-1]
            X[i,:,:]=feature
            y[i,:,:]=target.unsqueeze(0).t()
        return X,y

#-----------------------------------------------------------------------------------------------------------------------


def creation(dataset_, lookback, p=1):
    '''
    Staring from a pandas.DataFrame object, divide it in train and test set, putting together a number of "lookback" consecutive record.
    Returns a tuple of tuple, where each of the two tuple are the torch.tensor relative to train and test set.
    
    '''
    dataset=dataset_.copy()
    train_size = int(len(dataset) * p)
    test_size = len(dataset) - train_size
    if p==1:
         train_size=len(dataset)

    # in futuro bisogna controllare l'eccezione sul drop e non sulla trasformazione in numpy
    try:
        train, test = torch.tensor(dataset.to_numpy()[:train_size]),torch.tensor(dataset.to_numpy()[train_size:])
    except TypeError:
        # Datetime never dropped, drop it now.
        dataset.drop(columns="valid_at",inplace=True)
        train, test = torch.tensor(dataset.to_numpy()[:train_size]),torch.tensor(dataset.to_numpy()[train_size:])
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    X_train, y_train = create_dataset(train, lookback=lookback)
    X_train, y_train = X_train.to(device), y_train.to(device)

    if p!=1:
        X_test, y_test = create_dataset(test, lookback=lookback)
        X_test, y_test = X_test.to(device), y_test.to(device)
        return (X_train, y_train),(X_test, y_test)
    else:
        return (X_train,y_train)

#-----------------------------------------------------------------------------------------------------------------------


def load_csv_dataset(file_path):
    '''
    Carica un dataset da un file CSV e restituisce un torch.tensor.
    '''


    df = pd.read_csv(file_path)
    #df = df[["pm2p5_x", "pm2p5_y"]]
    #df = df[['pm2p5_x', 'relative_humidity', 'temperature', "pm2p5_y"]]
    df = df[['pm1', 'pm2p5_x', 'pm4', 'pm10', 'relative_humidity', 'temperature', 'pressure', 'wind_speed', 'cloud_coverage', "pm2p5_y"]]

    return torch.tensor(df.values)


def creation_v2(train_file, test_file, lookback):
    '''
    Carica i dataset di addestramento e test da file CSV, unendo un numero di record consecutivi pari a "lookback".
    Restituisce un tuple di tuple, in cui ciascuna delle due tuple contiene i torch.tensor relativi al set di addestramento e test.
    '''
    # Carica i dataset da file CSV
    train = load_csv_dataset(train_file)
    test = load_csv_dataset(test_file)

    device = torch.device("cpu")

    # Crea i tensori X e y per il set di addestramento
    X_train, y_train = create_dataset(train, lookback=lookback)
    X_train, y_train = X_train.to(device), y_train.to(device)

    # Crea i tensori X e y per il set di test
    X_test, y_test = create_dataset(test, lookback=lookback)
    X_test, y_test = X_test.to(device), y_test.to(device)

    return (X_train, y_train), (X_test, y_test)

