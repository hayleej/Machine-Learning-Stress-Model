import accuracy as a
import matplotlib.pyplot as plt
import os.path as path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split 
import copy
import tqdm
from math import ceil


#! IMPORTANT VARIABLES START
#* for model
numLayers = 3
batch_size = 8
lookback = 5
hiddenLayerSize = 100
shuffle = False

#* for data
path_to_data = "Data" # folder the data is stored in
dataFiles = ['/kinetic/24-08-experiment1/F670_Info.csv', '/kinetic/24-08-experiment1/FD36_Logistics.csv', '/kinetic/23-08-experiment1/FD36_Logistics.csv', '/kinetic/24-08-experiment1/FD32_Fire.csv', '/kinetic/23-08-experiment1/F670_Info.csv', '/kinetic/23-08-experiment1/F831_Leader.csv', '/kinetic/23-08-experiment1/FD32_Fire.csv', '/kinetic/24-08-experiment1/FCB1_Rescue.csv', '/kinetic/24-08-experiment1/F831_Leader.csv', '/kinetic/23-08-experiment1/FCB1_Rescue.csv']    
#* for code
numPrint = 5 # how many times you print per training

#* for model saving
SAVED = True
path_to_model = "models/StressModel_" + str(numLayers) + "_" + str(lookback) + "_" + str(batch_size) + ".pth"
#! IMPORTANT VARIABLES END

class StressModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StressModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        #self.linear = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x, _ = self.lstm(x)
        # Extract the last time step's output
        x = x[:, -1, :]
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        #x = self.linear(x[:, -1, :])
        return x

def saveModelToFile(model, model_path):
    # Save the model
    torch.save(model.state_dict(), model_path)

def loadModel(model_path):
    # Load the model
    loaded_model = StressModel(6,hiddenLayerSize,numLayers,lookback)  # Create an instance of your model
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()  # Set the model to evaluation mode
    return loaded_model



def multiFeature_create_dataset(datasetX, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    
    X, y = [], []
    for i in range(len(datasetX)-lookback):
        feature = datasetX[i:i+lookback,:-1]
        target = datasetX[i:i+lookback,-1]
        X.append(feature)
        y.append(target)
    X = np.array(X)
    y = np.array(y)
    return torch.tensor(X), torch.tensor(y)


def createData(filename):
    df = pd.read_csv(path_to_data + filename,header=0)
    timeseries = df.values.astype('float32')
    lookback_data, labels = multiFeature_create_dataset(timeseries, lookback=lookback)
    return lookback_data, labels

def createTestData(filenames):
    X, y = [], []
    i = 0
    for file in filenames:
        i = i + 1
        df = pd.read_csv(path_to_data + file,header=0)
        timeseries = df.values.astype('float32')
        
        for i in range(len(timeseries)-lookback):
            feature = timeseries[i:i+lookback,:-1]
            target = timeseries[i:i+lookback,-1]
            X.append(feature)
            y.append(target)
    X = np.array(X)
    y = np.array(y)
    
    return torch.tensor(X), torch.tensor(y)
    

def retrainModel(model, lookbackData, labelData):
    print("\n\n------------Training model------------\n")
    X_train, X_val, y_train, y_val = train_test_split(lookbackData, labelData, test_size=0.2)

    # Define data loaders for training and validation
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    model, acc = trainStressModel(model, train_loader, val_loader)
    print("Accuracy: %.2f" % (acc))
    
    return model, acc
    





def trainStressModel(model, trainLoader, valLoader):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.BCELoss() # binary cross entropy

    n_epochs = 250
    # Print max 10 evenly spaced epoch
    printEpoch = int(n_epochs / numPrint)

    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None
 
    for epoch in range(n_epochs):
        model.train() # Set the model in training mode

        with tqdm.tqdm(total=len(trainLoader), unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for inputs, targets in trainLoader:
                # forward pass
                y_pred = model(inputs)
                loss = loss_fn(y_pred, targets)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                acc = (y_pred.round() == targets).float().mean()
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
        # evaluate accuracy at end of each epoch
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            acc = 0.0

            for inputs, targets in valLoader:  # Iterate through the validation data loader
                y_pred = model(inputs)
                val_loss += loss_fn(y_pred, targets).item()
                acc += (y_pred.round() == targets).float().mean().item()

            val_loss /= len(valLoader)
            acc /= len(valLoader)

        # Print progress for every 25th epoch
        
        if epoch % printEpoch == 0:
            print(f"Epoch {epoch} - Validation Loss: {val_loss}, Validation Accuracy: {acc}")
           
        # Check if the current model has a better validation accuracy
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())

    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return model, best_acc

def myLSTMbinaryClassification():
    trainIndex = ceil(0.8 * len(dataFiles))
    if path.exists(path_to_model) and SAVED:
        model = loadModel(path_to_model)
    else:
        model = StressModel(6,hiddenLayerSize,numLayers,lookback)
        cv_scores = []
        print(dataFiles)
        for file in dataFiles[:trainIndex]:
            lookback_data, labels = createData(file)
            model, acc = retrainModel(model, lookback_data, labels)
            cv_scores.append(acc)

        # evaluate the model
        model_acc = np.mean(cv_scores)
        model_std = np.std(cv_scores)
        print("\n\nFinal Model: %.2f%% (+/- %.2f%%)\n\n" % (model_acc*100, model_std*100))

        saveModelToFile(model,path_to_model)

    test_data, test_labels = createTestData(dataFiles[trainIndex:])
    test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=batch_size)
    model.eval()

    # Create empty lists to store true labels and predicted probabilities
    true_labels = []
    predicted_probs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            # Forward pass to get predicted probabilities
            y_pred = model(inputs)
            
            # Convert tensors to numpy arrays
            true_labels.extend(targets.cpu().numpy())
            predicted_probs.extend(y_pred.cpu().numpy())
    
    true_labels = np.array(true_labels)
    predicted_probs = np.array(predicted_probs)
    threshold = 0.203
    predicted_labels = (predicted_probs[:, -1] >= threshold).astype(int)


    #? accuracy stuff
    a.precisionRecallCalc(true_labels, predicted_labels)
    a.f1Score(true_labels, predicted_labels)
    rocFig = a.rocCurve(true_labels, predicted_probs, numLayers, lookback, batch_size, hiddenLayerSize)
    prFig = a.prCurve(true_labels, predicted_probs, numLayers, lookback, batch_size, hiddenLayerSize)
    plt.show()
    

myLSTMbinaryClassification()
