import csv, sys, math, json, time, torch, random, os.path, warnings, argparse, importlib, torchvision, pdb
import numpy as np
import pandas as pd
from torch import nn
from typing import List
from torch.utils.data import *
import matplotlib.pyplot as plt
from torch.utils import data as td
from torch.autograd import Variable
import datasets as d
import torch.nn.functional as F
import utilities as utils


device = torch.device("cpu")
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CPU_DEVICE = torch.device("cpu")
GET_VALUE = lambda x: x.to(CPU_DEVICE).data.numpy().reshape(-1)[0]
print(f"\n *** \n Currently running on {device}\n *** \n")


class My_dataLoader:
    def __init__(self, batch_size : int, df_data: pd.DataFrame, df_label:pd.DataFrame):
        '''
            Creates train and test loaders from local files, to be easily used by torch.nn
            
            :batch_size: int for size of training batches
            :data_path: path to csv where data is. 2d file
            :label_path: csv containing labels. line by line equivalent to data_path file
        '''
        self.batch_size = batch_size
        total_length = df_data.shape[0]
        self.train_size = int(total_length * 0.8)
        self.valid_size = int(total_length * 0.9)
        df_data = df_data.sample(frac=1, random_state=2).reset_index(drop=True)
        self.trdata = torch.tensor(df_data.values[:self.train_size,:])
        self.trlabels = torch.tensor(df_label.values[:self.train_size]) # also has too be 2d

        self.vadata = torch.tensor(df_data.values[self.train_size:self.valid_size,:])
        self.valabels = torch.tensor(df_label.values[self.train_size:self.valid_size]) # also has too be 2d  

        self.tedata = torch.tensor(df_data.values[self.valid_size:,:]) # where data is 2d [D_train_size x features]
        self.telabels = torch.tensor(df_label.values[self.valid_size:]) # also has too be 2d  

        self.train_dataset = torch.utils.data.TensorDataset(self.trdata, self.trlabels)
        self.valid_dataset = torch.utils.data.TensorDataset(self.vadata, self.valabels)
        self.test_dataset = torch.utils.data.TensorDataset(self.tedata, self.telabels)
        

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            num_workers=1,
            pin_memory=False,
            shuffle=True
        )
        self.valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=args.batch_size,
            num_workers=1,
            pin_memory=False,
            shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=args.test_batch_size,
            num_workers=1,
            pin_memory=False,
            shuffle=True
        )

class BlackBox(nn.Module):
    def __init__(self, in_dim: int, hidden=254,  dropout=0.5):
        super(BlackBox, self).__init__()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(dropout)
        self.l1 = nn.Linear(in_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)

        self.l2 = nn.Linear(hidden,hidden//2)
        self.bn2 = nn.BatchNorm1d(hidden//2)

        self.l3 = nn.Linear(hidden//2, hidden//4)
        self.bn3 = nn.BatchNorm1d(hidden//4)

        self.l4 = nn.Linear(hidden//4, hidden//8)
        self.bn4 = nn.BatchNorm1d(hidden//8)
        
        self.out = nn.Linear(hidden//8, 2)
        # self.sm = nn.Softmax()
        

    def forward(self, xin):
        x = F.leaky_relu(self.l1(xin))
        x = self.drop1(self.bn1(x)) 

        x = F.leaky_relu(self.l2(x))
        x = self.drop1(self.bn2(x))    
        
        x = F.leaky_relu(self.l3(x))
        x = self.drop2(self.bn3(x))    
        
        x = F.leaky_relu(self.l4(x))
        x = self.drop2(self.bn4(x))      


        x = self.out(x)
        return x
    
def train(model: torch.nn.Module, train_loader:torch.utils.data.DataLoader, optimizer:torch.optim, loss_fn) -> int:
    model.train()
    train_loss = []
    for batch_idx, (inputs, target) in enumerate(train_loader.train_loader):
        inputs, target = inputs.to(device), target.to(device)        
        optimizer.zero_grad()
        output = model(inputs.float())
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.detach())
    
    
    return sum(train_loss)/len(train_loss)


def valid(model: torch.nn.Module, test_loss_fn, dataloader: My_dataLoader, epoch:int) -> int:
    '''
        Does the test loop and if last epoch, decodes data, generates new data, returns and saves both 
        under ./experiments/<experiment_name>/

        :PARAMS
        model: torch model to use
        test_loss_fn: Loss function from torch.nn 

        :RETURN
        int: average loss on this epoch
        pd.DataFrame: generated data in a dataframe with encoded data
    '''
    model.eval()
    
    test_loss = 0
    correct = 0
    test_size = 0
    with torch.no_grad():
        for inputs, target in dataloader.valid_loader:
            inputs, target = inputs.to(device), target.to(device)
            output = model(inputs.float())
            test_size += len(inputs)

            test_loss += test_loss_fn(output, target).item() 
            pred = output.max(1, keepdim=True)[1] 
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    # At this point `test_size` == bs * len(valid_loader) == rows passed in this loop ^
    #  Hence test_loss value will be an average over each data point
    test_loss /= test_size
    accuracy = correct / test_size
    
    print('\nTest set epoch {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, test_loss, correct, test_size,
        100. * accuracy))
    
    return test_loss, accuracy


def train_model(model, dataloader, loss_fn, test_loss_fn, optimizer, num_epochs):
        '''
            Takes care of all model training, testing and generation of data
            Generates data every epoch but only saves on file if loss is lowest so far.
        '''
        start = time.time()
        lowest_test_loss = 9999999999999999
        test_accuracy = []
        test_losses = []
        ave_train_loss = []
        lowest_loss_ep = -1

        for epoch in range(num_epochs):
            # Iterate on train set with SGD (adam)
            batch_ave_tr_loss = train(model, dataloader, optimizer, loss_fn)
            ave_train_loss.append(batch_ave_tr_loss.cpu().numpy().item())

        #     # Check test set metrics (+ generate data if last epoch )
            loss, accuracy = valid(model, test_loss_fn, dataloader, epoch)
            
            #saving model and epoch if lowest loss value so far
            if loss < lowest_test_loss:
                if os.path.isfile(path_to_exp+f"lowest-test-loss-model_ep{lowest_loss_ep}.pth"):
                    os.remove(path_to_exp+f"lowest-test-loss-model_ep{lowest_loss_ep}.pth")
                lowest_test_loss = loss
                fm = open(path_to_exp+f"lowest-test-loss-model_ep{epoch}.pth", "wb")
                torch.save(model.state_dict(), fm)
                fm.close()
                lowest_loss_ep = epoch
            test_accuracy.append(accuracy*100)
            test_losses.append(loss)
        with open(path_to_exp+f"metadata.txt", 'w+') as f:
            f.write(f"loss: {test_losses} \n \n")
            f.write(f"accuracies: {test_accuracy} \n \n")
            
        fm = open(path_to_exp+f"final-model_{num_epochs}ep.pth", "wb")
        torch.save(model.state_dict(), fm)
        end = time.time()

        print(f"Total time to train: {(end-start)/60:.2f} minutes.")
        return test_losses, test_accuracy

def split_data_labels(full_data:pd.DataFrame, column_label:str)-> (pd.DataFrame, pd.DataFrame):
    '''
        Takes in a dataframe and the column name to split out. Meant to be used with My_dataloader which will create a train, valid, test split with those. 
        Shuffles full_data on rows before splitting

        :PARAMS
        full_data: Dataframe containing all data
        column_label: Column in full_data that we want as target

        :RETURN
        2 dataframes, one with data, one with target
    '''
    # Shuffle rows
    data = full_data.sample(frac=1)
    try:
        labels_df = data[column_label]
    except:
        raise ValueError("column name not contained in Dataframe")

    data.drop(column_label, 1, inplace=True)

    return data, labels_df

def gen_loss_graphs(losses, accuracies):
    x_axis = np.arange(1,args.epochs+1)
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(x_axis, losses)
    plt.xlabel("Epochs")
    plt.ylabel("L1 Loss")
    plt.title("Test Loss")

    plt.subplot(1,2,2)
    plt.plot(x_axis, accuracies)
    plt.xlabel("Epochs")
    plt.ylabel("L1 Loss")
    plt.title("Test Accuracies")
    plt.savefig(path_to_exp+f"{args.learning_rate}lr_{args.batch_size}.png")

def main():
    # Import data at ../GeneralDatasets/Csv/Adult_NotNA.csv
    data = pd.read_csv(f'../GeneralDatasets/Csv/{args.input_dataset}.csv')

    if 'disp_impact'  not in args.input_dataset:
        # Encode Data with dummy variables
        data.to_csv(path_to_exp+'junk_raw_data.csv', index=False)
        df = pd.read_csv(path_to_exp+'junk_raw_data.csv')
        data_encoder = d.Encoder(df)

        data, labels = split_data_labels(data_encoder.df, args.target)

        if os.path.exists(path_to_exp+'parameters.prm'):
            data_encoder.load_parameters(path_to_exp, 'parameters.prm')
            data_encoder.transform()
        else:
            data_encoder.fit_transform()
            data_encoder.save_parameters(path_to_exp)

        encoded_data = data_encoder.df

    #Work with disp_impact input, already encoded
    else:
        data, labels = split_data_labels(data, args.target)
        encoded_data = data

    print("Data Encoded")
    # print(f"Headers: {encoded_data.columns}")
    dataloader = My_dataLoader(batch_size=args.batch_size, df_data=encoded_data, df_label=labels)
    print(f"Dataloader built")

    loss_fn = torch.nn.CrossEntropyLoss()
    test_loss_fn = nn.CrossEntropyLoss(reduction='sum')   
    learning_rate = args.learning_rate
    num_epochs = args.epochs
    weight_decay = args.weight_decay

    model = BlackBox(encoded_data.shape[1]).to(device)
    print(f"Model Built")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
 
    losses, accuracies = train_model(model, dataloader, loss_fn, test_loss_fn, optimizer, num_epochs)
    os.remove(path_to_exp+'junk_raw_data.csv')

    gen_loss_graphs(losses, accuracies)

parser = argparse.ArgumentParser()
args, path_to_exp = utils.parse_arguments(parser)
main()
