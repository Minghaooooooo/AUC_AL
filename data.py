from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from config import get_args
from util import *

args = get_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


def read_dataset(data_name):
    # read npy files from folder dataMats
    x_path = './data/' + data_name + '_X.npy'
    y_path = './data/' + data_name + '_Y.npy'
    x = np.load(x_path)
    y = np.load(y_path)
    return x, y


class MyDataset(Dataset):
    def __init__(self, x, y, transform=None, target_transform=None):
        self.x = x.type(torch.float)
        self.y = y.type(torch.float)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        instance = self.x[idx, :]
        label = self.y[idx]
        if self.transform:
            instance = self.transform(instance)
        if self.target_transform:
            label = self.target_transform(label)
        return instance, label

    def label_length(self):
        return len(self.y[0])

    def get_xtest(self):
        return self.x

    def get_ytest(self):
        return self.y

def split(data,label=None,train_rate=0.1,candidate_rate=0.6,
          test_rate=0.3,seed=25,even=False):
    # initial split for AL
    # both train and test include all labels (if possible) in this version
    index=np.arange(data.shape[0])
    np.random.seed(seed)
    np.random.shuffle(index)
    if(even is False):#do not require each label have a positive instance in train.
        train_index=index[0:int(len(index)*train_rate)]
        candidate_index=index[int(len(index)*train_rate):int(len(index)*(train_rate+candidate_rate))]
        test_index=index[int(len(index)*(train_rate+candidate_rate)):]
        return list(train_index),list(candidate_index),list(test_index)
    elif(even is True):#scan index to determine the train index first.
        label_stack=label[0,:]#label_stack is 0/1 L length vector, indicating whether a label has already appreared in the training
        train_index=[index[0]]
        orilen = len(index)
        i=0
        while (len(train_index)<int(len(index)*train_rate)):
            i=i+1
            current_label=np.sum(label_stack)
            if(current_label<label.shape[1]):  #then need a new training data with new positive label
                updated_label=np.sum(np.logical_or(label[index[i],:],label_stack))
                if(updated_label>current_label):#if introducing the next data introduce a new label(s),add it.
                    train_index.append(index[i])
                    label_stack=np.logical_or(label[index[i],:],label_stack)#update label stack
                    #print(np.sum(label_stack))
                else:#skip this data point
                    pass
            else:
                train_index.append(index[i])        #delete the train index from index
        index=[x for x in index if x not in train_index]
        test_index = [index[0]]
        label_stack=label[test_index[0],:]
        i=0
        while (len(test_index)<int(orilen*test_rate)):
            i=i+1
            current_label=np.sum(label_stack)
            if(current_label<label.shape[1]):  #then need a new training data with new positive label
                updated_label=np.sum(np.logical_or(label[index[i],:],label_stack))
                if(updated_label>current_label):#if introducing the next data introduce a new label(s),add it.
                    test_index.append(index[i])
                    label_stack=np.logical_or(label[index[i],:],label_stack)#update label stack
                    #print(np.sum(label_stack))
                else:#skip this data point
                    pass
            else:
                test_index.append(index[i])

        #delete the candidate index from index
        candidate_index=[x for x in index if x not in test_index]
        candidate_index = candidate_index[:int(orilen*candidate_rate)]
        return list(train_index),list(candidate_index),list(test_index)


def get_data():
    x, y = read_dataset(args.data_name)
    # print(x.shape)
    train = args.train
    pool = args.pool
    test = args.test
    deterministic(args.seed)
    train_index, candidate_index, test_index = split(x, label=y, train_rate=train,
                                                     candidate_rate=pool, test_rate=test,
                                                     seed=args.seed, even=True)
    print('training datasize', len(train_index))
    print('testing datasize', len(test_index))
    print('pooling datasize', len(candidate_index))

    xtrain = torch.tensor(x[train_index])
    xtest = torch.tensor(x[test_index])
    xcan = torch.tensor(x[candidate_index])
    ytrain = torch.tensor(y[train_index])
    ytest = torch.tensor(y[test_index])
    ycan = torch.tensor(y[candidate_index])

    train_data = MyDataset(xtrain.to(device),  ytrain.to(device))
    test_data = MyDataset(xtest.to(device),  ytest.to(device))

    return train_data, test_data


#
# # Assuming you have your data x and y ready
# xx = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# yy = torch.tensor([0, 1, 0])
#
# # Instantiate the myDataset class
# dataset = MyDataset(xx, yy)
#
# # Optionally, apply transformations if needed
# # transform = ...
# # target_transform = ...
# # dataset = myDataset(x, y, transform=transform, target_transform=target_transform)
#
# # Use a DataLoader to iterate over batches of your dataset
# # Assuming batch size is 2
# batch_size = 2
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
# # Iterate over batches
# for batch in dataloader:
#     instances, labels = batch
#     print("Instances:", instances)
#     print("Labels:", labels)


print(' dataset is called: ', args.data_name)
features, labels = read_dataset(args.data_name)
print(' features shape:', features.shape)
print('training features shape:', labels.shape)

