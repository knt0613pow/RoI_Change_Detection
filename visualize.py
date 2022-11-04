import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import pickle
import os
import numpy as np
import pygmtools as pygm
import json
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
pygm.BACKEND = 'pytorch'

def Path_img2npy(imgpath):
    return imgpath + '.npy'

def Path_img2json(imgpath):
    return imgpath + '.json'

def read_json(fname):
    with open(fname, 'r') as json_file:
        json_data = json.load(json_file)
    return json_data

def read_npy(fname):
    return np.load(fname)

def collate_fn(batch):
    return tuple(zip(*batch))



class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform):
        self.transform = transform
        self.root = data_dir
        relation_path = os.path.join(self.root, 'relation.pkl')
        with open(relation_path,"rb") as fr:
            data = pickle.load(fr)
        self.relation = data

    def __len__(self):
        return len(self.relation)
    def __getitem__(self, idx):
        pair = self.relation[idx]
        feature1_path = Path_img2npy(os.path.join(self.root, pair[0]))
        feature2_path = Path_img2npy(os.path.join(self.root, pair[1]))
        json1_path = Path_img2json(os.path.join(self.root, pair[0]))
        json2_path = Path_img2json(os.path.join(self.root, pair[1]))
        

        return read_npy(feature1_path), read_npy(feature2_path), read_json(json1_path), read_json(json2_path) , self.relation[idx]    



batch_size = 1
trsfm = transforms.Compose([transforms.ToTensor(), ])
data_path = 'data/Daesan2'

device = torch.device("cuda")

trainset = FeatureDataset(data_path, trsfm)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, collate_fn = collate_fn)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                          shuffle=False, num_workers=2)

class GMNet(torch.nn.Module):
    def __init__(self):
        super(GMNet, self).__init__()
        self.gm_net = pygm.utils.get_network(pygm.ipca_gm, pretrain=False) # fetch the network object

    def forward(self, f1, f2, A1, A2):

        node1 = f1
        node2 = f2
        X = pygm.ipca_gm(node1, node2, A1, A2, network=self.gm_net) # the network object is reused
        return X

net = GMNet()


net.load_state_dict(torch.load('saved_ipca_35.pth'))
# criterion = pygm.utils.permutation_loss
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# for epoch in tqdm(range(0,40)):  # loop over the dataset multiple times

#     running_loss = 0.0
num_pair = 0
num_correct_pair =0
for i, data in enumerate(trainloader, 0):
    f1, f2, j1, j2 , relation= data
    f1, f2, j1, j2 = torch.Tensor(f1[0]), torch.Tensor(f2[0]), j1[0], j2[0] 
    f1 = torch.nn.functional.normalize(f1 , dim=1, eps=1e-12, out=None)
    f2 = torch.nn.functional.normalize(f2 , dim=1, eps=1e-12, out=None)
    obj1, obj2 = j1['shapes'], j2['shapes']
    # f1, f2, j1, j2 = torch.Tensor(f1), torch.Tensor(f2), j1, j2 
    # obj1, obj2 = [obj_j1['shapes'] for obj_j1 in j1], [obj_j2['shapes'] for obj_j2 in j2]
    label1 = [lb['label'] for lb in obj1]
    label2 = [lb['label'] for lb in obj2]
    pt1 =  [lb['points'] for lb in obj1]
    pt2 =  [lb['points'] for lb in obj2]
    if len(label1) == 0 or len(label2) == 0 : continue
    X_gt = torch.zeros((len(label1), len(label2)))
    for idx1, lb1 in enumerate(label1):
        for idx2, lb2 in enumerate(label2):
            if lb1 == lb2 : X_gt[idx1, idx2] = 1
    A1 = torch.ones((len(label1), len(label1)))
    A2 = torch.ones((len(label2), len(label2)))

    num_pair += torch.sum(X_gt)

    outputs = net(f1, f2, A1, A2)
    breakpoint()

    # loss = criterion(outputs, X_gt)
    # loss.backward()
    # optimizer.step()
    # breakpoint()
    num_correct_pair += (torch.round(torch.matmul(f1, f2.T)) *(pygm.hungarian(outputs) * X_gt)).sum()
    # print(f'[{epoch + 1}] loss: {running_loss / 2000:.3f}')
    # running_loss = 0.0
    
    # PATH = f'./saved_{epoch}.pth'
print(num_pair)
print(num_correct_pair)
print(num_correct_pair/num_pair)
#     if epoch % 5 ==0 : torch.save(net.state_dict(), PATH)

# print('Finished Training')

########################################################################
# Let's quickly save our trained model:


