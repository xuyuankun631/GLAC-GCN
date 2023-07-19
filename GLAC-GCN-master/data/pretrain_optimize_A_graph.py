import numpy as np
#import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import Linear
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from evaluation import eva
import numpy as np
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, z, enc_h3


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


topk = 3

adj = np.loadtxt('acm_graph.txt', dtype=int)

def findPair(target,array):
    for i in range(len(array)):
        if target[0] == array[i][0]:
            if target[1] == array[i][1]:
                return True
        if i+1 == len(array):
            return False
        if target[0] == array[i][0] and target[0] < array[i+1][0]:
            return False


def pretrain_ae(model, dataset, y, clusterNum):
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=1e-3)
    for epoch in range(40):
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.cuda()

            x_bar, _, _ = model(x)
            loss = F.mse_loss(x_bar, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            if epoch != 39:
                x = torch.Tensor(dataset.x).cuda().float()
                x_bar, z, enc_h3 = model(x)
                loss = F.mse_loss(x_bar, x)
                print('{} loss: {}'.format(epoch, loss))
                kmeans = KMeans(n_clusters=clusterNum, n_init=20).fit(z.data.cpu().numpy()) # 簇的个数
                eva(y, kmeans.labels_, epoch)
            if epoch == 39:
                print("Do kmeans and K-NN to optimize the neighbor graph!")
                x = torch.Tensor(dataset.x).cuda().float()
                x_bar, z, enc_h3 = model(x)
                loss = F.mse_loss(x_bar, x)
                print('{} loss: {}'.format(epoch, loss))
                kmeans1 = KMeans(n_clusters=clusterNum, n_init=20).fit(z.data.cpu().numpy())  # 簇的个数
                eva(y, kmeans.labels_, epoch)
                kmeans2 = KMeans(n_clusters=clusterNum, n_init=20).fit(z.data.cpu().numpy())  # 簇的个数
                eva(y, kmeans2.labels_, epoch)
                kmeans3 = KMeans(n_clusters=clusterNum, n_init=20).fit(z.data.cpu().numpy())  # 簇的个数
                eva(y, kmeans3.labels_, epoch)

                construct_graphMulti_A_15(acm, kmeans1.labels_, kmeans2.labels_, kmeans3.labels_, 'cos')

    torch.save(model.state_dict(), 'acm_{}.pkl'.format(epoch + 1))



model = AE(
        n_enc_1=500,
        n_enc_2=500,
        n_enc_3=2000,
        n_dec_1=2000,
        n_dec_2=500,
        n_dec_3=500,
        n_input=1870,
        n_z=3,).cuda()


x = np.loadtxt('acm.txt', dtype=float)
y = np.loadtxt('acm_label.txt', dtype=int)

def construct_graphMulti_A_15(features, label1, label2, label3, method='heat'):
    fname = 'acm15_graph.txt'
    num = len(label1)
    dist = None

    if method == 'heat':
        dist = -0.5 * pair(features) ** 2
        dist = np.exp(dist)
    elif method == 'cos':
        features[features > 0] = 1
        dist = np.dot(features, features.T)
    elif method == 'ncos':
        features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)

    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(15+1))[-(15+1):]
        inds.append(ind)

        f = open(fname, 'w')
    counter = 0
    A = np.zeros_like(dist)
    for i, v in enumerate(inds):
        mutual_knn = False
        for vv in v:
            if vv == i:
                pass
            else:
                if label1[vv] != label1[i]:
                    counter += 1
                if (label1[vv] == label1[i] and label2[i] == label2[vv] and label3[vv]== label3[vv]) or (findPair([i,vv],adj)):
                    f.write('{} {}\n'.format(i, vv))
    f.close()
    print('m15+: {}'.format(counter / (num * 15)))


acm = np.loadtxt('acm.txt', dtype=float)
acm_label = np.loadtxt('acm_label.txt', dtype=int)

dataset = LoadDataset(x)
pretrain_ae(model, dataset, y, 3)


