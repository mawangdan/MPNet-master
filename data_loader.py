import h5py
import torch
import os
import numpy as np
import os.path
import random
from torch.autograd import Variable
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Environment Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.demention = 3
        self.encoder = nn.Sequential(nn.Linear(1400 * self.demention, 786), nn.PReLU(), nn.Linear(786, 512), nn.PReLU(),
                                     nn.Linear(512, 256), nn.PReLU(), nn.Linear(256, 60))

    def forward(self, x):
        x = self.encoder(x)
        return x




# N=number of environments; NP=Number of Paths
def load_dataset(N=135, NP=45):
    Q = Encoder()
    Q.load_state_dict(torch.load('./models/cae_encoder.pkl'))
    if torch.cuda.is_available():
        Q.cuda()

    with h5py.File(f"./AE/point_cloud.hdf5", "r") as f:
        m_length = f["m_length"][:]
        N = int(m_length.__len__()/NP)
        obstacles_point_list = f["obstacles_point_list"][:]
        solutions = f["solutions"][:]
        end_effector_positions=f["end_effector_positions"][:]
        obs_rep = np.zeros((N, 60), dtype=np.float32)
        for i in range(0, N,45):
            # load obstacle point cloud
            # temp = np.fromfile('../../dataset/obs_cloud/obc' + str(i) + '.dat')
            # temp = temp.reshape(len(temp) / 2, 2)
            # obstacles = np.zeros((1, 2800), dtype=np.float32)
            # obstacles[0] = temp.flatten()
            inp = torch.from_numpy(obstacles_point_list[i])
            inp = Variable(inp).cuda()
            output = Q(inp)
            output = output.data.cpu()
            obs_rep[i] = output.numpy()

    ## calculating length of the longest trajectory
    max_length = 300
    path_lengths = np.zeros((N, NP), dtype=np.int64)
    for i in range(0, N):
        for j in range(0, NP):
            # fname = '../../dataset/e' + str(i) + '/path' + str(j) + '.dat'
            # if os.path.isfile(fname):
            #     path = np.fromfile(fname)
            #     path = path.reshape(len(path) / 2, 2)
            #     path_lengths[i][j] = len(path)
            path_lengths[i][j]=min(300,int(m_length[i*45+j][0]))

    paths = np.zeros((N, NP, max_length, 7), dtype=np.float32)  ## padded paths

    for i in range(0, N):
        for j in range(0, NP):
            # fname = '../../dataset/e' + str(i) + '/path' + str(j) + '.dat'
            # if os.path.isfile(fname):
            #     path = np.fromfile(fname)
            #     path = path.reshape(len(path) / 2, 2)
            paths[i,j,:max_length,:]=solutions[i*45+j,:max_length,:]

    dataset = []
    targets = []
    for i in range(0, N):
        for j in range(0, NP):
            if path_lengths[i][j] > 0:
                for m in range(0, path_lengths[i][j] - 1):
                    data = np.zeros(60+7+7, dtype=np.float32)
                    data[0:60] = obs_rep[i, :]
                    data[60:60+7]=paths[i,j,m,:]
                    data[60+7:60+7+7] = paths[i,j,path_lengths[i][j] - 1,:]

                    targets.append(paths[i][j][m + 1])
                    dataset.append(data)

    data = list(zip(dataset, targets))
    random.shuffle(data)
    dataset, targets = zip(*data)
    return np.asarray(dataset), np.asarray(targets)


# N=number of environments; NP=Number of Paths; s=starting environment no.; sp=starting_path_no
# Unseen_environments==> N=10, NP=2000,s=100, sp=0
# seen_environments==> N=100, NP=200,s=0, sp=4000
def load_test_dataset(N=100, NP=200, s=0, sp=4000):
    obc = np.zeros((N, 7, 2), dtype=np.float32)
    temp = np.fromfile('../../dataset/obs.dat')
    obs = temp.reshape(len(temp) / 2, 2)

    temp = np.fromfile('../../dataset/obs_perm2.dat', np.int32)
    perm = temp.reshape(77520, 7)

    ## loading obstacles
    for i in range(0, N):
        for j in range(0, 7):
            for k in range(0, 2):
                obc[i][j][k] = obs[perm[i + s][j]][k]

    Q = Encoder()
    Q.load_state_dict(torch.load('../models/cae_encoder.pkl'))
    if torch.cuda.is_available():
        Q.cuda()

    obs_rep = np.zeros((N, 28), dtype=np.float32)
    k = 0
    for i in range(s, s + N):
        temp = np.fromfile('../../dataset/obs_cloud/obc' + str(i) + '.dat')
        temp = temp.reshape(len(temp) / 2, 2)
        obstacles = np.zeros((1, 2800), dtype=np.float32)
        obstacles[0] = temp.flatten()
        inp = torch.from_numpy(obstacles)
        inp = Variable(inp).cuda()
        output = Q(inp)
        output = output.data.cpu()
        obs_rep[k] = output.numpy()
        k = k + 1
    ## calculating length of the longest trajectory
    max_length = 0
    path_lengths = np.zeros((N, NP), dtype=np.int8)
    for i in range(0, N):
        for j in range(0, NP):
            fname = '../../dataset/e' + str(i + s) + '/path' + str(j + sp) + '.dat'
            if os.path.isfile(fname):
                path = np.fromfile(fname)
                path = path.reshape(len(path) / 2, 2)
                path_lengths[i][j] = len(path)
                if len(path) > max_length:
                    max_length = len(path)

    paths = np.zeros((N, NP, max_length, 2), dtype=np.float32)  ## padded paths

    for i in range(0, N):
        for j in range(0, NP):
            fname = '../../dataset/e' + str(i + s) + '/path' + str(j + sp) + '.dat'
            if os.path.isfile(fname):
                path = np.fromfile(fname)
                path = path.reshape(len(path) / 2, 2)
                for k in range(0, len(path)):
                    paths[i][j][k] = path[k]

    return obc, obs_rep, paths, path_lengths
