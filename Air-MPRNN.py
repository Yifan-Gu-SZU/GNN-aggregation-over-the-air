import scipy.io as sio                     
import numpy as np                         
import matplotlib.pyplot as plt
from yaml import DirectiveToken           
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN
import wireless_networks_generator as wg
import helper_functions
import time

class init_parameters():
    def __init__(self):
        # wireless network settings
        self.n_links = train_K
        self.field_length = 500
        self.shortest_directLink_length = 2
        self.longest_directLink_length = 40
        self.shortest_crossLink_length = 1
        self.bandwidth = 5e6
        self.carrier_f = 2.4e9
        self.tx_height = 1.5
        self.rx_height = 1.5
        self.antenna_gain_decibel = 2.5
        self.tx_power_milli_decibel = 40
        self.tx_power = np.power(10, (self.tx_power_milli_decibel-30)/10)
        self.noise_density_milli_decibel = -169
        self.input_noise_power = np.power(10, ((self.noise_density_milli_decibel-30)/10)) * self.bandwidth
        self.output_noise_power = self.input_noise_power
        self.setting_str = "{}_links_{}X{}_{}_{}_length".format(self.n_links, self.field_length, self.field_length, self.shortest_directLink_length, self.longest_directLink_length)

def get_directLink_channel_losses(channel_losses):
    return np.diagonal(channel_losses, axis1=2, axis2=3)

def normalize_directlink(train_directlink_data,test_directlink_data):
    train_copy = np.copy(train_directlink_data)
    train_mean = np.sum(train_copy)/train_layouts/frame_num/train_K
    train_var = np.sqrt(np.sum(np.square(train_copy-train_mean))/train_layouts/frame_num/train_K)
    norm_train = (train_directlink_data - train_mean)/train_var
    norm_test = (test_directlink_data - train_mean)/train_var 
    return norm_train, norm_test

def normalize_agg_constants(train_data):
    mask = np.eye(train_K)
    train_copy = np.copy(train_data)
    diag_H = np.multiply(mask,train_copy)

    off_diag = train_copy - diag_H
    off_diag_mean = np.sum(off_diag)/train_layouts/frame_num/train_K/(train_K-1)
    off_diag_var = np.sqrt(np.sum(np.square(off_diag-off_diag_mean))/train_layouts/frame_num/train_K/(train_K-1))
    
    return off_diag_mean, off_diag_var

def proc_data(HH, norm_HH, K):
    n = HH.shape[0]
    data_list = []
    for i in range(n):
            data = build_graph_sequence(HH[i,:,:,:], norm_HH[i,:,:], K)
            data_list.append(data)
    return data_list

def build_graph_sequence(loss, norm_loss, K):
    n = loss.shape[0]
    x = np.zeros((n,K,graph_embedding_size+2))
    adj = np.zeros((n,2,K*(K-1)))
    edge_attr = np.zeros((n,K*(K-1),1))
    y = np.zeros((n,1,K,K))
    for i in range(n):
        if i == 0:
            x1 = np.expand_dims(norm_loss[i,:],axis=1)
        else:
            x1 = np.expand_dims(norm_loss[i-1,:],axis=1)
        x2 = np.expand_dims(norm_loss[i,:],axis=1)
        x3 = np.zeros((K,graph_embedding_size))
        #x1, x2 is the local CSI in the previous frame, and current frame respectively
        #note that in Air-MPRNN we use the local CSI in the previous frame to determine pilot transmit power
        x[i,:,:] = np.concatenate((x1,x2,x3),axis=1)

        #conisder fully connected graph
        loss2 = np.copy(loss[i,:,:])
        loss3 = np.copy(loss[i,:,:])
        mask = np.eye(K)
        diag_loss2 = np.multiply(mask,loss2)
        loss2 = loss2 - diag_loss2
        attr_ind = np.nonzero(loss2)
        edge_attr_tmp = loss3[attr_ind]
        edge_attr_tmp = np.expand_dims(edge_attr_tmp, axis = -1)
        edge_attr[i:,:] = edge_attr_tmp
    
        attr_ind = np.array(attr_ind)
        adj[i,0,:] = attr_ind[1,:]
        adj[i,1,:] = attr_ind[0,:]
        y[i,:,:,:] = np.expand_dims(loss[i,:,:],axis=0)
        
    x = torch.tensor(x, dtype=torch.float)
    x = torch.transpose(x,0,1)

    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    edge_attr = torch.transpose(edge_attr,0,1)

    edge_index = torch.tensor(adj, dtype=torch.long)
    edge_index = torch.transpose(edge_index,0,1)

    y = torch.tensor(y, dtype=torch.float)
    y = torch.transpose(y,0,1)
    data = Data(x=x, edge_index=edge_index.contiguous(),edge_attr = edge_attr, y = y)
    return data

class AirConv(MessagePassing):
    def __init__(self, mlp1, mlp2, **kwargs):
        super(AirConv, self).__init__(aggr='add', **kwargs)

        self.mlp1 = mlp1
        self.mlp2 = mlp2
        
    def update(self, air_agg, x):
        norm_air_agg = (air_agg - agg_mean)/agg_var
        #use CSI in the current frame for graph embedding update
        tmp = torch.cat([x[:,1:], norm_air_agg], dim=1)
        comb = self.mlp2(tmp)
        return comb
        
    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        #use local CSI and graph embedding in the previous frame to generate pilot transmit power
        tmp = torch.cat([x_j[:,:1],x_j[:,2:]], dim=1)
        agg = self.mlp1(tmp)*edge_attr
        return agg

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.mlp1,self.mlp2)

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU())
        for i in range(1, len(channels))
    ])

class AirMPRNN(torch.nn.Module):
    def __init__(self):
        super(AirMPRNN, self).__init__()
        self.hidden_size = graph_embedding_size 

        self.mlp1 = MLP([graph_embedding_size+1, 32, 32])
        self.mlp1 = Seq(*[self.mlp1,Seq(Lin(32, 1, bias = True), Sigmoid())])

        self.mlp2 = MLP([1+1+graph_embedding_size, 32 ,graph_embedding_size])

        self.conv = AirConv(self.mlp1,self.mlp2)   
        self.h2o = MLP([graph_embedding_size, 16])
        self.h2o = Seq(*[self.h2o,Seq(Lin(16, 1, bias = True), Sigmoid())])
        self.tanh = nn.Tanh()
    
    def forward(self, data):
        hidden = Variable(torch.zeros(links*batches, self.hidden_size))
        hidden = hidden.to(device)
        outputs = torch.zeros(frame_num,links*batches,1)
        outputs = outputs.to(device)
        x, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        for t in range(frame_num):
            x_t = x[:,t,:]
            hidden_t = hidden
            x_and_hidden = torch.cat([x_t[:,:2], hidden_t],dim=1)
            hidden = self.tanh(self.conv(x = x_and_hidden, 
                     edge_index = edge_index[:,t,:], edge_attr = edge_attr[:,t,:]))
            output = self.h2o(hidden)
            outputs[t,:,:] = output
        return outputs     

def sr_loss(data, out, K):
    loss = 0
    for i in range(frame_num):
        power = out[i,:]
        power = torch.reshape(power, (-1, K, 1)) 
        abs_H_2 = data.y[:,i,:,:]
        abs_H_2 = abs_H_2.permute(0,2,1)
        rx_power = torch.mul(abs_H_2, power)
        mask = torch.eye(K)
        mask = mask.to(device)
        valid_rx_power = torch.sum(torch.mul(rx_power, mask), 1)
        interference = torch.sum(torch.mul(rx_power, 1 - mask), 1) + var
        rate = torch.log2(1 + torch.div(valid_rx_power, interference))*overhead_ratio
        sr = torch.mean(torch.sum(rate, 1))
        loss += torch.neg(sr)
    loss = loss/frame_num
    return loss

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = sr_loss(data,out,links)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / train_layouts

def test():
    model.eval()
    total_loss = 0
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
            loss = sr_loss(data,out,links)
            total_loss += loss.item() * data.num_graphs
    return total_loss / test_layouts

train_K = 20
test_K = 20
train_layouts = 2000
test_layouts = 500
frame_num = 10
test_config = init_parameters()
train_config = init_parameters()
var = train_config.output_noise_power / train_config.tx_power
frame_length = 3000
graph_embedding_size = 8
overhead_csi = 1
overhead_mp = 5

print('Data generation')
#Data generation
#Train data
layouts, train_dists = wg.generate_layouts(train_config, train_layouts)
train_path_losses = wg.compute_path_losses(train_config, train_dists)
train_channel_losses = helper_functions.add_fast_fading_sequence(frame_num, train_path_losses)

#Test data 
layouts, test_dists = wg.generate_layouts(test_config, test_layouts)
test_path_losses = wg.compute_path_losses(test_config, test_dists)
test_channel_losses = helper_functions.add_fast_fading_sequence(frame_num,test_path_losses)

#Data normalization
#Normalization of directlink CSIs
train_directlink_losses = get_directLink_channel_losses(train_channel_losses)
test_directlink_losses = get_directLink_channel_losses(test_channel_losses)
norm_train_directlink_losses, norm_test_directlink_losses = normalize_directlink(np.sqrt(train_directlink_losses), np.sqrt(test_directlink_losses))
#Calculate normalization constants for the aggregated message from training samples
agg_mean, agg_var = normalize_agg_constants(train_channel_losses)

print('Graph data processing')
#Graph data processing
train_data_list = proc_data(train_channel_losses, norm_train_directlink_losses, train_K)
test_data_list = proc_data(test_channel_losses, norm_test_directlink_losses, test_K)

airmprnn_overhead_ratio = (frame_length-overhead_csi*train_K)/frame_length

#train of Air-MPRNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AirMPRNN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

links = train_K
batches = 50
#assert train_layouts%batches==0
#assert test_layouts%batches==0
train_loader = DataLoader(train_data_list, batch_size=50, shuffle=True,num_workers=0)
test_loader = DataLoader(test_data_list, batch_size=50, shuffle=False, num_workers=0)

#Total 2000 samples, each epoch with 2000/50 = 40 iterations
for epoch in range(1,51):
    links = train_K
    overhead_ratio = airmprnn_overhead_ratio
    loss1 = train()
    loss2 = test()
    print('Epoch {:03d}, Train Loss: {:.4f}, Test Loss: {:.4f}'.format(
           epoch, loss1, loss2))
    scheduler.step()

#Test for scalability and various system parameters, an example
gen_tests = [10, 15, 20, 25, 30, 35, 40]
overhead_csi = 2
overhead_mp = 20
frame_length = 3000
frame_num = 10
density = train_config.field_length**2/train_K
for test_K in gen_tests:
    print('<<<<<<<<<<<<<< Num of Links is {:03d} >>>>>>>>>>>>>:'.format(test_K))
    # generate test data
    test_config.n_links = test_K
    field_length = int(np.sqrt(density*test_K))
    test_config.field_length = field_length
    layouts, test_dists = wg.generate_layouts(test_config, test_layouts)
    test_path_losses = wg.compute_path_losses(test_config, test_dists)
    test_channel_losses = helper_functions.add_fast_fading_sequence(frame_num,test_path_losses)

    airmprnn_overhead_ratio = (frame_length-overhead_csi*test_K)/frame_length
    airmprnn_overhead_ratio = max(airmprnn_overhead_ratio,0)

    test_directlink_losses = get_directLink_channel_losses(test_channel_losses)
    norm_train_directlink_losses, norm_test_directlink_losses = normalize_directlink(np.sqrt(train_directlink_losses), np.sqrt(test_directlink_losses))
    test_data_list = proc_data(test_channel_losses, norm_test_directlink_losses, test_K)
    
    links = test_K
    batches = 50
    #assert test_layouts%test_batchsize==0
    test_loader = DataLoader(test_data_list, batch_size=batches, shuffle=False, num_workers=0)
    links = test_K
    
    overhead_ratio = airmprnn_overhead_ratio
    loss2 = test()
    print('Air-MPRNN average sum rate:', -loss2)