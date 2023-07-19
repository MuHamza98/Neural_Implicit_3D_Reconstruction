import os 
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl




##############################################          Pytorch modules            ##############################################

class NDF(nn.Module):


    def __init__(self, hidden_dim=256):
        super(NDF, self).__init__()

        self.conv_in = nn.Conv3d(1, 16, 3, padding=1, padding_mode='replicate')  # out: 256 ->m.p. 128;  replicate, border
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1, padding_mode='replicate')  # out: 128
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1, padding_mode='replicate')  # out: 128 ->m.p. 64
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1, padding_mode='replicate')  # out: 64
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1, padding_mode='replicate')  # out: 64 -> mp 32
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1, padding_mode='replicate')  # out: 32
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 32 -> mp 16
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 16
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 16 -> mp 8
        self.conv_4 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 8
        self.conv_4_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 8

        feature_size = (1 +  16 + 32 + 64 + 128 + 128 + 128) * 7 + 3
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim * 2, 1)
        self.fc_1 = nn.Conv1d(hidden_dim *2, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim , hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)
        self.conv4_1_bn = nn.BatchNorm3d(128)


        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = torch.Tensor(displacments)#.cuda()

    def encoder(self,x):

        x = x.unsqueeze(1)
        f_0 = x

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        f_1 = net
        net = self.maxpool(net)  # out 128

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        f_2 = net
        net = self.maxpool(net)  # out 64

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        f_3 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        f_4 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        f_5 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_4(net))
        net = self.actvn(self.conv_4_1(net))
        net = self.conv4_1_bn(net)
        f_6 = net

        return f_0, f_1, f_2, f_3, f_4, f_5, f_6


    def decoder(self, p, f_0, f_1, f_2, f_3, f_4, f_5, f_6):



        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1) 
        # print('p aka Grid coordinates:', p.shape)
        # print('p aka Grid coordinates:', p)
        #if device !=None:
        # dev = p.get_device()

        displ = self.displacments#.to(dev)
        if p.get_device() != 'cpu':
            dev = p.get_device()
            displ = displ.to(dev)
        
        p=torch.cat([p+d for d in displ] , dim=2)#.to(dev)
        # print('p after displacements: ', p)      
        #else:
        #    p = torch.cat([p + d for d in self.displacments, dim=2)
        # feature extraction
        feature_0 = F.grid_sample(f_0, p, padding_mode='border')
        feature_1 = F.grid_sample(f_1, p, padding_mode='border')
        feature_2 = F.grid_sample(f_2, p, padding_mode='border')
        feature_3 = F.grid_sample(f_3, p, padding_mode='border')
        feature_4 = F.grid_sample(f_4, p, padding_mode='border')
        feature_5 = F.grid_sample(f_5, p, padding_mode='border')
        feature_6 = F.grid_sample(f_6, p, padding_mode='border')

        # print('f_0: ', f_0.shape)
        # print('feature_0: ',feature_0.shape)

        # here every channel corresponds to one feature.

        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6),
                             dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.actvn(self.fc_out(net))
        out = net.squeeze(1)

        return  out

    def forward(self, p, x):
        ##torch.cuda.set_device(device)
        # x is the occupany grid
        # p is grid coordinates
        out = self.decoder(p,  *self.encoder(x))
        return out



class IM_Net(nn.Module):

    def __init__(self, ):
        super(IM_Net, self).__init__()

        self.ef_dim = 32
        self.z_dim = 256
        self.gf_dim = 128

        ############## ENCODER #################
        self.conv1 = nn.Conv3d(1, self.ef_dim, kernel_size=4, stride=2, padding=1)
        self.in1 = nn.InstanceNorm3d(self.ef_dim)
        self.conv2 = nn.Conv3d(self.ef_dim, self.ef_dim*2, kernel_size=4, stride=2, padding=1)
        self.in2 = nn.InstanceNorm3d(self.ef_dim*2)
        self.conv3 = nn.Conv3d(self.ef_dim*2, self.ef_dim*4, kernel_size=4, stride=2, padding=1)
        self.in3 = nn.InstanceNorm3d(self.ef_dim*4)
        self.conv4 = nn.Conv3d(self.ef_dim*4, self.ef_dim*8, kernel_size=4, stride=2, padding=1)
        self.in4 = nn.InstanceNorm3d(self.ef_dim*8)
        self.conv5 = nn.Conv3d(self.ef_dim*8, self.z_dim, kernel_size=4, stride=1, padding=0)




        ############# Generator #################
        self.fc1 = nn.Linear(self.z_dim + 3, self.gf_dim*8)  # Assuming points is 3D
        self.fc2 = nn.Linear(self.gf_dim*8, self.gf_dim*8)
        self.fc3 = nn.Linear(self.gf_dim*8, self.gf_dim*8)
        self.fc4 = nn.Linear(self.gf_dim*8, self.gf_dim*4)
        self.fc5 = nn.Linear(self.gf_dim*4, self.gf_dim*2)
        self.fc6 = nn.Linear(self.gf_dim*2, self.gf_dim)
        self.fc7 = nn.Linear(self.gf_dim, 1)

    def encoder(self, x):
        '''
        Takes in the input voxel grid; 
        applies a number of 3d convolutions and then 
        '''
        x = x.unsqueeze(1)

        x = F.leaky_relu(self.in1(self.conv1(x)))

        x = F.leaky_relu(self.in2(self.conv2(x)))

        x = F.leaky_relu(self.in3(self.conv3(x)))

        x = F.leaky_relu(self.in4(self.conv4(x)))

        x = torch.sigmoid(self.conv5(x))

        x = x.view(x.size(0), -1)


        return x
    
    def generator(self, points, z):
        '''
        Takes in points and voxel encoding 
        concatenates them and then passes them through a number of layers linear layers with learky relu activations
        
        
        '''
        batch_size = z.size(0)
        num_points = points.size(1)
        z = z.repeat(1, num_points).view(batch_size, num_points, -1)
        # z = z.expand(batch_size, -1)
        pointz = torch.cat([points, z], -1)
        
        pointz = pointz.view(-1, 3 + 256)

        x = F.leaky_relu(self.fc1(pointz))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = F.leaky_relu(self.fc5(x))
        x = F.leaky_relu(self.fc6(x))
        x = self.fc7(x)
        x = torch.clamp(x, min=0, max=1)
        # may need to change shape

        return x.view(batch_size, num_points, 1)


    def forward(self, points, voxels):


        z =  self.encoder(voxels)
        out = self.generator(points, z).squeeze(-1)
        return out




