import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
#from matplotlib import pyplot as plt
import pdb


class GTN(nn.Module):
    
    def __init__(self, num_edge, num_channels, w_in, w_out, num_class,num_layers,norm):
        super(GTN, self).__init__()
        self.num_edge = num_edge
        self.num_channels = num_channels
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.num_layers = num_layers
        self.important_grad = None
        self.update_x = None
        self.hnorm = None
        self.hx = None
        #self.is_norm = norm
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(num_edge, num_channels, first=True))
            else:
                layers.append(GTLayer(num_edge, num_channels, first=False))
        self.layers = nn.ModuleList(layers)
        # this weight is used for the single gcn layer
        self.weight = nn.Parameter(torch.Tensor(w_in, w_out))
        #torch.manual_seed(0)
        #self.weight = nn.Parameter(torch.Tensor(w_in, self.num_class))
        #self.bias = nn.Parameter(torch.Tensor(w_out))
        self.loss = nn.CrossEntropyLoss()
        torch.manual_seed(0)
        self.linear1 = nn.Linear(self.w_out*self.num_channels, self.w_out, bias=False)
        #self.linear1 = nn.Linear(self.w_out*self.num_channels, self.w_out, bias=False)
        #self.linear1 = nn.Linear(self.w_out*self.num_channels, self.w_out)
        torch.manual_seed(0)
        self.linear2 = nn.Linear(self.w_out, self.num_class, bias=False)
        #self.linear2 = nn.Linear(self.w_in, self.num_class, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        torch.manual_seed(0)
        nn.init.xavier_uniform_(self.weight)
        #torch.set_printoptions(threshold=1000000000)
        #print("main", self.weight)
        #print("lin1", self.linear1.weight)
        #print("line2", self.linear2.weight)
        #nn.init.zeros_(self.bias)

    def printgrad(self):
        print("H grad", self.important_grad.grad)
        #print("H norm grad", self.hnorm.grad)
        #print("linear 2 grad", self.linear2.weight.grad)
        #first = True
        #for layer in self.layers:
        #    print("Score grad", layer.conv1.weight.grad)
        #    if first:
        #        print("Score grad", layer.conv2.weight.grad)
        #    first = False

    def gcn_conv(self,X,H):
        #H.requires_grad_(True)
        #H.retain_grad()
        #self.important_grad = H
        #self.weight.retain_grad()
        #print("Conv weights", self.weight)
        #print("Features", X)
        # linear xform before aggregation
        #print("features ", X)
        X = torch.mm(X, self.weight)
        #print("after xwith features ", X)
        #self.update_x = X
        #self.update_x.requires_grad_(True)
        #self.update_x.retain_grad()
        #print("Result", X)
        # normalization of scores by in-degree (functoin will transpose matrix
        #H = self.norm(H, add=True)
        #print("Metapaths", H)
        Hnorm = self.norm(H, add=False)
        #self.hnorm = Hnorm
        #self.hnorm.requires_grad_(True)
        #self.hnorm.retain_grad()
        #print("Metapaths after norm", Hnorm)
        # aggregate over in-edges
        #return torch.mm(H.t(),X)
        self.hx = torch.mm(Hnorm,X)
        #self.hx.requires_grad_(True)
        #self.hx.retain_grad()
        return self.hx

    def normalization(self, H):
        for i in range(self.num_channels):
            if i==0:
                H_ = self.norm(H[i,:,:]).unsqueeze(0)
            else:
                H_ = torch.cat((H_,self.norm(H[i,:,:]).unsqueeze(0)), dim=0)
        return H_

    def norm(self, H, add=False):
        # transpose to get in matrix
        #H = H.t()

        #if add == False:
        #    print(H)
        #    H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor))
        #    print(H)
        #else:
        #    H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor)) + torch.eye(H.shape[0]).type(torch.FloatTensor)

        # get inverse in-degrees
        # put it into a diagonal matrix
        deg = torch.sum(H, dim=1)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        deg_inv = deg_inv*torch.eye(H.shape[0]).type(torch.FloatTensor)
        # multiply each row (in metaedges) by corresponding degree inverse
        H = torch.mm(deg_inv,H)
        #H = H.t()
        return H

    def forward(self, A, X, target_x, target):
        #print("main", self.weight)
        #print("lin1", self.linear1.weight)
        #print("line2", self.linear2.weight)

        A = A.unsqueeze(0).permute(0,3,1,2) 
        Ws = []
        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A)
            else:
                # TODO ??????
                # this is the degree normalization; getting rid of it
                # for apples to apples since we do nothing like this for
                # metapath graph
                #H = self.normalization(H)
                H, W = self.layers[i](A, H)
            Ws.append(W)

        #print("H is", H)
        
        #H,W1 = self.layer1(A)
        #H = self.normalization(H)
        #H,W2 = self.layer2(A, H)
        #H = self.normalization(H)
        #H,W3 = self.layer3(A, H)
        for i in range(self.num_channels):
            if i==0:
                X_ = F.relu(self.gcn_conv(X,H[i]))
                #X_ = self.gcn_conv(X,H[i])
                #print("Final result", X_)
            else:
                # channels: concat similar to SAGE
                X_tmp = F.relu(self.gcn_conv(X,H[i]))
                X_ = torch.cat((X_,X_tmp), dim=1)
        #print("after layer 1", X_)
        #y = X_[target_x]
        # feature size is original * num channels: maps itself back to
        # original feature length
        X_ = self.linear1(X_)
        X_ = F.relu(X_)
        #print(target_x)
        # map to # of labels for cross entropy classification
        y = self.linear2(X_[target_x])
        #y = self.linear2(X_[target_x]).requires_grad_(True)
        #print(target)
        loss = self.loss(y, target)
        #print(loss)
        return loss, y, Ws

class GTLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels)
            self.conv2 = GTConv(in_channels, out_channels)
        else:
            self.conv1 = GTConv(in_channels, out_channels)
    
    def forward(self, A, H_=None):
        if self.first == True:
            a = self.conv1(A)
            b = self.conv2(A)
            H = torch.bmm(a,b)
            #print(H)
            #print(self.conf1.weight)
            #print(self.conf2.weight)
            W = [self.conv1.weight, self.conv2.weight]
            #W = [(F.softmax(self.conv1.weight, dim=1)).detach(),(F.softmax(self.conv2.weight, dim=1)).detach()]
        else:
            a = self.conv1(A)
            H = torch.bmm(H_,a)
            #print(self.conf1.weight)
            W = [self.conv1.weight]
            #W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
        return H,W

class GTConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,1,1))
        #self.weight = torch.Tensor(out_channels,in_channels,1,1)
        self.bias = None
        #self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        # inits all scores to 0.1
        self.reset_parameters()
    def reset_parameters(self):
        n = self.in_channels
        nn.init.constant_(self.weight, 0.1)
        #if self.bias is not None:
        #    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        #    bound = 1 / math.sqrt(fan_in)
        #    nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        #print(F.softmax(self.weight, dim=1))
        #print(A * F.softmax(self.weight, dim=1))
        # weight each edge differently
        A = torch.sum(A*F.softmax(self.weight, dim=1), dim=1)
        return A
