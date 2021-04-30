import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model import GTN
import pdb
import pickle
import argparse
import scipy.sparse as sparse
from utils import f1_score
from utils import accuracy
import time

def StaticFeatures(num_features):
    feats = []
    for i in range(num_features):
        feat = [i % 10] * 3
        feats.append(feat)
    return np.array(feats)

def ReadNodeFeaturesText(filename):
    feature_file = open(filename, "r")
    
    node_count = 0
    all_features = []
    for cur_feature in feature_file.readlines():
        array_of_features = [float(num) for num in cur_feature.strip().split(' ')]
        all_features.append(array_of_features)
        node_count += 1
    return all_features, node_count

def ReadNodeLabelsText(basename):
    train_file = open(basename + "_labels_train.txt", "r")
    val_file = open(basename + "_labels_val.txt", "r")
    test_file = open(basename + "_labels_test.txt", "r")
    # temp storage
    wset = []
    train_set = np.array([])
    val_set = np.array([])
    test_set = np.array([])

    for label in train_file.readlines():
        node_and_label = [int(num) for num in label.strip().split(' ')]
        wset.append(node_and_label)
    train_set = np.array(wset)
    wset.clear()

    for label in val_file.readlines():
        node_and_label = [int(num) for num in label.strip().split(' ')]
        wset.append(node_and_label)
    val_set = np.array(wset)
    wset.clear()

    for label in test_file.readlines():
        node_and_label = [int(num) for num in label.strip().split(' ')]
        wset.append(node_and_label)
    test_set = np.array(wset)
    wset.clear()

    return train_set, val_set, test_set

def ReadEdgesText(filename, num_nodes, num_edge_types):
    edgelist = open(filename, "r")
    
    edge_count = 0
    rows = []
    columns = []
    for i in range(num_edge_types):
        rows.append([])
        columns.append([])

    # read all edges
    for cur_edge in edgelist.readlines():
        read_edge = [int(num) for num in cur_edge.strip().split(' ')]
        edge_type = read_edge.pop()
        rows[edge_type].append(read_edge[0])
        columns[edge_type].append(read_edge[1])

    return_csrs = []
    for edge_type in range(num_edge_types):
        # create data matrix of ones
        assert len(rows[edge_type]) == len(columns[edge_type])
        data = [1] * len(rows[edge_type])
        return_csrs.append(sparse.csr_matrix((data, (rows[edge_type], columns[edge_type])), shape=(num_nodes,num_nodes)))
    return return_csrs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        help='Dataset')
    parser.add_argument('--epoch', type=int, default=40,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=64,
                        help='Node dimension')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='number of channels')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='l2 reg')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layer')
    parser.add_argument('--norm', type=str, default='true',
                        help='normalization')
    parser.add_argument('--adaptive_lr', type=str, default='false',
                        help='adaptive learning rate')

    args = parser.parse_args()
    print(args)
    epochs = args.epoch
    node_dim = args.node_dim
    num_channels = args.num_channels
    lr = args.lr
    weight_decay = args.weight_decay
    num_layers = args.num_layers
    norm = args.norm
    adaptive_lr = args.adaptive_lr

    with open('data/'+args.dataset+'/node_features.pkl','rb') as f:
        node_features = pickle.load(f)
    with open('data/'+args.dataset+'/edges.pkl','rb') as f:
        edges = pickle.load(f)
    with open('data/'+args.dataset+'/labels.pkl','rb') as f:
        labels = pickle.load(f)
################################################################################
    #som = np.fromfile("tester-feats.bin", dtype=np.single)

    #node_features, num_nodes = np.array(ReadNodeFeaturesText("tester-feats.txt"))
    #node_features = StaticFeatures(295912)
    #print(node_features)
    #num_nodes = 295912
    #edges = ReadEdgesText("chembio/chembio_edge_labels.txt", num_nodes, 4)
    #train_set, val_set, test_set = ReadNodeLabelsText("chembio/chembio_node")
    #labels = [train_set, val_set, test_set]
    #exit()

    #print(labels)
    #tester = [[0,0,0],[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6]]
    #node_features = np.array(tester)

    #row1 = np.array([0, 1, 2, 3, 4, 5])
    #col1 = np.array([1, 0, 1, 2, 3, 4])
    #data1 = np.array([1, 1, 1, 1, 1, 1])
    #even_edges = sparse.csr_matrix((data1, (row1, col1)), shape=(7,7))
    #row2 = np.array([1, 2, 3, 4, 5, 6])
    #col2 = np.array([2, 3, 4, 5, 6, 5])
    #data2 = np.array([1, 1, 1, 1, 1, 1])
    #odd_edges = sparse.csr_matrix((data2, (row2, col2)), shape=(7,7))
    #edges = [even_edges, odd_edges]
    num_nodes = edges[0].shape[0]

    #train_set = np.array([np.array([0, 0]),np.array([1, 1]),np.array([2, 2]),np.array([3, 3]),np.array([4, 4])])
    #val_set = np.array([np.array([5, 5])])
    #test_set = np.array([np.array([6, 6])])
    #labels = [train_set, val_set, test_set]

################################################################################

    for i,edge in enumerate(edges):
        if i ==0:
            A = torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)
        else:
            A = torch.cat([A,torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
    A = torch.cat([A,torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
    
    node_features = torch.from_numpy(node_features).type(torch.FloatTensor)
    train_node = torch.from_numpy(np.array(labels[0])[:,0]).type(torch.LongTensor)
    train_target = torch.from_numpy(np.array(labels[0])[:,1]).type(torch.LongTensor)
    valid_node = torch.from_numpy(np.array(labels[1])[:,0]).type(torch.LongTensor)
    valid_target = torch.from_numpy(np.array(labels[1])[:,1]).type(torch.LongTensor)
    test_node = torch.from_numpy(np.array(labels[2])[:,0]).type(torch.LongTensor)
    test_target = torch.from_numpy(np.array(labels[2])[:,1]).type(torch.LongTensor)

    #exit()
    
    #num_classes = 7
    num_classes = torch.max(train_target).item()+1
    #num_classes1 = torch.max(train_target).item()+1
    #num_classes2 = torch.max(valid_target).item()+1
    #num_classes3 = torch.max(test_target).item()+1
    final_f1 = 0
    for l in range(1):
        model = GTN(num_edge=A.shape[-1],
                            num_channels=num_channels,
                            w_in = node_features.shape[1],
                            w_out = node_dim,
                            num_class=num_classes,
                            num_layers=num_layers,
                            norm=norm)
        if adaptive_lr == 'false':
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
        else:
            optimizer = torch.optim.Adam([{'params':model.weight},
                                        {'params':model.linear1.parameters()},
                                        {'params':model.linear2.parameters()},
                                        {"params":model.layers.parameters(), "lr":0.5}
                                        ], lr=0.01, weight_decay=0.0)
        loss = nn.CrossEntropyLoss()
        # Train & Valid & Test
        best_val_loss = 10000
        best_test_loss = 10000
        best_train_loss = 10000
        best_train_f1 = 0
        best_val_f1 = 0
        best_test_f1 = 0
        
        total_training_time_so_far = 0.0

        for i in range(epochs):
            #for param_group in optimizer.param_groups:
            #    if param_group['lr'] > 0.005:
            #        param_group['lr'] = param_group['lr'] * 0.9
            train_epoch_start = time.time()
            model.zero_grad()
            model.train()
            loss,y_train,Ws = model(A, node_features, train_node, train_target)
            #print(y_train)
            #print(Ws)
            #train_f1 = torch.mean(f1_score(torch.argmax(y_train.detach(),dim=1), train_target, num_classes=num_classes)).cpu().numpy()
            #print('Train - Loss: {}, Macro_F1: {}'.format(loss.detach().cpu().numpy(), train_f1))
            print('Epoch {} Train - Loss: {}, Accuracy: {}'.format(i, loss.detach().cpu().numpy(), accuracy(torch.argmax(y_train.detach(),dim=1), train_target)), flush=True)
            #print(loss)
            y_train.retain_grad()
            loss.backward()
            #print("Y GRAD", y_train.grad)
            #model.printgrad()
            optimizer.step()
            train_epoch_stop = time.time()
            elapsed_train_epoch_time = train_epoch_stop - train_epoch_start
            total_training_time_so_far += elapsed_train_epoch_time

            model.eval()
            if (i % 5) == 0 or (i + 1) == epochs:
            #if (i +1) == epochs:
              # Valid
              with torch.no_grad():
                  #val_loss, y_valid,_ = model.forward(A, node_features, valid_node, valid_target)
                  #val_f1 = torch.mean(f1_score(torch.argmax(y_valid,dim=1), valid_target, num_classes=num_classes)).cpu().numpy()
                  #print('Valid - Loss: {}, Macro_F1: {}'.format(val_loss.detach().cpu().numpy(), val_f1))
                  test_loss, y_test,W = model.forward(A, node_features, test_node, test_target)
                  test_f1 = torch.mean(f1_score(torch.argmax(y_test,dim=1), test_target, num_classes=num_classes)).cpu().numpy()
                  #print('Test - Loss: {}, Macro_F1: {}\n'.format(test_loss.detach().cpu().numpy(), test_f1))
                  print('Test EndEpoch {} - Loss: {}, Accuracy: {}, Elapsed Time: {}'.format(i, test_loss.detach().cpu().numpy(), accuracy(torch.argmax(y_test.detach(),dim=1), test_target), total_training_time_so_far), flush=True)
            #if val_f1 > best_val_f1:
            #    best_val_loss = val_loss.detach().cpu().numpy()
            #    best_test_loss = test_loss.detach().cpu().numpy()
            #    best_train_loss = loss.detach().cpu().numpy()
            #    best_train_f1 = train_f1
            #    best_val_f1 = val_f1
            #    best_test_f1 = test_f1 
        #print('---------------Best Results--------------------')
        #print('Train - Loss: {}, Macro_F1: {}'.format(best_train_loss, best_train_f1))
        #print('Valid - Loss: {}, Macro_F1: {}'.format(best_val_loss, best_val_f1))
        #print('Test - Loss: {}, Macro_F1: {}'.format(best_test_loss, best_test_f1))
        #final_f1 += best_test_f1
