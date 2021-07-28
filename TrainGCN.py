import pandas as pd
import numpy as np
import os
import torch
import dgl
import time

from torch_geometric.data import Data
from Preprocess import Preprocessor
from utils import get_metrics
from GraphConstruct import Transfer2Graph
from sklearn.metrics import confusion_matrix
from Pytorch_model import HeteroRGCN
from SaveModel import save_model

def pre_construct_graph(target_node_type,nodes):

    GraphConstructor = Transfer2Graph()
    edgelists, id_to_node = {}, {}
    relations = list(filter(lambda x: x.startswith('relation'), os.listdir(GraphConstructor.datapath)))
    for relation in relations:
        edgelist, rev_edgelist, id_to_node, src, dst = GraphConstructor.parse_edgelist(relation, id_to_node,
                                                                                       header=True)
        if src == target_node_type:
            src = 'target'
        if dst == target_node_type:
            dst = 'target'

        if src == 'target' and dst == 'target':
            print("Will add self loop for target later......")
        else:
            if (src, src + '<>' + dst, dst) in edgelists:
                edgelists[(src, src + '<>' + dst, dst)] = edgelists[(src, src + '<>' + dst, dst)] + edgelist
                edgelists[(dst, dst + '<>' + src, src)] = edgelists[(dst, dst + '<>' + src, src)] + rev_edgelist
                print("Append edges for {} from edgelist: {}".format(src + '<>' + dst, relation))
            else:
                edgelists[(src, src + '<>' + dst, dst)] = edgelist
                edgelists[(dst, dst + '<>' + src, src)] = rev_edgelist
                print("Read edges for {} from edgelist: {}".format(src + '<>' + dst, relation))

    # get features for target nodes
    features, new_nodes = GraphConstructor.get_features(id_to_node[target_node_type], nodes)
    print("Read in features for target nodes")

    # add self relation
    edgelists[('target', 'self_relation', 'target')] = [(t, t) for t in id_to_node[target_node_type].values()]

    return features, id_to_node, edgelists

def get_labels(id_to_node, n_nodes, target_node_type, labels_files, masked_nodes_files, additional_mask_rate=0):
    """

    :param id_to_node: dictionary mapping node names(id) to dgl node idx
    :param n_nodes: number of user nodes in the graph
    :param target_node_type: column name for target node type
    :param labels_path: filepath containing labelled nodes
    :param masked_nodes_path: filepath containing list of nodes to be masked
    :param additional_mask_rate: additional_mask_rate: float for additional masking of nodes with labels during training
    :return: (list, list) train and test mask array
    """

    def read_masked_nodes(masked_nodes_files):
        """
        Returns a list of nodes extracted from the path passed in

        :param masked_nodes_path: filepath containing list of nodes to be masked i.e test users
        :return: list
        """
        masked_nodes = []
        for f in masked_nodes_files:
            with open(f, "r") as fh:
                masked_nodes += [line.strip() for line in fh]
        return masked_nodes

    def _get_mask(id_to_node, node_to_id, num_nodes, masked_nodes, additional_mask_rate):
        """
        Returns the train and test mask arrays

        :param id_to_node: dictionary mapping node names(id) to dgl node idx
        :param node_to_id: dictionary mapping dgl node idx to node names(id)
        :param num_nodes: number of user/account nodes in the graph
        :param masked_nodes: list of nodes to be masked during training, nodes without labels
        :param additional_mask_rate: float for additional masking of nodes with labels during training
        :return: (list, list) train and test mask array
        """
        train_mask = np.ones(num_nodes)
        test_mask = np.zeros(num_nodes)
        #只要在train中出现过的标记为masked
        for node_id in masked_nodes:
            train_mask[id_to_node[node_id]] = 0
            test_mask[id_to_node[node_id]] = 1
        #是否需要额外mask
        if additional_mask_rate and additional_mask_rate < 1:
            unmasked = np.array([idx for idx in range(num_nodes) if node_to_id[idx] not in masked_nodes])
            yet_unmasked = np.random.permutation(unmasked)[:int(additional_mask_rate * num_nodes)]
            train_mask[yet_unmasked] = 0
        return train_mask, test_mask

    node_to_id = {v: k for k, v in id_to_node.items()}

    #为了在多标签下也适用，这里使用concat，但在这里只有isFraud这一个标签
    #user_to_label 是TransactionID与label的对应关系。ex：  [2987000              0] [2987001              0]
    labels_df_from_files = pd.read_csv(labels_files)
    if not isinstance(labels_df_from_files, list):
        labels_df_from_files = [labels_df_from_files]
    user_to_label = pd.concat(labels_df_from_files, ignore_index=True).set_index(target_node_type)

    #为了针对semi-supervised情况这样写
    #取user_to_label中在node_to_id中的line，即有transactionID和label的记录（其实在这里是废话，这个数据集每个TransactionID都有label）
    labels = user_to_label.loc[map(int, pd.Series(node_to_id)[np.arange(n_nodes)].values)].values.flatten()

    #读取test集
    masked_nodes = read_masked_nodes([masked_nodes_files])

    train_mask, test_mask = _get_mask(id_to_node, node_to_id, n_nodes, masked_nodes,
                                      additional_mask_rate=additional_mask_rate)
    return labels, train_mask, test_mask

def construct_graph(features,edgelists,id_to_node):

    #以边建立异构图并标准化node feature
    g = dgl.heterograph(edgelists)
    node_feature = torch.tensor(features, dtype=torch.float)
    mean = torch.mean(node_feature,dim = 0)
    stdev = torch.sqrt(torch.sum((node_feature - mean) ** 2, dim=0) / node_feature.shape[0])
    features_normlized = (node_feature - mean) / stdev
    g.nodes['target'].data['features'] = features_normlized

    #获取标签与测试集合  mask？
    n_nodes = g.number_of_nodes('target')
    target_id_to_node = id_to_node[target_node_type]#一个TransactionID与节点ID的对应关系。ex：'2987000': 0, '2987001': 1 ...
    labels, _, test_mask = get_labels(target_id_to_node,n_nodes,target_node_type,labels_files = './PreprocessedData/tags.csv',
                                      masked_nodes_files = './PreprocessedData/test.csv')
    print("Got labels")
    labels = torch.from_numpy(labels).float()
    test_mask = torch.from_numpy(test_mask).float()

    n_nodes = torch.sum(torch.tensor([g.number_of_nodes(n_type) for n_type in g.ntypes]))
    n_edges = torch.sum(torch.tensor([g.number_of_edges(e_type) for e_type in g.etypes]))

    print("""----Data statistics------'
                    #Nodes: {}
                    #Edges: {}
                    #Features Shape: {}
                    #Labeled Test samples: {}""".format(n_nodes,
                                                        n_edges,
                                                        features_normlized.shape,
                                                        test_mask.sum()))
    return labels,test_mask,features_normlized,g

def get_model(ntype_dict, etypes, in_feats, n_classes, device):
    #hidden_size, out_size, n_layers, embedding_size
    model = HeteroRGCN(ntype_dict, etypes, in_feats, hidden_size = 16, out_size = n_classes, n_layers = 3 , embedding_size = in_feats)
    model = model.to(device)
    return model

def get_model_class_predictions(model, g, features, labels, device, threshold=None):
    unnormalized_preds = model(g, features.to(device))
    pred_proba = torch.softmax(unnormalized_preds, dim=-1)
    if not threshold:
        return unnormalized_preds.argmax(axis=1).detach().numpy(), pred_proba[:,1].detach().numpy()
    return np.where(pred_proba.detach().numpy() > threshold, 1, 0), pred_proba[:,1].detach().numpy()

def get_f1_score(y_true, y_pred):
    """
    Only works for binary case.
    Attention!
    tn, fp, fn, tp = cf_m[0,0],cf_m[0,1],cf_m[1,0],cf_m[1,1]

    :param y_true: A list of labels in 0 or 1: 1 * N
    :param y_pred: A list of labels in 0 or 1: 1 * N
    :return:
    """
    # print(y_true, y_pred)

    cf_m = confusion_matrix(y_true, y_pred)
    # print(cf_m)

    precision = cf_m[1,1] / (cf_m[1,1] + cf_m[0,1] + 10e-5)
    recall = cf_m[1,1] / (cf_m[1,1] + cf_m[1,0])
    f1 = 2 * (precision * recall) / (precision + recall + 10e-5)

    return precision, recall, f1

def evaluate(model, g, features, labels, device):
    "Compute the F1 value in a binary classification case"

    preds = model(g, features.to(device))
    preds = torch.argmax(preds, dim = 1).numpy()
    precision, recall, f1 = get_f1_score(labels, preds)

    return f1

def train_fg(model, optim, loss, features, labels, train_g, test_g, test_mask,
             device, n_epochs, thresh, compute_metrics=True):
    """
    A full graph verison of RGCN training
    """

    duration = []
    for epoch in range(n_epochs):
        tic = time.time()
        loss_val = 0.

        pred = model(train_g, features.to(device))

        l = loss(pred, labels)

        optim.zero_grad()
        l.backward()
        optim.step()

        loss_val += l

        duration.append(time.time() - tic)
        metric = evaluate(model, train_g, features, labels, device)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | f1 {:.4f} ".format(
                epoch, np.mean(duration), loss_val, metric))

    class_preds, pred_proba = get_model_class_predictions(model,
                                                          test_g,
                                                          features,
                                                          labels,
                                                          device,
                                                          threshold=thresh)

    if compute_metrics:
        acc, f1, p, r, roc, pr, ap, cm = get_metrics(class_preds, pred_proba, labels.numpy(), test_mask.numpy(), './')
        print("Metrics")
        print("""Confusion Matrix:
                                {}
                                f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, acc: {:.4f}, roc: {:.4f}, pr: {:.4f}, ap: {:.4f}
                             """.format(cm, f1, p, r, acc, roc, pr, ap))

    return model, class_preds, pred_proba

if __name__ == '__main__':


    target_node_type = 'TransactionID'
    nodes = ['features.csv']

    #简单处理原始CSV
    #Processor = Preprocessor('./IEEE-CIS_Fraud_Detection')
    #Processor.work()

    #真正建图前的准备
    features, id_to_node, edgelists = pre_construct_graph(target_node_type, nodes)
    labels,test_mask,features,g = construct_graph(features,edgelists,id_to_node)

    print("Initializing Model")
    device = torch.device('cuda:0')
    in_feats = features.shape[1]
    n_classes = 2
    ntype_dict = {n_type: g.number_of_nodes(n_type) for n_type in g.ntypes}

    #g.etypes为图g上的相连关系
    model = get_model(ntype_dict, g.etypes, in_feats, n_classes, device)
    print("Initialized Model")
    features = features.to(device)
    labels = labels.long().to(device)
    test_mask = test_mask.to(device)
    g = g.to(device)

    loss = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    print("Starting Model training")

    model, class_preds, pred_proba = train_fg(model, optim, loss, features, labels, g, g,
                                              test_mask, device, n_epochs = 100,
                                              thresh = 0, compute_metrics=True)

    print("Finished Model training")

    print("Saving model")
    save_model(g, model, './Model', id_to_node)
    print("Model and metadata saved")




