import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv, GATConv, SAGEConv
import dgl.function as fn


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict({
            name: nn.Linear(in_size, out_size) for name in etypes
        })

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        # 这里feat_dict就是feature
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h, 遍历所有关系返回元组形式：('entity', 'relation', 'entity')
            if srctype in feat_dict:
                Wh = self.weight[etype](feat_dict[srctype])
                # Save it in graph for message passing
                G.nodes[srctype].data['Wh_%s' % etype] = Wh
                # Specify per-relation message passing functions: (message_func, reduce_func).
                funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        # Trigger message passing of multiple types.
        G.multi_update_all(funcs, 'sum')
        # return the updated node feature dictionary
        return {ntype: G.nodes[ntype].data['h'] for ntype in G.ntypes if 'h' in G.nodes[ntype].data}


class HeteroRGCN(nn.Module):
    def __init__(self, ntype_dict, etypes, in_size, hidden_size, out_size, n_layers, embedding_size):
        '''
        :param ntype_dict: {节点类型：种类数目}
        :param etypes:每个line的数据个数，这里会有问题，如果没有填充0这就是空的
        :param in_size:
        :param hidden_size:
        :param out_size:
        :param n_layers:
        :param embedding_size:
        '''
        super(HeteroRGCN, self).__init__()
        # 这个矩阵完成Embedding操作，定义完成后使用xavier初始化 这相当于是原文中对应不同关系的W_r
        # 只要不是traget节点，均建立这样一个就建立一个矩阵
        embed_dict = {ntype: nn.Parameter(torch.Tensor(num_nodes_cata, in_size))
                      for ntype, num_nodes_cata in ntype_dict.items() if ntype != 'target'}
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.embed = nn.ParameterDict(embed_dict)

        # 一层weight将embedding size -> hidden_size,后面接3层的隐藏层，后再接一个weight将其映射到out-size
        # create layers
        self.layers = nn.ModuleList()
        self.layers.append(HeteroRGCNLayer(embedding_size, hidden_size, etypes))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(HeteroRGCNLayer(hidden_size, hidden_size, etypes))
        # output layer,映射回2分类
        self.layers.append(nn.Linear(hidden_size, out_size))

    def forward(self, g, features):
        # 这里先做一个转换，features到h_dict的target元素，其余元素为
        h_dict = {ntype: emb for ntype, emb in self.embed.items()}
        # feat_para = torch.tensor(features)
        h_dict['target'] = features

        # pass through all layers

        for i, layer in enumerate(self.layers[:-1]):
            if i != 0:
                h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
            h_dict = layer(g, h_dict)

        # get user logits
        # 注意形状，是括号里的矩阵乘layer[-1]
        return self.layers[-1](h_dict['target'])
