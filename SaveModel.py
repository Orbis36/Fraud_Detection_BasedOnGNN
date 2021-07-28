import os
import pandas as pd
import torch as th
import pickle

def save_model(g, model, model_dir, id_to_node):

    # Save Pytorch model's parameters to model.pth
    th.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))

    # Save graph's structure information to metadata.pkl for inference codes to initialize RGCN model.
    etype_list = g.canonical_etypes
    ntype_cnt = {ntype: g.number_of_nodes(ntype) for ntype in g.ntypes}
    with open(os.path.join(model_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump({'etypes': etype_list,
                     'ntype_cnt': ntype_cnt}, f)

    # Save original IDs to Node_ids, and trained embedding for non-target node type
    # Covert id_to_node into pandas dataframes
    for ntype, mapping in id_to_node.items():

        # ignore target node
        if ntype == 'target':
            continue

        # retrieve old and node id list
        old_id_list, node_id_list = [], []
        for old_id, node_id in mapping.items():
            old_id_list.append(old_id)
            node_id_list.append(node_id)

        # retrieve embeddings of a node type
        node_feats = model.embed[ntype].detach().numpy()

        # get the number of nodes and the dimension of features
        num_nodes = node_feats.shape[0]
        num_feats = node_feats.shape[1]

        # create id dataframe
        node_ids_df = pd.DataFrame({'~label': [ntype] * num_nodes})
        node_ids_df['~id_tmp'] = old_id_list
        node_ids_df['~id'] = node_ids_df['~label'] + '-' + node_ids_df['~id_tmp']
        node_ids_df['node_id'] = node_id_list

        # create feature dataframe columns
        cols = {'val' + str(i + 1) + ':Double': node_feats[:, i] for i in range(num_feats)}
        node_feats_df = pd.DataFrame(cols)

        # merge id with feature, where feature_df use index
        node_id_feats_df = node_ids_df.merge(node_feats_df, left_on='node_id', right_on=node_feats_df.index)
        # drop the id_tmp and node_id columns to follow the Grelim format requirements
        node_id_feats_df = node_id_feats_df.drop(['~id_tmp', 'node_id'], axis=1)

        # dump the embeddings to files
        node_id_feats_df.to_csv(os.path.join(model_dir, ntype + '.csv'),
                                index=False, header=True, encoding='utf-8')