import numpy as np
import os


class Transfer2Graph():
    def __init__(self):
        self.nodetype = 'TransactionID'
        self.datapath = './PreprocessedData'

    def get_features(self, id_to_node, node_feature_files):
        """

        :param id_to_node: dictionary mapping node names(id) to dgl node idx
        :param node_features: path to file containing node features
        :return: (np.ndarray, list) node feature matrix in order and new nodes not yet in the graph
        """
        indices, features, new_nodes = [], [], []
        max_node = max(id_to_node.values())

        for node_file in node_feature_files:
            is_1st_line = True
            with open(os.path.join(self.datapath, node_file), "r") as fh:
                for line in fh:
                    # hard-coding to ignore the 1st line of header
                    if is_1st_line:
                        is_1st_line = False
                        continue

                    node_feats = line.strip().split(",")
                    node_id = node_feats[0]
                    feats = np.array(list(map(float, node_feats[1:])))
                    features.append(feats)
                    if node_id not in id_to_node:
                        max_node += 1
                        id_to_node[node_id] = max_node
                        new_nodes.append(max_node)

                    indices.append(id_to_node[node_id])

        features = np.array(features).astype('float32')
        features = features[np.argsort(indices), :]
        return features, new_nodes

    def parse_edgelist(self, edges_path, id_to_node, header=False, source_type='user', sink_type='user'):
        """
            Parse an edgelist path file and return the edges as a list of tuple
            :param edges: path to comma separated file containing bipartite edges with header for edgetype
            :param id_to_node: dictionary containing mapping for node names(id) to dgl node indices, init with empty


            :param header: boolean whether or not the file has a header row
            :param source_type: type of the source node in the edge. defaults to 'user' if no header
            :param sink_type: type of the sink node in the edge. defaults to 'user' if no header.
            :return: (list, dict) a list containing edges of a single relationship type as tuples and updated id_to_node dict.
        """
        edge_list = []
        rev_edge_list = []
        source_pointer, sink_pointer = 0, 0
        with open(os.path.join(self.datapath,edges_path), "r") as fh:#某个维度信息和TransactionID的csv
            for i, line in enumerate(fh):
                source, sink = line.strip().split(",")#去处头尾空格以逗号隔开，事实上获得了两个column的名字
                if i == 0:#初始化
                    if header:
                        source_type, sink_type = source, sink
                    if source_type in id_to_node:#如果这个ID在id_to_node字典中已经存在
                        source_pointer = max(id_to_node[source_type].values()) + 1 #source指示器加1
                    if sink_type in id_to_node:
                        sink_pointer = max(id_to_node[sink_type].values()) + 1
                    continue

                source_node, id_to_node, source_pointer = self._get_node_idx(id_to_node, source_type, source, source_pointer)
                if source_type == sink_type:
                    sink_node, id_to_node, source_pointer = self._get_node_idx(id_to_node, sink_type, sink, source_pointer)
                else:
                    sink_node, id_to_node, sink_pointer = self._get_node_idx(id_to_node, sink_type, sink, sink_pointer)

                edge_list.append((source_node, sink_node))
                rev_edge_list.append((sink_node, source_node))

        return edge_list, rev_edge_list, id_to_node, source_type, sink_type

    def _get_node_idx(self,id_to_node, node_type, node_id, ptr):
        if node_type in id_to_node:
            if node_id in id_to_node[node_type]:
                node_idx = id_to_node[node_type][node_id]
            else:
                id_to_node[node_type][node_id] = ptr
                node_idx = ptr
                ptr += 1
        else:
            id_to_node[node_type] = {}
            id_to_node[node_type][node_id] = ptr
            node_idx = ptr
            ptr += 1

        return node_idx, id_to_node, ptr
