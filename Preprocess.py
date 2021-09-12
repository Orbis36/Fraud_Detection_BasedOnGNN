import pandas as pd
import numpy as np
import os
from itertools import combinations
from ML_Baseline.GetData import Preprocessor_ML

class Preprocessor(Preprocessor_ML):

    def __init__(self, data_dir, path):

        super().__init__(path)
        self.output_dir = './PreprocessedData'
        
        #check
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        # extract out transactions for test/validation
        n_train = int(self.train_transaction.shape[0] * self.train_data_ratio)
        test_transactions = self.train_transaction.TransactionID.values[n_train:]
        with open(os.path.join(self.output_dir, 'test.csv'), 'w') as f:
            f.writelines(map(lambda x: str(x) + "\n", test_transactions))
        print('initialization over')

    def work(self):
        self.get_feature_and_labels()
        edges = self.get_relations_and_edgelist()
        #self.create_homogeneous_edgelist(edges)
        print('done')

    def get_feature_and_labels(self):
        non_feature_cols = ['isFraud', 'TransactionDT'] + self.id_cols.split(",")
        feature_cols = [col for col in self.train_transaction.columns if col not in non_feature_cols]
        # 将除了上面--id-cols的内容和isFraud，Transaction之外的表行数据作为feature；id即identity
        # one-hot编码
        features = pd.get_dummies(self.train_transaction[feature_cols], columns=self.cat_cols.split(",")).fillna(0)
        # 取常用对数放缩AMT值
        features['TransactionAmt'] = features['TransactionAmt'].apply(np.log10)
        self.train_transaction[['TransactionID', 'isFraud']].to_csv(os.path.join(self.output_dir, 'tags.csv'), index=False,header=True)
        features.to_csv(os.path.join(self.output_dir, 'features.csv'), index=False, header=True)
        print('label produced')

    def get_relations_and_edgelist(self):
        edge_types = self.id_cols.split(",") + list(self.train_identity.columns)
        id_cols = ['TransactionID'] + self.id_cols.split(",")
        # 合并identity表与transaction表的所有行数据
        full_identity_df = self.train_transaction[id_cols].merge(self.train_identity, on='TransactionID', how='left')

        # extract edges
        edges = {}
        for etype in edge_types:
            edgelist = full_identity_df[['TransactionID', etype]].dropna()  # 将合并后的表内每一列和ID列组合并保存
            edgelist.to_csv(os.path.join(self.output_dir, 'relation_{}_edgelist.csv').format(etype), index=False, header=True)
            edges[etype] = edgelist
        print('edges produced')
        return edges

    def create_homogeneous_edgelist(self,edges):
        homogeneous_edges = []
        total = len(edges)
        count = 0
        for etype, relations in edges.items():
            count+=1
            for edge_relation, frame in relations.groupby(etype):
                new_edges = [(a, b) for (a, b) in combinations(frame.TransactionID.values, 2)
                             if (a, b) not in homogeneous_edges and (b, a) not in homogeneous_edges]
                homogeneous_edges.extend(new_edges)
            print('Complete {} of {}'.format(count,total))
        with open(os.path.join(self.output_dir, 'homogeneous_edgelist.csv'), 'w') as f:
            f.writelines(map(lambda x: "{}, {}\n".format(x[0], x[1]), homogeneous_edges))


