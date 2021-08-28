import pandas as pd
import numpy as np
import os
from itertools import combinations


class Preprocessor():

    def __init__(self,data_dir):
        self.train_data_ratio = 0.8
        self.id_cols = 'card1,card2,card3,card4,card5,card6,ProductCD,addr1,addr2,P_emaildomain,R_emaildomain'
        self.cat_cols = 'M1,M2,M3,M4,M5,M6,M7,M8,M9'
        self.output_dir = './PreprocessedData'
        self.identity_df = pd.read_csv(os.path.join(data_dir, 'train_identity.csv'))
        self.transaction_df = pd.read_csv(os.path.join(data_dir, 'train_transaction.csv'))
        print('load successfully')
        
        #check
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        # extract out transactions for test/validation
        n_train = int(self.transaction_df.shape[0] * self.train_data_ratio)
        test_transactions = self.transaction_df.TransactionID.values[n_train:]
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
        feature_cols = [col for col in self.transaction_df.columns if col not in non_feature_cols]
        # 将除了上面--id-cols的内容和isFraud，Transaction之外的表行数据作为feature；id即identity
        # one-hot编码
        features = pd.get_dummies(self.transaction_df[feature_cols], columns=self.cat_cols.split(",")).fillna(0)
        # 取常用对数放缩AMT值
        features['TransactionAmt'] = features['TransactionAmt'].apply(np.log10)
        self.transaction_df[['TransactionID', 'isFraud']].to_csv(os.path.join(self.output_dir, 'tags.csv'), index=False,header=True)
        features.to_csv(os.path.join(self.output_dir, 'features.csv'), index=False, header=True)
        print('label produced')

    def get_relations_and_edgelist(self):
        edge_types = self.id_cols.split(",") + list(self.identity_df.columns)
        id_cols = ['TransactionID'] + self.id_cols.split(",")
        # 合并identity表与transaction表的所有行数据
        full_identity_df = self.transaction_df[id_cols].merge(self.identity_df, on='TransactionID', how='left')

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


