import pandas as pd
import numpy as np
import os
from itertools import combinations
from ML_Baseline.GetData import Preprocessor_ML

class Preprocessor(Preprocessor_ML):

    def __init__(self, path):

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

    def GNN_Pre(self, mode, ToOC, Separate, Bagging):
        if Bagging:
            self.train_transaction = self.BaggingData()
        #R-GCN mode 并不需要identity表
        Raw_data = self.Work(mode = mode, ToObjectColumn = ToOC, SeparateTrainData = Separate)
        Raw_data[['TransactionID','isFraud']].to_csv(os.path.join(self.output_dir, 'tags.csv'),index=False, header=True)
        Raw_data.to_csv(os.path.join(self.output_dir, 'features.csv'), index=False, header=True)
        print('label produced')
        self.get_relations_and_edgelist()
        print('done')

    def BaggingData(self):



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



