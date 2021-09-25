import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ML_Baseline.GetData import Preprocessor_ML

class Preprocessor(Preprocessor_ML):

    def __init__(self, path, stratify = False):

        super().__init__(path)
        self.output_dir = './PreprocessedData'
        self.n_train = n_train = int(self.train_transaction.shape[0] * self.train_data_ratio)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        #check
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        # extract out transactions for test/validation
        if stratify:
            X_train, X_test, y_train, y_test  = \
                train_test_split(self.train_transaction,self.train_transaction['isFraud'],stratify=self.train_transaction['isFraud'])
            test_transactions = X_test.TransactionID.values
        else:
            test_transactions = self.train_transaction.TransactionID.values[self.n_train:]

        with open(os.path.join(self.output_dir, 'test.csv'), 'w') as f:
            f.writelines(map(lambda x: str(x) + "\n", test_transactions))
        print('initialization over')

    def GNN_Pre(self, EasyMode = True):
        #self.Bagging()
        #R-GCN mode 这里并不需要identity表
        Raw_data = self.R_GCN_Preprocess(OneHot = False, Fillna = True)
        Raw_data[['TransactionID','isFraud']].to_csv(os.path.join(self.output_dir, 'tags.csv'),index=False, header=True)
        Raw_data.to_csv(os.path.join(self.output_dir, 'features.csv'), index=False, header=True)
        print('label produced')
        self.get_relations_and_edgelist(Raw_data, EasyMode)
        print('done')

    def Bagging(self):
        pass

    def R_GCN_Preprocess(self, OneHot = True, Fillna = True):
        # transaction表中不适合做编码的值
        num_cols = self.SelectNumColumn()
        Raw_data = self.train_transaction
        Raw_data['TransactionAmt'] = Raw_data['TransactionAmt'].apply(np.log10)
        #做直接映射还是OneHot编码
        #只选取cols字符串内的几个column做处理，不是全部需要，这里先分开再合并考虑了dummies的执行时间？
        if OneHot:
            OneHotFeature = pd.get_dummies(self.train_transaction[self.catagorical_column])
            Raw_data = pd.merge(self.train_transaction[num_cols], OneHotFeature,on='TransactionID', how='left')
        else:
            encoder = LabelEncoder()
            for col in self.catagorical_column:
                Raw_data[col] = encoder.fit_transform(Raw_data[col].astype(str).values)
        if Fillna:
            Raw_data = Raw_data.fillna(-1)

        return Raw_data

    def get_relations_and_edgelist(self, Raw_data, EasyMode):
        edge_types = Raw_data.columns.tolist() + list(self.train_identity.columns)
        if EasyMode:
            edge_types = [x for x in edge_types if x[0] != 'V'] #V都不要了
        edge_types.remove(['isFraud','TransactionDT'])
        # 合并identity表与transaction表的所有行数据
        full_identity_df = Raw_data.merge(self.train_identity, on='TransactionID', how='left')
        # extract edges
        edges = {}
        for etype in edge_types:
            edgelist = full_identity_df[['TransactionID', etype]].dropna()  # 将合并后的表内每一列和ID列组合并保存
            edgelist.to_csv(os.path.join(self.output_dir, 'relation_{}_edgelist.csv').format(etype), index=False, header=True)
            edges[etype] = edgelist
        print('edges produced')



