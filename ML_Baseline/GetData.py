from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import os

class Preprocessor_ML():
    def __init__(self,path):
        self.train_transaction = pd.read_csv(os.path.join(path,'train_transaction.csv'))
        self.train_identity = pd.read_csv(os.path.join(path,'train_identity.csv'))
        #这里只有下面两个字符串内的column可以与identity合并组成features表(弃用这种方案，先做完整映射)
        self.transactions_id_cols = 'card1,card2,card3,card4,card5,card6,ProductCD,addr1,addr2,P_emaildomain,R_emaildomain'
        self.cat_cols = 'M1,M2,M3,M4,M5,M6,M7,M8,M9'
        self.train_data_ratio = 0.8
        self.GetObjectColumn()
        print('initialization success')

    def GetColNull(self, precentage):
        cols_lotof_nulls = [c for c in self.train_transaction if
                                 (self.train_transaction[c].isnull().sum() / self.train_transaction.shape[0]) > precentage]
        return cols_lotof_nulls

    def GetObjectColumn(self):
        cat_col = []
        for col in self.train_transaction:
            if self.train_transaction[col].dtype == 'object':
                cat_col.append(col)
        self.catagorical_column = cat_col

    def SelectNumColumn(self):
        #如果不想用所有列可以在这里声明
        return [col for col in self.train_transaction.columns if col not in self.catagorical_column]


    def Work(self, mode='merge_raw', ToObjectColumn='Ignore', ThresholdDrop = 1):

        #mode actually decide what columns in data we need use
        if mode == 'merge_raw':
            Raw_data = pd.merge(self.train_transaction, self.train_identity, on='TransactionID', how='left')
        elif mode == 'raw':
            Raw_data = self.train_transaction
        elif mode == 'DropMajorNull':
            #Don't use any column which ThresholdDrop% is Null
            Raw_data = pd.merge(self.train_transaction, self.train_identity, on='TransactionID', how='left')
            cols_lotof_nulls = self.GetColNull(ThresholdDrop)
            print('columns need to drop:{}'.format(cols_lotof_nulls))
            Raw_data = Raw_data.drop(columns=cols_lotof_nulls)
        else:
            print('mode error, it can only been choosed in raw/merge-raw/RGCN')
            return
        Raw_data['TransactionAmt'] = Raw_data['TransactionAmt'].apply(np.log10)


        #To process categorical data
        if ToObjectColumn == 'Ignore':
            print('columns need to drop:{}'.format(self.catagorical_column))
            Raw_data = Raw_data.drop(columns=self.catagorical_column)
        elif ToObjectColumn == 'EncodeDirect':
            encoder = LabelEncoder()
            for col in self.catagorical_column:
                Raw_data[col] = encoder.fit_transform(Raw_data[col].astype(str).values)
        elif ToObjectColumn == 'Onehot':
            #get_dummies 只会对cata类别做 保留其他column
            Raw_data = pd.get_dummies(self.train_transaction,columns=self.catagorical_column).fillna(0)
        else:
            print('parameter error, it can only been choosed in Ignore/Encode')
            return


        # 在GNN中不应该对缺失值做填充
        Raw_data = Raw_data.fillna(-1)
        Label = Raw_data['isFraud'].to_numpy()
        Raw_data = Raw_data.drop(columns = 'isFraud').to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(Raw_data, Label,test_size=0.2, stratify=Label, random_state=1)

        '''
        # X_train, X_test, y_train, y_test = train_test_split(Raw_data, Label,test_size=0.2, random_state=1)
        
        n_train = int(Raw_data.shape[0] * self.train_data_ratio)
        X_train = Raw_data[:n_train]
        y_train = Label[:n_train]
        X_test = Raw_data[n_train:]
        y_test = Label[n_train:]
        '''

        return X_train, X_test, y_train, y_test