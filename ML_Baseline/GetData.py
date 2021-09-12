from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import os

class Preprocessor_ML():
    def __init__(self,path):
        self.train_transaction = pd.read_csv(os.path.join(path,'train_transaction.csv'))
        self.train_identity = pd.read_csv(os.path.join(path,'train_identity.csv'))
        self.id_cols = 'card1,card2,card3,card4,card5,card6,ProductCD,addr1,addr2,P_emaildomain,R_emaildomain'
        self.cat_cols = 'M1,M2,M3,M4,M5,M6,M7,M8,M9'
        self.train_data_ratio = 0.8
        print('initialization success')

    def Work(self, mode='merge_raw', ToObjectColumn='Ignore'):

        if mode == 'merge_raw':
            Raw_data = pd.merge(self.train_transaction, self.train_identity, on='TransactionID', how='left')

        elif mode == 'raw':
            Raw_data = self.train_transaction

        elif mode == 'R-GCN':
            non_feature_cols = ['isFraud', 'TransactionDT'] + self.id_cols.split(",")#找到那些不适合做onehot的值
            feature_cols = [col for col in self.train_transaction.columns if col not in non_feature_cols]#适合的，为上取补
            # 将除了上面id-cols的内容和isFraud，Transaction之外的表行数据作为feature(为什么id_column不算很疑惑)；以one-hot编码
            OneHotFeature = pd.get_dummies(self.train_transaction[feature_cols], columns=self.cat_cols.split(","))
            #再将原来列放回
            Raw_data = pd.merge(self.train_transaction[non_feature_cols+['TransactionID']], OneHotFeature, on='TransactionID', how='left')
            Raw_data['TransactionAmt'] = Raw_data['TransactionAmt'].apply(np.log10)

        elif mode == 'DropMajorNull':#去掉百分之90以上均是空值的列
            Raw_data = pd.merge(self.train_transaction, self.train_identity, on='TransactionID', how='left')
            cols_lotof_nulls  = [c for c in self.train_transaction if (self.train_transaction[c].isnull().sum() / self.train_transaction.shape[0]) > 0.90]
            print('columns need to drop:{}'.format(cols_lotof_nulls))
            Raw_data = Raw_data.drop(columns=cols_lotof_nulls)

        else:
            print('mode error, it can only been choosed in raw/merge-raw/RGCN')
            return

        cat_col = []
        for col in Raw_data:
            if Raw_data[col].dtype == 'object':
                cat_col.append(col)

        if ToObjectColumn == 'Ignore':
            print('columns need to drop:{}'.format(cat_col))
            Raw_data = Raw_data.drop(columns=cat_col)

        elif ToObjectColumn == 'Encode':
            encoder = LabelEncoder()
            for col in cat_col:
                Raw_data[col] = encoder.fit_transform(Raw_data[col].astype(str).values)
        else:
            print('parameter error, it can only been choosed in Ignore/Encode')
            return


        Raw_data = Raw_data.fillna(-1)
        Label = Raw_data['isFraud'].to_numpy()
        Raw_data = Raw_data.drop(columns = 'isFraud').to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(Raw_data, Label,test_size=0.2, stratify=Label, random_state=1)
        #X_train, X_test, y_train, y_test = train_test_split(Raw_data, Label,test_size=0.2, random_state=1)
        '''
        n_train = int(Raw_data.shape[0] * self.train_data_ratio)

        X_train = Raw_data[:n_train]
        y_train = Label[:n_train]

        X_test = Raw_data[n_train:]
        y_test = Label[n_train:]
        '''




        return X_train, X_test, y_train, y_test