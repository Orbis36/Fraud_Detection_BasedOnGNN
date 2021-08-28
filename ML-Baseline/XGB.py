import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
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
        else:
            print('mode error, it can only been choosed in raw/merge-raw/RGCN')
            return

        if ToObjectColumn == 'Ignore':
            cat_col = []
            for col in Raw_data:
                if Raw_data[col].dtype == 'object':
                    cat_col.append(col)
            print('columns need to drop:{}'.format(cat_col))
            Raw_data = Raw_data.drop(columns=cat_col)

        Raw_data = Raw_data.fillna(0)
        Label = Raw_data['isFraud'].to_numpy()
        Raw_data = Raw_data.drop(columns = 'isFraud').to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(Raw_data, Label,test_size=0.2, stratify=Label, random_state=1)
        return X_train, X_test, y_train, y_test


if __name__ == '__main__':

    Preprocessor = Preprocessor_ML('../IEEE-CIS_Fraud_Detection')
    test_mode = ['merge_raw','raw','R-GCN']
    color_need = ['limegreen','salmon','cyan']
    plt.figure(figsize=(6, 6))
    plt.title('Prediction of XGB')


    for mode,color in zip(test_mode,color_need):
        X_train, X_test, y_train, y_test = Preprocessor.Work(mode = mode)
        scale_pos_weight = np.sqrt((len(y_train) - sum(y_train))/sum(y_train))

        model = XGBClassifier(max_depth=5,subsample=0.8,eta=0.2,gamma=4,
                              min_child_weight=6,objective='binary:logistic',
                              scale_pos_weight=scale_pos_weight)

        model.fit(X_train,y_train)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        fpr,tpr,threshold = metrics.roc_curve(y_test,y_pred_prob)
        roc_auc = metrics.auc(fpr,tpr)
        plt.plot(fpr, tpr, color, label='{} Mode Val AUC = {}'.format(mode,roc_auc))
        plt.legend(loc='lower right')

    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.show()

