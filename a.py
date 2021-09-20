import pandas as pd
import numpy as np

transaction_data = pd.read_csv('./IEEE-CIS_Fraud_Detection/train_transaction.csv')
transaction_data.fillna(-1)
col = dict(transaction_data.dtypes)
result = []
for key,value in col.items():
    if value == 'float64':
        print(key)
        result.append(len(pd.value_counts(transaction_data[key].tolist()))/len())

print(result)