from GetData import Preprocessor_ML
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn import metrics
import torch
import matplotlib.pyplot as plt

def show_metrics(cm):
    tp = cm[1, 1]
    fn = cm[1, 0]
    fp = cm[0, 1]
    tn = cm[0, 0]
    print('Precision = %.03f' % (tp / (tp + fp)))
    print('Recall    = %.03f' % (tp / (tp + fn)))
    print('F1_score  = %.03f' % (2 * (((tp / (tp + fp)) * (tp / (tp + fn))) /
                                      ((tp / (tp + fp)) + (tp / (tp + fn))))))

if __name__ == '__main__':
    device = 'cuda'
    EPOCHS = 6
    Thres = 0.50
    Preprocessor = Preprocessor_ML('../IEEE-CIS_Fraud_Detection')
    X_train, X_valid, y_train, y_valid = Preprocessor.Work(mode='DropMajorNull', ToObjectColumn='Encode')

    model = TabNetClassifier(n_d=8, n_a=8, n_steps=1, gamma=1.3,
                           lambda_sparse=0, optimizer_fn=torch.optim.Adam,
                           optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                           mask_type='entmax', device_name=device, output_dim=1,
                           scheduler_params=dict(milestones=[100, 150], gamma=0.9),
                           scheduler_fn=torch.optim.lr_scheduler.MultiStepLR)

    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)],max_epochs=EPOCHS,patience=20,
              batch_size=1024, virtual_batch_size=128, eval_name=['train','valid'])

    preds = model.predict_proba(X_valid)
    preds_label = preds[:,1] > Thres
    cm = metrics.confusion_matrix(y_valid,preds_label)
    print(cm)
    show_metrics(cm)
    fpr, tpr, threshold = metrics.roc_curve(y_valid, preds[:,1])
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.title('Prediction of TabNet')
    plt.plot(fpr, tpr, 'salmon', label='TabNet Prediction, AUC is {}'.format(roc_auc))
    plt.legend(loc='lower right')
    plt.show()
