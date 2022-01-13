
import os 
import pandas as pd
import utils as ut
import dataset as dt
import models as md
import numpy as np
from tqdm import tqdm
import  sklearn.metrics as  m
import sys

result_path = '../results/default-bert_prt_small-default_bert_mlp-2022.01.13 - 10.55.26'

if len(sys.argv) == 2:
    result_path = sys.argv[1]
    
def predict(model, test_ds):

    pred_all = []
    true_all = []

    for text_batch, label_batch in tqdm(test_ds):
        pred_label = model.predict(text_batch)  
        pred_all.extend(pred_label.flatten().tolist())
        true_all.extend(label_batch.numpy().tolist())

    true_all = np.asarray(true_all)
    pred_all = np.asarray(pred_all)

    return true_all, pred_all

if __name__ == '__main__':

    test_path_env = os.path.join(result_path, 'env.yaml')
    params = ut.get_env_params(pathname=test_path_env)
    
    bert_model_name = params['model']['bert_model_name']
    mlp_model_name = params['model']['mlp_model_name']    

    test_ds = dt.get_fakenews_test_dataset_tf(pathname=test_path_env)

    model = md.build_bert_model(bert_model_name, mlp_model_name)
    model.load_weights(os.path.join(result_path, 'weights.h5'))
    
    y_true, y_pred_prob  = predict(model, test_ds)
    y_pred_bin  = (y_pred_prob > 0.5  ).astype(np.uint8)    

    # calculate metrics

    metrics = {}
    print("#-- Metrics --#")
    fpr, tpr, thresholds = m.roc_curve(y_true, y_pred_prob)
    auc = m.auc(fpr, tpr)
    metrics['auc'] = auc

    ut.save_roc_curve(os.path.join(result_path, 'roc_curve.png'), fpr, tpr, auc)

    acc = m.accuracy_score(y_true, y_pred_bin)
    metrics['acc'] = acc

    for key, val in metrics.items():
        print('{}: {:.4f}'.format(key, val))

    
    keys = metrics.keys()
    df = pd.DataFrame(columns=keys)
    df = df.append(metrics, ignore_index=True)
    df.to_csv(os.path.join(result_path, 'metrics.csv'), sep=';', decimal=',', index=False)