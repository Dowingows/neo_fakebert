import os
import yaml
import time
import matplotlib.pyplot as plt
import shutil

def create_nested_dir(path):
    try:
        os.makedirs(path)
    except:
        pass    

def get_env_params(pathname = r'./config/env.yaml'):
    params = {}
    with open(pathname) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    return params

def save_env_in_results(out_dir):
    shutil.copy(r'./config/env.yaml', os.path.join(out_dir,'env.yaml'))

def save_graphs(out_dir, history):
    for key in history:
        save_training_graph(out_dir, history, key)

def sanitize(strname):
    strname = strname.replace("/", "_").replace("-","_")
    return strname


def gen_outdirname_by_env():
    params = get_env_params()['model']
    return sanitize(params['tag']) +'-'+ sanitize(params['bert_model_name']) + "-" + sanitize(params['mlp_model_name']) + "-" +  time.strftime("%Y.%m.%d - %H.%M.%S")

def save_train_status(out_dir, model, history):
    save_env_in_results(out_dir)
    model.save_weights(os.path.join(out_dir, 'train_weights_final.h5'))
    save_graphs(out_dir, history)

def save_training_graph(out_dir, history, variable):
    out_dir = out_dir + '/training_graphs'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.title('model {}'.format(variable))
    plt.ylabel('{}'.format(variable))
    plt.xlabel('epoch')
    plt.plot(history['{}'.format(variable)])
    try:
        plt.plot(history['val_{}'.format(variable)])
    except:
        pass
    plt.savefig(os.path.join(out_dir, '{}.png'.format(variable)))

    plt.cla()
    plt.clf()
    plt.close('all')

def save_roc_curve(out_dir, fpr, tpr, auc)   :

    plt.title('Curva ROC')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Taxa de verdadeiros positivos')
    plt.xlabel('Taxa de falsos positivos')
    plt.savefig(out_dir)