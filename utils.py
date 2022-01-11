import os
import yaml
import time
import matplotlib.pyplot as plt

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
    out_dir = os.path.join(out_dir, gen_outdirname_by_env())
    create_nested_dir(out_dir)
    model.save_weights(os.path.join(out_dir, 'weights.h5'))
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