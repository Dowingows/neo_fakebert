from official.nlp import optimization  # to create AdamW optimizer
import os
import tensorflow as tf
import models as md
import dataset as dt
import utils as ut 

def run(bert_model_name, mlp_model_name, train_params):

    train_ds, val_ds = dt.get_fakenews_train_dataset_tf(train_params['batch_size'])
    epochs = train_params['epochs']
    
    model = md.build_bert_model(bert_model_name, mlp_model_name)
    
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    metrics = [tf.metrics.BinaryAccuracy(), tf.metrics.AUC()]

    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)

    init_lr = 3e-5
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                            num_train_steps=num_train_steps,
                                            num_warmup_steps=num_warmup_steps,
                                            optimizer_type='adamw')

    """# Compila e Treina"""

    model.compile(optimizer=optimizer,
                            loss=loss,
                            metrics=metrics)

    md_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(train_params['out_dir'], 'weights.h5'), 
            save_best_only=True, verbose=1, 
            save_weights_only=True, monitor='val_loss', mode='min')
    
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')

    history = model.fit(x=train_ds,
                                validation_data=val_ds,
                                epochs=epochs, callbacks=[md_checkpoint, earlyStopping ])
    

    ut.save_train_status(train_params['out_dir'], model, history.history)

if __name__ == '__main__':

    params = ut.get_env_params()
    
    bert_model_name = params['model']['bert_model_name']
    mlp_model_name = params['model']['mlp_model_name']
    
    out_dir = os.path.join(params['train']['out_pathname'], ut.gen_outdirname_by_env())

    ut.create_nested_dir(out_dir)

    train_params = {
        'epochs': params['train']['epochs'],
        'out_dir': out_dir,
        'batch_size': params['train']['batch_size']
    }

    print(f'Training model {bert_model_name}')
    run(bert_model_name, mlp_model_name, train_params)
