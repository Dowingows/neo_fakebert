import utils as ut

def get_fakenews_train_dataset_tf(batch_size = 32, seed = 42, pathname =  r'./config/env.yaml'):
    
    import tensorflow as tf

    dataset_path = ut.get_env_params(pathname)['dataset']['fakenews_train']

    train_ds = tf.keras.utils.text_dataset_from_directory(
    dataset_path,
    batch_size=batch_size,
    validation_split=0.11,
    subset='training',
    seed=seed)

    val_ds = tf.keras.utils.text_dataset_from_directory(
        dataset_path,
        batch_size=batch_size,
        validation_split=0.11,
        subset='validation',
        seed=seed)

    return train_ds, val_ds    

def get_fakenews_test_dataset_tf(batch_size = 32, seed = 42, pathname =  r'./config/env.yaml'):
    
    import tensorflow as tf

    dataset_path = ut.get_env_params(pathname)['dataset']['fakenews_test']

    raw_test_ds = tf.keras.utils.text_dataset_from_directory(
        dataset_path,
        batch_size=batch_size,
        validation_split=None,
        subset=None,
        seed=seed
    )

    return raw_test_ds        