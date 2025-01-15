import argparse
import pandas as pd
import h5py
import numpy as np
from os.path import  isdir, join
from os import mkdir
from sys import argv
from functools import partial
from datetime import datetime
import os
import tensorflow as tf
import warnings
from Model.SEcnnlstm_channel import cnn_lstm_attention

warnings.filterwarnings("ignore")
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
seed = 42
tf.random.set_seed(seed)

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def ppg_hdf2tfrecord(h5_file, tfrecord_path, samp_idx):
    N_samples = len(samp_idx)

    with h5py.File(h5_file, 'r') as f:

        ppg_h5 = f.get('/ppg')
        BP = f.get('/label')
        subject_idx = f.get('/subject_idx')
        writer = tf.io.TFRecordWriter(tfrecord_path)

        for i in np.nditer(samp_idx):
            ppg = np.array(ppg_h5[i, :])
            target = np.array(BP[i, :], dtype=np.float32)
            sub_idx = np.array(subject_idx[i])

            data = \
                {'ppg': _float_feature(ppg.tolist()),
                 'label': _float_feature(target.tolist()),
                 'subject_idx': _float_feature(sub_idx.tolist()),
                 'Nsamples': _float_feature([N_samples])}

            feature = tf.train.Features(feature=data)
            example = tf.train.Example(features=feature)
            serialized = example.SerializeToString()

            writer.write(serialized)
        writer.close()

def ppg_hdf2tfrecord_sharded(h5_file, tf_path, samp_idx, Nsamp_per_shard, modus='train'):

    tf_name = join(tf_path, modus)
    N_samples = len(samp_idx)
    N_shards = np.ceil(N_samples / Nsamp_per_shard).astype(int)

    # iterate over every shard
    for i in range(N_shards):
        idx_start = i * Nsamp_per_shard
        idx_stop = (i + 1) * Nsamp_per_shard
        if idx_stop > N_samples:
            idx_stop = N_samples
        idx_curr = samp_idx[idx_start:idx_stop]
        output_filename = '{0}_{1:05d}_of_{2:05d}.tfrecord'.format(tf_name, i + 1, N_shards)
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(dt_string, ': processing ',modus,' shard ', str(i + 1), ' of ', str(N_shards))
        ppg_hdf2tfrecord(h5_file, output_filename, idx_curr)


def h5_to_tfrecords(SourceFile, tfrecordsPath):

    Nsamp_per_shard = 500
    with h5py.File(SourceFile, 'r') as f:
        BP = np.array(f.get('/label'))
        BP = np.round(BP)
        BP = np.transpose(BP)

        subject_idx = np.squeeze(np.array(f.get('/subject_idx')))

    N_samp_total = BP.shape[1]
    subject_idx = subject_idx[:N_samp_total]

    valid_idx = np.arange(subject_idx.shape[-1])
    subject_labels = np.unique(subject_idx)
    print(subject_labels)

    for i in subject_labels:

        tfrecordsPath_cv=join(tfrecordsPath, 'round_{}'.format(i))
        if not isdir(tfrecordsPath_cv):
            mkdir(tfrecordsPath_cv)
        tfrecord_path_train = join(tfrecordsPath_cv, 'train')
        if not isdir(tfrecord_path_train):
            mkdir(tfrecord_path_train)
        tfrecord_path_test = join(tfrecordsPath_cv, 'test')
        if not isdir(tfrecord_path_test):
            mkdir(tfrecord_path_test)


        subjects_test_labels = i

        idx_test = valid_idx[np.isin(subject_idx, subjects_test_labels)]
        train_test_labels = np.setdiff1d(subject_labels, subjects_test_labels)  #
        idx_train = valid_idx[np.isin(subject_idx, train_test_labels)]
        
        # save ground truth BP values of training, validation and test set in csv-files for future reference
        BP_train = BP[:, idx_train]
        d = {"SBP": np.transpose(BP_train[0, :]), "DBP": np.transpose(BP_train[1, :]), "MBP": np.transpose(BP_train[2, :])}
        train_set = pd.DataFrame(d)
        train_set.to_csv(join(tfrecordsPath_cv,'rPPG_cycle_trainset.csv'))

        BP_test = BP[:, idx_test]
        d = {"SBP": np.transpose(BP_test[0, :]), "DBP": np.transpose(BP_test[1, :]), "MBP": np.transpose(BP_test[2, :])}
        train_set = pd.DataFrame(d)
        train_set.to_csv(join(tfrecordsPath_cv,'rPPG_cycle_testset.csv'))

    # create tfrecord dataset
    # ----------------------------
        np.random.shuffle(idx_train)
        ppg_hdf2tfrecord_sharded(SourceFile, tfrecord_path_test, idx_test,  Nsamp_per_shard, modus='test')
        ppg_hdf2tfrecord_sharded(SourceFile, tfrecord_path_train, idx_train, Nsamp_per_shard, modus='train')
    print("Script finished")

def read_tfrecord(example, win_len):
    tfrecord_format = (
        {
            'ppg': tf.io.FixedLenFeature([win_len], tf.float32),
            'label': tf.io.FixedLenFeature([3], tf.float32)
        }
    )
    parsed_features = tf.io.parse_single_example(example, tfrecord_format)

    return parsed_features['ppg'], (parsed_features['label'][0], parsed_features['label'][1], parsed_features['label'][2])


def create_dataset(tfrecords_dir, win_len, batch_size, modus='train'):

    pattern = join(tfrecords_dir, modus, modus + "_?????_of_?????.tfrecord")
    dataset = tf.data.TFRecordDataset.list_files(pattern)

    if modus == 'train':
        dataset = dataset.shuffle(10000, reshuffle_each_iteration=True)
        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=400,
            block_length=200)
    else:
        dataset = dataset.interleave(
            tf.data.TFRecordDataset)

    dataset = dataset.map(partial(read_tfrecord, win_len=win_len), num_parallel_calls=2)
    dataset = dataset.shuffle(700, reshuffle_each_iteration=True)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.repeat()
    return dataset

def get_model(architecture, input_shape):
    return {
        'cnn_lstm_attention': cnn_lstm_attention(input_shape),
    }[architecture]

def train_one_subject(idx,architecture,
                        DataDir,
                        DataDir_cv,
                        experiment_name,
                        win_len,
                        batch_size,
                        lr,
                        N_epochs,
                        Ntrain,
                        Ntest):

    data_in_shape = (win_len, 1)
    model = get_model(architecture, data_in_shape)

    if idx == 1:
        for layer in model.layers:
            print(layer.name)
        print(model.summary())

    test_dataset = create_dataset(DataDir_cv, win_len, batch_size, modus='test')
    train_dataset = create_dataset(DataDir_cv, win_len, batch_size, modus='train')

    csvLogger_cb = tf.keras.callbacks.CSVLogger(
        filename=join(ResultsDir, 'learning_curve', experiment_name + str(int(idx)) + '_learningcurve.csv'))

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=join(DataDir_cv, str(int(idx)) + '_cb.h5'),
        save_best_only=False)

    tensorbard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=join(ResultsDir, 'tensorboard', experiment_name),
        write_images=True,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch')

    earlystopping=False
    EarlyStopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True)

    opt = tf.keras.optimizers.Adam(lr)

    losses = {
        'SBP': 'mean_squared_error',
        'DBP': 'mean_squared_error',
        'MBP': 'mean_squared_error'
    }

    metrics = {
        'SBP': ['mae'],
        'DBP': ['mae'],
        'MBP': ['mae']
    }

    equal_weights = 1.0
    loss_weights = {
        'SBP': equal_weights,
        'DBP': equal_weights,
        'MBP': equal_weights
    }

    model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights, metrics=metrics)

    cb_list = [checkpoint_cb,
               tensorbard_cb,
               csvLogger_cb,
               EarlyStopping_cb if earlystopping == True else []]

    validation_steps = max(1, Ntest // batch_size) if Ntest >= batch_size else 1
    model.fit(
        train_dataset,
        steps_per_epoch=Ntrain // batch_size,
        epochs=N_epochs,
        validation_data=test_dataset,
        validation_steps=validation_steps,
        callbacks=cb_list)

    # Predictions on the testset
    model.load_weights(checkpoint_cb.filepath)
    test_results = pd.DataFrame({'SBP_true' : [],
                                 'DBP_true' : [],
                                 'MBP_true': [],
                                 'SBP_pre' : [],
                                 'DBP_pre' : [],
                                 'MBP_pre' : []})
    # store predictions on the test set as well as the corresponding ground truth in a csv file
    test_dataset = iter(test_dataset)
    for i in range(int(Ntest//batch_size)):
        ppg_test, BP_true = test_dataset.next()
        BP_est = model.predict(ppg_test)
        TestBatchResult = pd.DataFrame({'SBP_true' : BP_true[0].numpy(),
                                        'DBP_true' : BP_true[1].numpy(),
                                        'MBP_true': BP_true[2].numpy(),

                                        'SBP_pre' : np.squeeze(BP_est[0]),
                                        'DBP_pre' : np.squeeze(BP_est[1]),
                                        'MBP_pre' : np.squeeze(BP_est[2])})
        test_results = test_results._append(TestBatchResult)
    if Ntest % batch_size != 0:
        ppg_test, BP_true = test_dataset.next()
        BP_est = model.predict(ppg_test)
        TestBatchResult = pd.DataFrame({'SBP_true' : BP_true[0].numpy(),
                                        'DBP_true' : BP_true[1].numpy(),
                                        'MBP_true': BP_true[2].numpy(),

                                        'SBP_pre' : np.squeeze(BP_est[0]),
                                        'DBP_pre' : np.squeeze(BP_est[1]),
                                        'MBP_pre' : np.squeeze(BP_est[2])})
    test_results = test_results._append(TestBatchResult)

    ResultsFile = join(DataDir, '{}_test_results.csv'.format(idx))
    test_results.to_csv(ResultsFile)

def train_rppg_cycle(architecture,
                        DataDir,
                        SourceFile,
                        experiment_name,
                        win_len,
                        batch_size,
                        lr,
                        N_epochs,
                        ):
    # fixed random seed
    seed = 42
    tf.random.set_seed(seed)
    h5_to_tfrecords(SourceFile, DataDir)

    with h5py.File(SourceFile, 'r') as f:
        BP = np.array(f.get('/label'))
        BP = np.round(BP)
        BP = np.transpose(BP)
        subject_idx = np.squeeze(np.array(f.get('/subject_idx')))

    N_samp_total = BP.shape[1]
    subject_idx = subject_idx[:N_samp_total]

    # divide the subjects into training, validation and test subjects
    subject_labels = np.unique(subject_idx)
    print(subject_labels)


    for i in subject_labels:
        idx = i

        DataDir_cv = join(DataDir,'round_{}'.format(i))
        identifier_train = 'rPPG_cycle_trainset'
        identifier_test = 'rPPG_cycle_testset'

        csv_files = [file for file in os.listdir(DataDir_cv) if file.endswith('.csv')]
        Ntrain = 0
        Ntest  = 0

        for file in csv_files:
            if identifier_train in file:
                file_path_trian = os.path.join(DataDir_cv, file)
                Ntrain = sum(1 for line in open(file_path_trian))-1
                print('number of Trainset',Ntrain)

            if identifier_test in file:
                file_path_tset = os.path.join(DataDir_cv, file)
                Ntest = sum(1 for line in open(file_path_tset))-1
                print('number of Testset',Ntest)

        train_one_subject(idx, architecture,
                            DataDir,
                            DataDir_cv,
                            experiment_name,
                            win_len,
                            batch_size,
                            lr,
                            N_epochs,
                            Ntrain,
                            Ntest)
        print('{}_Finished'.format(i))
        tf.keras.backend.clear_session()

if __name__ == "__main__":

    if len(argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument('--input', type=str, help="Path to the .h5 file containing the dataset")
        parser.add_argument('--datadir', type=str, help="folder containing the train, val and test tfrecord files")
        parser.add_argument('--expname', type=str, help="unique name for the training")
        parser.add_argument('--resultsdir', type=str, help="Directory in which results are stored")
        parser.add_argument('--arch', type=str, default="cnn_lstm_attention", help="neural architecture used for training")
        parser.add_argument('--winlen', type=int, default="50", help="length_of_signal")

        args = parser.parse_args()
        architecture = args.arch
        win_len = args.winlen
        SourceFile = args.input
        DataDir = args.datadir
        ResultsDir = args.resultsdir
        experiment_name = args.expname
        experiment_name = datetime.now().strftime("%Y-%d-%m") + '_' + architecture + '_' + experiment_name

        #  default paras
        lr = 0.0001
        batch_size = 128
        N_epochs = 50

        train_rppg_cycle(architecture,
                        DataDir,
                        SourceFile,
                        experiment_name,
                        win_len,
                        batch_size,
                        lr,
                        N_epochs)