from tensorflow.keras.layers import Input, LayerNormalization, BatchNormalization, LSTM, Bidirectional, Conv1D, Dropout, Flatten, GlobalAveragePooling1D, Reshape, Dense, multiply
from tensorflow.keras import regularizers
from tensorflow.keras import Model
import numpy as np

def se_block(input_tensor, ratio=20):

    squeeze = GlobalAveragePooling1D()(input_tensor)
    excitation = Dense(units=int(input_tensor.shape[-1]) // ratio, activation='relu')(squeeze)
    excitation = Dense(units=int(input_tensor.shape[-1]), activation='sigmoid')(excitation)

    excitation = Reshape((1, int(input_tensor.shape[-1])))(excitation)

    scaled = multiply([input_tensor, excitation])
    return scaled

def cnn_lstm_attention(data_in_shape):

    X_input = Input(shape=data_in_shape)
    X = Conv1D(filters=32, kernel_size=3, strides=1, padding='causal', activation='relu')(X_input)
    X = LayerNormalization()(X)
    X = Conv1D(filters=64, kernel_size=3, strides=1, padding='causal', activation='relu')(X)
    X = LayerNormalization()(X)
    X = Conv1D(filters=128, kernel_size=3, strides=1, padding='causal', activation='relu')(X)
    X = se_block(X)
    X = Conv1D(filters=256, kernel_size=3, strides=1, padding='causal', activation='relu')(X)


    X = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.04)))(X)
    X = se_block(X)
    X = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.04)))(X)

    X = Flatten()(X)

    X = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.04))(X)
    X = LayerNormalization()(X)
    X = Dropout(0.2)(X)
    X = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.04))(X)
    X = LayerNormalization()(X)
    X = Dropout(0.2)(X)

    X_SBP = Dense(1, name='SBP')(X)
    X_DBP = Dense(1, name='DBP')(X)
    X_MBP = Dense(1, name='MBP')(X)

    model = Model(inputs=X_input, outputs=[X_SBP, X_DBP, X_MBP], name='CNN_BiLSTM_SE_channel')

    return model

if __name__ == "__main__":
    # test
    data_in_shape = (50, 1)
    model = cnn_lstm_attention(data_in_shape)
    model.summary()