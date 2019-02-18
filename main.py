import os

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.datasets import mnist
from keras.engine.saving import load_model
from keras.layers import Input, Dense
from keras.models import Model

TRAINED_MODELS_PATH = './trained_models/'


def plot_training_loss(training_history):
    plt.plot(training_history['loss'], linewidth=2, label='Train')
    plt.plot(training_history['val_loss'], linewidth=2, label='Test')
    plt.legend(loc='upper right')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()


def plot_output(data_test, decoded):
    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        # print(data_test.shape)
        plt.imshow(data_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def autoencoder_numbers(num_test):
    if not os.path.exists(TRAINED_MODELS_PATH):
        os.makedirs(TRAINED_MODELS_PATH)

    # Inicializar Autoencoder
    # this is the size of our encoded representations
    encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # this is our input placeholder
    input_layer = Input(shape=(784,))
    # "encoded" is the encoded representation of the input
    encoding_layer = Dense(encoding_dim, activation='relu')(input_layer)
    # "decoded" is the lossy reconstruction of the input
    decoding_layer = Dense(784, activation='sigmoid')(encoding_layer)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_layer, decoding_layer)

    autoencoder.compile(optimizer='adadelta', loss='categorical_crossentropy')

    cp = ModelCheckpoint(filepath=TRAINED_MODELS_PATH + 'auto_' + str(num_test) + '.h5',
                         save_best_only=True,
                         verbose=0)

    tb = TensorBoard(log_dir='./logs',
                     histogram_freq=0,
                     write_graph=True,
                     write_images=True)

    ea = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Procesamiento de datos
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    # num_x_test = x_test
    num_filter = np.where(y_train == num_test)
    others_filter = np.where(y_train != num_test)
    num_x_train = x_train[num_filter]
    others_x_train = x_train[others_filter]
    split_size = int(num_x_train.shape[0] * 0.7)
    num_x_train, num_x_test = np.split(num_x_train, [split_size])

    files = os.listdir(TRAINED_MODELS_PATH)
    exists = False
    for file in files:
        if ('auto_' + str(num_test) + '.h5') == file:
            exists = True

    if not exists:
        history = autoencoder.fit(num_x_train, num_x_train,
                                  epochs=300,
                                  batch_size=256,
                                  shuffle=True,
                                  validation_data=(num_x_test, num_x_test),
                                  callbacks=[cp, tb, ea]).history
        plot_training_loss(history)

    autoencoder = load_model(TRAINED_MODELS_PATH + 'auto_' + str(num_test) + '.h5')
    num_x_reconstruction = autoencoder.predict(num_x_test)
    others_x_reconstruction = autoencoder.predict(others_x_train)

    num_mse = np.mean(np.power(num_x_test - num_x_reconstruction, 2), axis=1)
    others_mse = np.mean(np.power(others_x_train - others_x_reconstruction, 2), axis=1)

    plot_output(num_x_test, num_x_reconstruction)

    print("Creating error plot")

    threshold = np.mean(num_mse) + np.std(num_mse)
    fig, ax = plt.subplots()

    ax.plot(num_mse, marker='o', ms=1.5, linestyle='', label="Digit %s" % num_test)

    others_mse = np.random.choice(others_mse, num_mse.size)

    ax.plot(others_mse, marker='o', ms=1, linestyle='', label='Others')

    ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
    ax.legend()
    plt.title("Reconstruction error for %s and others" % num_test)
    plt.ylabel("Reconstruction error")
    plt.xlabel("Data point index")
    plt.show()

    # Evaluar el modelo
    # mse = np.mean(np.power(num_x_test - num_x_reconstruction, 2), axis=1)
    # error_df = pd.DataFrame({'Reconstruction_error': mse,
    # 'True_class': num_x_test})
    # error_df.describe()


for i in range(10):
    autoencoder_numbers(i)
