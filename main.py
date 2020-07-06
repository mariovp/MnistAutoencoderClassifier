import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model

TRAINED_MODELS_PATH = './trained_models/'


def plot_training_loss(training_history, digit):
    plt.plot(training_history['loss'], linewidth=2, label='Train')
    plt.plot(training_history['val_loss'], linewidth=2, label='Test')
    plt.legend(loc='upper right')
    plt.title('Model loss for %s' % digit)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()


def plot_output(data_test, decoded, n_samples):
    n = n_samples  # how many digits we will display
    plt.figure(figsize=(n*2, 4))
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


def get_train_test_data_for_digit(digit):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    digit_train_filter = np.where(y_train == digit)
    others_train_filter = np.where(y_train != digit)

    digit_x_train = x_train[digit_train_filter]
    others_x_train = x_train[others_train_filter]

    digit_test_filter = np.where(y_test == digit)
    digit_x_test = x_test[digit_test_filter]

    return digit_x_train, digit_x_test, others_x_train


def get_random_test_images(n):
    (_, _), (x_test, _) = mnist.load_data()
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    return x_test[np.random.choice(x_test.shape[0], n, replace=False), :]


def train_autoencoder_for_number(digit):
    if not os.path.exists(TRAINED_MODELS_PATH):
        os.makedirs(TRAINED_MODELS_PATH)

    # Inicializar Autoencoder
    # this is the size of our encoded representations
    # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
    encoding_dim = 32

    # this is our input placeholder
    input_layer = Input(shape=(784,))
    # "encoded" is the encoded representation of the input
    encoding_layer = Dense(encoding_dim, activation='relu')(input_layer)
    # "decoded" is the lossy reconstruction of the input
    decoding_layer = Dense(784, activation='sigmoid')(encoding_layer)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_layer, decoding_layer)

    autoencoder.compile(optimizer='adadelta', loss='categorical_crossentropy')

    cp = ModelCheckpoint(filepath=TRAINED_MODELS_PATH + 'auto_' + str(digit) + '.h5',
                         save_best_only=True,
                         verbose=0)

    tb = TensorBoard(log_dir='./logs',
                     histogram_freq=0,
                     write_graph=True,
                     write_images=True)

    ea = EarlyStopping(monitor='val_loss', patience=3,
                       restore_best_weights=True)

    digit_x_train, digit_x_test, others_x_train = get_train_test_data_for_digit(
        digit)

    files = os.listdir(TRAINED_MODELS_PATH)
    exists = False
    for file in files:
        if ('auto_' + str(digit) + '.h5') == file:
            exists = True

    if not exists:
        history = autoencoder.fit(digit_x_train, digit_x_train,
                                  epochs=300,
                                  batch_size=64,
                                  shuffle=True,
                                  validation_data=(digit_x_test, digit_x_test),
                                  callbacks=[cp, tb, ea]).history
        plot_training_loss(history, digit)

    autoencoder = load_model(TRAINED_MODELS_PATH +
                             'auto_' + str(digit) + '.h5')
    num_x_reconstruction = autoencoder.predict(digit_x_test)
    others_x_reconstruction = autoencoder.predict(others_x_train)

    num_mse = np.mean(np.power(digit_x_test - num_x_reconstruction, 2), axis=1)
    others_mse = np.mean(
        np.power(others_x_train - others_x_reconstruction, 2), axis=1)

    plot_output(digit_x_test, num_x_reconstruction, 10)

    print("Creating error plot")

    threshold = np.mean(num_mse) + np.std(num_mse)
    fig, ax = plt.subplots()

    ax.plot(num_mse, marker='o', ms=1.5,
            linestyle='', label="Digit %s" % digit)

    others_mse = np.random.choice(others_mse, num_mse.size)

    ax.plot(others_mse, marker='o', ms=1, linestyle='', label='Others')

    ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[
              1], colors="r", zorder=100, label='Threshold')
    ax.legend()
    plt.title("Reconstruction error for %s and others" % digit)
    plt.ylabel("Reconstruction error")
    plt.xlabel("Data point index")
    plt.show()

    """print("Hacer prediccion con todos los autoencoders")
    image = digit_x_test[0:1:1]
    predict(image)"""

    # Evaluar el modelo
    # mse = np.mean(np.power(digit_x_test - num_x_reconstruction, 2), axis=1)
    # error_df = pd.DataFrame({'Reconstruction_error': mse,
    # 'True_class': digit_x_test})
    # error_df.describe()


def train_all():
    for i in range(10):
        train_autoencoder_for_number(i)


def predict(image):
    autoencoders = []
    for i in range(10):
        autoencoders.append(load_model(
            TRAINED_MODELS_PATH + 'auto_' + str(i) + '.h5'))

    reconstruction_errors = []
    for i in range(10):
        reconstruction = autoencoders[i].predict(image)
        mse = np.mean(np.power(image - reconstruction, 2), axis=1)
        plot_output(image, reconstruction, 1)
        reconstruction_errors.append(mse[0])

    print(reconstruction_errors)


def classification_test():
    test_images = get_random_test_images(10)

    autoencoders = []
    for i in range(10):
        autoencoders.append(load_model(
            TRAINED_MODELS_PATH + 'auto_' + str(i) + '.h5'))

    reconstruction_errors = []
    for i in range(10):
        reconstructions = autoencoders[i].predict(test_images)
        mse = np.mean(np.power(test_images - reconstructions, 2), axis=1)
        # plot_output(image, reconstruction, 1)
        reconstruction_errors.append(mse)

    reconstruction_errors = np.array(reconstruction_errors)

    for i, image in enumerate(test_images):
        plt.figure(figsize=(5, 4))
        ax = plt.subplot(2, 1, 1)
        plt.imshow(image.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        row_labels = np.arange(10)
        image_errors = reconstruction_errors[:, i]

        min_index = np.argmin(image_errors)

        cell_colors = []
        for j in range(10):
            cell_colors.append("w")

        cell_colors[min_index] = "#66CC00"
        cell_colors = np.array(cell_colors)

        image_errors = image_errors.astype(np.str)

        print(cell_colors)

        plt.table(cellText=image_errors.reshape(10, 1),
                  cellColours=cell_colors.reshape(10, 1), rowLabels=row_labels)

        plt.title("Reconstruction error for each digit autoencoder")

        plt.show()

    print(reconstruction_errors)
    print("==========")
    print(reconstruction_errors[:, 1])


classification_test()
# train_autoencoder_for_number(0)
