import json
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import tensorflow.keras as keras
import matplotlib.pyplot as plt

DATASET_PATH = "/Users/andrea/Documents/DHDK/Thesis/Music Emotion/mfccs.json"
classes = ['HAPPY', 'SAD', 'TENDER', 'FEAR', 'ANGER', 'SURPRISE', 'HIGH VALENCE', 'LOW VALENCE', 'HIGH ENERGY',
         'LOW ENERGY', 'HIGH TENSION', 'LOW TENSION']


def load_data(dataset_path):
    with open(dataset_path, 'r') as fp:
        data = json.load(fp)

        # convert lists into numpy arrays
        X = np.array(data["mfcc"])
        y = np.array(data["labels"])

        return X, y


def plot_history(history):
    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error subplot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=40)
    plt.xlabel('Predicted label', fontsize=40)
    plt.tight_layout()


def prepare_dataset(test_size, validation_size):
    # load data
    X, y = load_data(DATASET_PATH)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # create train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    # RNN . LSTM model
    # create model
    model = keras.Sequential()

    # 2 LSTM layers
    model.add(keras.layers.LSTM(128, input_shape=input_shape, return_sequences=True))  # True = s2s, False = s2v
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.LSTM(64, return_sequences=True))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.LSTM(32, return_sequences=False))
    model.add(keras.layers.Dropout(0.1))

    # dense layer
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(32, activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    # output
    model.add(keras.layers.Dense(12, activation="softmax"))

    return model


def confusion(model, X, y):
    # X = X[np.newaxis, ...]
    prediction = model.predict(X)

    # extract index with the max value
    predicted_index = np.argmax(prediction, axis=1)

    predicted = predicted_index
    actual = y

    return actual, predicted


def predict(model, X, y):
    X = X[np.newaxis, ...]
    prediction = model.predict(X)

    # extract index with the max value
    predicted_index = np.argmax(prediction, axis=1)
    # predicted_class = classes[predicted_index]
    # actual_class = classes[y]

    print("The expected index is {}\nThe predicted index is {}".format(y, predicted_index))


if __name__ == "__main__":
    # create train, validation and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_dataset(0.25, 0.2)

    # build the CNN net
    input_shape = (X_train.shape[1], X_train.shape[2])  # could be train, validation, test
    model = build_model(input_shape)

    # compile the net
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # train the CNN
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)

    # evaluate the CNN on the testset
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on testset is: {}".format(test_accuracy))

    # stop the training if the validation score doesn't increase

    # make a prediction on a sample
    X = X_validation[10]
    y = y_validation[10]
    predict(model, X, y)

    # make a prediction on a sample
    X = X_test[100]
    y = y_test[100]
    predict(model, X, y)

    # plot model summary
    model.summary()

    # plot accuracy and error over epochs
    plot_history(history)
    # Compute confusion matrix

    actual, predicted = confusion(model, X_validation, y_validation)

    cnf_matrix = confusion_matrix(actual, predicted)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure(figsize=(32, 24), dpi=200)
    plot_confusion_matrix(cnf_matrix, classes=classes,
                          title='Confusion matrix, without normalization')
    plt.show()