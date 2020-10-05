import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

def train(x_train, x_test, y_train, y_test):
    np.random.seed(123)
    tf.random.set_seed(123)
    print('First 3 labels: ', y_train[:3])

    # initialize model
    model = keras.models.Sequential()

    # add input layer
    model.add(keras.layers.Dense(
        units=3000,
        input_dim=len(x_train[0]),
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='tanh') 
    )
    # add hidden layer
    model.add(
        keras.layers.Dense(
            units=1024,
            input_dim=1024,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation='tanh')
        )
    model.add(
        keras.layers.Dense(
            units=512,
            input_dim=1024,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation='tanh')
        )
    # add output layer
    model.add(
        keras.layers.Dense(106)
        )

    # define SGD optimizer
    sgd_optimizer = keras.optimizers.SGD(
        lr=0.0005, decay=1e-7, momentum=0.6
    )
    # compile model
    model.compile(
        optimizer=sgd_optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

    )




    # train model
    history = model.fit(
        x_train, y_train,
        batch_size=50, epochs=200,
        verbose=1, validation_split=0.1
    )


    y_train_pred = model.predict_classes(x_train, verbose=0)
    print('First 3 predictions: ', y_train_pred[:3])

    # calculate training accuracy
    y_train_pred = model.predict_classes(x_train, verbose=0)
    correct_preds = np.sum(y_train == y_train_pred, axis=0)
    train_acc = correct_preds / y_train.shape[0]

    print(f'Training accuracy: {(train_acc * 100):.2f}')

    # calculate testing accuracy
    y_test_pred = model.predict_classes(x_test, verbose=0)
    print(y_test_pred[:8])
    print(y_test[:8])

    correct_preds = np.sum(y_test == y_test_pred, axis=0)
    test_acc = correct_preds / y_test.shape[0]

    print(f'Test accuracy: {(test_acc * 100):.2f}')
    return model
