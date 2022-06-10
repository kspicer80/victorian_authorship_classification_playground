import numpy as np
import tensorflow as tf
import load_and_shuffle_data
import vectorize_data
import build_model
import load_and_shuffle_data
import explore_data

def train_ngram_model(data,
                      learning_rate=1e-3,
                      epochs=1000,
                      batch_size=128,
                      layers=2,
                      units=64,
                      dropout_rate=0.2):
    (train_texts, train_labels), (val_texts, val_labels) = data

    # Verify that validation labels are in the same range as training labels.
    #num_classes = explore_data.get_num_classes(train_labels) # This is the number of authors in the training_dataset
    #num_classes = len(train_labels)
    #unexpected_labels = [v for v in val_labels if v not in num_classes]
    #if len(unexpected_labels):
        #raise ValueError('Unexpected label values found in the validation set:'
                        #' {unexpected_labels}. Please make sure that the '
                        #'labels in the validation set are in the same range '
                        #'as training labels.'.format(
                        #unexpected_labels=unexpected_labels))
    num_classes = 51
    # Vectorize texts.
    x_train, x_val = vectorize_data.ngram_vectorizer(
        train_texts, train_labels, val_texts)

    # Create model instance.
    model = build_model.mlp_model(layers=layers,
                                  units=units,
                                  dropout_rate=dropout_rate,
                                  input_shape=x_train.shape[1:],
                                  num_classes=num_classes)

    # Compile model with learning parameters.
    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2)]

    # Train and validate model.
    history = model.fit(
            x_train,
            train_labels,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(x_val, val_labels),
            verbose=2,  # Logs once per epoch.
            batch_size=batch_size)

    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # Save model.
    model.save('victorian_authorship_attribution_mlp_model.h5')
    return history['val_acc'][-1], history['val_loss'][-1]