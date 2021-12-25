import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json

DATA_FOLDER = "./GTZAN/mfcc"
LABELS = {"blues":0, "classical":1, "country":2, "disco":3, "hiphop":4, "jazz":5, "metal":6, "pop":7, "reggae":8, "rock":9}

IMAGE_HEIGHT = 400
IMAGE_WIDTH = 600
TEST_RATIO = 0.25
VAL_RATIO = 0.15

BATCH_SIZE = 16
EPOCHS = 50

MODEL_NAME = "mfccModel"

def load_image(filename, label):
    image = tf.io.read_file(filename)
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    return image, label

def normalize(input_image, label):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, label

def filenamesAndLabels(path):
    filenames = []
    labels = []
    for f in os.listdir(path):
        if "png" in f:
            filename = os.path.join(path, f)
            filenames.append(filename)
            label = LABELS[f.split(".")[0]]
            labels.append(label)
    return filenames, labels

def display_images_from_dataset(dataset):
    plt.figure(figsize=(13,13))
    subplot=231
    for i, (image, label) in enumerate(dataset):
        plt.subplot(subplot)
        plt.axis('off')
        plt.imshow(image.numpy().astype(np.uint8))
        plt.title(label.numpy(), fontsize=16)
        subplot += 1
        if i==6:
            break
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()



# Source: https://github.com/chittalpatel/Music-Genre-Classification-GTZAN/blob/master/Music%20Genre%20Classification/CNN_train(1).ipynb
def conv_block(x, n_filters,filter_size=(3, 3), pool_size=(2, 2),stride=(1, 1)):
    x = tf.keras.layers.Conv2D(n_filters, filter_size, strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=stride)(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    return x

def build_model(input_shape):
    inpt = tf.keras.layers.Input(shape=input_shape)
    x = conv_block(inpt, 16,stride=(2,2))
    x = conv_block(x, 32,filter_size=(3,3),stride=(2,2))
    x = conv_block(x, 64, stride=(2,2))
    x = conv_block(x, 128,filter_size=(3,3),stride=(2,2))
    x = conv_block(x, 256,stride=(2,2))

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu', 
              kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    predictions = tf.keras.layers.Dense(10, 
                        activation='softmax', 
                        kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    
    model = tf.keras.Model(inputs=inpt, outputs=predictions)
    return model


if __name__ == "__main__":
    if not os.path.exists(f"./GTZAN/checkpoints/{MODEL_NAME}"):
        os.makedirs(f"./GTZAN/checkpoints/{MODEL_NAME}")

    filenames, labels = filenamesAndLabels(DATA_FOLDER)
    filenames_train, filenames_test, labels_train, labels_test = train_test_split(filenames, labels, test_size=TEST_RATIO, 
                                                    random_state=42, shuffle=True, stratify=labels)
    filenames_train, filenames_val, labels_train, labels_val = train_test_split(filenames_train, labels_train, test_size=VAL_RATIO,
                                                    random_state=42, shuffle=True, stratify=labels_train)
    
    dataset_train = tf.data.Dataset.from_tensor_slices((filenames_train, labels_train))
    train_images = dataset_train.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    # display_images_from_dataset(train_images)

    dataset_val = tf.data.Dataset.from_tensor_slices((filenames_val, labels_val))
    val_images = dataset_val.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    TRAIN_LENGTH = len(filenames_train)
    BUFFER_SIZE = 1000
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

    train_batches = (
        train_images
        .cache()
        .batch(BATCH_SIZE)
        .repeat()
        .map(normalize)
        .prefetch(buffer_size=tf.data.AUTOTUNE))

    val_batches = val_images.batch(BATCH_SIZE).map(normalize)

    model = build_model((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    model.compile(optimizer="adam",
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])
    model.summary()

    chechPoint_callback = tf.keras.callbacks.ModelCheckpoint("./GTZAN/checkpoints/"+MODEL_NAME+"/model{epoch:04d}.h5",
                                        save_weights_only=False, period=10)

    model_history = model.fit(train_batches, epochs=EPOCHS,steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=val_batches, callbacks=[chechPoint_callback])

    history_dict = model_history.history
    json.dump(history_dict, open(f"./GTZAN/checkpoints/{MODEL_NAME}/modelHistory.json", 'w'))

    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    plt.figure()
    plt.plot(model_history.epoch, loss, 'r', label='Training loss')
    plt.plot(model_history.epoch, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./GTZAN/checkpoints/{MODEL_NAME}/loss.jpg")


    
    