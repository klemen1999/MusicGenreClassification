import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import json
from tqdm import tqdm

DATA_FOLDER = "./GTZAN/melSpectrogram"
LABELS = {"blues":0, "classical":1, "country":2, "disco":3, "hiphop":4, "jazz":5, "metal":6, "pop":7, "reggae":8, "rock":9}

IMAGE_HEIGHT = 400
IMAGE_WIDTH = 600
TEST_RATIO = 0.25
VAL_RATIO = 0.15

MODEL_NAME = "melSpectrogramModel"

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

def getMetrics(y_pred, y_test, modelName, returnRow=False):
    acc = accuracy_score(y_pred, y_test)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    if returnRow:
        return [modelName, round(acc,4), round(precision,4), round(recall,4), round(f1,4)]
    else:
        print(f"Model: {modelName}")
        print(f"- CA: {round(acc,4)}")
        print(f"- Precision: {round(precision,4)}")
        print(f"- Recall: {round(recall,4)}")
        print(f"- F1: {round(f1,4)}")

if __name__ == "__main__":
    filenames, labels = filenamesAndLabels(DATA_FOLDER)
    filenames_train, filenames_test, labels_train, labels_test = train_test_split(filenames, labels, test_size=TEST_RATIO, 
                                                    random_state=42, shuffle=True, stratify=labels)

    dataset_test = tf.data.Dataset.from_tensor_slices((filenames_test, labels_test))
    test_images = dataset_test.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).map(normalize)

    model = tf.keras.models.load_model(f"./GTZAN/checkpoints/{MODEL_NAME}/model0050.h5")

    labels_pred = []
    for element in tqdm(test_images.as_numpy_iterator()):
        image, label = element
        imageToPredict = image[None, :,:,:]
        currLabel = np.argmax(model.predict(imageToPredict)[0])
        labels_pred.append(currLabel)

    getMetrics(labels_pred, labels_test, MODEL_NAME)

