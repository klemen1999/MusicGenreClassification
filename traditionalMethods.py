import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import preprocessing

LABELS = ["Blues", "Classical", "Country", "Disco", "HipHop", "Jazz", "Metal", "Pop", "Reaggea", "Rock"]

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

def vizConfusionMat(y_pred, y_test, currentModel, save=False):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS, )  
    disp.plot()
    plt.title(currentModel)
    plt.xticks(rotation = 90)
    plt.tight_layout()
    plt.savefig(f"./results/{currentModel}.jpg") if save else plt.show()

def evalBaseline(y_test):
    y_pred = np.array([0]*y_test.shape[0])
    return y_pred

def evalKnn(X_train, X_test, y_train, k=5):
    knn = KNeighborsClassifier(n_neighbors=k)
    model = knn.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def evalRandomForest(X_train, X_test, y_train, n=100):
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    model = rf.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def evalLogRegression(X_train, X_test, y_train):
    logReg = LogisticRegression(random_state=42)
    model = logReg.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def evalSVM(X_train, X_test, y_train, kernel="rbf",c=1):
    svm = SVC(C=c, kernel=kernel, random_state=42)
    model = svm.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def evalGB(X_train, X_test, y_train, n=100, rate=1.0):
    xgb = GradientBoostingClassifier(n_estimators=n, learning_rate=rate, random_state=42)
    model = xgb.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def evalPerceptron(X_train, X_test, y_train, hLayerSize=30, numIter=200):
    mp = MLPClassifier(hidden_layer_sizes=(hLayerSize), max_iter=numIter, random_state=42)
    model = mp.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

if __name__ == "__main__":
    # Load data
    df = pd.read_pickle("rawFeaturesGTZAN.pkl")

    # Split into train and test
    data = df.drop(["path", "class"], axis=1)

    X = data.to_numpy()
    y = df["class"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True, stratify=y)

    gaussScaler = preprocessing.PowerTransformer(method='yeo-johnson', standardize=True).fit(X_train)
    X_train = gaussScaler.transform(X_train)
    X_test = gaussScaler.transform(X_test)

    models = ["baseline","kNN","randomForest","logRegression","SVM","GradientBoosting","MultilayerPerceptron"]
    results = pd.DataFrame(columns=["modelName","acc","precision","recall","f1"])
    for currentModel in tqdm(models):
        if currentModel == "baseline":
            y_pred = evalBaseline(y_test)
        elif currentModel == "kNN":
            y_pred = evalKnn(X_train, X_test, y_train, 11)
        elif currentModel == "randomForest":
            y_pred = evalRandomForest(X_train, X_test, y_train, 200)
        elif currentModel == "logRegression":
            y_pred = evalLogRegression(X_train, X_test, y_train)
        elif currentModel == "SVM":
            y_pred = evalSVM(X_train, X_test, y_train, kernel="rbf", c=4.4)
        elif currentModel == "GradientBoosting":
            y_pred = evalGB(X_train, X_test, y_train, rate=0.5)
        elif currentModel == "MultilayerPerceptron":
            y_pred = evalPerceptron(X_train, X_test, y_train, hLayerSize=30, numIter=200)

        vizConfusionMat(y_pred, y_test, currentModel, save=True)
        newRow = getMetrics(y_pred, y_test, currentModel, returnRow=True)
        results.loc[len(results)] = newRow

    results.to_csv("./results/tabel.csv", index=False)