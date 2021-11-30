import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

LABELS = ["Blues", "Classical", "Country", "Disco", "HipHop", "Jazz", "Metal", "Pop", "Reaggea", "Rock"]

def getMetrics(y_pred, y_test, modelName):
    acc = accuracy_score(y_pred, y_test)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Model: {modelName}")
    print(f"- CA: {round(acc,4)}")
    print(f"- Precision: {round(precision,4)}")
    print(f"- Recall: {round(recall,4)}")
    print(f"- F1: {round(f1,4)}")

def vizConfusionMat(y_pred, y_test, currentModel, save=False, saveName=None):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)  
    disp.plot()
    plt.title(currentModel)
    plt.savefig(f"./images/{saveName}.jpg") if save else plt.show()

def evalBaseline(y_test):
    y_pred = np.array([0]*y_test.shape[0])
    return y_pred

def evalKnn(X_train, X_test, y_train, n=5):
    knn = KNeighborsClassifier(n_neighbors=n)
    model = knn.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def evalRandomForest(X_train, X_test, y_train, n=100):
    rf = RandomForestClassifier(n_estimators=n)
    model = rf.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def evalLogRegression(X_train, X_test, y_train):
    logReg = LogisticRegression()
    model = logReg.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def evalSVM(X_train, X_test, y_train):
    svm = SVC(kernel='linear')
    model = svm.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def evalXGB(X_train, X_test, y_train):
    xgb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    model = xgb.fit(X_train, y_train)
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

    currentModel = "XGB"
    if currentModel == "baseline":
        y_pred = evalBaseline(y_test)
    elif currentModel == "kNN":
        y_pred = evalKnn(X_train, X_test, y_train, 5)
    elif currentModel == "randomForest":
        y_pred = evalRandomForest(X_train, X_test, y_train, 100)
    elif currentModel == "logRegression":
        y_pred = evalLogRegression(X_train, X_test, y_train)
    elif currentModel == "SVM":
        y_pred = evalSVM(X_train, X_test, y_train)
    elif currentModel == "XGB":
        y_pred = evalXGB(X_train, X_test, y_train)

    getMetrics(y_pred, y_test, currentModel)
    vizConfusionMat(y_pred, y_test, currentModel)