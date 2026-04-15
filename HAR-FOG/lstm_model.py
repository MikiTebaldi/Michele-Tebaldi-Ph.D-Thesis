import pandas as pd
import numpy as np
import scipy.io
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    '''Load and split the data'''
    X, y, ids = load_data()
    trainInd, valInd, testInd = train_test(ids)

    X_train, y_train = X[trainInd], y[trainInd]
    X_val, y_val = X[valInd], y[valInd]
    X_test, y_test = X[testInd], y[testInd]

    '''Preprocess labels'''
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train.ravel())
    y_val_enc = le.transform(y_val.ravel())
    y_test_enc = le.transform(y_test.ravel())
    num_classes = len(np.unique(y_train_enc))

    '''Train and evaluate LSTM'''
    train_lstm_model(X_train, y_train_enc, X_val, y_val_enc, X_test, y_test_enc, num_classes)

def load_data():
    fileName = 'data.mat'
    mat = scipy.io.loadmat(fileName)
    data = mat['data']['data'][0][0]
    labels = mat['data']['activity'][0][0]
    ids = mat['data']['subject'][0][0]
    return data, labels, ids

def train_test(ids):
    trainPd = range(1, 40)
    valPd = range(40, 50)
    testPd = range(40, 60)

    trainInd = np.concatenate([np.where(ids == pd)[0] for pd in trainPd])
    valInd = np.concatenate([np.where(ids == pd)[0] for pd in valPd])
    testInd = np.concatenate([np.where(ids == pd)[0] for pd in testPd])
    return trainInd, valInd, testInd

def train_lstm_model(X_train, y_train, X_val, y_val, X_test, y_test, num_classes):
    input_shape = (X_train.shape[1], X_train.shape[2])

    model = models.Sequential([
        layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        layers.LSTM(64),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("\nTraining LSTM model...")
    model.fit(X_train, y_train, epochs=5, batch_size=128,
              validation_data=(X_val, y_val), verbose=1)

    print("\nEvaluating on test set...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    activity_mapping = {
        'SitToStand': 0,
        'Sitting': 1,
        'Standing': 2,
        'TandemWalk': 3,
        'Turn': 4,
        'TurnToSit': 5,
        'Walk': 6
    }
    # Create inverse mapping to get names from indices
    inv_activity_mapping = {v: k for k, v in activity_mapping.items()}
    plt.figure(figsize=(6, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[inv_activity_mapping[i] for i in range(7)],
                yticklabels=[inv_activity_mapping[i] for i in range(7)])
    plt.xlabel('Predicted Activity')
    plt.ylabel('True Activity')
    plt.show()

main()
