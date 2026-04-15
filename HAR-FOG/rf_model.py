import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np
import scipy.io

def main():
    '''Load data'''
    X_train,y_train,idTrain= load_data()
    '''Create training, validation, and test data'''
    trainInd,valInd,testInd = train_test(idTrain)
    X_val, y_val = X_train[valInd,:,:], y_train[valInd,:]
    '''Random forest'''
    rf_model = evaluate_each_sensor(X_train[trainInd,:,:], y_train[trainInd,:], X_train[testInd,:,:], y_train[testInd,:])

def load_data():
    fileName = 'data.mat'
    #fileName = 'rempark_segmented_32Hz_2s_50overlap.mat'
    mat = scipy.io.loadmat(fileName)
    dataTrain,labelTrain,idTrain = mat['data']['data'][0][0],mat['data']['activity'][0][0],mat['data']['subject'][0][0]
    return (dataTrain,labelTrain,idTrain)

def train_test(id):
    trainPd = range(1, 40)
    valPd = range(40, 50)
    testPd = range(50, 60)
    trainInd = np.concatenate([np.where(id == pd)[0] for pd in trainPd])
    valInd = np.concatenate([np.where(id == pd)[0] for pd in valPd])
    testInd = np.concatenate([np.where(id == pd)[0] for pd in testPd])
    return trainInd, valInd, testInd

def evaluate_each_sensor(X_train, y_train, X_test, y_test, output_csv='sensor_evaluation_gyro.csv'):
    sensor_map = {
        'LowerBack': list(range(3, 6)),
        'R_Wrist': list(range(9, 12)),
        'L_Wrist': list(range(15, 18)),
        'R_MidLatThigh': list(range(21, 24)),
        'L_MidLatThigh': list(range(27, 30)),
        'R_LatShank': list(range(33, 36)),
        'L_LatShank': list(range(39, 42)),
        'R_DorsalFoot': list(range(45, 48)),
        'L_DorsalFoot': list(range(51, 54)),
        'R_Ankle': list(range(57, 60)),
        'L_Ankle': list(range(63, 66)),
        'Xiphoid': list(range(69, 72)),
        'Forehead': list(range(75, 78)),
    }

    results = []

    for sensor_name, channel_indices in sensor_map.items():
        print(f"Evaluating sensor: {sensor_name}")

        X_train_sensor = X_train[:, :, channel_indices]
        y_train_sensor = y_train

        X_test_sensor = X_test[:, :, channel_indices]
        y_test_sensor = y_test

        # Reshape for RF input
        X_train_flat = X_train_sensor.reshape(X_train_sensor.shape[0], -1)
        X_test_flat = X_test_sensor.reshape(X_test_sensor.shape[0], -1)

        y_train_flat = y_train_sensor.ravel()
        y_test_flat = y_test_sensor.ravel()

        # Train RF
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_flat, y_train_flat)
        y_pred = rf.predict(X_test_flat)

        acc = accuracy_score(y_test_flat, y_pred)
        f1_macro = f1_score(y_test_flat, y_pred, average='macro')
        f1_weight = f1_score(y_test_flat, y_pred, average='weighted')

        results.append({
            'sensor_location': sensor_name,
            'accuracy': acc,
            'f_macro': f1_macro,
            'f_weight': f1_weight
        })
        print(results)

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nSensor evaluation completed. Results saved to {output_csv}.")

main()