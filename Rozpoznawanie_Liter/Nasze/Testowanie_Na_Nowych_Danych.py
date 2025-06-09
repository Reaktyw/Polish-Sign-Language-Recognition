from keras.models import load_model
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.utils.np_utils import to_categorical

directory = os.path.dirname(__file__)
model = load_model(f'{directory}/model_fine_tuning_04_06_2025_1.keras')

rozpoznawanie_liter = os.path.dirname(directory)
pslr = os.path.dirname(rozpoznawanie_liter)

X = []
y = []
data_path = os.path.join(directory, 'dane')
for letter in os.listdir(data_path):
    curr_path = os.path.join(data_path, letter)
    for num in os.listdir(curr_path):
        curr_patha = os.path.join(curr_path, num)
        data = np.load(curr_patha, allow_pickle=True)

        X.append(data)
        y.append(letter)


X = np.array(X)
y = np.array(y)
print(X.shape)

ls = LabelEncoder()
y_encoded = ls.fit_transform(y)
y_onehot = to_categorical(y_encoded, num_classes=36)


y_pred = model.predict(X)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_onehot, axis=1)

# Wyświetlenie przykładowych predykcji
for i in range(len(y_pred)):
    expected = ls.inverse_transform([y_true_labels[i]])[0]
    predicted = ls.inverse_transform([y_pred_labels[i]])[0]
    confidence = y_pred[i][y_pred_labels[i]] * 100  # Pewność dla przewidywanej litery

    true_index = y_true_labels[i]
    true_letter_confidence = y_pred[i][true_index] * 100  # Pewność dla prawdziwej litery

    if expected != predicted: print(f"Expected: {ls.inverse_transform([y_true_labels[i]])[0]}, Predicted: {ls.inverse_transform([y_pred_labels[i]])[0]}, Confidence: {confidence:.2f}%, Confidence of real letter: {true_letter_confidence:.2f}%")
    else: print(f"Expected: {ls.inverse_transform([y_true_labels[i]])[0]}, Predicted: {ls.inverse_transform([y_pred_labels[i]])[0]}, Confidence: {confidence:.2f}%")
    



from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true_labels, y_pred_labels)
print(f"\nDokładność predykcji: {accuracy * 100:.2f}%")