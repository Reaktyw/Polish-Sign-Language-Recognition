import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import TimeDistributed, Flatten, Input, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import pickle
from keras.models import load_model
from sklearn.metrics import accuracy_score



class PSLR_Model:
    def __init__(self, ls):
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.ls = ls

    def _train_test_split(self, X, Y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    def train_model(self, X, Y, model_name):
        self._train_test_split(X, Y)
        model = Sequential()
        model.add(Input((60, 21, 3)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(64))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(36, activation='softmax'))

        model.compile(optimizer='adamw', loss='categorical_crossentropy', metrics=['accuracy'])

        es = EarlyStopping(patience=7, restore_best_weights=True)
        history = model.fit(self.X_train, self.y_train, epochs=100, batch_size=128, validation_split=0.12, callbacks=[es])

        model.save(model_name)
        with open('label_encoder_1.pkl', 'wb') as f:
            pickle.dump(self.ls, f)

        model = self.load_model(model_name)

        y_pred = model.predict(self.X_test)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(self.y_test, axis=1)

        # Wyświetlenie przykładowych predykcji
        for i in range(len(y_pred)):
            expected = self.ls.inverse_transform([y_true_labels[i]])[0]
            predicted = self.ls.inverse_transform([y_pred_labels[i]])[0]
            confidence = y_pred[i][y_pred_labels[i]] * 100  # Pewność dla przewidywanej litery

            true_index = y_true_labels[i]
            true_letter_confidence = y_pred[i][true_index] * 100  # Pewność dla prawdziwej litery

            if expected != predicted: print(f"Expected: {self.ls.inverse_transform([y_true_labels[i]])[0]}, Predicted: {self.ls.inverse_transform([y_pred_labels[i]])[0]}, Confidence: {confidence:.2f}%, Confidence of real letter: {true_letter_confidence:.2f}%")
            else: print(f"Expected: {self.ls.inverse_transform([y_true_labels[i]])[0]}, Predicted: {self.ls.inverse_transform([y_pred_labels[i]])[0]}, Confidence: {confidence:.2f}%")

        accuracy = accuracy_score(y_true_labels, y_pred_labels)
        print(f"\nDokładność predykcji: {accuracy * 100:.2f}%")

    def load_model(self, model_name):
        return load_model(model_name)