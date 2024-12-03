import sys
import os
import pickle
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
)


class InstrumentRecognizerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Instrument Recognizer")
        self.setGeometry(100, 100, 400, 300)

        # Layout and Widgets
        self.layout = QVBoxLayout()

        self.label = QLabel("Select an audio file to predict the instrument:")
        self.layout.addWidget(self.label)

        self.upload_button = QPushButton("Upload Audio File")
        self.upload_button.clicked.connect(self.upload_audio)
        self.layout.addWidget(self.upload_button)

        self.predict_button = QPushButton("Predict Instrument")
        self.predict_button.clicked.connect(self.predict_instrument)
        self.predict_button.setEnabled(False)  # Disabled until file is uploaded
        self.layout.addWidget(self.predict_button)

        self.result_label = QLabel("")
        self.layout.addWidget(self.result_label)

        self.setLayout(self.layout)

        # Variables
        self.audio_file = None
        self.svm_model = None
        self.scaler = None
        self.label_encoder = None

        # Load Pre-trained SVM Model and Scaler
        self.load_model()

    def load_model(self):
        try:
            # Load the pre-trained SVM model
            with open("pickle_model.pkl", "rb") as file:
                self.svm_model = pickle.load(file)
            # Load the StandardScaler used during training
            with open("scaler.pkl", "rb") as file:
                self.scaler = pickle.load(file)
            # Load the label encoder
            with open("label_encoder.pkl", "rb") as file:
                self.label_encoder = pickle.load(file)

            QMessageBox.information(self, "Model Loaded", "SVM model and scaler loaded successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model or scaler: {str(e)}")
            sys.exit()

    def upload_audio(self):
        # Open file dialog to select audio file
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", "Audio Files (*.wav *.mp3)", options=options
        )
        if file_path:
            self.audio_file = file_path
            self.label.setText(f"Selected File: {os.path.basename(file_path)}")
            self.predict_button.setEnabled(True)
        else:
            QMessageBox.warning(self, "No File", "Please select a valid audio file.")

    def extract_features(self, file_path):
        # Load the audio file
        y, sr = librosa.load(file_path, sr=44100)

        # Extract features
        rms = np.mean(librosa.feature.rms(y=y))
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_means = [np.mean(e) for e in mfcc]

        # Combine all features into a single array
        features = [rms, spec_cent, spec_bw, rolloff, zcr] + mfcc_means
        return np.array(features).reshape(1, -1)

    def predict_instrument(self):
        if not self.audio_file or not self.svm_model or not self.scaler:
            QMessageBox.warning(self, "Error",
                                "Please upload an audio file and ensure the model and scaler are loaded.")
            return

        try:
            # Extract features
            features = self.extract_features(self.audio_file)

            # Scale features
            features_scaled = self.scaler.transform(features)

            # Predict instrument
            predicted_label = self.svm_model.predict(features_scaled)[0]
            predicted_instrument = self.label_encoder.inverse_transform([predicted_label])[0]  # Map to instrument name

            # Confidence scores
            if hasattr(self.svm_model, "decision_function"):
                confidences = self.svm_model.decision_function(features_scaled)
            elif hasattr(self.svm_model, "predict_proba"):
                confidences = self.svm_model.predict_proba(features_scaled)

            # Format results
            result_text = f"Predicted Instrument: {predicted_instrument}\nConfidence Scores:\n"
            for i, confidence in enumerate(confidences[0]):
                instrument_name = self.label_encoder.inverse_transform([i])[0]
                result_text += f"{instrument_name}: {confidence:.2f}\n"

            self.result_label.setText(result_text)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process the audio file: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = InstrumentRecognizerApp()
    main_window.show()
    sys.exit(app.exec_())