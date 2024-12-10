import sys
import os
import pickle
import numpy as np
import pandas as pd
import librosa
import librosa.display
import torch
import torch.nn as nn
import torch.nn.functional as F
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QScrollArea,
    QHBoxLayout,
    QComboBox,
    QTextEdit  # Add this import
)
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class InstrumentCNN(nn.Module):
    def __init__(self, num_classes):
        super(InstrumentCNN, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.dropout2 = nn.Dropout(0.25)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.dropout3 = nn.Dropout(0.25)

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.fc1 = nn.Linear(256, 1024)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool3(x)
        x = self.dropout3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))

        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)

        return x


def nn_preprocess_audio(file_path, target_sr=22050, n_fft=1024, hop_length=512, n_mels=128, max_length=3.0):
    try:
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

        max_length_samples = int(target_sr * max_length)
        if len(y) > max_length_samples:
            y = y[:max_length_samples]
        else:
            pad_width = max_length_samples - len(y)
            y = np.pad(y, (0, pad_width), mode="constant")

        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=target_sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=target_sr / 2
        )
        log_mel_spectrogram = np.log(mel_spectrogram + 1e-9)
        return log_mel_spectrogram
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def svm_preprocess_audio(file_path):

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


class InstrumentRecognizerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Instrument Recognizer")
        self.setGeometry(200, 200, 1200, 800)

        # Main layout
        self.layout = QHBoxLayout()

        # Left layout (Waveform and Controls)
        left_layout = QVBoxLayout()

        # Instruction Label
        self.label = QLabel("Select an audio file to predict the instrument:")
        left_layout.addWidget(self.label)

        # Upload Button
        self.upload_button = QPushButton("Upload Audio File")
        self.upload_button.clicked.connect(self.upload_audio)
        left_layout.addWidget(self.upload_button)

        # Model Selection
        model_selection_layout = QHBoxLayout()
        self.model_selector = QComboBox()
        self.model_selector.addItems(["SVM", "Neural Network"])
        model_selection_layout.addWidget(QLabel("Choose Model:"))
        model_selection_layout.addWidget(self.model_selector)
        left_layout.addLayout(model_selection_layout)

        # Predict Button
        self.predict_button = QPushButton("Predict Instrument")
        self.predict_button.clicked.connect(self.predict_instrument)
        self.predict_button.setEnabled(False)
        left_layout.addWidget(self.predict_button)

        # Feature Extraction Button
        self.feature_button = QPushButton("Feature Extraction")
        self.feature_button.clicked.connect(self.extract_and_display_features)
        self.feature_button.setEnabled(False)
        left_layout.addWidget(self.feature_button)

        # Result Label
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignLeft)
        left_layout.addWidget(self.result_label)

        # Waveform Plot
        self.figure = Figure(figsize=(10, 4))
        self.canvas = FigureCanvas(self.figure)
        left_layout.addWidget(self.canvas)

        # Add left layout to main layout
        self.layout.addLayout(left_layout)

        # Right layout (Features)
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Extracted Features (SVM):"))

        self.features_text = QTextEdit()
        self.features_text.setReadOnly(True)
        self.features_text.setMinimumHeight(400)
        self.features_text.setVisible(False)  # Hide by default
        right_layout.addWidget(self.features_text)

        # Add right layout to main layout
        self.layout.addLayout(right_layout)

        # Set main layout
        self.setLayout(self.layout)

        # Variables
        self.audio_file = None
        self.svm_model = None
        self.scaler = None
        self.label_encoder = None
        self.nn_model = None
        self.nn_label_encoder = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nn_model_loaded = False

        # Load Models
        self.load_svm_model()

    def upload_audio(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.mp3 *.flac *.aac)", options=options)
        if file_path:
            self.audio_file = file_path
            self.label.setText(f"Selected File: {os.path.basename(file_path)}")
            self.predict_button.setEnabled(True)
            self.feature_button.setEnabled(True)
            self.result_label.setText("")
            self.figure.clear()
            self.canvas.draw()
        else:
            QMessageBox.warning(self, "No File", "Please select a valid audio file.")

    def extract_and_display_features(self):
        """
        Extract features and display them in the features_text widget.
        Only available when SVM is selected.
        """
        if self.audio_file is None:
            QMessageBox.warning(self, "Error", "Please upload an audio file first.")
            return

        selected_model = self.model_selector.currentText()
        if selected_model != "SVM":
            QMessageBox.warning(self, "Error", "Feature extraction is only available for the SVM model.")
            return

        # Extract features for SVM
        features = svm_preprocess_audio(self.audio_file)
        if features is None:
            QMessageBox.critical(self, "Error", "Failed to extract features.")
            return

        # Display extracted features
        self.features_text.setVisible(True)
        feature_names = [
            "RMS",
            "Spectral Centroid",
            "Spectral Bandwidth",
            "Spectral Rolloff",
            "Zero Crossing Rate",
        ]
        mfcc_names = [f"MFCC_{i + 1}" for i in range(20)]
        all_feature_names = feature_names + mfcc_names
        feature_values = features.flatten()

        feature_dict = dict(zip(all_feature_names, feature_values))
        feature_df = pd.DataFrame(
            list(feature_dict.items()), columns=["Feature", "Value"]
        )
        self.features_text.setText(feature_df.to_string(index=False))

    def load_svm_model(self):
        try:
            with open("pickle_model.pkl", "rb") as file:
                self.svm_model = pickle.load(file)
            with open("scaler.pkl", "rb") as file:
                self.scaler = pickle.load(file)
            with open("label_encoder.pkl", "rb") as file:
                self.label_encoder = pickle.load(file)
            QMessageBox.information(self, "Model Loaded", "SVM model loaded successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load SVM model or scaler: {str(e)}")

    def load_nn_model(self):
        if self.nn_model_loaded:
            return
        try:
            with open("nn_label_encoder.pkl", "rb") as file:
                self.nn_label_encoder = pickle.load(file)
            num_classes = len(self.nn_label_encoder.classes_)
            self.nn_model = InstrumentCNN(num_classes=num_classes)
            self.nn_model.load_state_dict(torch.load("final_model.pth", map_location=self.device))
            self.nn_model = self.nn_model.to(self.device)
            self.nn_model.eval()
            self.nn_model_loaded = True
            QMessageBox.information(self, "NN Model Loaded", "Neural Network model loaded successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load Neural Network model: {str(e)}")


    def predict_instrument(self):
        if not self.audio_file:
            QMessageBox.warning(self, "Error", "Please upload an audio file first.")
            return

        selected_model = self.model_selector.currentText()
        if selected_model == "SVM":
            self.predict_with_svm()
        elif selected_model == "Neural Network":
            self.predict_with_nn()

    def predict_with_svm(self):
        if self.svm_model is None or self.scaler is None or self.label_encoder is None:
            QMessageBox.warning(self, "Error", "SVM model, scaler, or label encoder not loaded.")
            return

        features = svm_preprocess_audio(self.audio_file)
        if features is None:
            return
        features_scaled = self.scaler.transform(features)

        try:
            # Predict probabilities
            probabilities = self.svm_model.predict_proba(features_scaled)[0]
            top_index = np.argmax(probabilities)
            top_instrument = self.label_encoder.inverse_transform([top_index])[0]
            confidence = probabilities[top_index]
            result_text = f"Predicted Instrument: {top_instrument}\nConfidence: {confidence:.2f}"
            self.result_label.setStyleSheet("font-weight: bold; color: green;")
            self.result_label.setText(result_text)

            # Display waveform
            self.plot_waveform()

            # Display features
            feature_names = [
                "RMS",
                "Spectral Centroid",
                "Spectral Bandwidth",
                "Spectral Rolloff",
                "Zero Crossing Rate",
            ]
            mfcc_names = [f"MFCC_{i+1}" for i in range(20)]
            all_feature_names = feature_names + mfcc_names
            feature_values = features.flatten()

            feature_dict = dict(zip(all_feature_names, feature_values))
            feature_df = pd.DataFrame(
                list(feature_dict.items()), columns=["Feature", "Value"]
            )
            self.features_text.setText(feature_df.to_string(index=False))

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process with SVM: {str(e)}")

    def predict_with_nn(self):
        self.load_nn_model()
        if self.nn_model is None or self.nn_label_encoder is None:
            QMessageBox.warning(self, "Error", "Neural Network model or label encoder not loaded.")
            return

        mel_spec = nn_preprocess_audio(self.audio_file)
        if mel_spec is None:
            QMessageBox.critical(self, "Error", "Failed to preprocess audio for Neural Network.")
            return

        tensor_input = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.nn_model(tensor_input)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]

        threshold = 0.5
        predicted_indices = np.where(probs >= threshold)[0]

        if len(predicted_indices) == 0:
            # No instruments above threshold, pick the highest probability
            top_index = np.argmax(probs)
            top_instrument = self.nn_label_encoder.inverse_transform([top_index])[0]
            confidence = probs[top_index]
            result_text = f"Predicted Instrument (NN): {top_instrument}\nConfidence: {confidence:.2f}"
        else:
            predicted_instruments = self.nn_label_encoder.inverse_transform(predicted_indices)
            result_text = "Predicted Instruments (NN):\n"
            for i, inst_idx in enumerate(predicted_indices):
                inst_name = predicted_instruments[i]
                confidence = probs[inst_idx]
                result_text += f"{inst_name}: {confidence:.2f}\n"

        self.result_label.setStyleSheet("font-weight: bold; color: blue;")
        self.result_label.setText(result_text)

        # Display both waveform and mel spectrogram for NN predictions
        self.plot_waveform_and_mel(mel_spec)

    def plot_waveform(self):
        """
        Plot the waveform of the audio file.
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        y, sr = librosa.load(self.audio_file, sr=44100)
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title("Audio Waveform")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        self.canvas.draw()

    def plot_waveform_and_mel(self, mel_spec):
        """
        Plot both the waveform and mel spectrogram.
        """
        self.figure.clear()
        y, sr = librosa.load(self.audio_file, sr=22050)
        ax1 = self.figure.add_subplot(121)
        librosa.display.waveshow(y, sr=sr, ax=ax1)
        ax1.set_title("Audio Waveform")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")

        ax2 = self.figure.add_subplot(122)
        librosa.display.specshow(mel_spec, sr=22050, hop_length=512, x_axis="time", y_axis="mel", cmap="magma", ax=ax2)
        ax2.set_title("Mel Spectrogram (Log Scale)")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Mel Bands")
        self.canvas.draw()




if __name__ == "__main__":
        app = QApplication(sys.argv)
        main_window = InstrumentRecognizerApp()
        main_window.show()
        sys.exit(app.exec_())