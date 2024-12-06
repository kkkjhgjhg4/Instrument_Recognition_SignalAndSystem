import sys
import os
import pickle
import numpy as np
import librosa
import librosa.display
import pandas as pd
from sklearn.preprocessing import StandardScaler
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QTextEdit,
    QScrollArea,
)
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class InstrumentRecognizerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Instrument Recognizer")
        self.setGeometry(200, 200, 1200, 800)  # Increased size for better layout

        # Main layout
        self.layout = QVBoxLayout()

        # Instruction Label
        self.label = QLabel("Select an audio file to predict the instrument:")
        self.layout.addWidget(self.label)

        # Upload Button
        self.upload_button = QPushButton("Upload Audio File")
        self.upload_button.clicked.connect(self.upload_audio)
        self.layout.addWidget(self.upload_button)

        # Playback Controls Layout
        self.playback_layout = QVBoxLayout()

        # Play Button
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play_audio)
        self.play_button.setEnabled(False)  # Disabled until file is uploaded
        self.playback_layout.addWidget(self.play_button)

        # Pause Button
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_audio)
        self.pause_button.setEnabled(False)  # Disabled until audio is playing
        self.playback_layout.addWidget(self.pause_button)

        self.layout.addLayout(self.playback_layout)

        # Extract Features Button
        self.extract_features_button = QPushButton("Extract and Display Features")
        self.extract_features_button.clicked.connect(self.extract_and_display_features)
        self.extract_features_button.setEnabled(False)
        self.layout.addWidget(self.extract_features_button)

        # Predict Button
        self.predict_button = QPushButton("Predict Instrument")
        self.predict_button.clicked.connect(self.predict_instrument)
        self.predict_button.setEnabled(False)  # Disabled until file is uploaded
        self.layout.addWidget(self.predict_button)

        # Result Label
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.result_label)

        # Scroll Area for Features and Waveform
        self.scroll_area = QScrollArea()
        self.scroll_area_widget = QWidget()
        self.scroll_layout = QVBoxLayout()

        # Text Area for Features
        self.features_text = QTextEdit()
        self.features_text.setReadOnly(True)
        self.features_text.setMinimumHeight(150)
        self.scroll_layout.addWidget(QLabel("Extracted Features:"))
        self.scroll_layout.addWidget(self.features_text)

        # Matplotlib Figure for Waveform
        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)
        self.scroll_layout.addWidget(QLabel("Audio Waveform:"))
        self.scroll_layout.addWidget(self.canvas)

        # Set layout to scroll area
        self.scroll_area_widget.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_area_widget)
        self.scroll_area.setWidgetResizable(True)
        self.layout.addWidget(self.scroll_area)

        self.setLayout(self.layout)

        # Variables to store audio data and features
        self.audio_file = None
        self.svm_model = None
        self.scaler = None
        self.label_encoder = None
        self.y = None  # Audio time series
        self.sr = None  # Sampling rate
        self.features = None  # Extracted features

        # Media Player for Audio Playback
        self.player = QMediaPlayer()
        self.player.stateChanged.connect(self.update_buttons)
        self.player.error.connect(self.handle_playback_error)

        # Load Pre-trained SVM Model and Scaler
        self.load_model()

    def load_model(self):
        """
        Load the pre-trained SVM model, scaler, and label encoder from pickle files.
        """
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

            QMessageBox.information(
                self, "Model Loaded", "SVM model, scaler, and label encoder loaded successfully!"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to load model, scaler, or label encoder: {str(e)}"
            )
            sys.exit()

    def upload_audio(self):
        """
        Open a file dialog to select an audio file and initialize UI elements accordingly.
        """
        # Open file dialog to select audio file
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.aac)",
            options=options,
        )
        if file_path:
            self.audio_file = file_path
            self.label.setText(f"Selected File: {os.path.basename(file_path)}")
            self.play_button.setEnabled(True)
            self.extract_features_button.setEnabled(True)
            self.predict_button.setEnabled(True)
            self.result_label.setText("")
            self.features_text.clear()
            self.figure.clf()
            self.canvas.draw()
            # Load audio for waveform visualization
            self.load_audio()
        else:
            QMessageBox.warning(self, "No File", "Please select a valid audio file.")

    def load_audio(self):
        """
        Load the audio file using librosa and plot its waveform.
        """
        try:
            self.y, self.sr = librosa.load(self.audio_file, sr=44100)
            # Plot waveform
            self.plot_waveform()
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to load audio for waveform: {str(e)}"
            )

    def plot_waveform(self):
        """
        Plot the waveform of the loaded audio file using matplotlib.
        """
        self.figure.clf()  # Clear previous plots
        ax = self.figure.add_subplot(111)
        librosa.display.waveshow(self.y, sr=self.sr, ax=ax)
        ax.set_title("Audio Waveform")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        self.canvas.draw()

    def play_audio(self):
        """
        Play the loaded audio file using QMediaPlayer.
        """
        if self.audio_file:
            url = QUrl.fromLocalFile(self.audio_file)
            if not url.isValid():
                QMessageBox.critical(self, "Invalid URL", "The audio file URL is invalid.")
                return
            content = QMediaContent(url)
            self.player.setMedia(content)
            self.player.play()
        else:
            QMessageBox.warning(self, "No Audio", "No audio file loaded to play.")

    def pause_audio(self):
        """
        Pause the currently playing audio.
        """
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()

    def update_buttons(self, state):
        """
        Update the enabled/disabled state of Play and Pause buttons based on the player's state.
        """
        if state == QMediaPlayer.PlayingState:
            self.play_button.setEnabled(False)
            self.pause_button.setEnabled(True)
        elif state == QMediaPlayer.PausedState:
            self.play_button.setEnabled(True)
            self.pause_button.setEnabled(False)
        elif state == QMediaPlayer.StoppedState:
            self.play_button.setEnabled(True)
            self.pause_button.setEnabled(False)

    def handle_playback_error(self, error):
        """
        Handle errors emitted by QMediaPlayer.
        """
        if error == QMediaPlayer.NoError:
            return
        error_message = self.player.errorString()
        QMessageBox.critical(self, "Playback Error", f"Error playing audio: {error_message}")

    def extract_features(self):
        """
        Extract features from the loaded audio file.
        """
        if not self.audio_file:
            QMessageBox.warning(
                self, "No File", "Please upload an audio file first."
            )
            return None

        try:
            # Extract features
            features = self.extract_features_from_audio(self.audio_file)
            self.features = features
            return features
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to extract features: {str(e)}"
            )
            return None

    def extract_and_display_features(self):
        """
        Extract features and display them in the text area.
        """
        features = self.extract_features()
        if features is not None:
            # Display features in text area
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

    def extract_features_from_audio(self, file_path):
        """
        Extract relevant audio features from the given audio file.

        Parameters:
            file_path (str): Path to the audio file.

        Returns:
            np.ndarray: Extracted features as a 2D NumPy array.
        """
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
        """
        Predict the instrument present in the loaded audio file using the pre-trained SVM model.
        Displays only the top prediction with its normalized probability.
        """
        if not self.audio_file or not self.svm_model or not self.scaler:
            QMessageBox.warning(
                self,
                "Error",
                "Please upload an audio file and ensure the model and scaler are loaded.",
            )
            return

        try:
            # Extract features
            features = self.extract_features()
            if features is None:
                return

            # Scale features
            features_scaled = self.scaler.transform(features)

            # Predict instrument probabilities
            if hasattr(self.svm_model, "predict_proba"):
                probabilities = self.svm_model.predict_proba(features_scaled)[0]
                # Get the index of the highest probability
                top_index = np.argmax(probabilities)
                top_probability = probabilities[top_index]
                # Get the instrument name
                top_instrument = self.label_encoder.inverse_transform([top_index])[0]

                # Format the result
                result_text = f"Predicted Instrument: {top_instrument}\n"
                result_text += f"Confidence: {top_probability:.2f}"

                # Optionally, style the result label
                self.result_label.setStyleSheet("font-weight: bold; color: green;")
                self.result_label.setText(result_text)
            else:
                QMessageBox.warning(
                    self,
                    "Probability Not Available",
                    "The loaded SVM model does not support probability estimates.",
                )
                # Optionally, you can still provide the decision function scores
                # Here, we'll just display the predicted instrument without confidence
                predicted_label = self.svm_model.predict(features_scaled)[0]
                predicted_instrument = self.label_encoder.inverse_transform([predicted_label])[0]
                result_text = f"Predicted Instrument: {predicted_instrument}\n"
                self.result_label.setStyleSheet("font-weight: bold; color: green;")
                self.result_label.setText(result_text)

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to process the audio file: {str(e)}"
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = InstrumentRecognizerApp()
    main_window.show()
    sys.exit(app.exec_())