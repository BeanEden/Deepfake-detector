import gradio as gr
import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification, \
    pipeline
import cv2
import moviepy as mp
import librosa
import numpy as np
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from moviepy.video.io.VideoFileClip import VideoFileClip
import torchaudio


class ModelHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config, num_labels):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)

        return hidden_states, logits_age, logits_gender


# Function to extract frames from video
def extract_frames(video, frame_interval=10):
    cap = cv2.VideoCapture(video)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        frame_count += 1

    cap.release()
    return frames


# Function to extract audio from video (if needed)
def extract_audio(video):
    video = mp.VideoFileClip(video)
    audio = video.audio
    audio_file = "extracted_audio.wav"
    audio.write_audiofile(audio_file)
    return audio_file


# Step 1: Check Deepfake Quality
def check_deepfake_quality(image, video):
    # Load model and processor
    model_name = "prithivMLmods/Deepfake-Quality-Assess-Siglip2"
    model = SiglipForImageClassification.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    labels = model.config.id2label
    """Detect deepfake quality scores for an image or video."""
    print("image", image)
    print("video", video)
    if isinstance(image, Image.Image):  # Image input
        print('image detected, processing...')
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits,
                                                dim=1).squeeze().tolist()

        labels = model.config.id2label
        predictions = {labels[i]: round(probs[i], 3) for i in
                       range(len(probs))}
        print(predictions)


        if predictions["Issue In Deepfake"] > predictions["High Quality Deepfake"]:
            return predictions
        else:
            return ("Proceeding to next step.", "Pas de deepfake évident détecté")

    if isinstance(video, str):  # Video input (file path)
        print('video detected, processing...')
        frames = extract_frames(video)
        max_prob = -1  # Initialize the maximum probability to a very low value
        final_prediction = None  # Initialize the final prediction variable

        for frame in frames:
            inputs = processor(images=frame, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits,
                                                    dim=1).squeeze().tolist()

            # Check the probability of the "Deepfake" class (if it exists in the labels)
            deepfake_prob = probs[model.config.id2label[
                "Deepfake"]] if "Deepfake" in model.config.id2label else max(
                probs)

            # If this frame has a higher probability than the current max, update the final prediction
            if deepfake_prob > max_prob:
                max_prob = deepfake_prob
                predicted_class_index = probs.index(max_prob)
                labels = model.config.id2label
                final_prediction = labels[predicted_class_index]
                if final_prediction == "High Quality Deepfake":
                    final_prediction = "Pas de deepfake évident détecté"
        # Optionally, extract audio if needed
        if (final_prediction) == "Pas de deepfake évident détecté":
            print("Proceeding to next step.")
            return ("Proceeding to next step.", "Pas de deepfake évident détecté")
        else:
            return f"Deepfake detection on frames completed {final_prediction}"


# Step 2: AI-Generated or Deepfake Detection
def check_ai_generated(image, video):
    print("Step 2: Checking if AI-generated or deepfake...")

    print("image", image)
    print("video", video)
    model_name = "prithivMLmods/AI-vs-Deepfake-vs-Real"
    print("model loaded")

    pipe = pipeline("image-classification", model=model_name, device=-1)
    print("pipeline loaded")

    if isinstance(image, Image.Image):  # Image input
        print('image detected, processing...')
        result = pipe(image)

        predicted_label = result[0]['label']
        print(f"AI vs. Deepfake vs. Real Prediction: {predicted_label}")

        if predicted_label in ["AI-generated", "Artificial", "Real"]:
            return ("Done",f"Image deemed {predicted_label}")
        else:
            return ("Proceeding to next step.",f"Image deemed {predicted_label}")

    if isinstance(video, str):  # Video input (file path)
        print('video detected, processing...')
        frames = extract_frames(video)
        final_prediction = None  # Initialize the final prediction variable
        previous_label = None
        consecutive_frames = 0
        for frame in frames:
            result = pipe(frame)

            predicted_label = result[0]['label']
            print(f"AI vs. Deepfake vs. Real Prediction: {predicted_label}")

            if previous_label == predicted_label :
                consecutive_frames +=1

            else:
                previous_label = predicted_label
            if consecutive_frames == 6:
                if predicted_label in ["Deepfake", "AI-generated", "Artificial"]:
                    return ("Done",
                            f"Deepfake detected in AI classification {predicted_label}")
        final_prediction = "Réelle"
        return ("Proceeding to next step.", f"Deepfake detection on frames completed {final_prediction}")


# Step 3: Audio Deepfake Detection
def check_audio(audio):
    print("Step 3: Checking audio authenticity...")
    y, sr = librosa.load(audio, sr=None)
    # Corrected code for extracting MFCC features using librosa
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

    # Placeholder: Replace with actual deepfake audio detection model
    fake_probability = np.random.rand()  # Random value for now
    print(f"Audio deepfake probability: {fake_probability}")

    if fake_probability > 0.5:
        return "Deepfake detected in audio. Exiting..."
    return "Proceeding to next step."


# Step 4: Age and Gender Prediction from Audio (Wav2Vec2)
def check_age_gender(video):
    print("Step 4: Checking age and gender from audio...")

    # Load model from hub
    device = "cpu"
    model_name = "audeering/wav2vec2-large-robust-24-ft-age-gender"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = AgeGenderModel.from_pretrained(model_name)

    def process_func(
            x: np.ndarray,
            sampling_rate: int,
            embeddings: bool = False,
    ) -> np.ndarray:
        r"""Predict age and gender or extract embeddings from raw audio signal."""

        # run through processor to normalize signal
        # always returns a batch, so we just get the first entry
        # then we put it on the device
        y = processor(x, sampling_rate=sampling_rate)
        y = y["input_values"][0]
        y = y.reshape(1, -1)
        y = torch.from_numpy(y).to(device)

        # run through model
        with torch.no_grad():
            y = model(y)
            if embeddings:
                y = y[0]
            else:
                y = torch.hstack([y[1], y[2]])

        # convert to numpy
        y = y.detach().cpu().numpy()

        return y

    def extract_audio_from_video(video_path, target_sr=16000):
        """Extrait l'audio d'une vidéo et le convertit en un tableau NumPy avec un taux d'échantillonnage donné."""

        # Charger la vidéo et extraire l’audio
        video = VideoFileClip(video_path)
        audio_path = "temp_audio.wav"
        video.audio.write_audiofile(audio_path,
                                    fps=target_sr)  # Sauvegarder en 16 kHz

        # Charger l'audio et convertir en tenseur
        waveform, sr = torchaudio.load(
            audio_path)  # Chargement avec TorchAudio
        waveform = torchaudio.transforms.Resample(orig_freq=sr,
                                                  new_freq=target_sr)(
            waveform
        )  # Rééchantillonnage

        # Convertir en NumPy (format attendu par le modèle)
        signal = waveform.numpy()

        return signal, target_sr

    signal, sr = extract_audio_from_video(video)
    output = process_func(signal, sr)
    print("Predicted age and gender:", output)

    return f"Predicted age and gender [Age Female Male Child]: {output}"


# Main Deepfake Detection Function
def deepfake_detector(image, video):
    Quality_check_result = check_deepfake_quality(image,
                                         video)

    if Quality_check_result[0] != "Proceeding to next step.":
        return Quality_check_result

    First_Message = {"First test - Quality": Quality_check_result[1]}
    print(First_Message,"\n Proceeding to next step.")
    if isinstance(video, str):
        print("audio consulted")
        Audio_check = check_age_gender(video)
        First_Message["Audio test"] = Audio_check
    Source_check_result = check_ai_generated(image, video)
    First_Message["Second test - Source"] = Source_check_result[1]
    if Source_check_result[0] != "Proceeding to next step.":
        return First_Message
    print(First_Message, "\n Proceeding to next step.")

    return First_Message


# Create Gradio interface
iface = gr.Interface(
    fn=deepfake_detector,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Video(label="Upload Video"),
    ],
    outputs=gr.JSON(label="Prediction Scores"),
    title="Deepfake Detection",
    description="Upload an image or video to check for deepfake and AI-related scores. For videos, multiple frames will be processed."
)

# Launch the app
if __name__ == "__main__":
    iface.launch(share=True)

