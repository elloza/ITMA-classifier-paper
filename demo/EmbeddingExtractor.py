'''

EmbeddingExtractor.py

This file contains a class for extracting embeddings from a given music file (wav or mp3)

The available embeddings are:

    - Jukemir (Requeries 15 GB of RAM and 13 VRAM) 4800 dimensions
    - Mule 1728 dimensions
    - MERT 1024 dimensions

In the constructor of the class, the model to be used for the embedding extraction is loaded.

The embedding is returned as a numpy array.

'''

import librosa
import numpy as np
import requests
import os

import jukemirlib

from scooch import Config
import click

from transformers import Wav2Vec2FeatureExtractor, AutoModel
import torch
import torchaudio
from torch import nn
import torchaudio.transforms as T
from datasets import load_dataset
import subprocess

class EmbeddingExtractor:
    """
    This class is used to extract embeddings from a given music file (wav or mp3)
    """
    def __init__(self, embedding_model="jukemir"):

        # Load the embedding model
        if embedding_model == "jukemir":
            self.embedding_model = Jukemir()

        elif embedding_model == "mule":
            
            self.embedding_model = Mule()

        elif embedding_model == "mert":
            self.embedding_model = MERT()

    def get_embedding(self, music_file):
        # Extract the embedding
        return self.embedding_model.get_embedding(music_file)


# Base class for the embedding models
class EmbeddingModel:
    def __init__(self):
        pass

    # get_embedding is the method that must be implemented by the child classes
    def get_embedding(self, file):
        pass


# Jukemir embedding model
class Jukemir(EmbeddingModel):
    def __init__(self):
        super().__init__()

    def get_embedding(self, file):
        # Extract the embedding
        audio = jukemirlib.load_audio(file)
        # meanpool (if not it is a embeeding of 8192x4800)
        reps = jukemirlib.extract(audio, layers=[36], meanpool=True)
        embedding = np.array(reps[36])
        return embedding
    

# Mule embedding model
class Mule(EmbeddingModel):
    def __init__(self):
        super().__init__()
        from mule import Analysis
        # Get home directory
        home = os.path.expanduser("~")
        # join with /.cache/music-audio-representations/supporting_data/configs/mule_embedding_timeline.yml
        config_path = os.path.join(home, ".cache/music-audio-representations/supporting_data/configs/mule_embedding_average.yml")
        self.cfg = Config(config_path)

        # {'Analysis': {'source_feature': {'AudioWaveform': {'input_file': {'AudioFile': {'sample_rate': 44100}}, 'sample_rate': 16000}}, 'feature_transforms': [{'MelSpectrogram': {'n_fft': 2048, 'hop_length': 160, 'win_length': 400, 'window': 'hann', 'n_mels': 96, 'fmin': 0.0, 'fmax': 8000.0, 'norm': 2.0, 'mag_compression': 'log10_nonneg', 'htk': True, 'power': 2.0, 'mag_range': None, 'extractor': {'BlockExtractor': None}}}, {'EmbeddingFeature': {'model_location': './supporting_data/model/', 'extractor': {'SliceExtractor': {'look_forward': 150, 'look_backward': 150, 'hop': 200}}}}]}}
        # Change 'EmbeddingFeature': {'model_location': './supporting_data/model/'
        # to the current path of the models config file
        print(os.path.join(home, ".cache/music-audio-representations/supporting_data/model/"))
        self.cfg['Analysis']['feature_transforms'][1]['EmbeddingFeature']['model_location'] = os.path.join(home, ".cache/music-audio-representations/supporting_data/model/")

        print(self.cfg)
        self.analysis = Analysis(self.cfg)

    def get_embedding(self, file):
        # Extract the embedding

        # Check if the file is longer than 30 seconds
        # If it is, extract the embedding from the first 30 seconds
        # If not, extract the embedding from the whole file
        audio, sr = librosa.load(file)

        if librosa.get_duration(audio, sr) > 30:
            # Cut with ffmpeg in a temporary folder in the same directory
            # as the file
            # Get the directory of the file
            directory = os.path.dirname(file)

            # Get the name of the file
            filename = os.path.basename(file)

            subprocess.run(["ffmpeg", "-i", file, "-ss", "00:00:00", "-t", "00:00:30", "-c:a", "copy", os.path.join(directory, "temp_" + filename), "-y"])
            new_file = os.path.join(directory, "temp_" + filename)

            feat = self.analysis.analyze(new_file)
            array = np.array(feat.data.tolist())
            mean_pooled_embedding = np.mean(array, axis=1)

            os.remove(new_file)

            return mean_pooled_embedding
        else:
            feat = self.analysis.analyze(file)
            array = np.array(feat.data.tolist())
            mean_pooled_embedding = np.mean(array, axis=1)
            return mean_pooled_embedding

    # Destroy resources when done
    def __del__(self):
        del self.analysis
    

# MERT embedding model
class MERT(EmbeddingModel):
    def __init__(self):
        super().__init__()
        # Loading the model and processor
        self.model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Pass the model to the cuda GPU if available
        self.model.to(self.device)
        print("Model loaded to the device", self.device)

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "m-a-p/MERT-v1-330M", trust_remote_code=True
        )

    def get_embedding(self, audio_file_path, layer=17, resample_rate=24000):

        # Extract the embedding
        # Load and preprocess the audio file
        waveform, sampling_rate = torchaudio.load(audio_file_path, normalize=True)
        if resample_rate != sampling_rate:
            resampler = T.Resample(sampling_rate, resample_rate)
            waveform = resampler(waveform)

        # Load and preprocess the audio file
        resample_rate = self.processor.sampling_rate

        # Crop the audio to 30 seconds if it is longer than that
        target_duration = 30  # seconds
        target_num_frames = int(target_duration * resample_rate)
        if waveform.size(1) > target_num_frames:
            waveform = waveform[:, :target_num_frames]

        waveform = waveform.mean(dim=0, keepdim=True)

        # Flatten the mono_waveform tensor to a single dimension
        waveform = waveform.view(-1)

        # Extract features using the Wav2Vec2 processor
        inputs = self.processor(
            waveform.numpy(), sampling_rate=resample_rate, return_tensors="pt"
        )

        inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # take a look at the output shape, there are 25 layers of representation
        # each layer performs differently in different downstream tasks, you should choose empirically
        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()

        # Promedio de la dimension temporal
        time_reduced_hidden_states = all_layer_hidden_states.mean(-2)

        # Numpy array
        embedding = time_reduced_hidden_states[layer, :].cpu().detach().numpy()

        return embedding
