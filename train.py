from datasets import load_dataset
import librosa
import torch
from src import WavAutoEncoderConfig, WavAutoEncoderModel
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Config
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn


class WavDataset(Dataset):
    def __init__(
        self,
        dataset_name: str = "patrickvonplaten/librispeech_asr_dummy",
        feature_extractor_name: str = "facebook/wav2vec2-base-960h",
        dataset_split: str = "validation",
        dataset_subset: str = "clean",
    ):
        self.dataset = load_dataset(dataset_name, dataset_subset, split=dataset_split)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            feature_extractor_name
        )
        self.dataset = self.dataset.remove_columns(
            ["file", "text", "speaker_id", "chapter_id", "id"]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        audio = item["audio"]["array"]
        return audio



