from datasets import load_dataset
import librosa
import torch
from src import WavAutoEncoderConfig, WavAutoEncoderModel
from transformers import Wav2Vec2FeatureExtractor
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


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


def _collate_fn(batch, max_length=120000):

    batch_size = len(batch)

    input_values = [batch[i]["audio"]["array"] for i in range(batch_size)]
    input_values = [torch.tensor(each_input) for each_input in input_values]
    input_values = pad_sequence(input_values, batch_first=True)[:, :max_length]

    attention_mask = [torch.ones_like(each_input) for each_input in input_values]
    attention_mask = pad_sequence(attention_mask, batch_first=True)[:, :max_length]

    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
    }


if __name__ == "__main__":
    dataset = WavDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=_collate_fn,
    )

    data = next(iter(dataloader))
    print(data)