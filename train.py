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


def compute_mask_indices(
    attention_mask: torch.Tensor,
    mask_prob: float,
    mask_length: int,
    attention_window: int,
    min_masks: int = 0,
) -> torch.Tensor:
    batch_size, sequence_length = attention_mask.shape
    attention_mask = attention_mask.view(batch_size, sequence_length)

    # compute how many tokens we want to mask
    num_masked_tokens = int(mask_prob * sequence_length)
    num_masked_tokens = max(num_masked_tokens, min_masks)

    # make sure num masked indices <= sequence_length
    if num_masked_tokens > sequence_length:
        num_masked_tokens = sequence_length

    # SpecAugment mask to fill
    spec_aug_mask = torch.zeros((batch_size, sequence_length), dtype=torch.bool, device=attention_mask.device)

    # uniform distribution to sample from, make sure that offset samples are < sequence_length
    uniform_dist = torch.ones((batch_size, sequence_length - (mask_length - 1)), device=attention_mask.device)

    # get random indices to mask
    spec_aug_mask_idxs = torch.multinomial(uniform_dist, num_masked_tokens)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = spec_aug_mask_idxs.unsqueeze(-1).expand((batch_size, num_masked_tokens, mask_length))
    spec_aug_mask_idxs = spec_aug_mask_idxs.contiguous().view(batch_size, num_masked_tokens * mask_length)

    offsets = torch.arange(mask_length, device=attention_mask.device).unsqueeze(0).expand((num_masked_tokens, mask_length))
    offsets = offsets.contiguous().view(-1)

    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # scatter indices to mask
    spec_aug_mask = spec_aug_mask.scatter(1, spec_aug_mask_idxs, True)

    return spec_aug_mask


if __name__=="__main__":
    input = torch.rand(1, 16000)
    mask = compute_mask_indices(input, 0.2, 10, 10)
    input_masked = input.masked_fill(mask, 0)
    print(input_masked)

if __name__ == "__main__":
    input_values = torch.rand(2, 1, 120000)
    encoder = Encoder()
    print(encoder(input_values).shape)
