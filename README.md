## WavAutoEncoder 

This repository is an implementation of  autoencoder based on Transformer architecture for self supervised speech representation . 

## Install

```bash
$ pip install wav-autoencoder
```


## Usage

Simple example for using the model

```python
import torch
from wav_autoencoder import WavAutoEncoderConfig ,  WavAutoEncoderModel 

>> config = WavAutoEncoderConfig()
>> model = WavAutoEncoderModel(config)
>> wav_signal = torch.randn(2, 1, 16000)
>> outputs = model(wav_signal)
>> print(outputs.shape)
```

## Todo

## Citations

```bibtex
@misc{
  title  = {WavAutoencoder: A Self-Supervised Framework for Learning Audio Representations},
  author = {Abdou Aziz DIOP},
  year   = {2022}
}
```
