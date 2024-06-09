# Pretrained CED on 🤗 Hugging Face

[![arxiv](https://img.shields.io/badge/arXiv-2308.11957-b31b1b)](https://arxiv.org/abs/2308.11957)
[![dasheng](https://img.shields.io/badge/CED-tiny-ffcc66)](https://huggingface.co/mispeech/ced-tiny)
[![dasheng](https://img.shields.io/badge/CED-mini-ffcc66)](https://huggingface.co/mispeech/ced-mini)
[![dasheng](https://img.shields.io/badge/CED-small-ffcc66)](https://huggingface.co/mispeech/ced-small)
[![dasheng](https://img.shields.io/badge/CED-base-ffcc66)](https://huggingface.co/mispeech/ced-base)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

CED are simple ViT-Transformer-based models for audio tagging.

- **Model Cards:** https://huggingface.co/models?search=mispeech%2Fced
- **Original Repository:** https://github.com/RicherMans/CED
- **Paper:** [CED: Consistent ensemble distillation for audio tagging](https://arxiv.org/abs/2308.11957)
- **Demo:** https://huggingface.co/spaces/mispeech/ced-base

## Install
```bash
cd hf_transformers_custom_model_ced
pip install .
```

## Inference

```python
>>> from ced_model.feature_extraction_ced import CedFeatureExtractor
>>> from ced_model.modeling_ced import CedForAudioClassification

>>> model_name = "mispeech/ced-mini"
>>> feature_extractor = CedFeatureExtractor.from_pretrained(model_name)
>>> model = CedForAudioClassification.from_pretrained(model_name)

>>> import torchaudio
>>> audio, sampling_rate = torchaudio.load("resources/JeD5V5aaaoI_931_932.wav")
>>> assert sampling_rate == 16000
>>> inputs = feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt")

>>> import torch
>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = torch.argmax(logits, dim=-1).item()
>>> model.config.id2label[predicted_class_id]
'Finger snapping'
```

## Fine-tuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jimbozhang/hf_transformers_custom_model_ced/blob/main/example_finetune_esc50.ipynb)

[`example_finetune_esc50.ipynb`](https://github.com/jimbozhang/hf_transformers_custom_model_ced/blob/main/example_finetune_esc50.ipynb) demonstrates how to train a linear head on the ESC-50 dataset with the CED encoder frozen.
