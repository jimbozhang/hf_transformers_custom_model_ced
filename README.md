# Pretrained CED on Hugging Face

CED are simple ViT-Transformer-based models for audio tagging.

- **Model Cards:** https://huggingface.co/models?search=mispeech%2Fced
- **Original Repository:** https://github.com/RicherMans/CED
- **Paper:** [CED: Consistent ensemble distillation for audio tagging](https://arxiv.org/abs/2308.11957)
- **Demo:** https://huggingface.co/spaces/mispeech/ced-base

## Install
```bash
pip install -r requirements.txt
```

## Inference

```python
>>> from ced_model.feature_extraction_ced import CedFeatureExtractor
>>> from ced_model.modeling_ced import CedForAudioClassification

>>> model_id = "mispeech/ced-tiny"  # or "mispeech/ced-mini", "mispeech/ced-small", "mispeech/ced-base"
>>> feature_extractor = CedFeatureExtractor.from_pretrained(model_id)
>>> model = CedForAudioClassification.from_pretrained(model_id)

>>> import torchaudio
>>> audio, sampling_rate = torchaudio.load("resources/JeD5V5aaaoI_931_932.wav")

>>> inputs = feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt")
>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> import torch
>>> predicted_class_ids = torch.argmax(logits, dim=-1).item()
>>> model.config.id2label[predicted_class_ids]
'Finger snapping'
```

## Finetuning

[`example_finetune_esc50.ipynb`](https://github.com/jimbozhang/hf_transformers_custom_model_ced/blob/main/example_finetune_esc50.ipynb) demonstrates how to train a linear head on the ESC-50 dataset with the CED encoder frozen.
