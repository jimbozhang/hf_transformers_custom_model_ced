---
license: apache-2.0
datasets:
- AudioSet
metrics:
- mAP
pipeline_tag: audio-classification
---

# Pretrained CED on HuggingFace

CED are simple ViT-Transformer-based models for audio tagging. Notable differences from other available models include:
1. Simplification for finetuning: Batchnormalization of Mel-Spectrograms. During finetuning one does not need to first compute mean/variance over the dataset, which is common for AST.
1. Support for variable length inputs. Most other models use a static time-frequency position embedding, which hinders the model's generalization to segments shorter than 10s. Many previous transformers simply pad their input to 10s in order to avoid the performance impact, which in turn slows down training/inference drastically.
1. Training/Inference speedup: 64-dimensional mel-filterbanks and 16x16 patches without overlap, leading to 248 patches from a 10s spectrogram. In comparison, AST uses 128 mel-filterbanks with 16x16 (10x10 overlap) convolution, leading to 1212 patches during training/inference. CED-Tiny runs on a common CPU as fast as a comparable MobileNetV3.
1. Performance: CED with 10M parameters outperforms the majority of previous approaches (~80M).

The abstract from the paper is the following:

Augmentation and knowledge distillation (KD) are well-established techniques employed in audio classification tasks, aimed at enhancing performance and reducing model sizes on the widely recognized Audioset (AS) benchmark. Although both techniques are effective individually, their combined use, called consistent teaching, hasn't been explored before. This paper proposes CED, a simple training framework that distils student models from large teacher ensembles with consistent teaching. To achieve this, CED efficiently stores logits as well as the augmentation methods on disk, making it scalable to large-scale datasets. Central to CED's efficacy is its label-free nature, meaning that only the stored logits are used for the optimization of a student model only requiring 0.3\% additional disk space for AS. The study trains various transformer-based models, including a 10M parameter model achieving a 49.0 mean average precision (mAP) on AS.

### Model Sources

- **Original Repository:** https://github.com/RicherMans/CED
- **Paper:** [CED: Consistent ensemble distillation for audio tagging](https://arxiv.org/abs/2308.11957)
- **Demo:** https://huggingface.co/spaces/mispeech/ced-base

## Uses

```bash
pip install -r requirements.txt
```

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
