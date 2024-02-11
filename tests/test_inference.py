import unittest

import torch
import torchaudio
from ced_model.feature_extraction_ced import CedFeatureExtractor
from ced_model.modeling_ced import CedForAudioClassification


class TestInference(unittest.TestCase):

    def test_ced_for_audio_classification(self):
        model_id = "mispeech/ced-tiny"
        feature_extractor = CedFeatureExtractor.from_pretrained(model_id)
        model = CedForAudioClassification.from_pretrained(model_id)

        audio, sampling_rate = torchaudio.load("resources/JeD5V5aaaoI_931_932.wav")

        inputs = feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_class_ids = torch.argmax(logits, dim=-1).item()
        predicted_class = model.config.id2label[predicted_class_ids]
        assert predicted_class == "Finger snapping"


if __name__ == "__main__":
    """
    Run the tests:
        python -m unittest tests/*.py
    """

    unittest.main()
