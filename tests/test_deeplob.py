import torch

from genlob.models.deeplob import DeepLOBStyleClassifier


def test_deeplob_style_classifier_forward_shape():
    model = DeepLOBStyleClassifier(input_dim=40, conv_channels=8, inception_channels=4, lstm_hidden=8)
    x = torch.randn(3, 100, 40)
    logits = model(x)
    assert logits.shape == (3, 3)


def test_deeplob_style_classifier_rejects_wrong_feature_dim():
    model = DeepLOBStyleClassifier(input_dim=40, conv_channels=8, inception_channels=4, lstm_hidden=8)
    x = torch.randn(3, 100, 41)
    try:
        model(x)
    except ValueError as exc:
        assert "input_dim=40" in str(exc)
    else:
        raise AssertionError("expected ValueError for wrong feature dimension")
