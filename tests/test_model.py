from models.train_model import train_model

def test_model_training():
    model, trainer = train_model()
    assert model is not None
    assert trainer is not None
