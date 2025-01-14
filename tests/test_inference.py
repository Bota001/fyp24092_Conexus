from scripts.inference import load_model

def test_load_model():
    classifier = load_model()
    assert classifier is not None

def test_prediction():
    classifier = load_model()
    prediction = classifier(["Free gift card for you! Click now!"])
    assert prediction[0]['label'] in ["LABEL_0", "LABEL_1"]
