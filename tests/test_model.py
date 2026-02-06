import os

def test_model_file():
    assert os.path.exists("models/model.pkl")