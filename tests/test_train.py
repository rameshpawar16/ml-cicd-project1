import os 
import joblib
from src.train import train_model

def test_model_traning():
    train_model()
    
    assert os.path.exists("models/model.pkl"),"model file not found"
    model = joblib.load("models/model.pkl")
    assert hasattr(model,'predict'),'model does not have a predict method'
    
    
    
    