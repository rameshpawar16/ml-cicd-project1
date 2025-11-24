import joblib
import numpy as np

def evaluate_model():
    # load the trained model
    model  = joblib.load('models/model.pkl')
    test_x = np.array([[6],[7],[8]])
    preds = model.predict(test_x)
    print('Prediction:',preds)
    return preds

if __name__ == "__main__":
    evaluate_model()