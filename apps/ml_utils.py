# apps/ml_utils.py
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

model_dict = {
    'RandomForest': RandomForestClassifier,
    'LogisticRegression': LogisticRegression
}

def create_model(model_type:str):
    cls = model_dict.get(model_type)
    if cls:
        return cls()
    return None

def dump_model(model):
    return pickle.dumps(model)

def load_model(blob):
    return pickle.loads(blob)
