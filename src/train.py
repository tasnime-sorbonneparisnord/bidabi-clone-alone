from src.model import build_model

def train_model(X_train, y_train):
    model = build_model()
    model.fit(X_train, y_train)
    return model
