from src.data_loader import load_data
from src.train import train_model
from src.evaluate import evaluate_model
import joblib

def main():
    X_train, X_test, y_train, y_test = load_data('data/raw.csv')

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    joblib.dump(model, 'model.pkl')
    print('Model saved successfully')

if __name__ == '__main__':
    main()
