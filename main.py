from src.preprocessing import load_data, split_data
from src.train_model import train_model, evaluate_model, save_model

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)

if __name__ == "__main__":
    main()
