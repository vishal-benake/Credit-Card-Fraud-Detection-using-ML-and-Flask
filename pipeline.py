from modules.model_training import load_data, train_model, save_model
from modules.model_evaluation import evaluate_model

DATA_PATH = 'data/creditcard.csv'
MODEL_PATH = 'model.pkl'

def run_pipeline():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Training model...")
    model, x_test, y_test = train_model(df)

    print("Evaluating model...")
    evaluate_model(model, x_test, y_test)

    print(f"Saving model to {MODEL_PATH} ...")
    save_model(model, MODEL_PATH)

    print("Pipeline finished successfully.")

if __name__ == "__main__":
    run_pipeline()
