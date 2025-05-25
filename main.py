import training.py as tr
import data_treatment as dt

if __name__ == "__main__":
   
    X_train, y_train, X_test, y_test = dt.load_data()
    dt.plot_data(X_train, y_train, X_test, y_test)

    print("Starting training...")
    weights, error = tr.learning(X_train, y_train)
    print("Training completed.")
    print("Training error:", error)
    print("Final weights:", weights)