import training as tr
import data_treatment as dt

if __name__ == "__main__":
   
    print("Starting training...")
    weights, error = tr.learning()
    print("Training completed.")
    print("Training error:", error)
    print("Final weights:", weights)

    X_train, y_train = dt.generate_linear_data()
    X_test, y_test = dt.generate_linear_data(True)
    dt.plot_data(X_train, y_train, X_test, y_test)