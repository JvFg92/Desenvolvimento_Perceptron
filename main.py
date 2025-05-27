import training as tr
import data_treatment as dt

if __name__ == "__main__":
    perceptron = tr.Perceptron(lr=0.1)
    weights, error = perceptron.learning(epochs=30)
    print("Learned Weights:", weights)
    print("Final Error:", error)


    X_train, y_train = dt.load_data()
    X_test, y_test = dt.load_data(True)
    dt.plot_data(X_train, y_train, X_test, y_test)