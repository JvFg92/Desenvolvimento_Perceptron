import training as tr
import data_treatment as dt

if __name__ == "__main__":
    perceptron = tr.Perceptron(lr=0.1)
    weights, error = perceptron.learning()
    print("Learned Weights:", weights)
    print("Final Error:", error)
    print("Number of epochs:", perceptron.get_epochs())