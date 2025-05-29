import training as tr

if __name__ == "__main__":
    perceptron = tr.Perceptron(lr=0.1)
    perceptron.plot_data()
    weights, error, accuracy = perceptron.learning()
    print("Learned Weights:", weights)
    print("Final Error:", error)
    print("Number of epochs:", perceptron.get_epochs())
    print(f"Final Accuracy: {accuracy * 100:.2f}%")
    
    perceptron.plot_decision_boundary()