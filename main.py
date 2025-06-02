import training as tr

if __name__ == "__main__":
    """
    Main entry point for training and evaluating the Perceptron model.
    """
    perceptron = tr.Perceptron(generated=True, linear=True)
    perceptron.plot_data()
    perceptron.learning()
    weights, error, accuracy, epochs = perceptron.get_post_train(plot=True)
    accuracies = perceptron.cross_validate(plot=True)
    predict = perceptron.think()
    
    print("\nLearned Weights:", weights)
    print("Final Error:", error)
    print("Number of epochs:", epochs)
    print(f"Final Accuracy: {accuracy*100:.2f}%")
    print("\nPredictions Test:", predict)
    print("Cross-validation accuracies:", accuracies)
    
    perceptron.__del__()