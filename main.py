import training as tr

if __name__ == "__main__":
    
    perceptron = tr.Perceptron(generated=True, linear=True)
    perceptron.learning()
    weights, error, accuracy, epochs = perceptron.get_post_train()
    predict = perceptron.think()
    
    print("\nLearned Weights:", weights)
    print("Final Error:", error)
    print("Number of epochs:", epochs)
    print(f"Final Accuracy: {accuracy*100:.2f}%")
    print("\nPredictions:", predict)
    
    accuracies = perceptron.cross_validate()
    print("Cross-validation accuracies:", accuracies)
    perceptron.plot_data()
    perceptron.plot_decision_boundary()
    perceptron.plot_errors()
    perceptron.plot_accuracy()
    perceptron.plot_weights()
    
    perceptron.__del__()