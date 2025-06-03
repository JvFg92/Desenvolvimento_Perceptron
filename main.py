import training as tr

if __name__ == "__main__":
    """
    Main entry point for training and evaluating the Perceptron model.
    """
    perceptron = tr.Perceptron(generated=True, linear=True)
    perceptron.plot_data()
    perceptron.learning()
    weights, error, accuracy, epochs, recall = perceptron.get_post_train(plot=True)
    perceptron.cross_validate(plot=True)
    predict = perceptron.think()
    
    print("\nFinal Learned Weights:", weights)
    print("Final Error:", error)
    print("Number of epochs:", epochs)
    print(f"Final Accuracy: {accuracy*100:.2f}%")
    print(f"Final Recall: {recall*100:.2f}%")
    print("\nPredictions Test:", predict)

    perceptron.__del__()