import training as tr

if __name__ == "__main__":
    """
    Main entry point for training and evaluating the Perceptron model.
    """
    while True:
        input_data = input("Select dataset \n'1' for Iris Setosa x no Setosa; \n'2' for Iris Versicolor x Virginica; \n'3' for non linear Breast Cancer; \n'4' for linear Breast Cancer. \n'5' for exit")

        if input_data == '1':
            perceptron = tr.Perceptron(generated=False, linear=True)
        elif input_data == '2':
            perceptron = tr.Perceptron(generated=False, linear=False)
        elif input_data == '3':
            perceptron = tr.Perceptron(generated=True, linear=False)
        elif input_data == '4':
            perceptron = tr.Perceptron(generated=True, linear=True)
        elif input_data == '5':
            print("Exiting the program.")
            break
        else:
            print("Invalid input. Please try again.")
            continue

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