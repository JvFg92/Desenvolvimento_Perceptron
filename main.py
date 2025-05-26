import training as tr
import data_treatment as dt

if __name__ == "__main__":
   
    print("Starting training...")
    weights, error = tr.learning()
    print("Training completed.")
    print("Training error:", error)
    print("Final weights:", weights)