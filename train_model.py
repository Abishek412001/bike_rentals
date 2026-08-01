from src.models.train_pipeline import run_training_pipeline

if __name__ == "__main__":
    print("Executing Enterprise Bike Rental Demand Model Training Pipeline...")
    bundle = run_training_pipeline()
    print("Training complete. Champion model saved and report figures updated.")
