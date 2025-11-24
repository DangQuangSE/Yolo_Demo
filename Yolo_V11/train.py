from ultralytics import YOLO

# Load a model
model = YOLO('yolo11x.pt')  # Load pretrained model (n=nano, s=small, m=medium, l=large, x=xlarge)

# Train the model
results = model.train(
    data='dataset/data.yaml',      # Path to data config file
    epochs=100,                      # Number of epochs
    imgsz=640,                       # Image size
    batch=16,                        # Batch size (adjust based on GPU memory)
    name='yolo11_ui_detection',      # Experiment name
    patience=50,                     # Early stopping patience
    save=True,                       # Save checkpoints
    device='cpu',                    # GPU device (0 for first GPU, 'cpu' for CPU)
    workers=8,                       # Number of workers for data loading
    project='runs/detect',           # Project folder
    exist_ok=True                    # Overwrite existing project
)

print("Training completed!")
print(f"Best model saved at: {results.save_dir}")
