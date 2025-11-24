from ultralytics import YOLO

# Load a model - Dùng nano (n) để nhẹ nhất, tránh crash
model = YOLO('yolo11n.pt')  # Load pretrained model (n=nano, s=small, m=medium, l=large, x=xlarge)

# Train the model with optimized config for accuracy on 8GB RAM
results = model.train(
    data='dataset/data.yaml',      # Path to data config file
    epochs=100,                      # Increase to 100 epochs for better accuracy
    imgsz=640,                       # Increase to 640 for better accuracy (standard YOLO size)
    batch=8,                         # Increase batch to 8 (safe for 8GB RAM)
    name='yolo11_ui_detection',      # Experiment name
    patience=30,                     # Increase patience for better convergence
    save=True,                       # Save checkpoints
    device='cpu',                    # GPU device (0 for first GPU, 'cpu' for CPU)
    workers=4,                       # Increase workers for faster data loading
    project='runs/detect',           # Project folder
    exist_ok=True,                   # Overwrite existing project
    cache=False,                     # Don't cache to save RAM
    amp=False,                       # Disable AMP for CPU
    verbose=True,                    # Show detailed training process
    
    # Optimizer settings for better accuracy
    optimizer='AdamW',               # AdamW often better than SGD
    lr0=0.001,                       # Initial learning rate
    lrf=0.01,                        # Final learning rate
    
    # Data augmentation for better generalization
    hsv_h=0.015,                     # HSV-Hue augmentation
    hsv_s=0.7,                       # HSV-Saturation augmentation
    hsv_v=0.4,                       # HSV-Value augmentation
    degrees=10.0,                    # Rotation augmentation
    translate=0.1,                   # Translation augmentation
    scale=0.5,                       # Scale augmentation
    fliplr=0.5,                      # Flip left-right probability
    mosaic=1.0,                      # Mosaic augmentation
    mixup=0.1,                       # Mixup augmentation
    
    # Validation and save settings
    val=True,                        # Validate during training
    save_period=10,                  # Save every 10 epochs
    plots=True                       # Save training plots
)

print("Training completed!")
print(f"Best model saved at: {results.save_dir}")
