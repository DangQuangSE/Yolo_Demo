"""
Flask API để upload ảnh và detect layout
"""
from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
import os
from pathlib import Path
import json
from datetime import datetime

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
MODEL_PATH = 'runs/detect/yolo11_ui_detection/weights/best.pt'

# Create folders if not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model (lazy loading - only when needed)
model = None

def get_model():
    """Load model on demand"""
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            # Try alternative paths
            alternative_paths = [
                'runs/detect/yolo11_ui_detection/weights/best.pt',
                'runs/detect/train/weights/best.pt',
                'best.pt',
                'yolo11n.pt'
            ]
            
            for path in alternative_paths:
                if os.path.exists(path):
                    MODEL_PATH_ACTUAL = path
                    break
            else:
                raise FileNotFoundError(
                    f"Model not found! Please train the model first.\n"
                    f"Expected path: {MODEL_PATH}\n"
                    f"Run: python train.py or python train_ultra_light.py"
                )
        else:
            MODEL_PATH_ACTUAL = MODEL_PATH
        
        print(f"Loading model from: {MODEL_PATH_ACTUAL}")
        model = YOLO(MODEL_PATH_ACTUAL)
        print("Model loaded successfully!")
    
    return model


@app.route('/')
def home():
    return """
    <html>
    <head>
        <title>YOLO11 UI Detection</title>
        <style>
            body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #333; }
            .upload-box { border: 2px dashed #ccc; padding: 30px; text-align: center; }
            button { background: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; }
            button:hover { background: #45a049; }
            #result { margin-top: 20px; }
            img { max-width: 100%; }
            .detection-item { background: #f5f5f5; padding: 10px; margin: 5px 0; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>YOLO11 UI Layout Detection</h1>
        
        <div class="upload-box">
            <h3>Upload image to detect layout</h3>
            <input type="file" id="imageFile" accept="image/*">
            <br><br>
            <button onclick="uploadImage()">Detect Layout</button>
        </div>
        
        <div id="result"></div>
        
        <script>
            async function uploadImage() {
                const fileInput = document.getElementById('imageFile');
                const file = fileInput.files[0];
                
                if (!file) {
                    alert('Please select an image!');
                    return;
                }
                
                const formData = new FormData();
                formData.append('image', file);
                
                document.getElementById('result').innerHTML = '<p>Processing...</p>';
                
                try {
                    const response = await fetch('/detect', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        displayResult(data);
                    } else {
                        document.getElementById('result').innerHTML = 
                            '<p style="color:red">Error: ' + data.error + '</p>';
                    }
                } catch (error) {
                    document.getElementById('result').innerHTML = 
                        '<p style="color:red">Error: ' + error.message + '</p>';
                }
            }
            
            function displayResult(data) {
                let html = '<h2>Detection Results</h2>';
                
                // Display image
                html += '<h3>Result Image:</h3>';
                html += '<img src="/output/' + data.output_image + '" alt="Result">';
                
                // Summary info
                html += '<h3>Summary:</h3>';
                html += '<p><strong>Total objects:</strong> ' + data.total_objects + '</p>';
                html += '<p><strong>Image size:</strong> ' + 
                        data.image_size.width + ' x ' + data.image_size.height + '</p>';
                
                // Object details
                html += '<h3>Object Details:</h3>';
                data.detections.forEach(det => {
                    html += '<div class="detection-item">';
                    html += '<strong>[' + det.id + '] ' + det.class_name + '</strong> ';
                    html += '(Confidence: ' + (det.confidence * 100).toFixed(1) + '%)<br>';
                    html += 'Position: (' + det.bbox.x1 + ', ' + det.bbox.y1 + ') to (' + 
                            det.bbox.x2 + ', ' + det.bbox.y2 + ')<br>';
                    html += 'Size: ' + det.bbox.width + ' x ' + det.bbox.height + '<br>';
                    html += 'Center: (' + det.center.x + ', ' + det.center.y + ')';
                    html += '</div>';
                });
                
                // Download JSON link
                html += '<br><a href="/output/' + data.output_json + '" download>';
                html += '<button>Download JSON</button></a>';
                
                document.getElementById('result').innerHTML = html;
            }
        </script>
    </body>
    </html>
    """


@app.route('/detect', methods=['POST'])
def detect():
    """
    API endpoint to detect layout from uploaded image
    """
    try:
        # Get model
        model = get_model()
        
        # Check file
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image uploaded'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected'})
        
        # Save uploaded file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Predict with lower confidence threshold
        results = model(filepath, conf=0.1, iou=0.4, imgsz=640)
        result = results[0]
        
        # Save result image
        output_image = f"{timestamp}_result.jpg"
        output_image_path = os.path.join(OUTPUT_FOLDER, output_image)
        result.save(output_image_path)
        
        # Get detailed info
        detections = []
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            width = x2 - x1
            height = y2 - y1
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = result.names[class_id]
            
            detection = {
                'id': i,
                'class_id': class_id,
                'class_name': class_name,
                'confidence': round(confidence, 4),
                'bbox': {
                    'x1': round(x1, 2),
                    'y1': round(y1, 2),
                    'x2': round(x2, 2),
                    'y2': round(y2, 2),
                    'width': round(width, 2),
                    'height': round(height, 2)
                },
                'center': {
                    'x': round(center_x, 2),
                    'y': round(center_y, 2)
                }
            }
            detections.append(detection)
        
        # Create JSON output
        output_data = {
            'success': True,
            'image_path': filepath,
            'output_image': output_image,
            'image_size': {
                'width': result.orig_shape[1],
                'height': result.orig_shape[0]
            },
            'total_objects': len(detections),
            'detections': detections
        }
        
        # Save JSON
        output_json = f"{timestamp}_result.json"
        output_json_path = os.path.join(OUTPUT_FOLDER, output_json)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        output_data['output_json'] = output_json
        
        return jsonify(output_data)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/output/<filename>')
def get_output(filename):
    """
    Serve output files
    """
    return send_file(os.path.join(OUTPUT_FOLDER, filename))


if __name__ == '__main__':
    print("\n" + "="*60)
    print("YOLO11 UI Detection API Server")
    print("="*60)
    print(f"\nModel: {MODEL_PATH}")
    print("\nHow to use:")
    print("   1. Open browser: http://localhost:5000")
    print("   2. Upload image")
    print("   3. Get detection results with positions\n")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
