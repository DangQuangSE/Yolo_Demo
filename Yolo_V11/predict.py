"""
Script để sử dụng model YOLO11 đã train
Detect layout và trả về position của từng class
"""
from ultralytics import YOLO
import json

def predict_image(image_path, model_path='runs/detect/yolo11_ui_detection/weights/best.pt'):
    """
    Predict trên 1 ảnh và trả về kết quả
    
    Args:
        image_path: Đường dẫn đến ảnh cần detect
        model_path: Đường dẫn đến model đã train
    
    Returns:
        dict: Kết quả detection với position và class
    """
    # Load model đã train
    model = YOLO(model_path)
    
    # Predict with lower confidence threshold for better detection
    results = model(image_path, conf=0.1, iou=0.4, imgsz=640)
    
    # Lấy kết quả đầu tiên
    result = results[0]
    
    # Chuẩn bị output
    detections = []
    
    # Lấy thông tin từng object được detect
    for i, box in enumerate(result.boxes):
        # Lấy tọa độ (xyxy format - góc trên trái, góc dưới phải)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        # Tính center và size
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Lấy class và confidence
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
    
    # Metadata
    output = {
        'image_path': image_path,
        'image_size': {
            'width': result.orig_shape[1],
            'height': result.orig_shape[0]
        },
        'total_objects': len(detections),
        'detections': detections
    }
    
    return output


def predict_and_save(image_path, output_image='output.jpg', output_json='output.json', 
                     model_path='runs/detect/yolo11_ui_detection/weights/best.pt'):
    """
    Predict và lưu kết quả ra file ảnh + JSON
    
    Args:
        image_path: Đường dẫn ảnh cần detect
        output_image: Đường dẫn lưu ảnh kết quả (có box)
        output_json: Đường dẫn lưu JSON kết quả
        model_path: Đường dẫn model
    """
    # Load model
    model = YOLO(model_path)
    
    # Predict with lower confidence threshold
    results = model(image_path, conf=0.1, iou=0.4, imgsz=640)
    
    # Lưu ảnh có bounding boxes
    results[0].save(output_image)
    print(f"✅ Đã lưu ảnh kết quả: {output_image}")
    
    # Lấy thông tin chi tiết
    output = predict_image(image_path, model_path)
    
    # Lưu JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"✅ Đã lưu kết quả JSON: {output_json}")
    
    # Hiển thị kết quả
    print(f"\n📊 Kết quả detection:")
    print(f"   Tổng số objects: {output['total_objects']}")
    print(f"\n📋 Chi tiết:")
    
    for det in output['detections']:
        print(f"\n   [{det['id']}] {det['class_name']} (confidence: {det['confidence']})")
        print(f"       Bounding Box: ({det['bbox']['x1']}, {det['bbox']['y1']}) -> ({det['bbox']['x2']}, {det['bbox']['y2']})")
        print(f"       Size: {det['bbox']['width']} x {det['bbox']['height']}")
        print(f"       Center: ({det['center']['x']}, {det['center']['y']})")
    
    return output


if __name__ == "__main__":
    import sys
    
    # Sử dụng:
    # python predict.py <path_to_image>
    
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [model_path]")
        print("\nExample:")
        print("  python predict.py test_image.jpg")
        print("  python predict.py test_image.jpg runs/detect/yolo11_ui_detection/weights/best.pt")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'runs/detect/yolo11_ui_detection/weights/best.pt'
    
    print("="*60)
    print("🚀 YOLO11 UI Detection - Prediction")
    print("="*60)
    print(f"\n📷 Image: {image_path}")
    print(f"🤖 Model: {model_path}")
    print()
    
    result = predict_and_save(
        image_path=image_path,
        output_image='output_detected.jpg',
        output_json='output_detected.json',
        model_path=model_path
    )
    
    print("\n" + "="*60)
    print("✅ Hoàn tất!")
    print("="*60)
