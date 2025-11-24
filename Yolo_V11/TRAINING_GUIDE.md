# 🚀 Hướng Dẫn Training YOLO11 (Tối Ưu)

## ✅ Đã Hoàn Thành

### 1. Kiểm tra Labels

- ✅ Tất cả labels đã đúng format YOLO11
- ✅ Tên file labels khớp với tên file images
- ✅ Dataset: 84 train, 7 valid, 8 test

### 2. Tối Ưu Cấu Hình Training

File `train.py` đã được cập nhật với cấu hình nhẹ:

| Tham số    | Giá trị Cũ | Giá trị Mới | Lý do                                   |
| ---------- | ---------- | ----------- | --------------------------------------- |
| `epochs`   | 100        | **50**      | Giảm thời gian train                    |
| `imgsz`    | 640        | **416**     | Tiết kiệm RAM ~40%                      |
| `batch`    | 16         | **4**       | **QUAN TRỌNG** - Tránh crash do hết RAM |
| `workers`  | 8          | **2**       | Giảm tải CPU                            |
| `patience` | 50         | **20**      | Early stopping sớm hơn                  |
| `cache`    | -          | **False**   | Không cache để tiết kiệm RAM            |
| `amp`      | -          | **False**   | Tắt AMP cho CPU                         |

## 🎯 Cách Chạy Training

### Bước 1: Kích hoạt virtual environment (nếu chưa)

```bash
source .venv/Scripts/activate
```

### Bước 2: Chạy training

```bash
python train.py
```

## 📊 Theo Dõi Training

Training sẽ hiển thị:

- **Epoch**: Số epoch hiện tại
- **GPU_mem**: 0GB (vì dùng CPU)
- **box_loss**: Loss của bounding box
- **cls_loss**: Loss của classification
- **dfl_loss**: Distribution focal loss
- **instances**: Số objects trong batch
- **Size**: Kích thước ảnh (416x416)

## ⚙️ Điều Chỉnh Nếu Vẫn Bị Crash

### Nếu vẫn thiếu RAM:

Mở file `train.py` và sửa:

```python
batch=2,      # Giảm xuống 2 (hoặc 1 nếu cần)
imgsz=320,    # Giảm xuống 320
workers=1,    # Chỉ dùng 1 worker
```

### Nếu muốn train nhanh hơn (ít chính xác hơn):

```python
epochs=25,    # Chỉ train 25 epochs
patience=10,  # Dừng sớm hơn
```

### Nếu có GPU (NVIDIA):

```python
device=0,     # Thay vì 'cpu'
batch=8,      # Có thể tăng batch lên
amp=True,     # Bật AMP để train nhanh hơn
```

## 📁 Kết Quả Training

Sau khi training, kết quả sẽ ở:

```
runs/detect/yolo11_ui_detection/
├── weights/
│   ├── best.pt       ← Model tốt nhất (dùng cái này)
│   └── last.pt       ← Model ở epoch cuối
├── results.png       ← Biểu đồ training
├── confusion_matrix.png
└── ...
```

## 🧪 Test Model Sau Khi Train

Tạo file `test.py`:

```python
from ultralytics import YOLO

# Load model đã train
model = YOLO('runs/detect/yolo11_ui_detection/weights/best.pt')

# Test trên ảnh
results = model('dataset/test/images/33_png.rf.37e99851dde9d58a479f37c3fa746359.jpg')

# Hiển thị kết quả
results[0].show()

# Hoặc lưu kết quả
results[0].save('result.jpg')
```

## 📈 Đánh Giá Hiệu Suất

```python
from ultralytics import YOLO

model = YOLO('runs/detect/yolo11_ui_detection/weights/best.pt')

# Validate trên test set
metrics = model.val(data='dataset/data.yaml', split='test')

print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
```

## ⏱️ Thời Gian Dự Kiến

Với cấu hình hiện tại (CPU):

- **Mỗi epoch**: ~5-10 phút (tùy CPU)
- **Tổng thời gian**: ~4-8 giờ (50 epochs)
- **Early stopping**: Có thể dừng sớm hơn nếu không cải thiện

## 💡 Tips

1. **Chạy qua đêm**: Training trên CPU mất nhiều thời gian
2. **Theo dõi RAM**: Dùng Task Manager kiểm tra
3. **Lưu checkpoint**: Model tự động lưu sau mỗi epoch
4. **Dừng giữa chừng**: Có thể Ctrl+C, model vẫn được lưu

## 🆘 Troubleshooting

### Lỗi: "Out of Memory"

→ Giảm `batch=1` hoặc `imgsz=320`

### Lỗi: "CUDA out of memory"

→ Đổi `device='cpu'`

### Training quá chậm

→ Giảm `epochs=25` hoặc dùng GPU

### Kết quả không tốt

→ Tăng `epochs=100`, `imgsz=640` nếu máy cho phép

## ✨ So Sánh Cấu Hình

| Cấu hình     | RAM cần | Tốc độ     | Độ chính xác | Khuyến nghị    |
| ------------ | ------- | ---------- | ------------ | -------------- |
| **Hiện tại** | ~2-3GB  | Trung bình | Tốt          | ✅ Máy yếu     |
| Ultra nhẹ    | ~1-2GB  | Nhanh      | Trung bình   | Máy rất yếu    |
| Cân bằng     | ~4-6GB  | Chậm       | Rất tốt      | Máy trung bình |
| Tối đa       | ~8-16GB | Rất chậm   | Xuất sắc     | Có GPU         |

---

**Chúc bạn training thành công! 🎉**
