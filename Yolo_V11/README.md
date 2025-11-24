# Hướng Dẫn Setup YOLO11 Training Demo

## Mục Lục

- [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
- [Bước 1: Tạo Virtual Environment](#bước-1-tạo-virtual-environment)
- [Bước 2: Cài Đặt Thư Viện](#bước-2-cài-đặt-thư-viện)
- [Bước 3: Chuẩn Bị Dataset](#bước-3-chuẩn-bị-dataset)
- [Bước 4: Cấu Hình Dataset](#bước-4-cấu-hình-dataset)
- [Bước 5: Tạo File Training](#bước-5-tạo-file-training)
- [Bước 6: Chạy Training](#bước-6-chạy-training)
- [Kết Quả](#kết-quả)
- [Troubleshooting](#troubleshooting)

---

## Yêu Cầu Hệ Thống

- **Python**: 3.8 trở lên (khuyến nghị 3.10+)
- **RAM**: Tối thiểu 8GB
- **Disk**: ~5GB cho môi trường và dataset
- **GPU**: Không bắt buộc (có thể train trên CPU)

---

## Bước 1: Tạo Virtual Environment

### 1.1 Tạo môi trường ảo

```bash
python -m venv .venv
```

### 1.2 Kích hoạt virtual environment

**Trên Windows (Command Prompt):**

```cmd
.venv\Scripts\activate
```

**Trên Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

**Trên Windows (Git Bash/WSL):**

```bash
source .venv/Scripts/activate
```

**Trên Linux/Mac:**

```bash
source .venv/bin/activate
```

---

## Bước 2: Cài Đặt Thư Viện

### 2.1 Nâng cấp pip

```bash
python -m pip install --upgrade pip
```

### 2.2 Cài đặt Ultralytics (YOLO11)

```bash
pip install ultralytics
```

Thư viện này sẽ tự động cài đặt các dependencies cần thiết:

- PyTorch
- OpenCV
- NumPy
- Matplotlib
- PyYAML
- v.v.

---

## Bước 3: Chuẩn Bị Dataset

### 3.1 Cấu trúc thư mục

Giải nén dataset **cùng cấp với thư mục `.venv`**:

```
Yolo_V11/
├── .venv/                    # Virtual environment
├── dataset/                  # Dataset ở đây
│   ├── train/
│   │   ├── images/          # Ảnh training
│   │   └── labels/          # Labels (YOLO format .txt)
│   ├── val/                 # hoặc valid/
│   │   ├── images/
│   │   └── labels/
│   ├── test/                # (Optional)
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml           # File cấu hình dataset
├── train.py                 # Script training (tạo ở bước 5)
└── README.md               # File này
```

### 3.2 Format Labels

Mỗi ảnh cần có file `.txt` tương ứng với format YOLO:

```
<class_id> <x_center> <y_center> <width> <height>
```

Ví dụ (`image1.txt`):

```
0 0.5 0.5 0.3 0.4
1 0.7 0.3 0.2 0.25
```

---

## Bước 4: Cấu Hình Dataset

### 4.1 Tạo/Chỉnh sửa file `data.yaml`

Tạo file `dataset/data.yaml` với nội dung:

```yaml
# Đường dẫn (relative từ vị trí file data.yaml)
train: train/images
val: valid/images # hoặc val/images
test: test/images # Optional

# Số lượng classes
nc: 22

# Tên các classes
names: ["class1", "class2", "class3", ...]
```

**Ví dụ cụ thể:**

```yaml
train: train/images
val: valid/images
test: test/images

nc: 22
names:
  [
    "chs-toggle",
    "data-body-row",
    "data-header-cell",
    "data-table-header",
    "datatable-body",
    "div-toogle",
    "dropdown",
    "dropdown-label",
    "floating-label",
    "label",
    "mat-card",
    "mat-chip-grid",
    "mat-chip-row",
    "mat-form-field",
    "mat-icn",
    "mat-icon",
    "mat-label",
    "mat-option",
    "mat-slde",
    "mat-slide-toggle",
    "ngx-datatable",
    "toggle-label",
  ]
```

---

## Bước 5: Tạo File Training

### 5.1 Tạo file `train.py`

Tạo file `train.py` trong thư mục gốc với nội dung:

```python
from ultralytics import YOLO

# Load model (khuyến nghị dùng version nhẹ: n, s hoặc m)
model = YOLO('yolo11n.pt')  # nano - nhẹ nhất, nhanh nhất
# model = YOLO('yolo11s.pt')  # small
# model = YOLO('yolo11m.pt')  # medium

# Train the model
results = model.train(
    data='dataset/data.yaml',      # Path to data config file
    epochs=100,                      # Number of epochs
    imgsz=640,                       # Image size
    batch=16,                        # Batch size (giảm xuống nếu thiếu RAM)
    name='yolo11_training',          # Experiment name
    patience=50,                     # Early stopping patience
    save=True,                       # Save checkpoints
    device='cpu',                    # 'cpu' hoặc 0 (cho GPU)
    workers=8,                       # Number of workers
    project='runs/detect',           # Project folder
    exist_ok=True                    # Overwrite existing project
)

print("Training completed!")
print(f"Best model saved at: {results.save_dir}")
```

### 5.2 Chọn Model Size

**Khuyến nghị:**

| Model        | Size   | Speed      | Accuracy       | Use Case                 |
| ------------ | ------ | ---------- | -------------- | ------------------------ |
| `yolo11n.pt` | ~5MB   | ⚡⚡⚡⚡⚡ | ⭐⭐⭐         | **Demo, CPU, Real-time** |
| `yolo11s.pt` | ~20MB  | ⚡⚡⚡⚡   | ⭐⭐⭐⭐       | **Cân bằng tốt**         |
| `yolo11m.pt` | ~50MB  | ⚡⚡⚡     | ⭐⭐⭐⭐⭐     | Production với GPU       |
| `yolo11l.pt` | ~100MB | ⚡⚡       | ⭐⭐⭐⭐⭐⭐   | Độ chính xác cao         |
| `yolo11x.pt` | ~110MB | ⚡         | ⭐⭐⭐⭐⭐⭐⭐ | Chỉ dùng khi có GPU mạnh |

**⚠️ Lưu ý:** Nếu train trên CPU, nên dùng `yolo11n.pt` hoặc `yolo11s.pt`

---

## Bước 6: Chạy Training

### 6.1 Chạy script

```bash
python train.py
```

### 6.2 Theo dõi tiến trình

Training sẽ hiển thị:

- Loss values (box, cls, dfl)
- mAP (mean Average Precision)
- Precision & Recall
- Training progress bar

### 6.3 Điều chỉnh nếu gặp vấn đề

**Thiếu RAM/Memory:**

```python
batch=8,  # Giảm từ 16 xuống 8
# hoặc
batch=4,  # Giảm xuống 4
```

**Training quá chậm (CPU):**

- Giảm `epochs=50` (thay vì 100)
- Giảm `imgsz=416` (thay vì 640)
- Dùng model nhỏ hơn (`yolo11n.pt`)

**Có GPU CUDA:**

```python
device=0,  # Thay vì 'cpu'
```

---

## Kết Quả

Sau khi training xong, kết quả sẽ được lưu trong:

```
runs/detect/yolo11_training/
├── weights/
│   ├── best.pt          # Model tốt nhất
│   └── last.pt          # Model ở epoch cuối
├── args.yaml            # Training arguments
├── results.csv          # Training metrics
├── results.png          # Training curves
├── confusion_matrix.png # Confusion matrix
├── F1_curve.png         # F1 score curve
├── PR_curve.png         # Precision-Recall curve
└── val_batch*.jpg       # Validation predictions
```

### Sử dụng model đã train

```python
from ultralytics import YOLO

# Load model đã train
model = YOLO('runs/detect/yolo11_training/weights/best.pt')

# Predict trên ảnh mới
results = model('path/to/image.jpg')

# Hiển thị kết quả
results[0].show()

# Lưu kết quả
results[0].save('output.jpg')
```

---

## Troubleshooting

### 1. Lỗi: "command not found" khi activate venv

**Trên Git Bash:**

```bash
source .venv/Scripts/activate
```

**Trên PowerShell (nếu bị chặn):**

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\Activate.ps1
```

### 2. Lỗi: "CUDA not available"

Nếu không có GPU, đổi `device=0` thành `device='cpu'` trong `train.py`

### 3. Lỗi: "Dataset not found"

Kiểm tra:

- File `data.yaml` có đúng vị trí không
- Đường dẫn trong `data.yaml` có chính xác không
- Thư mục `images/` và `labels/` có tồn tại không

### 4. Training quá chậm

- Giảm `epochs`, `batch`, `imgsz`
- Dùng model nhỏ hơn (`yolo11n.pt`)
- Nếu có GPU, cài PyTorch với CUDA support

### 5. Out of Memory

```python
batch=4,      # Giảm batch size
workers=2,    # Giảm số workers
```

---

## Tài Liệu Tham Khảo

- [Ultralytics YOLO11 Docs](https://docs.ultralytics.com/)
- [YOLO11 GitHub](https://github.com/ultralytics/ultralytics)
- [Training Guide](https://docs.ultralytics.com/modes/train/)
- [Dataset Format](https://docs.ultralytics.com/datasets/detect/)

---

## License

Tuân theo license của dataset và Ultralytics YOLO.

---

**Chúc bạn training thành công! 🚀**
