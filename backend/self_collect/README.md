# Self-Collect Paired (PPG, BP) Data — Quick Start

## Tổng quan

Workflow thu data **local** (không qua Render/cloud) cho thesis MVP:

```
ESP32 (firmware v4.1.0)
 │ HTTP POST batch JSON 5s
 ▼
Laptop FastAPI (log_ppg_local.py)
 │ + Manual nhập SBP/DBP từ Omron HEM-7143T1
 ▼
collected_data/<subject_id>_<timestamp>.csv (raw IR+Red + metadata header)
 │
 ▼ (sau 30-50 subjects)
python ../../ML\ code/ml/train_ppg_bp.py (auto-detect collected_data/)
 │
 ▼
ml/models/random_forest_models.pkl (combined PPG-BP + self-collected)
 │
 ▼ git commit + push
Render auto-redeploy
```

## File trong folder này

| File | Mục đích |
|---|---|
| `log_ppg_local.py` | FastAPI server local thay Render khi thu data |
| `subjects_metadata.xlsx` | Excel template log per-subject metadata + 2 sheet phụ (BP categories, Protocol checklist) |
| `consent_form.md` | Mẫu phiếu đồng ý tham gia (in 50 bản trước) |
| `README.md` | File này |

## Quick start — 5 bước trước session đầu tiên

### Bước 1: Cài dependency (~5 phút)

```bash
pip install pyserial fastapi uvicorn openpyxl pandas
```

(FastAPI + uvicorn đã có sẵn từ backend project.)

### Bước 2: Sửa firmware ESP32 → trỏ về laptop (~10 phút, 1 lần)

Trong `arduino/ppg_v4_firmware/ppg_v4_firmware.ino`, tìm `serverUrl` và đổi:

```cpp
// Trước (production Render):
const char* serverUrl = "https://ppg-ml.onrender.com/api/ppg/upload";

// Sau (local thu data):
const char* serverUrl = "http://192.168.1.X:8080/api/ppg/upload";
// ^^^^^^^^^^^^ thay bằng IP laptop của bạn
```

→ Chạy `log_ppg_local.py` 1 lần, console sẽ in IP laptop ra. Copy IP đó vào firmware.

Build + flash ESP32 1 lần. Sau khi xong tất cả thu data, nhớ đổi URL về Render và flash lại.

### Bước 3: Print + Test consent form (~30 phút)

```bash
# Convert MD → PDF (optional, dùng Pandoc hoặc online converter)
pandoc consent_form.md -o consent_form.pdf

# Hoặc: in trực tiếp file Markdown từ VS Code/editor
```

In **50 bản** consent form (1 bản subject giữ, 1 bản bạn lưu).

### Bước 4: Chạy thử workflow với 1-2 người nhà (~30 phút)

```bash
cd "backend/self_collect"
python log_ppg_local.py
```

Console sẽ hiển thị banner + IP. Verify ESP32 connect được:
- ESP32 reset → kết nối WiFi → POST batch đầu lên laptop
- Console laptop in `[BATCH #001] S001 fs=100 +500 → total=500 samples`
- Nếu KHÔNG thấy batch in ra: check WiFi cùng mạng, firewall laptop tắt cho port 8080.

Nhập metadata test 1-2 lần để familiar workflow.

### Bước 5: Recruit 30-50 subjects + lên lịch

Theo BP_Reference sheet trong `subjects_metadata.xlsx`:
- 30% normotensive (sinh viên trẻ)
- 30% pre-hypertensive (trung niên)
- **30% hypertensive** ⭐ (lớn tuổi, dùng thuốc) — quan trọng nhất, PPG-BP dataset thiếu nhóm này
- 10% diverse age/sex

## Workflow per subject (~12-15 phút)

Theo `Protocol_Checklist` sheet trong `subjects_metadata.xlsx`:

```
[Trước đo — 5 phút]
 ☐ Subject ngồi yên, không nói, không điện thoại
 ☐ Tay đặt ngang tim

[Đo BP cuff Omron — 4 phút] (AHA 2025: discard reading 1 do white-coat effect)
 ☐ Đo lần 1 → BỎ (không note, không lưu)
 ☐ Đợi 1 phút
 ☐ Đo lần 2 → ghi nhớ giá trị (SBP2, DBP2)
 ☐ Đợi 1 phút
 ☐ Đo lần 3 → ghi nhớ giá trị (SBP3, DBP3)
 ☐ Tính tay: sbp_baseline_mean = round((SBP2+SBP3)/2), dbp_baseline_mean tương tự

[Đo PPG — 5 phút]
 ☐ Trên console laptop: nhập subject_id + age + sex + height + weight
 ☐ Nhập sbp_baseline_mean + dbp_baseline_mean (lấy từ Excel)
 ☐ MAX30102 ở ngón giữa, tay không cử động
 ☐ Press ENTER → ESP32 stream
 ☐ Sau ~5 phút: Press ENTER lần nữa → save CSV

[Đo BP post — 1 phút]
 ☐ Đo Omron lần 3 → nhập sbp_post + dbp_post vào console
 ☐ Save tự động

[Cập nhật Excel]
 ☐ Update Excel: csv_file name, duration_s, signal_quality_notes
```

## Sau khi xong tất cả subjects

### Train model với data combined

```bash
cd "ml/ml"
python train_ppg_bp.py
```

Script tự động:
1. Load 219 subjects PPG-BP transmission
2. Load N subjects self-collected từ `collected_data/`
3. Combine → train CV → fit best model → save pkl

Expected output:
```
Step 2b — Combine PPG-BP (219) + self-collected (50) = 269 total subjects
...
Bundle hợp nhất (SBP=svr, DBP=random_forest) -> random_forest_models.pkl
```

### Deploy lên Render

```bash
cd "ML code"
git add ml/models/random_forest_models.pkl ml/models/random_forest_models.pkl.sha256
git commit -m "Retrain BP with $(ls ../../collected_data/ | wc -l) self-collected subjects"
git push
# Render auto-redeploy ~3-5 phút
```

### Đổi firmware về Render production

Đổi `serverUrl` ESP32 về `https://ppg-ml.onrender.com/api/ppg/upload`. Build + flash. Test BP có giá trị tốt hơn.

## Troubleshooting

| Vấn đề | Giải pháp |
|---|---|
| ESP32 không POST được lên laptop | Check WiFi cùng mạng, ping IP laptop từ điện thoại, tắt firewall port 8080 |
| `[BATCH] received` xuất hiện trước khi nhập metadata | Console chưa nhập subject → 503 error, ESP32 sẽ retry. Nhập metadata nhanh, ESP32 tiếp tục stream |
| File CSV bị 0 samples | ESP32 không POST được trong 5 phút stream (WiFi rớt). Test lại |
| Train script lỗi `No module named 'pandas'` | `pip install pandas openpyxl` |
| Server lỗi `OSError: [Errno 98] Address already in use` | Port 8080 đang dùng. Đổi `PORT=8090 python log_ppg_local.py` |

## Checklist tổng final trước session

```
☐ Laptop sạc đầy + adapter
☐ ESP32 cable USB + nguồn (powerbank dự phòng)
☐ Omron HEM-7143T1 + 4 pin AA + 4 pin dự phòng
☐ MAX30102 sensor sạch (lau bằng cồn 70% trước session)
☐ Phòng yên tĩnh, có bàn ghế thoải mái
☐ Excel template `subjects_metadata.xlsx` mở sẵn
☐ Console `python log_ppg_local.py` đang chạy
☐ 50 bản consent form đã in, bút ký, kẹp file
☐ Backup: USB / Google Drive cho cuối ngày
```
