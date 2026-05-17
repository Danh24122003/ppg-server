# QUY TRÌNH THU MẪU PPG-BP — THESIS PILOT STUDY

> **Phiên bản:** v1.2 (01/05/2026 — chiều) | **Mục tiêu:** N=15 subjects × 2 sessions = 30 paired recordings
> **In file này ra A4 mang theo khi đo. Tick ☐ → ✅ từng bước.**
>
> **Changelog v1.2 (01/05 chiều):** HTN-on-medication clarification + per-subject calibration mode:
> - HTN-treated subjects: ✅ INCLUDE (trừ beta-blockers), label theo BP **measured**, không phải diagnosis
> - Document medication chi tiết: dose_mg, hours_since_dose, years_since_diagnosis
> - Per-subject calibration: session 1 = anchor (auto via `/api/ml/calibrate`), session 2 = validation
> - Tất cả sessions phải đo CẢ cuff CẢ PPG (không phải chỉ PPG)
>
> **Changelog v1.1 (01/05):** Cập nhật theo researcher audit AHA 2025 + Liang 2018 + ESH 2023:
> - Resting 5 phút → **10 phút** (Liang 2018 protocol — match training data)
> - 2 sessions: cùng ngày → **khác ngày 2-7 ngày, cùng khung giờ ±1h** (intra-subject variability)
> - Thêm: inter-arm BP screening, beta-blockers exclusion, cuff circumference check, đi tiểu trước, eyes open
> - Demographic: ≥3 subjects ≥40 tuổi (capture HTN morphology)

---

## 0. CHUẨN BỊ TRƯỚC NGÀY ĐO (1 lần duy nhất)

### Hardware
- ☐ Sạc đầy laptop + adapter
- ☐ ESP32 + cáp USB + powerbank dự phòng
- ☐ MAX30102 sensor sạch (lau cồn 70% sau mỗi subject)
- ☐ Omron HEM-7143T1 + 4 pin AA + 4 pin dự phòng
- ☐ Băng dán y tế (giữ sensor ổn định nếu cần)

### Software
- ☐ Verify firmware ESP32 đã flash bản:
 - HTTP/HTTPS auto-detect (line 780)
 - NTP bypass (line 743 = `if (false && curOffset == 0)`)
- ☐ Test self-test (chính bản thân) thành công ≥ 1 session FULL 5 phút
- ☐ Tera Term encoding: Receive UTF-8, Transmit UTF-8 (Setup → Encoding)

### Recruit
- ☐ List ≥ 20 ứng viên tiềm năng (gia đình, bạn, đồng nghiệp)
- ☐ Schedule lịch hẹn 15 subjects:
 - 7 đã có (S001-S009) — toàn Normal/Elevated BP measured
 - 8 cần tuyển ưu tiên BP cao: 3-4 Stage 2 HTN measured + 3-4 Stage 1 HTN measured
- ☐ **Tuyển HTN subjects (v1.2):**
 - Ưu tiên 1: HTN-untreated (chưa start thuốc / mới chẩn đoán) — true Stage 1/2 BP measured
 - Ưu tiên 2: HTN-treated KHÔNG dùng beta-blockers (xem section 2.1.a) — đo trước dose buổi sáng
 - Cấm: beta-blockers, digoxin, severe HTN (>180/120)
- ☐ **Demographic balance (AHA 2025 Guideline):**
 - ≥ 5 nam + ≥ 5 nữ (≥33% mỗi gender)
 - **≥ 5 subjects ≥ 40 tuổi** ⚠️ (sinh viên 18-25 KHÔNG capture HTN morphology — cần recruit faculty/người thân lớn tuổi)
- ☐ **Đo cánh tay circumference dự kiến** từng ứng viên: Omron HEM-7143T1 fit 22-32 cm. Nếu ngoài range → cần cuff size khác hoặc loại.

### Phòng đo
- ☐ Phòng yên tĩnh, nhiệt độ 20-25°C
- ☐ Bàn + ghế tựa lưng + tựa tay
- ☐ Tránh ánh sáng mạnh chiếu vào ngón tay
- ☐ Có ổ điện gần (cho ESP32 + laptop)

---

## 1. KHỞI ĐỘNG ĐẦU NGÀY (1 lần/ngày)

```bash
# Terminal 1 (giữ chạy xuyên suốt ngày)
cd "c:/Users/Acer/Desktop/PPG monitor/backend/self_collect"
python log_ppg_local.py
```

- ☐ Verify banner hiện URL `http://192.168.1.90:8080/api/ppg/upload` 
 *(hoặc IP WiFi LAN hiện tại của laptop — KHÔNG dùng 172.16.x.x)*
- ☐ Power on ESP32 → Tera Term verify:
 ```
 Server : http://192.168.1.90:8080/api/ppg/upload
 [READY] Place finger on sensor
 ```
- ☐ Test 1 batch dummy: đặt ngón tay 10s → Terminal 1 thấy `[BATCH #001] ...` → ✅ OK
- ☐ Mở Excel `subjects_metadata.xlsx` (sheet **Subjects**) sẵn sàng

---

## 2. QUY TRÌNH PER SUBJECT (~25 phút × 2 sessions)

### 2.1. Tiêu chí LOẠI subject (kiểm tra trước khi đo)

❌ **KHÔNG đo nếu subject có 1 trong các điều kiện:**
- ☐ Rung nhĩ (atrial fibrillation) đã chẩn đoán → PPG peak detection fail
- ☐ Hội chứng Raynaud / tay chân lạnh mãn tính → vasoconstriction nặng
- ☐ **Đang dùng beta-blockers** (atenolol, metoprolol, bisoprolol, propranolol, carvedilol, nebivolol) → giảm HR chronically → HRV features lệch
- ☐ Đang dùng Digoxin / antiarrhythmics (amiodarone, flecainide) → ảnh hưởng nhịp tim bất thường
- ☐ Sốt > 37.5°C
- ☐ Vận động mạnh < 30 phút trước đó
- ☐ Uống cà phê / trà / hút thuốc < 30 phút trước
- ☐ Bữa ăn nhiều muối < 30 phút trước
- ☐ SBP > 180 hoặc DBP > 120 mmHg (severe HTN — khuyên đi viện ngay)

→ Nếu LOẠI: ghi vào notes Excel, không tính vào N=15.

### 2.1.a. HTN-treated subjects — XỬ LÝ THẾ NÀO (v1.2)

🆕 **Quan trọng:** Nhiều subject HTN diagnosed đang dùng thuốc → BP đo có thể về Normal/Elevated. Cần phân loại đúng cho ML training.

#### Nguyên tắc cốt lõi

> **Label BP cho ML = BP MEASURED tại session đó (theo AHA 2025).**
> KHÔNG dùng diagnosis làm label.

Ví dụ: Subject 60 tuổi, diagnosed HTN 5 năm, đang uống Amlodipine. Đo BP = 122/78.
- ❌ KHÔNG label "Stage 2 HTN" (theo diagnosis cũ)
- ✅ Label "Elevated" (theo BP đo được hiện tại)

#### Thuốc HA — danh sách Allow / Exclude

| Loại thuốc | Tên thông dụng | Quyết định | Lý do |
|---|---|---|---|
| **Beta-blockers** | atenolol, metoprolol, bisoprolol, propranolol, carvedilol, nebivolol | ❌ **EXCLUDE** | Giảm HR chronically → HRV features lệch |
| **Antiarrhythmics** | amiodarone, flecainide, digoxin | ❌ **EXCLUDE** | Ảnh hưởng nhịp tim |
| ACE inhibitors | enalapril, lisinopril, captopril, ramipril, perindopril | ✅ **Include** | Không ảnh hưởng HR/HRV trực tiếp |
| ARBs | losartan, valsartan, telmisartan, candesartan, olmesartan | ✅ **Include** | Tương tự ACE-I |
| Calcium channel blockers | amlodipine, nifedipine, felodipine | ✅ **Include** | Không ảnh hưởng HR |
| CCB nhóm khác | diltiazem, verapamil | ⚠️ **Include với note** | Giảm HR nhẹ — flag trong xlsx |
| Diuretics | HCTZ, furosemide, indapamide, spironolactone | ✅ **Include** | Không ảnh hưởng PPG morphology |
| Combination | Telmisartan/HCTZ, Amlodipine/Valsartan, etc. | ✅ **Include** | Note loại nào trong cột medication_name |

#### Timing đo cho HTN-treated subjects

🎯 **Best practice cho thesis (capture BP cao hơn baseline-on-meds):**

```
Đo TRƯỚC liều buổi sáng — khi thuốc cũ đã wear off
 → BP có thể cao hơn baseline-on-meds 5-10 mmHg
 → Có cơ hội capture Stage 1/Stage 2 measured
 
Cấm đo ngay sau dose (2-4h):
 → Thuốc đang peak → BP thấp giả (artificial low)
 → Không reflect baseline BP của subject
```

#### Document chi tiết trong xlsx

Bắt buộc fill cột (xlsx schema v2 đã có):
- `hypertension`: Y/N (diagnosis)
- `medication`: Y/N
- `medication_name`: tên thuốc cụ thể
- `medication_dose_mg`: liều mg (vd "5", "10/12.5" cho combo)
- `medication_taken_today_hr_ago`: số giờ kể từ lần uống gần nhất (vd "8" nếu sáng mai chưa uống)
- `years_since_diagnosis`: số năm kể từ chẩn đoán
- `bp_category_diagnosed`: "None"/"Stage 1 HTN"/"Stage 2 HTN" (theo diagnosis)
- `bp_category_measured`: auto-derive từ sbp_baseline_mean

### 2.1.bis. Inter-arm BP screening (LẦN ĐẦU MỖI SUBJECT — 1 lần duy nhất)

🆕 **Thêm v1.1** — AHA 2005 (PMC8109470): inter-arm difference > 10 mmHg SBP có thể gợi ý subclavian stenosis.

```
☐ Đo Omron tay PHẢI 1 lần → ghi SBP_R, DBP_R
☐ Đo Omron tay TRÁI 1 lần → ghi SBP_L, DBP_L
☐ Tính |SBP_R − SBP_L|:
 - ≤ 10 mmHg: OK, dùng tay PHẢI cho tất cả sessions sau
 - > 10 mmHg: dùng tay có BP CAO HƠN cho tất cả sessions, NOTE rõ
 - > 20 mmHg: cân nhắc EXCLUDE subject + advise medical follow-up
```

→ Ghi vào Excel cột `inter_arm_diff_sbp` ở Session 1. Session 2 KHÔNG cần đo lại.

### 2.2. Lịch session

🆕 **v1.2 — Vai trò 2 sessions:**

| Session | Schedule | Vai trò trong calibration pipeline |
|---|---|---|
| **Session 1 (Anchor)** | Ngày X | Capture **calibration anchor** — đo cuff + PPG, server compute offset = real - predicted |
| **Session 2 (Validation)** | Ngày X+2 đến X+7, **cùng khung giờ ±1h** | **Validate** calibration — đo cuff + PPG (cuff làm ground truth), compute MAE post-calibration |

⚠️ **CẢ 2 sessions ĐỀU phải đo cuff + PPG** (không chỉ session 1). Session 2 cần cuff làm ground truth để eval calibration approach.

🆕 **v1.1:** 2 sessions **KHÁC NGÀY**, cách nhau **2-7 ngày**, **CÙNG khung giờ ±1h**.

**Lý do thay đổi (v1.1):**
- Cùng ngày sáng+chiều → fatigue + intra-day stress confound (không reflect intra-subject BP variability thực)
- Khác ngày cùng khung giờ → giảm circadian variation confound (BP sáng cao hơn chiều ~5 mmHg) + capture true day-to-day variability
- Reference: PracticalBP IMWUT 2025 (doi:10.1145/3749486) dùng sessions cách nhau nhiều ngày

→ Tối đa cách 7 ngày để tránh subject BP trend drift (vd thay đổi medication, lifestyle).

### 2.3. Pre-session (trước mỗi session, **10 phút** ⚠️ v1.1)

🆕 **v1.1 — UPDATED:** 5 phút → **10 phút** resting (Liang 2018 PPG-BP protocol — match training data).

- ☐ **Đi tiểu trước khi đo** (bàng quang đầy tăng BP +5-10 mmHg)
- ☐ Subject ngồi yên trên ghế tựa lưng
- ☐ Tay đặt trên bàn, lòng bàn tay ngửa lên, **cánh tay NGANG TIM** (ARMS Trial 2024 PMC11459360: arm unsupported = SBP +6.5 mmHg)
- ☐ Chân phẳng trên sàn, KHÔNG bắt chéo
- ☐ KHÔNG nói chuyện, KHÔNG xem điện thoại
- ☐ **Mở mắt, nhìn phía trước** (không nhắm mắt — giảm sympathetic arousal làm BP thấp giả ~3-5 mmHg)
- ☐ Đợi **10 phút** resting
- ☐ Verify ngón tay ấm (sờ thấy không lạnh) — nếu lạnh, bảo subject xoa tay 30-60s
- ☐ Ghi nhiệt độ phòng → cột `room_temp_c` Excel (target 20-25°C)

### 2.4. Đo BP cuff Omron (3 lần) — sequencing CONTRALATERAL

🆕 **v1.1 — Clarify placement:** Cuff tay PHẢI, PPG ngón TRỎ tay TRÁI (contralateral, theo Liang 2018 + ESH 2023). KHÔNG cùng tay (cuff inflation block PPG flow).

```
☐ Đo arm circumference tay PHẢI bằng thước dây → ghi cột `arm_circumference_cm`
 - Nếu 22-32 cm: Omron HEM-7143T1 standard cuff OK
 - Nếu < 22 hoặc > 32 cm: cần cuff size khác hoặc EXCLUDE
☐ Đeo cuff tay PHẢI, mép dưới cách khuỷu tay 1-2cm
☐ Cuff vừa khít (lùa 2 ngón tay vào, không lỏng không quá chặt)
☐ Cánh tay PHẢI đặt trên bàn, cuff NGANG TIM (đỉnh ngực)
 ⚠️ ARMS Trial 2024: arm unsupported = SBP +6.5 mmHg

Đo lần 1 → ghi SBP1, DBP1, HR1 vào Excel
Đợi 1 phút (subject vẫn ngồi yên)
Đo lần 2 → ghi SBP2, DBP2, HR2
Đợi 1 phút
Đo lần 3 → ghi SBP3, DBP3, HR3

→ TÍNH MEAN(lần 2, lần 3): KHÔNG dùng lần 1 (white-coat alerting effect)
 sbp_baseline = round((SBP2 + SBP3) / 2)
 dbp_baseline = round((DBP2 + DBP3) / 2)

⚠️ Reject criterion (ISO 81060-2:2018):
 - Nếu |SBP2 − SBP3| > 10 mmHg HOẶC |DBP2 − DBP3| > 8 mmHg:
 → Đo thêm lần 4, dùng MEAN(lần 3, lần 4)
 - Nếu cả 3 lần chênh > 15 mmHg: invalidate session, nghỉ 15 phút retry
```

### 2.5. Đo PPG (5 phút streaming) — gap ≤ 2 phút sau Omron lần 3

🆕 **v1.1 — Timing:** Bắt đầu PPG **trong vòng 2 phút** sau khi đo Omron lần 3 xong. Gap > 5 phút là không acceptable (BP có thể drift).

```
☐ Đeo MAX30102 ngón TRỎ TAY TRÁI (contralateral với cuff phải)
 ⚠️ KHÔNG dùng ngón cái (arterial supply khác)
☐ Áp lực vừa-mạnh (không quá nhẹ, không quá chặt)
☐ Subject thở **NORMAL BREATHING** — KHÔNG paced 6 BPM
 (paced breathing tăng HF HRV nhân tạo → bias LF/HF features cho ML)
☐ Mở mắt, nhìn phía trước; không nhìn màn hình điện thoại

☐ Trên Terminal 1 (laptop), nhập metadata khi prompt:
 Subject ID: S001 (mỗi session 1 ID riêng, vd S001a + S001b)
 Age: [tuổi]
 Sex: M/F
 Height: [cm]
 Weight: [kg]
 SBP baseline: [từ MEAN lần 2-3]
 DBP baseline: [từ MEAN lần 2-3]
 Hypertension: y/n
 Medication: y/n
 Smoking: y/n
 Finger: index (mặc định)
 Cuff arm: left (hoặc right consistent với cuff)
 Notes: [vd "session 1 sáng"]

☐ ENTER → server hiện "READY: S001a"
☐ Bắt đầu đếm 5 phút (timer điện thoại)
```

### 2.6. Theo dõi quality trong 5 phút

```
Quan sát Terminal 1 mỗi 5s:
 [BATCH #001] S001a fs=100 +500 → total=500 samples
 [BATCH #002] S001a fs=100 +500 → total=1000 samples
 ...
 [BATCH #060] S001a fs=100 +500 → total=30000 samples

Quan sát Tera Term — mỗi batch in:
 AC%=X.XX% (n=520) [OK / too flat]

⭐ TARGET: AC% ≥ 0.5% (lý tưởng 1-5%)

Nếu AC% < 0.3% liên tục:
 ☐ ĐIỀU S01ỈNH placement: nhấn ngón tay chặt hơn 1 chút
 ☐ KHÔNG rút ngón tay ra (sẽ mất calibration)
 ☐ Đợi 30s xem AC% có tăng không
 ☐ Nếu vẫn poor sau 1 phút → STOP session, retry với placement mới
```

### 2.7. Kết thúc session (5 phút)

```
☐ Press ENTER trên Terminal 1 (lần 2)
☐ Server hỏi "SBP post (mmHg, ENTER = skip):"
☐ Đo Omron lần 4 NGAY (subject vẫn ngồi yên)
☐ Nhập SBP_post, DBP_post → ENTER
☐ Server log: "[SAVE] S001a_xxx.csv — 30000 samples"
☐ Verify CSV xuất hiện trong collected_data/
```

### 2.8. Post-session checks

```
☐ Update Excel:
 - csv_file: tên file vừa save
 - duration_s: thời gian thực
 - signal_quality_notes: "good" / "fair" / "có vài batch poor"
 - general_notes: gì đó đặc biệt

☐ Check BP_post vs BP_baseline:
 - Chênh ≤ 10 mmHg: OK, BP stable trong session
 - Chênh > 15 mmHg: cuff lỏng / lần đo có vấn đề / stress factor
 → Note vào Excel để loại nếu cần lúc training

☐ Cảm ơn subject + schedule session 2 (nếu chưa)
☐ Lau MAX30102 cồn 70% trước subject tiếp theo
```

---

## 3. DECISION TREE — Xử lý tình huống

### 3.1. Subject từ chối / không đến giờ hẹn
```
→ OK, không ép. Ghi nhận vào Excel "no-show" hoặc "declined"
→ Liên hệ ứng viên dự phòng từ list ban đầu
→ Mục tiêu: 15 subjects net cuối cùng
```

### 3.2. SBP > 180 hoặc DBP > 120
```
→ DỪNG đo NGAY (severe HTN, không tự ý đo nhiều lần gây stress)
→ Khuyên subject đến phòng khám/bệnh viện kiểm tra trong tuần
→ Ghi nhận: "excluded - severe HTN, recommended medical follow-up"
→ Tìm subject thay thế
```

### 3.3. Signal poor (AC% < 0.3% liên tục) trong 5 phút đầu
```
1. Điều chỉnh ngón tay (nhấn chặt hơn / nhẹ hơn)
2. Thử ngón giữa thay vì trỏ
3. Lau lại MAX30102 + đầu ngón tay subject
4. Verify finger nhiệt độ ấm (tay lạnh → vasoconstriction → AC thấp)
 → Nếu lạnh: bảo subject xoa tay 30s

→ Nếu sau 3 lần thử vẫn poor → loại session
→ Note Excel: "loại session 1 (signal poor), retry session 2 ngày sau"
```

### 3.4. > 10 batches `fails` trong 1 session
```
1. Verify Terminal 1 vẫn chạy log_ppg_local.py
2. Verify Tera Term hiện "[UPLOAD] 200 OK"
3. Nếu không: kiểm tra WiFi router, restart firmware ESP32

→ Loại session, restart fresh
```

### 3.5. Subject cử động tay nhiều
```
1. Nhắc nhẹ nhàng "Cố gắng giữ tay yên thêm 3 phút nữa nhé"
2. Cho subject xem điện thoại/đọc sách bằng tay KHÁC (không phải tay đo PPG)
3. Nếu vẫn cử động: dán băng y tế nhẹ giữ sensor (KHÔNG quá chặt — gây ischemia)

→ Nếu motion artifact > 50% session → loại session, schedule lại
```

### 3.6. ESP32 mất kết nối WiFi giữa session
```
1. Verify Tera Term thấy "[WIFI] Disconnected"
2. Đợi 30s → firmware tự reconnect
3. Nếu không reconnect: power cycle ESP32 → calibration phải làm lại
4. Loại session, schedule lại
```

### 3.7. BP_post chênh > 20 mmHg vs BP_baseline
```
→ Bình thường nếu khác giờ (sáng vs chiều).
→ Cùng session mà chênh nhiều: có thể cuff lỏng / lần đo lỗi
 - Đo lại 1 lần Omron để verify
 - Chọn giá trị consistent hơn cho training label
→ Note rõ trong Excel
```

### 3.8. Recruit < 12 subjects sau tuần 1
```
→ Snowball sampling: hỏi subjects đã có giới thiệu thêm
→ Liên hệ thầy hướng dẫn / lab mate
→ Extend timeline thêm 3-5 ngày
→ Ưu tiên Stage 2 HTN (group khó nhất)
```

---

## 4. CUỐI MỖI NGÀY (~10 phút)

### 4.1. Backup data
```bash
# Copy folder collected_data/ lên Google Drive
# Hoặc zip + upload thủ công
```

- ☐ Backup folder `collected_data/` (chứa tất cả CSV trong ngày)
- ☐ Backup file Excel `subjects_metadata.xlsx`
- ☐ Verify backup OK trên Drive (mở thử 1 file)

### 4.2. Quick QC

Run command:
```bash
python -c "
import pandas as pd
from pathlib import Path
for csv in sorted(Path(r'c:/Users/Acer/Desktop/PPG monitor/collected_data').glob('*.csv')):
 df = pd.read_csv(csv, comment='#')
 n = len(df)
 duration = (df.timestamp_ms.max() - df.timestamp_ms.min()) / 1000 if n > 0 else 0
 print(f'{csv.name}: {n} samples, {duration:.0f}s')
"
```

→ Verify:
- ☐ Mỗi file có ≥ 25,000 samples (5 phút @ 100Hz)
- ☐ Duration ≥ 250 giây
- ☐ Không có file 0 samples (failed save)

### 4.3. Update progress log
- ☐ Ghi vào Excel/notes: hôm nay đo được bao nhiêu subjects, sessions
- ☐ Plan cho ngày mai: subjects nào tiếp theo

---

## 5. TIMELINE 2 TUẦN

### Tuần 1 (Ngày 1-7) — Recruit + Pilot

| Ngày | Hoạt động |
|---|---|
| 1 | Verify firmware + self-test 2-3 lần để familiar |
| 2-3 | S001-S002 (gia đình normotensive) — 2 sessions/subject |
| 4-5 | S003-S005 (mở rộng normal group) |
| 6 | QC giữa tuần — verify pipeline + kiểm tra signal quality |
| 7 | S006-S007 (bắt đầu Stage 1 HTN group) |

### Tuần 2 (Ngày 8-14) — Hoàn thành + Train

| Ngày | Hoạt động |
|---|---|
| 8-10 | S008-S012 (Stage 1 HTN + bắt đầu Stage 2) |
| 11-12 | S013-S015 (hoàn thành Stage 2 HTN — group khó nhất) |
| 13 | QC final + chuẩn bị training pipeline |
| 14 | Run AutoGluon LOSO + sinh Bland-Altman + viết results |

---

## 6. METRICS TARGET (CHO THESIS)

Sau khi xong tất cả 15 subjects:

| Metric | Baseline (PPG-BP only) | Target (Combined + Calibration) |
|---|---|---|
| MAE SBP | 15.41 mmHg | **8-10 mmHg** |
| MAE DBP | 9.15 mmHg | **5-7 mmHg** |
| BHS Grade SBP | D | **B-C** |
| BHS Grade DBP | C | **A-B** |

→ Chấp nhận **MAE > 5 mmHg IEEE standard** — cite Moulaeifard, Charlton & Strodthoff 2025 (cross-dataset calibration-free MAE SBP ~15–25 mmHg điển hình toàn ngành cho 5 DL architectures: LeNet1D, XResNet1d50/101, Inception1D, S4 — arXiv:2502.19167).

---

## 7. ANCHOR POINTS — KHÔNG ĐƯỢC QUÊN

🔴 **Resting 10 PHÚT trước session** (v1.1, không phải 5 phút)
🔴 **Đo BP 3 lần, dùng MEAN(lần 2, 3) — LOẠI lần 1**
🔴 **CONTRALATERAL: Cuff tay PHẢI, PPG ngón TRỎ tay TRÁI** — KHÔNG cùng tay
🔴 **Tay PHẢI đặt trên bàn, cuff ngang tim** (ARMS 2024)
🔴 **Ngón TRỎ tay TRÁI — KHÔNG dùng ngón cái**
🔴 **Inter-arm BP screening lần đầu mỗi subject** (>10 mmHg note, >20 exclude)
🔴 **2 sessions KHÁC ngày, cách 2-7 ngày, cùng khung giờ ±1h** (v1.1)
🔴 **Đi tiểu trước khi đo + Mở mắt + Normal breathing**
🔴 **Loại beta-blockers + Raynaud's + AF + cuff size ngoài 22-32 cm**
🔴 **AC% ≥ 0.5% xuyên suốt 5 phút — không rút tay**
🔴 **Press ENTER 2 lần trên console laptop:**
 - Lần 1: sau khi nhập metadata
 - Lần 2: sau 5 phút đo PPG → save CSV
🔴 **Backup CSV cuối ngày — Google Drive**

---

## 8. CONTACT IF STUCK

- Pipeline error → tham khảo `README.md` (cùng folder)
- Firmware issue → đọc Serial Monitor + check log gần nhất trong `Log/`
- BP cuff issue → consult Omron HEM-7143T1 manual (kèm theo máy)

---

> **In file này, kẹp clipboard, mang theo khi đo.** ✊
> **Tick ☐ → ✅ từng bước, không skip.**
