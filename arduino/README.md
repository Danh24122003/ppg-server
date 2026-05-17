# Arduino Firmware

Embedded firmware for the wearable edge node. Acquires reflectance PPG from the MAX30102 optical front-end at 100 Hz, buffers five-second batches, and uploads JSON payloads to the cloud back-end over Wi-Fi.

## Hardware Targets

| Platform | Status | Pin map |
|----------|--------|---------|
| **NodeMCU-32S (Ai-Thinker)** | Empirical validation platform used throughout the thesis | I²C SDA=21, SCL=22, status LED=2 |
| **Seeed Studio XIAO ESP32-S3** | Target architecture for the production deliverable; firmware port prepared | I²C SDA=5, SCL=6, status LED=21 |

Both targets are functionally equivalent for the acquisition path (MAX30102 at 100 Hz, single-channel infrared at 880 nm, batch upload every five seconds). The empirical numbers reported in the thesis were collected on the NodeMCU-32S; the XIAO migration target is documented for the post-defense engineering phase.

## Files

| File | Purpose |
|------|---------|
| `ppg_v4_firmware.ino` | Production firmware. FreeRTOS dual-core task partitioning (`sampleTask`, `calibTask`, `sendTask`, `uploadTask`), automatic LED-current calibration, HTTP/HTTPS auto-detection, NTP-bypass safety fallback. |
| (XIAO-port `.ino`) | XIAO ESP32-S3 pin-mapped variant (compile-tested, flash test deferred). |
| `README_pinmap.md` | Detailed pin-mapping table for both platforms. |

## Build

Open `ppg_v4_firmware.ino` in the Arduino IDE (≥ 2.0) with the ESP32 core installed (≥ 2.0.5). Select the target board (NodeMCU-32S or XIAO ESP32-S3), open `Secrets.h` (a header you create locally) to fill in Wi-Fi credentials and back-end URL, then upload.

A template `Secrets.example.h` should be created if this firmware is shared with new contributors; it should declare four constants and **not** be committed with real credentials:

```cpp
#define WIFI_SSID "your_ssid"
#define WIFI_PASSWORD "your_password"
#define BACKEND_URL "https://ppg-backend-udze.onrender.com/api/ppg/upload"
#define DEVICE_AUTH_TOKEN "your_token"
```

## Configuration Defaults

| Parameter | Value | Source |
|-----------|------:|--------|
| Sample rate | 100 Hz | MAX30102 SR = 100, AVG = 8 |
| LED pulse width | 411 µs | maximum sensitivity for 18-bit ADC |
| LED current target | calibrated to mean(IR) ∈ [80 000, 140 000] | `calibTask` automatic sweep |
| Batch size | 500 samples (5 seconds) | matches cloud-side window length |
| Upload retry | exponential back-off, queue cap = 10 | survives transient Wi-Fi drops |

## References

- Chapter 3 §3.1.2 — MAX30102 specifications
- Chapter 3 §3.1.3 — ESP32 platforms (testing vs. target)
- Chapter 3 §3.2 — embedded firmware architecture
- Chapter 3 §3.2.2 — FreeRTOS task layout (Table 3.1)
- Chapter 3 §3.2.3 — automatic LED calibration
- Chapter 3 §3.2.4 — HTTP/HTTPS transport detection
- Chapter 3 §3.2.5 — NTP-bypass robustness
