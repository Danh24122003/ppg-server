/*
==================================================
 PPG v4 Firmware — ESP32 DevKit + MAX30102
 Tương thích Backend v4.3 + ML Server ppg-ml.onrender.com
==================================================
 Target board : ESP32 DevKit / NodeMCU-32S (WROOM-32)
 Sensor       : MAX30102 reflectance PPG
 I2C pins     : SDA=21, SCL=22 (default ESP32 DevKit)
 LED built-in : GPIO 2
 BOOT button  : GPIO 0 (giữ 5s → reset WiFi)
==================================================
 Thay đổi chính so với v3 (ppg_optimized.ino):
 [P0] LED_MODE=2 đúng cách — không toggle RED bằng tay
      (bỏ delayMicroseconds + loop 5-sample RED snapshot)
 [P0] calibTask tách khỏi sampleTask — không block Core 0
 [P0] CALIB_THRESHOLD 50000 (đúng spec 18-bit ADC)
 [P0] NTP offset = 0 (UTC) khớp timestamp format ".000Z"
 [P0] Default server → https://ppg-ml.onrender.com/api/ppg/upload
 [P0] Mutex bảo vệ batch[] + batchIndex
 [P1] Header X-Device-Token (optional qua WiFiManager portal)
 [P1] Validate device_id (regex [a-zA-Z0-9_-], len 1-64)
 [P1] sample_rate tính từ timestamps thực tế (không hardcode 100)
 [P1] gmtime_r + strftime (bỏ self-implement calendar)
 [P1] UPLOAD_TIMEOUT_MS giảm 60s → 15s
 [P2] JSON payload thêm has_gaps + drop_count
 [P2] Parse response v4.3: pnn20, reliability, lf_hf, ratio_r,
      perfusion_index, ml_predictions.blood_pressure
==================================================
 Version: 4.1.0
 v4.1.0 changes (fs 50Hz → 100Hz):
   - SENSOR_RATE_HZ 200 → 400 (avg=4 unchanged → effective 100Hz)
   - BATCH_SIZE 250 → 500 (5s window @ 100Hz)
   - PPG_QUEUE_SIZE 400 → 800 (~8s buffer @ 100Hz)
   - JSON_DOC_SIZE 16384 → 32768 (500 samples ~22.5KB payload)
   - MIN_VALID_PARTIAL 150 → 300 (3s minimum @ 100Hz)
   - Lý do: HRV pNN20 reliable hơn, BP ML model trained at 100Hz
==================================================
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <Wire.h>
#include "MAX30105.h"
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/queue.h>
#include <freertos/semphr.h>
#include <NTPClient.h>
#include <WiFiUdp.h>
#include <ArduinoJson.h>
#include "esp_timer.h"
#include <WiFiManager.h>
#include <Preferences.h>
#include <time.h>

// ================= FIRMWARE VERSION =================
#define FW_VERSION "4.1.0"

// ================= ESP32 DevKit PIN MAP =================
#define SDA_PIN        21     // ESP32 DevKit default SDA
#define SCL_PIN        22     // ESP32 DevKit default SCL
#define LED_PIN        2      // Built-in LED (GPIO 2)
#define BOOT_BTN       0      // Boot button (giữ 5s → reset WiFi)

// ================= SENSOR CONFIG =================
// v4.1.0: 400Hz / avg4 = 100Hz effective FIFO rate.
// FIFO depth = 32 slots → 320ms before overflow.
// sampleTask poll interval = 20ms → drains FIFO 16x faster than fill rate
// → safe margin. Khác với case cũ (400/avg3=133Hz overflow 240ms) vì:
//   (1) avg=4 giữ alignment đúng, không bị skip slot
//   (2) sampleTask đã tách khỏi calibTask (v4.0.9) → không bị block
// Trade-off so với 50Hz: payload 8KB→16KB JSON, heap usage +8KB/batch,
// nhưng HRV pNN20 reliable hơn (threshold 20ms = 2 samples vs 1 sample),
// và khớp BP ML model training fs=100Hz.
#define SENSOR_RATE_HZ     400
#define SAMPLE_AVG         4
#define LED_MODE_DUAL      2      // Dual LED IR + RED — sensor tự alternate
#define PULSE_WIDTH_US     411
#define ADC_RANGE          16384

#define IR_MIN             5000
#define IR_MAX             262000

// Finger detection — spec 18-bit: mean(IR) > 50,000
#define CALIB_THRESHOLD    50000

// Calibration DC targets
#define IR_DC_LOW          80000
#define IR_DC_HIGH         140000
// Bump per researcher recommendation (Bent 2021 + Analog Devices SNR guideline):
// Force calibration pick higher LED power → DC ~95k → AC scales linearly →
// noise floor electrical unchanged → SNR cải thiện → ratio_r ổn định hơn
#define RED_DC_LOW         80000
#define RED_DC_HIGH        160000  // 61% full scale 18-bit ADC, headroom AC 39%

// ================= SYSTEM CONFIG =================
#define BATCH_SIZE         500    // 5 giây @ 100Hz effective
#define PPG_QUEUE_SIZE     800    // Buffer ~8s @ 100Hz
#define UPLOAD_QUEUE_SIZE  10  // 50s buffer in-RAM cho WiFi disconnect ngắn
#define JSON_DOC_SIZE      32768  // 32KB — safe margin cho 500 samples (~22.5KB raw)
#define RESPONSE_DOC_SIZE  4096
#define WIFI_RESET_HOLD_MS 5000
#define NTP_RESYNC_MS      60000
#define STATS_INTERVAL_MS  5000
#define UPLOAD_RETRY       3
#define UPLOAD_TIMEOUT_MS  15000  // giảm từ 60s — fail fast

// Backend validator range
#define MIN_SAMPLE_RATE    25
#define MAX_SAMPLE_RATE    400
#define MIN_PARTIAL_FLUSH  50     // backend MIN_SAMPLES
#define MIN_VALID_PARTIAL  300    // 3s @ 100Hz — batch nhỏ hơn sẽ bị discard khi finger removed

// ================= STRUCTS =================
struct PPGSample {
  uint32_t ir;
  uint32_t red;
  uint64_t t;  // ms epoch
};

struct UploadJob {
  char* payload;
  size_t len;
};

// ================= FSM =================
enum SysState {
  WAIT_FINGER,
  CALIBRATING_IR,
  CALIBRATING_RED,
  STREAMING
};
volatile SysState sysState = WAIT_FINGER;

// ================= GLOBAL =================
MAX30105 particleSensor;
WiFiUDP ntpUDP;
NTPClient timeClient(ntpUDP, "pool.ntp.org", 0);  // UTC — khớp suffix "Z"
Preferences prefs;

QueueHandle_t ppgQueue = nullptr;
QueueHandle_t uploadQueue = nullptr;
SemaphoreHandle_t batchMutex = nullptr;

volatile bool streamingEnabled = false;
volatile bool calibrated = false;

uint8_t ledPower = 0x40;
uint8_t redPower = 0x40;

// timeOffsetMs là 64-bit, ESP32 LX6 là 32-bit CPU → ghi/đọc cần 2 word
// → bảo vệ bằng spinlock (portMUX) tránh word-tearing cross-core
volatile uint64_t timeOffsetMs = 0;
portMUX_TYPE timeMux = portMUX_INITIALIZER_UNLOCKED;
portMUX_TYPE dropsMux = portMUX_INITIALIZER_UNLOCKED;

// Batch buffer — SRAM heap
PPGSample* batch = nullptr;
int batchIndex = 0;

// WiFiManager params — persisted to Preferences
// Production (Render Cloud) — commented out, dùng cho deployment chính thức:
//char serverUrl[128] = "https://ppg-ml.onrender.com/api/ppg/upload";
// Self-collect local laptop (default) — đổi IP nếu DHCP đổi:
char serverUrl[128] = "http://192.168.1.100:8080/api/ppg/upload";
char deviceId[32]   = "esp32-001";
char apiKey[64]     = "";  // Rỗng = dev mode (backend không enforce auth)

// Stats
volatile uint32_t samplesOut   = 0;
volatile uint32_t dropsOut     = 0;
volatile uint32_t batchesSent  = 0;
volatile uint32_t uploadFails  = 0;
volatile uint32_t partialDiscards = 0;

// Debug/diag
unsigned long lastStatsTime = 0;
unsigned long lastNtpSync = 0;
uint32_t redSampleCount = 0;
uint32_t redMinValue = 999999;
uint32_t redMaxValue = 0;
uint64_t redSumValue = 0;  // uint64 tránh overflow

// Forward declarations (sendTask gọi printStats định nghĩa ở dưới)
void printStats();

// ================= HELPERS =================
inline uint64_t nowMs() {
  // Atomic read 64-bit qua spinlock — sampleTask Core 0 đọc lúc Core 1 đang ghi NTP
  portENTER_CRITICAL(&timeMux);
  uint64_t offset = timeOffsetMs;
  portEXIT_CRITICAL(&timeMux);
  return esp_timer_get_time() / 1000ULL + offset;
}

void debugPrint(const char* stage, const char* msg) {
  Serial.printf("[%s] %s\n", stage, msg);
}

void printMemoryInfo() {
  Serial.printf("  Heap Free: %u KB | Min: %u KB\n",
                ESP.getFreeHeap() / 1024, ESP.getMinFreeHeap() / 1024);
}

// ================= ISO TIMESTAMP (UTC) =================
// Dùng gmtime_r + strftime — không tự implement calendar loop
String getISOTimestamp() {
  uint64_t ms = nowMs();
  time_t sec = (time_t)(ms / 1000);
  struct tm t;
  gmtime_r(&sec, &t);
  char buf[32];
  strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S.000Z", &t);
  return String(buf);
}

// ================= CONFIG VALIDATION =================
// Device ID: chỉ chấp nhận [a-zA-Z0-9_-], len 1-64
bool isValidDeviceId(const char* s) {
  if (!s) return false;
  size_t n = strlen(s);
  if (n < 1 || n > 64) return false;
  for (size_t i = 0; i < n; i++) {
    char c = s[i];
    bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '_' || c == '-';
    if (!ok) return false;
  }
  return true;
}

// ================= PREFERENCES =================
void loadConfig() {
  prefs.begin("ppg", true);
  String url = prefs.getString("server_url", serverUrl);
  String did = prefs.getString("device_id", deviceId);
  String key = prefs.getString("api_key", apiKey);
  url.toCharArray(serverUrl, sizeof(serverUrl));
  did.toCharArray(deviceId, sizeof(deviceId));
  key.toCharArray(apiKey, sizeof(apiKey));
  prefs.end();

  // Guarantee null-terminated
  serverUrl[sizeof(serverUrl) - 1] = '\0';
  deviceId[sizeof(deviceId) - 1] = '\0';
  apiKey[sizeof(apiKey) - 1] = '\0';

  // Validate device_id — fallback nếu invalid
  if (!isValidDeviceId(deviceId)) {
    debugPrint("CONFIG", "Invalid device_id, using default");
    strncpy(deviceId, "esp32-001", sizeof(deviceId) - 1);
    deviceId[sizeof(deviceId) - 1] = '\0';
  }
}

void saveConfig() {
  prefs.begin("ppg", false);
  prefs.putString("server_url", serverUrl);
  prefs.putString("device_id", deviceId);
  prefs.putString("api_key", apiKey);
  prefs.end();
}

// ================= WIFI AUTO-RECONNECT =================
// Driver tự reconnect qua setAutoReconnect(true) — chỉ log thôi
void onWiFiDisconnect(WiFiEvent_t event, WiFiEventInfo_t info) {
  Serial.printf("[WIFI] Disconnected reason=%d — auto-reconnecting\n",
                (int)info.wifi_sta_disconnected.reason);
}

void onWiFiConnected(WiFiEvent_t event, WiFiEventInfo_t info) {
  Serial.printf("[WIFI] Reconnected, IP=%s RSSI=%d\n",
                WiFi.localIP().toString().c_str(), WiFi.RSSI());
}

// ================= WIFI MANAGER =================
void initWiFiManager() {
  debugPrint("WIFI", "Starting WiFiManager...");
  loadConfig();

  WiFiManager wm;
  wm.setConnectTimeout(15);
  wm.setConfigPortalTimeout(180);

  WiFiManagerParameter param_url("server_url",
      "Server URL (ppg-ml.onrender.com)", serverUrl, 128);
  WiFiManagerParameter param_did("device_id",
      "Device ID (a-z 0-9 _ -)", deviceId, 32);
  WiFiManagerParameter param_key("api_key",
      "API Key (optional, leave empty for dev)", apiKey, 64);

  wm.addParameter(&param_url);
  wm.addParameter(&param_did);
  wm.addParameter(&param_key);

  // BOOT giữ 5s → reset WiFi credentials
  pinMode(BOOT_BTN, INPUT_PULLUP);
  if (digitalRead(BOOT_BTN) == LOW) {
    unsigned long ps = millis();
    while (digitalRead(BOOT_BTN) == LOW && millis() - ps < WIFI_RESET_HOLD_MS) {
      delay(100);
    }
    if (millis() - ps >= WIFI_RESET_HOLD_MS) {
      debugPrint("WIFI", "BOOT 5s -> Reset credentials");
      wm.resetSettings();
    }
  }

  bool connected = wm.autoConnect("PPG-Setup");
  if (!connected) {
    debugPrint("WIFI", "Failed to connect -> Restart");
    ESP.restart();
  }

  // Đăng ký event auto-reconnect (thay cho gọi reconnect() thủ công trong uploadTask)
  WiFi.onEvent(onWiFiDisconnect, ARDUINO_EVENT_WIFI_STA_DISCONNECTED);
  WiFi.onEvent(onWiFiConnected,  ARDUINO_EVENT_WIFI_STA_GOT_IP);
  WiFi.setAutoReconnect(true);

  debugPrint("WIFI", "Connected");
  Serial.printf("  IP: %s RSSI: %d\n",
                WiFi.localIP().toString().c_str(), WiFi.RSSI());

  // Pull params từ portal vào buffer
  strncpy(serverUrl, param_url.getValue(), sizeof(serverUrl) - 1);
  serverUrl[sizeof(serverUrl) - 1] = '\0';

  const char* didRaw = param_did.getValue();
  if (isValidDeviceId(didRaw)) {
    strncpy(deviceId, didRaw, sizeof(deviceId) - 1);
    deviceId[sizeof(deviceId) - 1] = '\0';
  } else {
    Serial.printf("  WARNING: device_id '%s' invalid, keep '%s'\n",
                  didRaw, deviceId);
  }

  strncpy(apiKey, param_key.getValue(), sizeof(apiKey) - 1);
  apiKey[sizeof(apiKey) - 1] = '\0';

  saveConfig();

  Serial.printf("  Server : %s\n", serverUrl);
  Serial.printf("  Device : %s\n", deviceId);
  Serial.printf("  ApiKey : %s\n", strlen(apiKey) > 0 ? "(set)" : "(empty - dev mode)");

  WiFi.setSleep(false);
}

// ================= NTP =================
void syncNTP() {
  debugPrint("NTP", "Syncing UTC...");
  timeClient.begin();
  int attempts = 0;
  while (!timeClient.update() && attempts < 10) {
    delay(500);
    attempts++;
  }
  if (attempts >= 10) {
    debugPrint("NTP", "Sync failed -> local time (timestamp may be off)");
  } else {
    uint64_t ntpMs = (uint64_t)timeClient.getEpochTime() * 1000ULL;
    uint64_t espMs = esp_timer_get_time() / 1000ULL;
    portENTER_CRITICAL(&timeMux);
    timeOffsetMs = ntpMs - espMs;
    portEXIT_CRITICAL(&timeMux);
    Serial.printf("[NTP] Synced UTC epoch=%llu\n", ntpMs / 1000ULL);
  }
  lastNtpSync = millis();
}

// ================= CALIBRATION (chạy từ calibTask, KHÔNG block sampleTask) =================
bool calibrateIR() {
  debugPrint("CALIB_IR", "Starting...");
  for (uint8_t p = 0x20; p <= 0x80; p += 0x08) {
    particleSensor.setPulseAmplitudeIR(p);
    particleSensor.setPulseAmplitudeRed(0);
    delay(300);
    // Quick finger-gone check — dùng IR_MIN (5000), KHÔNG dùng CALIB_THRESHOLD (50000)
    // vì ở LED power thấp (p=0x20) reading có thể chỉ ~25000 dù finger vẫn ở
    particleSensor.check();
    if (particleSensor.available()) {
      uint32_t quickIR = particleSensor.getFIFOIR();
      particleSensor.nextSample();
      if (quickIR < IR_MIN) {
        particleSensor.setPulseAmplitudeIR(0);
        particleSensor.setPulseAmplitudeRed(0);
        debugPrint("CALIB", "Finger removed during calibration — abort");
        return false;
      }
    }
    particleSensor.clearFIFO();

    uint64_t sum = 0;
    int n = 0;
    unsigned long t0 = millis();
    while (n < 50 && millis() - t0 < 1000) {
      particleSensor.check();
      if (particleSensor.available()) {
        sum += particleSensor.getFIFOIR();  // tail position — sequential drain
        particleSensor.nextSample();
        n++;
      }
    }
    if (n == 0) continue;
    uint32_t dc = sum / n;
    Serial.printf("  IR=0x%02X DC=%lu", p, dc);

    if (dc > IR_DC_LOW && dc < IR_DC_HIGH) {
      ledPower = p;
      Serial.println(" OK");
      return true;
    }
    Serial.println(dc < IR_DC_LOW ? " (low)" : " (high)");
  }
  debugPrint("CALIB_IR", "FAILED");
  return false;
}

bool calibrateRED() {
  debugPrint("CALIB_RED", "Starting...");
  uint8_t startP = (uint8_t)(ledPower * 0.85);
  bool found = false;
  for (uint8_t p = startP; p <= 0x7F; p += 0x08) {
    // Giữ IR ở ledPower (calibrated) — match streaming LED_MODE=2 dual condition
    particleSensor.setPulseAmplitudeIR(ledPower);
    particleSensor.setPulseAmplitudeRed(p);
    delay(300);
    particleSensor.clearFIFO();
    delay(20);
    // Finger-gone check — IR_MIN (5000) consistent với calibrateIR
    particleSensor.check();
    if (particleSensor.available()) {
      uint32_t quickIR = particleSensor.getFIFOIR();
      particleSensor.nextSample();
      if (quickIR < IR_MIN) {
        // Cleanup: tắt cả 2 LED trước khi return
        particleSensor.setPulseAmplitudeIR(0);
        particleSensor.setPulseAmplitudeRed(0);
        debugPrint("CALIB_RED", "Finger removed — abort");
        return false;
      }
    }
    particleSensor.clearFIFO();

    uint64_t sum = 0;
    int n = 0;
    unsigned long t0 = millis();
    while (n < 50 && millis() - t0 < 1000) {
      particleSensor.check();
      if (particleSensor.available()) {
        sum += particleSensor.getFIFORed();  // tail position — sequential drain
        particleSensor.nextSample();
        n++;
      }
    }
    if (n == 0) continue;
    uint32_t dc = sum / n;
    Serial.printf("  RED=0x%02X DC=%lu", p, dc);

    if (dc > RED_DC_LOW && dc < RED_DC_HIGH) {
      redPower = p;
      Serial.println(" OK");
      found = true;
      break;
    }
    Serial.println(dc < RED_DC_LOW ? " (low)" : " (high)");
  }
  if (!found) {
    redPower = 0x60;
    debugPrint("CALIB_RED", "Fallback 0x60");
  }
  return found;
}

// Sau calibration — khởi động lại sensor ở LED_MODE=2 (Dual LED)
void startStreamingMode() {
  particleSensor.shutDown();
  delay(50);
  particleSensor.wakeUp();
  delay(50);
  // LED_MODE=2 — sensor tự bật cả IR + RED xen kẽ trong mỗi cycle
  particleSensor.setup(ledPower, SAMPLE_AVG, LED_MODE_DUAL,
                       SENSOR_RATE_HZ, PULSE_WIDTH_US, ADC_RANGE);
  particleSensor.setPulseAmplitudeIR(ledPower);
  particleSensor.setPulseAmplitudeRed(redPower);
  particleSensor.clearFIFO();

  calibrated = true;
  streamingEnabled = true;
  sysState = STREAMING;

  redSampleCount = 0;
  redMinValue = 999999;
  redMaxValue = 0;
  redSumValue = 0;
  lastStatsTime = millis();

  debugPrint("STREAM", "Started — LED_MODE=2 dual sampling");
  Serial.printf("  ledPower=0x%02X redPower=0x%02X\n", ledPower, redPower);
}

// ================= SAMPLE TASK (Core 0) — chỉ đọc FIFO khi STREAMING =================
void sampleTask(void*) {
  debugPrint("TASK", "Sample task started (Core 0)");
  PPGSample s;

  while (true) {
    // Chỉ đọc sensor khi đang ở WAIT_FINGER (check finger) hoặc STREAMING (sample)
    // Khi CALIBRATING_* → yield cho calibTask độc quyền sensor
    if (sysState == CALIBRATING_IR || sysState == CALIBRATING_RED) {
      vTaskDelay(pdMS_TO_TICKS(20));
      continue;
    }

    // Periodic sensor health check — recover từ I2C glitch / sensor reset
    static unsigned long lastHealthCheck = 0;
    if (millis() - lastHealthCheck > 10000) {
      uint8_t partId = particleSensor.readPartID();
      if (partId != 0x15) {  // MAX30102 expected = 0x15
        Serial.printf("[SENSOR] Lost (PartID=0x%02X) — reinit\n", partId);
        Wire.end();
        delay(100);
        Wire.begin(SDA_PIN, SCL_PIN);
        Wire.setClock(400000);
        if (particleSensor.begin(Wire, I2C_SPEED_FAST)) {
          particleSensor.setup(0x40, SAMPLE_AVG, LED_MODE_DUAL,
                               SENSOR_RATE_HZ, PULSE_WIDTH_US, ADC_RANGE);
          particleSensor.clearFIFO();
          Serial.println("[SENSOR] Reinit OK — restarting calibration");
          calibrated = false;
          streamingEnabled = false;
          sysState = WAIT_FINGER;
        } else {
          Serial.println("[SENSOR] Reinit FAILED — will retry next cycle");
        }
      }
      lastHealthCheck = millis();
    }

    particleSensor.check();
    while (particleSensor.available()) {
      // getFIFOIR/getFIFORed → đọc từ sense.tail (sequential drain buffer).
      // getIR/getRed lại trả về sense.head (latest only) — dùng sai sẽ lặp lại
      // sample mới nhất nhiều lần và mất mẫu cũ chưa đọc.
      uint32_t ir  = particleSensor.getFIFOIR();
      uint32_t red = particleSensor.getFIFORed();
      particleSensor.nextSample();  // advance tail

      // WAIT_FINGER → chờ ngón tay
      if (sysState == WAIT_FINGER) {
        if (ir > CALIB_THRESHOLD) {
          debugPrint("DETECT", "Finger detected -> calibrate");
          sysState = CALIBRATING_IR;  // calibTask pick up
          break;  // thoát vòng while(available) — nhường sensor
        }
        continue;
      }

      // STREAMING → finger removed → về WAIT
      if (sysState == STREAMING && ir < CALIB_THRESHOLD) {
        debugPrint("DETECT", "Finger removed");
        calibrated = false;
        streamingEnabled = false;
        sysState = WAIT_FINGER;
        break;
      }

      // STREAMING — push vào queue
      if (sysState == STREAMING) {
        if (ir < IR_MIN || ir > IR_MAX) continue;

        s.ir = ir;
        s.red = red;
        s.t = nowMs();

        // Diag RED
        redSampleCount++;
        redSumValue += red;
        if (red < redMinValue) redMinValue = red;
        if (red > redMaxValue) redMaxValue = red;

        if (xQueueSend(ppgQueue, &s, 0) != pdTRUE) {
          portENTER_CRITICAL(&dropsMux);
          dropsOut++;
          portEXIT_CRITICAL(&dropsMux);
        } else {
          samplesOut++;
        }
      }
    }
    taskYIELD();
  }
}

// ================= CALIB TASK (Core 0) — độc quyền sensor khi calibrate =================
void calibTask(void*) {
  debugPrint("TASK", "Calib task started (Core 0)");
  while (true) {
    if (sysState == CALIBRATING_IR) {
      bool ok = calibrateIR();
      if (ok) {
        sysState = CALIBRATING_RED;
      } else {
        sysState = WAIT_FINGER;  // lặp lại khi đặt tay
      }
    } else if (sysState == CALIBRATING_RED) {
      bool okRed = calibrateRED();
      if (okRed) {
        startStreamingMode();
      } else {
        sysState = WAIT_FINGER;
      }
    }
    vTaskDelay(pdMS_TO_TICKS(50));
  }
}

// ================= SEND BATCH — build JSON + enqueue upload =================
// Caller: sendTask. batchMutex phải đã acquired khi gọi.
void sendBatchLocked(int count) {
  if (count < MIN_PARTIAL_FLUSH) {
    Serial.printf("[SEND] skip tiny batch (%d < %d)\n", count, MIN_PARTIAL_FLUSH);
    return;
  }

  // Tính sample_rate thực tế từ timestamps
  // Fallback = SENSOR_RATE_HZ/SAMPLE_AVG (effective rate, ví dụ 400/4=100)
  // Hardcode 100 sẽ làm backend hiểu sai sample rate khi fallback trigger
  int fs = SENSOR_RATE_HZ / SAMPLE_AVG;  // fallback
  if (count >= 2) {
    uint64_t span_ms = batch[count - 1].t - batch[0].t;
    if (span_ms > 0) {
      float actual = ((float)(count - 1) * 1000.0f) / (float)span_ms;
      int rounded = (int)(actual + 0.5f);
      if (rounded >= MIN_SAMPLE_RATE && rounded <= MAX_SAMPLE_RATE) {
        fs = rounded;
      } else {
        Serial.printf("[SEND] actual_fs=%.1f ngoài range [%d,%d], fallback %d\n",
                      actual, MIN_SAMPLE_RATE, MAX_SAMPLE_RATE, fs);
      }
    }
  }

  // Build JSON — DynamicJsonDocument SRAM
  DynamicJsonDocument doc(JSON_DOC_SIZE);
  doc["device_id"]   = deviceId;
  doc["sample_rate"] = fs;
  doc["timestamp"]   = getISOTimestamp();
  // Snapshot + reset atomic
  portENTER_CRITICAL(&dropsMux);
  uint32_t dropSnap = dropsOut;
  dropsOut = 0;
  portEXIT_CRITICAL(&dropsMux);
  doc["has_gaps"]   = (dropSnap > 0);
  doc["drop_count"] = dropSnap;

  JsonArray irArr  = doc.createNestedArray("ir_values");
  JsonArray redArr = doc.createNestedArray("red_values");
  for (int i = 0; i < count; i++) {
    irArr.add(batch[i].ir);
    redArr.add(batch[i].red);
  }

  if (doc.overflowed()) {
    debugPrint("SEND", "JSON overflow — tăng JSON_DOC_SIZE");
    return;
  }

  size_t jsonLen = measureJson(doc);
  char* buf = (char*)malloc(jsonLen + 1);
  if (!buf) {
    debugPrint("SEND", "malloc failed — drop batch");
    return;
  }
  serializeJson(doc, buf, jsonLen + 1);

  UploadJob job{buf, jsonLen};
  if (xQueueSend(uploadQueue, &job, pdMS_TO_TICKS(500)) != pdTRUE) {
    free(buf);
    debugPrint("SEND", "Upload queue full — drop");
  } else {
    batchesSent++;
    Serial.printf("[SEND] batch=%d fs=%d bytes=%u queued\n",
                  count, fs, (unsigned)jsonLen);
  }
}

// ================= SEND TASK (Core 1) =================
void sendTask(void*) {
  debugPrint("TASK", "Send task started (Core 1)");
  PPGSample s;

  while (true) {
    // Partial flush khi nhấc tay
    if (!streamingEnabled || !calibrated) {
      xSemaphoreTake(batchMutex, portMAX_DELAY);
      if (batchIndex >= MIN_VALID_PARTIAL) {
        debugPrint("FLUSH", "Partial flush");
        sendBatchLocked(batchIndex);
      } else if (batchIndex > 0) {
        Serial.printf("[DISCARD] partial batch=%d too short, likely abrupt finger removal\n",
                      batchIndex);
        partialDiscards++;
      }
      batchIndex = 0;
      xSemaphoreGive(batchMutex);
      vTaskDelay(pdMS_TO_TICKS(50));
      continue;
    }

    // Drain queue
    if (xQueueReceive(ppgQueue, &s, pdMS_TO_TICKS(100)) == pdTRUE) {
      xSemaphoreTake(batchMutex, portMAX_DELAY);
      batch[batchIndex++] = s;
      if (batchIndex >= BATCH_SIZE) {
        sendBatchLocked(BATCH_SIZE);
        batchIndex = 0;
      }
      xSemaphoreGive(batchMutex);
    }

    // Stats
    if (millis() - lastStatsTime > STATS_INTERVAL_MS) {
      printStats();
      lastStatsTime = millis();
    }
  }
}

// ================= UPLOAD TASK (Core 1) =================
void uploadTask(void*) {
  debugPrint("TASK", "Upload task started (Core 1)");
  UploadJob job;

  while (true) {
    if (xQueueReceive(uploadQueue, &job, portMAX_DELAY) != pdTRUE) continue;

    // Block upload nếu chưa có NTP — tránh gửi timestamp 1970 corrupt backend history
    // BYPASS for self-collect (laptop server tự gen timestamp) — set guard = false để
    // payload luôn POST được kể cả khi pool.ntp.org không reachable. NHỚ revert thành
    // `if (curOffset == 0)` khi flash production trỏ về Render.
    portENTER_CRITICAL(&timeMux);
    uint64_t curOffset = timeOffsetMs;
    portEXIT_CRITICAL(&timeMux);
    if (false && curOffset == 0) {
      Serial.println("[UPLOAD] NTP chưa sync — drop payload + retry NTP background");
      free(job.payload);
      uploadFails++;
      // Retry NTP nhanh
      if (WiFi.status() == WL_CONNECTED) {
        if (timeClient.update()) {
          uint64_t ntpMs = (uint64_t)timeClient.getEpochTime() * 1000ULL;
          uint64_t espMs = esp_timer_get_time() / 1000ULL;
          portENTER_CRITICAL(&timeMux);
          timeOffsetMs = ntpMs - espMs;
          portEXIT_CRITICAL(&timeMux);
          Serial.printf("[NTP] Re-synced UTC epoch=%llu\n", ntpMs / 1000ULL);
        }
      }
      vTaskDelay(pdMS_TO_TICKS(2000));
      continue;
    }

    // WiFi check — nếu không WL_CONNECTED thì drop và chờ background tự reconnect.
    // WiFi.onEvent tự động reconnect qua SYSTEM_EVENT_STA_DISCONNECTED;
    // gọi reconnect() tay trong lúc driver đang connect gây "cannot set config" error.
    wl_status_t st = WiFi.status();
    if (st != WL_CONNECTED) {
      Serial.printf("[UPLOAD] WiFi status=%d — drop payload (RSSI=%d)\n",
                    (int)st, WiFi.RSSI());
      free(job.payload);
      uploadFails++;
      vTaskDelay(pdMS_TO_TICKS(2000));  // cooldown trước batch tiếp theo
      continue;
    }

    // Auto-detect http:// vs https:// để support cả production (Render TLS)
    // và self-collect local server (laptop FastAPI plain HTTP).
    HTTPClient http;
    WiFiClient plainClient;
    WiFiClientSecure tlsClient;
    bool isHttps = (strncmp(serverUrl, "https://", 8) == 0);
    if (isHttps) {
      tlsClient.setInsecure();  // không pin cert cho MVP
      http.begin(tlsClient, serverUrl);
    } else {
      http.begin(plainClient, serverUrl);  // HTTP plain cho local laptop
    }
    http.addHeader("Content-Type", "application/json");
    http.addHeader("X-Firmware-Version", FW_VERSION);
    if (strlen(apiKey) > 0) {
      http.addHeader("X-Device-Token", apiKey);
    }
    http.setTimeout(UPLOAD_TIMEOUT_MS);

    bool success = false;
    int code = -1;

    for (int attempt = 0; attempt < UPLOAD_RETRY; attempt++) {
      unsigned long t0 = millis();
      Serial.printf("[UPLOAD] POST %u bytes (try %d/%d)...\n",
                    (unsigned)job.len, attempt + 1, UPLOAD_RETRY);
      code = http.POST((uint8_t*)job.payload, job.len);
      unsigned long dt = millis() - t0;

      if (code == 200) {
        Serial.printf("[UPLOAD] 200 OK in %lums\n", dt);
        success = true;
        break;
      }
      Serial.printf("[UPLOAD] code=%d dt=%lums\n", code, dt);

      if (code == 403) {
        debugPrint("UPLOAD", "403 — kiểm tra X-Device-Token có khớp PPG_API_KEY?");
        break;  // không retry vì auth sai
      }
      if (code == 422 || code == 400) {
        String body = http.getString();
        Serial.printf("[UPLOAD] validator reject: %s\n", body.c_str());
        break;  // payload sai schema, retry vô nghĩa
      }
      if (attempt < UPLOAD_RETRY - 1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
      }
    }

    if (success) {
      String response = http.getString();
      DynamicJsonDocument resDoc(RESPONSE_DOC_SIZE);
      DeserializationError err = deserializeJson(resDoc, response);
      if (err) {
        Serial.printf("[UPLOAD] JSON parse error: %s\n", err.c_str());
      } else {
        // Core fields (backend + ML server cùng schema)
        const char* rid  = resDoc["reading_id"] | "";
        float hr         = resDoc["heart_rate"]     | 0.0f;
        float hrConf     = resDoc["hr_confidence"]  | 0.0f;
        float spo2       = resDoc["spo2"]           | 0.0f;
        float spo2Conf   = resDoc["spo2_confidence"]| 0.0f;
        float ratioR     = resDoc["ratio_r"]        | 0.0f;
        float pi         = resDoc["perfusion_index"]| 0.0f;
        const char* sq   = resDoc["signal_quality"] | "n/a";

        // HRV (v4.3 schema — nested)
        float sdnn = 0, rmssd = 0, pnn50 = 0, pnn20 = 0, lfhf = 0;
        int rrCount = 0;
        const char* reliab = "n/a";
        if (resDoc.containsKey("hrv")) {
          JsonObject hrv = resDoc["hrv"];
          sdnn   = hrv["sdnn_ms"]      | 0.0f;
          rmssd  = hrv["rmssd_ms"]     | 0.0f;
          pnn50  = hrv["pnn50_pct"]    | 0.0f;
          pnn20  = hrv["pnn20_pct"]    | 0.0f;
          rrCount= hrv["rr_count"]     | 0;
          reliab = hrv["reliability"]  | "n/a";
          // lf_hf có thể null
          if (!hrv["lf_hf"].isNull()) lfhf = hrv["lf_hf"].as<float>();
        }

        Serial.println("\n=== KET QUA DO ===");
        Serial.printf("  ID       : %s\n", rid);
        Serial.printf("  HR       : %.1f BPM (conf %.2f)\n", hr, hrConf);
        Serial.printf("  SpO2     : %.1f %% (conf %.2f, R=%.3f, PI=%.2f)\n",
                      spo2, spo2Conf, ratioR, pi);
        Serial.printf("  HRV      : SDNN=%.1f RMSSD=%.1f pNN50=%.1f pNN20=%.1f\n",
                      sdnn, rmssd, pnn50, pnn20);
        Serial.printf("  HRV rr_count=%d reliability=%s LF/HF=%.2f\n",
                      rrCount, reliab, lfhf);
        Serial.printf("  Quality  : %s\n", sq);

        // ML prediction (chỉ có khi dùng ML server)
        if (resDoc.containsKey("ml_predictions")) {
          JsonObject ml = resDoc["ml_predictions"];
          if (ml.containsKey("blood_pressure")) {
            JsonObject bp = ml["blood_pressure"];
            int sys = bp["systolic"]  | 0;
            int dia = bp["diastolic"] | 0;
            const char* bpUnit = bp["unit"] | "mmHg";
            float mlConf = ml["confidence"] | 0.0f;
            const char* mlModel = ml["model_used"] | "n/a";
            Serial.printf("  BP       : %d/%d %s (ml conf %.2f, %s)\n",
                          sys, dia, bpUnit, mlConf, mlModel);
          }
        }
        Serial.println("==================\n");
      }
    } else {
      uploadFails++;
      Serial.printf("[UPLOAD] FAILED code=%d totalFails=%lu\n",
                    code, (unsigned long)uploadFails);
      if (code == -1 || code == -11) {
        Serial.println("  Render cold start hoặc timeout — retry sau");
      }
    }

    http.end();
    free(job.payload);
  }
}

// ================= STATS =================
void printStats() {
  Serial.println("\n=== STATS ===");
  Serial.printf("  samples=%lu drops=%lu batches=%lu fails=%lu partialDiscards=%lu\n",
                (unsigned long)samplesOut,
                (unsigned long)dropsOut,
                (unsigned long)batchesSent,
                (unsigned long)uploadFails,
                (unsigned long)partialDiscards);

  if (redSampleCount > 0) {
    uint32_t redAvg = (uint32_t)(redSumValue / redSampleCount);
    uint32_t redRange = redMaxValue - redMinValue;
    // AC% = range/2 (amplitude) / avg (DC) × 100
    // Valid SpO2 needs AC/DC ≥ ~3% so RED gets ratio_r ≥ 0.4
    float redAcPct = redAvg > 0 ? (redRange * 50.0f / redAvg) : 0.0f;
    Serial.printf("  RED min=%lu avg=%lu max=%lu range=%lu AC%%=%.2f%% (n=%lu) %s\n",
                  (unsigned long)redMinValue,
                  (unsigned long)redAvg,
                  (unsigned long)redMaxValue,
                  (unsigned long)redRange,
                  redAcPct,
                  (unsigned long)redSampleCount,
                  redAcPct < 2.0f ? "<<< too flat — press harder" : "OK");
    redSampleCount = 0;
    redMinValue = 999999;
    redMaxValue = 0;
    redSumValue = 0;
  }

  printMemoryInfo();
  Serial.println("=============\n");
}

// ================= SETUP =================
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n============================================");
  Serial.printf("PPG Monitor v%s — ESP32 DevKit\n", FW_VERSION);
  Serial.println("============================================");
  printMemoryInfo();

  // I2C
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  // LED
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // WiFi Manager
  initWiFiManager();

  // NTP (UTC)
  syncNTP();

  // MAX30102
  debugPrint("INIT", "Initializing MAX30102...");
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    debugPrint("INIT", "MAX30102 NOT FOUND — check SDA=21 SCL=22 wiring");
    while (true) {
      delay(1000);
      Serial.println("  Sensor missing!");
    }
  }
  debugPrint("INIT", "MAX30102 OK");

  particleSensor.setup(0x40, SAMPLE_AVG, LED_MODE_DUAL,
                       SENSOR_RATE_HZ, PULSE_WIDTH_US, ADC_RANGE);
  particleSensor.clearFIFO();

  // Batch buffer — SRAM heap
  batch = (PPGSample*)malloc(BATCH_SIZE * sizeof(PPGSample));
  if (!batch) {
    debugPrint("INIT", "Batch malloc failed — abort");
    while (true) delay(1000);
  }
  Serial.printf("[INIT] batch=%u bytes SRAM\n",
                (unsigned)(BATCH_SIZE * sizeof(PPGSample)));

  // Mutex cho batch[]
  batchMutex = xSemaphoreCreateMutex();
  if (!batchMutex) {
    debugPrint("INIT", "batchMutex create failed — abort");
    while (true) delay(1000);
  }

  // Queues
  ppgQueue    = xQueueCreate(PPG_QUEUE_SIZE, sizeof(PPGSample));
  uploadQueue = xQueueCreate(UPLOAD_QUEUE_SIZE, sizeof(UploadJob));
  if (!ppgQueue || !uploadQueue) {
    debugPrint("INIT", "Queue create failed — abort");
    while (true) delay(1000);
  }

  // Tasks
  xTaskCreatePinnedToCore(sampleTask, "sample", 8192,  NULL, 5, NULL, 0);
  xTaskCreatePinnedToCore(calibTask,  "calib",  4096,  NULL, 4, NULL, 0);
  xTaskCreatePinnedToCore(sendTask,   "send",   16384, NULL, 3, NULL, 1);
  xTaskCreatePinnedToCore(uploadTask, "upload", 12288, NULL, 1, NULL, 1);

  Serial.println("\n============================================");
  debugPrint("READY", "Place finger on sensor");
  Serial.println("============================================\n");
}

// ================= LOOP =================
void loop() {
  // LED indicator state machine
  static unsigned long ledLastToggle = 0;
  static bool ledState = false;
  unsigned long now = millis();

  switch (sysState) {
    case WAIT_FINGER:
      digitalWrite(LED_PIN, LOW);
      break;
    case CALIBRATING_IR:
    case CALIBRATING_RED:
      if (now - ledLastToggle > 500) {
        ledState = !ledState;
        digitalWrite(LED_PIN, ledState);
        ledLastToggle = now;
      }
      break;
    case STREAMING:
      digitalWrite(LED_PIN, HIGH);
      break;
  }

  // NTP re-sync định kỳ (ngoài hot path)
  if (millis() - lastNtpSync > NTP_RESYNC_MS) {
    if (WiFi.status() == WL_CONNECTED) {
      if (timeClient.update()) {
        uint64_t ntpMs = (uint64_t)timeClient.getEpochTime() * 1000ULL;
        uint64_t espMs = esp_timer_get_time() / 1000ULL;
        portENTER_CRITICAL(&timeMux);
        timeOffsetMs = ntpMs - espMs;
        portEXIT_CRITICAL(&timeMux);
      }
    }
    lastNtpSync = millis();
  }

  // BOOT 5s → reset WiFi
  if (digitalRead(BOOT_BTN) == LOW) {
    unsigned long ps = millis();
    while (digitalRead(BOOT_BTN) == LOW && millis() - ps < WIFI_RESET_HOLD_MS) {
      delay(100);
    }
    if (millis() - ps >= WIFI_RESET_HOLD_MS) {
      debugPrint("WIFI", "Resetting credentials...");
      WiFiManager wm;
      wm.resetSettings();
      prefs.begin("ppg", false);
      prefs.clear();
      prefs.end();
      ESP.restart();
    }
  }

  delay(100);
}
