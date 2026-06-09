/*
==================================================
 PPG v4.1 Firmware — XIAO ESP32-S3 + MAX30102
 Tương thích Backend v4.3 + ML Server ppg-ml.onrender.com
==================================================
 Target board : Seeed Studio XIAO ESP32-S3 (ESP32-S3R8)
 PSRAM        : 8MB OPI (enable trong Board Settings)
 Form factor  : LK87 fingertip pulse oximeter retrofit (all-in-one)
 PPG Sensor   : MAX30102 reflectance @ I2C 0x57
 Display      : OLED 0.49" SSD1306 @ I2C 0x3C (optional, parse-only)
 I2C bus      : SDA=GPIO 5 (D4), SCL=GPIO 6 (D5) @ 400kHz
 LED built-in : GPIO 21 (orange LED)
 BOOT pad     : GPIO 0 (internal pad — short với GND nếu reset WiFi)
==================================================
 Thay đổi so với v4.0.5 (NodeMCU-32S):
 [PIN] SDA 21→5, SCL 22→6, LED 2→21
 [HW]  Power: pin LiPo 3.7V 600mAh cắm thẳng JST 1.25mm vào XIAO
       (XIAO có TP4056 charge IC + LDO 3.3V onboard — không cần module ngoài)
 [BUF]  Batch buffer dùng ps_malloc (PSRAM) — dư SRAM cho WiFi/HTTPS
 [BOARD] Compile settings cần thiết Arduino IDE 2.x:
       - Board: "XIAO_ESP32S3" (Seeed Studio)
       - USB CDC On Boot: ENABLED (bắt buộc cho Serial)
       - PSRAM: "OPI PSRAM"
       - Flash Size: "8MB (64Mb)"
       - Partition Scheme: "8M with spiffs (3MB APP/1.5MB SPIFFS)"
       - CPU Frequency: 240MHz
==================================================
 Library dependencies (Arduino Library Manager):
   - WiFiManager (tzapu)
   - SparkFun MAX3010x Pulse and Proximity Sensor Library
   - NTPClient
   - ArduinoJson (v6.x)
==================================================
 Version: 4.1.0-xiao
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
#include "esp_heap_caps.h"           // ps_malloc cho PSRAM batch buffer
#include <WiFiManager.h>
#include <Preferences.h>
#include <time.h>

// ================= FIRMWARE VERSION =================
#define FW_VERSION "4.1.0-xiao"

// ================= XIAO ESP32-S3 PIN MAP =================
#define SDA_PIN        5      // D4 — XIAO ESP32-S3 default I2C SDA
#define SCL_PIN        6      // D5 — XIAO ESP32-S3 default I2C SCL
#define LED_PIN        21     // Built-in orange LED (active HIGH)
#define BOOT_BTN       0      // BOOT pad on XIAO (pad nhỏ mặt sau, short với GND)

// ================= WIFI HARDCODE (default) =================
// Thử lần lượt các WiFi dưới đây. Nối được mạng nào → dùng mạng đó, bỏ qua portal.
// Tất cả fail → rơi về portal PPG-XIAO-Setup.
// LƯU Ý: ESP32-S3 chỉ bắt 2.4GHz → hotspot/router phải có băng 2.4GHz.
struct WifiCred { const char* ssid; const char* pass; };
static const WifiCred WIFI_LIST[] = {
  { "Be Dep",            "19892007"  },   // router nhà — ưu tiên (WPA2 ổn định cho ESP32)
  { "Redmi Note 13 Pro", "123456781" },   // hotspot điện thoại — dự phòng
};
static const int WIFI_LIST_N = sizeof(WIFI_LIST) / sizeof(WIFI_LIST[0]);
#define WIFI_TRY_TIMEOUT_MS   12000   // chờ tối đa 12s mỗi WiFi
// Giảm công suất phát WiFi -> tránh sụt áp/RF instability khi handshake.
// RSSI thường rất mạnh (-50..-65) nên giảm power KHÔNG ảnh hưởng tầm.
// Nếu vẫn fail, thử hạ tiếp: WIFI_POWER_5dBm / WIFI_POWER_2dBm.
#define WIFI_TX_POWER         WIFI_POWER_8_5dBm

// ================= SERVER (cloud primary + local fallback) =================
// Mặc định gửi Render (HTTPS). Fail liên tiếp >= ngưỡng → tự đổi sang server
// local trên laptop (HTTP). Fail tiếp ở local → quay lại cloud (ping-pong tự dò).
// >>> SỬA IP laptop bên dưới cho khớp LAN (Windows: ipconfig → IPv4 Address) <<<
#define CLOUD_SERVER_URL       "https://ppg-ml.onrender.com/api/ppg/upload"
#define LOCAL_SERVER_URL       "http://192.168.1.100:8080/api/ppg/upload"
#define SERVER_FALLBACK_AFTER  3       // số batch fail liên tiếp trước khi đổi đích

// ================= I2C ADDRESS MAP =================
// MAX30102:  0x57 (SparkFun lib auto-detect)
// SSD1306:   0x3C (OLED, không dùng trong firmware này — để main loop ignore)

// ================= SENSOR CONFIG =================
// v4.1.0: 400Hz / avg4 = 100Hz effective FIFO rate — parity NodeMCU baseline.
// Backend BP ML model trained @ 100Hz (Approach B); 50Hz break CLAUDE.md L114.
#define SENSOR_RATE_HZ     400
#define SAMPLE_AVG         4
#define LED_MODE_DUAL      2
#define PULSE_WIDTH_US     411
#define ADC_RANGE          16384

#define IR_MIN             5000
#define IR_MAX             262000

// Finger detection — spec 18-bit: mean(IR) > 50,000
#define CALIB_THRESHOLD    50000

// Calibration DC targets — matched NodeMCU baseline (Bent 2021 + ADI SNR guideline)
#define IR_DC_LOW          80000
#define IR_DC_HIGH         140000
#define RED_DC_LOW         80000
#define RED_DC_HIGH        160000  // 61% full scale 18-bit ADC, headroom AC 39%

// ================= SYSTEM CONFIG =================
#define BATCH_SIZE         500    // 5 giây @ 100Hz effective
#define PPG_QUEUE_SIZE     800    // Buffer ~8s @ 100Hz
#define UPLOAD_QUEUE_SIZE  3
#define JSON_DOC_SIZE      32768  // 32KB — safe margin cho 500 samples
#define RESPONSE_DOC_SIZE  4096
#define WIFI_RESET_HOLD_MS 5000
#define NTP_RESYNC_MS      60000
#define STATS_INTERVAL_MS  5000
#define UPLOAD_RETRY       3
#define UPLOAD_TIMEOUT_MS  15000

// Backend validator range
#define MIN_SAMPLE_RATE    25
#define MAX_SAMPLE_RATE    400
#define MIN_PARTIAL_FLUSH  50
#define MIN_VALID_PARTIAL  300    // 3s @ 100Hz

// ================= STRUCTS =================
struct PPGSample {
  uint32_t ir;
  uint32_t red;
  uint64_t t;        // ms epoch
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
NTPClient timeClient(ntpUDP, "pool.ntp.org", 0);  // UTC
Preferences prefs;

QueueHandle_t ppgQueue = nullptr;
QueueHandle_t uploadQueue = nullptr;
SemaphoreHandle_t batchMutex = nullptr;

volatile bool streamingEnabled = false;
volatile bool calibrated = false;

uint8_t ledPower = 0x40;
uint8_t redPower = 0x40;

volatile uint64_t timeOffsetMs = 0;
portMUX_TYPE timeMux = portMUX_INITIALIZER_UNLOCKED;
portMUX_TYPE dropsMux = portMUX_INITIALIZER_UNLOCKED;

// Batch buffer — PSRAM (XIAO có 8MB OPI PSRAM)
PPGSample* batch = nullptr;
int batchIndex = 0;

// WiFiManager params — persisted to Preferences
char serverUrl[128] = CLOUD_SERVER_URL;   // đích hiện hành (cloud mặc định, có thể đổi sang local)
char deviceId[32]   = "xiao-001";    // NEW v4.1: default đổi từ esp32-001 → xiao-001 (phân biệt v1/v2 trong backend)
char apiKey[64]     = "";

// Stats
volatile uint32_t samplesOut   = 0;
volatile uint32_t dropsOut     = 0;
volatile uint32_t batchesSent  = 0;
volatile uint32_t uploadFails  = 0;
volatile uint32_t partialDiscards = 0;
uint32_t consecutiveFails = 0;    // fail liên tiếp → trigger đổi cloud<->local (chỉ uploadTask đụng)

// Debug/diag
unsigned long lastStatsTime = 0;
unsigned long lastNtpSync = 0;
uint32_t redSampleCount = 0;
uint32_t redMinValue = 999999;
uint32_t redMaxValue = 0;
uint64_t redSumValue = 0;

// Forward declarations
void printStats();

// ================= HELPERS =================
inline uint64_t nowMs() {
  portENTER_CRITICAL(&timeMux);
  uint64_t offset = timeOffsetMs;
  portEXIT_CRITICAL(&timeMux);
  return esp_timer_get_time() / 1000ULL + offset;
}

void debugPrint(const char* stage, const char* msg) {
  Serial.printf("[%s] %s\n", stage, msg);
}

void printMemoryInfo() {
  Serial.printf("  Heap Free: %u KB | Min: %u KB | PSRAM Free: %u KB\n",
                ESP.getFreeHeap() / 1024,
                ESP.getMinFreeHeap() / 1024,
                ESP.getFreePsram() / 1024);
}

// ================= ISO TIMESTAMP (UTC) =================
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

  serverUrl[sizeof(serverUrl) - 1] = '\0';
  deviceId[sizeof(deviceId) - 1] = '\0';
  apiKey[sizeof(apiKey) - 1] = '\0';

  if (!isValidDeviceId(deviceId)) {
    debugPrint("CONFIG", "Invalid device_id, using default");
    strncpy(deviceId, "xiao-001", sizeof(deviceId) - 1);
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

// ================= WIFI EVENT HANDLERS =================
void onWiFiDisconnect(WiFiEvent_t event, WiFiEventInfo_t info) {
  Serial.printf("[WIFI] Disconnected reason=%d — auto-reconnecting\n",
                (int)info.wifi_sta_disconnected.reason);
}

void onWiFiConnected(WiFiEvent_t event, WiFiEventInfo_t info) {
  Serial.printf("[WIFI] Reconnected, IP=%s RSSI=%d\n",
                WiFi.localIP().toString().c_str(), WiFi.RSSI());
}

// ================= WIFI STATUS -> chuỗi dễ đọc (chẩn đoán) =================
const char* wifiStatusStr(wl_status_t s) {
  switch (s) {
    case WL_NO_SHIELD:       return "NO_SHIELD";
    case WL_IDLE_STATUS:     return "IDLE (dang thu)";
    case WL_NO_SSID_AVAIL:   return "KHONG THAY SSID (sai ten / 5GHz / qua xa)";
    case WL_SCAN_COMPLETED:  return "SCAN_DONE";
    case WL_CONNECTED:       return "CONNECTED";
    case WL_CONNECT_FAILED:  return "SAI MAT KHAU / connect failed";
    case WL_CONNECTION_LOST: return "CONNECTION_LOST";
    case WL_DISCONNECTED:    return "DISCONNECTED";
    default:                 return "UNKNOWN";
  }
}

// ================= WIFI MANAGER =================
void initWiFiManager() {
  debugPrint("WIFI", "Starting WiFiManager...");
  loadConfig();

  // Đăng ký handler SỚM để bắt LÝ DO disconnect khi thử hardcode (chẩn đoán)
  // reason=15 (HANDSHAKE_TIMEOUT) / reason=2 (AUTH_EXPIRE) → SAI MAT KHAU
  // reason=201 (NO_AP_FOUND) → khong thay AP | reason=205 (CONNECTION_FAIL)
  WiFi.onEvent(onWiFiDisconnect, ARDUINO_EVENT_WIFI_STA_DISCONNECTED);

  // --- Quét WiFi (chẩn đoán: XIAO thấy những mạng nào) ---
  WiFi.mode(WIFI_STA);
  WiFi.setTxPower(WIFI_TX_POWER);   // giảm TX power -> tránh sụt áp khi handshake
  WiFi.disconnect();
  delay(100);
  Serial.println("[WIFI] Scanning 2.4GHz networks...");
  int nFound = WiFi.scanNetworks();
  if (nFound <= 0) {
    Serial.println("  (!) Khong thay mang WiFi nao — anten / 2.4GHz?");
  } else {
    for (int i = 0; i < nFound; i++) {
      Serial.printf("  %2d) %-28s RSSI=%d %s\n",
                    i + 1, WiFi.SSID(i).c_str(), WiFi.RSSI(i),
                    WiFi.encryptionType(i) == WIFI_AUTH_OPEN ? "OPEN" : "ENC");
    }
  }
  WiFi.scanDelete();

  // --- Thử lần lượt từng WiFi trong WIFI_LIST ---
  for (int k = 0; k < WIFI_LIST_N; k++) {
    Serial.printf("[WIFI] (%d/%d) Connecting to '%s'...\n",
                  k + 1, WIFI_LIST_N, WIFI_LIST[k].ssid);
    WiFi.disconnect();
    delay(100);
    WiFi.begin(WIFI_LIST[k].ssid, WIFI_LIST[k].pass);
    WiFi.setTxPower(WIFI_TX_POWER);   // áp lại sau begin cho chắc
    unsigned long t0 = millis();
    while (WiFi.status() != WL_CONNECTED &&
           millis() - t0 < WIFI_TRY_TIMEOUT_MS) {
      delay(250);
      Serial.print(".");
    }
    Serial.println();
    if (WiFi.status() == WL_CONNECTED) break;
    Serial.printf("[WIFI] '%s' FAIL, status=%d (%s)\n",
                  WIFI_LIST[k].ssid, (int)WiFi.status(),
                  wifiStatusStr(WiFi.status()));
  }

  if (WiFi.status() == WL_CONNECTED) {
    WiFi.onEvent(onWiFiConnected, ARDUINO_EVENT_WIFI_STA_GOT_IP);
    WiFi.setAutoReconnect(true);
    debugPrint("WIFI", "Connected (hardcoded)");
    Serial.printf("  SSID   : %s\n", WiFi.SSID().c_str());
    Serial.printf("  IP: %s RSSI: %d\n",
                  WiFi.localIP().toString().c_str(), WiFi.RSSI());
    Serial.printf("  Server : %s\n", serverUrl);
    Serial.printf("  Device : %s\n", deviceId);
    Serial.printf("  ApiKey : %s\n",
                  strlen(apiKey) > 0 ? "(set)" : "(empty - dev mode)");
    WiFi.setSleep(false);
    return;  // bỏ qua portal
  }

  debugPrint("WIFI", "Tat ca WiFi hardcode fail -> mở portal PPG-XIAO-Setup");

  // --- Fallback: WiFiManager portal ---
  WiFiManager wm;
  wm.setConnectTimeout(15);
  wm.setConfigPortalTimeout(300);   // 5 phút (tăng từ 180s cho dễ nối kịp)

  WiFiManagerParameter param_url("server_url",
      "Server URL (ppg-ml.onrender.com)", serverUrl, 128);
  WiFiManagerParameter param_did("device_id",
      "Device ID (a-z 0-9 _ -)", deviceId, 32);
  WiFiManagerParameter param_key("api_key",
      "API Key (optional)", apiKey, 64);

  wm.addParameter(&param_url);
  wm.addParameter(&param_did);
  wm.addParameter(&param_key);

  // BOOT giữ 5s → reset WiFi credentials
  // XIAO ESP32-S3: GPIO 0 là pad nhỏ ở mặt sau, short với GND bằng nhíp
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

  // Dọn trạng thái STA + xóa creds đã lưu để portal bật NGAY
  // (không lãng phí 15s retry lại credential hardcode đã fail ở trên)
  WiFi.disconnect(true, true);
  delay(200);
  // startConfigPortal: bật AP PPG-XIAO-Setup ngay lập tức, KHÔNG thử creds cũ
  bool connected = wm.startConfigPortal("PPG-XIAO-Setup");
  if (!connected) {
    debugPrint("WIFI", "Portal timeout / failed -> Restart");
    ESP.restart();
  }

  WiFi.onEvent(onWiFiDisconnect, ARDUINO_EVENT_WIFI_STA_DISCONNECTED);
  WiFi.onEvent(onWiFiConnected,  ARDUINO_EVENT_WIFI_STA_GOT_IP);
  WiFi.setAutoReconnect(true);

  debugPrint("WIFI", "Connected");
  Serial.printf("  IP: %s RSSI: %d\n",
                WiFi.localIP().toString().c_str(), WiFi.RSSI());

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

// ================= CALIBRATION (calibTask) =================
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
        sum += particleSensor.getFIFOIR();
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
        sum += particleSensor.getFIFORed();
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

void startStreamingMode() {
  particleSensor.shutDown();
  delay(50);
  particleSensor.wakeUp();
  delay(50);
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

// ================= SAMPLE TASK (Core 0) =================
void sampleTask(void*) {
  debugPrint("TASK", "Sample task started (Core 0)");
  PPGSample s;

  while (true) {
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
      uint32_t ir  = particleSensor.getFIFOIR();
      uint32_t red = particleSensor.getFIFORed();
      particleSensor.nextSample();

      if (sysState == WAIT_FINGER) {
        if (ir > CALIB_THRESHOLD) {
          debugPrint("DETECT", "Finger detected -> calibrate");
          sysState = CALIBRATING_IR;
          break;
        }
        continue;
      }

      if (sysState == STREAMING && ir < CALIB_THRESHOLD) {
        debugPrint("DETECT", "Finger removed");
        calibrated = false;
        streamingEnabled = false;
        sysState = WAIT_FINGER;
        break;
      }

      if (sysState == STREAMING) {
        if (ir < IR_MIN || ir > IR_MAX) continue;

        s.ir = ir;
        s.red = red;
        s.t = nowMs();

        // RED diag stats
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

// ================= CALIB TASK (Core 0) =================
void calibTask(void*) {
  debugPrint("TASK", "Calib task started (Core 0)");
  while (true) {
    if (sysState == CALIBRATING_IR) {
      bool ok = calibrateIR();
      if (ok) {
        sysState = CALIBRATING_RED;
      } else {
        sysState = WAIT_FINGER;
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

// ================= SEND BATCH =================
// Caller: sendTask. batchMutex phải đã acquired khi gọi.
void sendBatchLocked(int count) {
  if (count < MIN_PARTIAL_FLUSH) {
    Serial.printf("[SEND] skip tiny batch (%d < %d)\n", count, MIN_PARTIAL_FLUSH);
    return;
  }

  // Tính sample_rate thực tế từ timestamps
  // Fallback = SENSOR_RATE_HZ/SAMPLE_AVG (effective rate, ví dụ 400/4=100)
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

  // Build JSON — DynamicJsonDocument SRAM (32KB)
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
  // Try PSRAM first, fallback SRAM nếu fail
  char* buf = (char*)heap_caps_malloc(jsonLen + 1, MALLOC_CAP_SPIRAM);
  if (!buf) buf = (char*)malloc(jsonLen + 1);
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

    if (xQueueReceive(ppgQueue, &s, pdMS_TO_TICKS(100)) == pdTRUE) {
      xSemaphoreTake(batchMutex, portMAX_DELAY);
      batch[batchIndex++] = s;
      if (batchIndex >= BATCH_SIZE) {
        sendBatchLocked(BATCH_SIZE);
        batchIndex = 0;
      }
      xSemaphoreGive(batchMutex);
    }

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

    // Đích hiện hành dùng HTTPS (cloud) hay HTTP (local laptop)?
    bool isHttps = (strncmp(serverUrl, "https://", 8) == 0);

    portENTER_CRITICAL(&timeMux);
    uint64_t curOffset = timeOffsetMs;
    portEXIT_CRITICAL(&timeMux);
    // Chỉ chặn khi gửi CLOUD (cần timestamp chuẩn). Local laptop tự gen timestamp → bỏ qua guard.
    if (isHttps && curOffset == 0) {
      Serial.println("[UPLOAD] NTP chưa sync (cloud) — drop payload + retry NTP background");
      free(job.payload);
      uploadFails++;
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

    wl_status_t st = WiFi.status();
    if (st != WL_CONNECTED) {
      Serial.printf("[UPLOAD] WiFi status=%d — drop payload (RSSI=%d)\n",
                    (int)st, WiFi.RSSI());
      free(job.payload);
      uploadFails++;
      vTaskDelay(pdMS_TO_TICKS(2000));
      continue;
    }

    // Auto-detect http:// (local laptop) vs https:// (Render cloud) — port từ firmware 4
    HTTPClient http;
    WiFiClient plainClient;
    WiFiClientSecure tlsClient;
    if (isHttps) {
      tlsClient.setInsecure();   // không pin cert cho MVP
      http.begin(tlsClient, serverUrl);
    } else {
      http.begin(plainClient, serverUrl);   // HTTP plain cho local laptop
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
        break;
      }
      if (code == 422 || code == 400) {
        String body = http.getString();
        Serial.printf("[UPLOAD] validator reject: %s\n", body.c_str());
        break;
      }
      if (attempt < UPLOAD_RETRY - 1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
      }
    }

    if (success) {
      consecutiveFails = 0;
      String response = http.getString();
      DynamicJsonDocument resDoc(RESPONSE_DOC_SIZE);
      DeserializationError err = deserializeJson(resDoc, response);
      if (err) {
        Serial.printf("[UPLOAD] JSON parse error: %s\n", err.c_str());
      } else {
        const char* rid  = resDoc["reading_id"] | "";
        float hr         = resDoc["heart_rate"]     | 0.0f;
        float hrConf     = resDoc["hr_confidence"]  | 0.0f;
        float spo2       = resDoc["spo2"]           | 0.0f;
        float spo2Conf   = resDoc["spo2_confidence"]| 0.0f;
        float ratioR     = resDoc["ratio_r"]        | 0.0f;
        float pi         = resDoc["perfusion_index"]| 0.0f;
        const char* sq   = resDoc["signal_quality"] | "n/a";

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

        // NEW v4.1: backend trả về calibrated flag (per-subject calibration)
        if (resDoc.containsKey("calibrated")) {
          bool cal = resDoc["calibrated"] | false;
          Serial.printf("  Cal      : %s\n", cal ? "YES (offset applied)" : "NO (raw pred)");
        }

        Serial.println("==================\n");
      }
    } else {
      uploadFails++;
      consecutiveFails++;
      Serial.printf("[UPLOAD] FAILED code=%d totalFails=%lu consec=%lu\n",
                    code, (unsigned long)uploadFails, (unsigned long)consecutiveFails);
      if (code == -1 || code == -11) {
        Serial.println("  Render cold start hoặc timeout — retry sau");
      }
      // Fail liên tiếp nhiều → đổi đích cloud <-> local (tự dò cái nào sống)
      if (consecutiveFails >= SERVER_FALLBACK_AFTER) {
        if (strncmp(serverUrl, "https://", 8) == 0) {
          strncpy(serverUrl, LOCAL_SERVER_URL, sizeof(serverUrl) - 1);
        } else {
          strncpy(serverUrl, CLOUD_SERVER_URL, sizeof(serverUrl) - 1);
        }
        serverUrl[sizeof(serverUrl) - 1] = '\0';
        consecutiveFails = 0;
        Serial.printf("[UPLOAD] >>> Doi dich sang: %s\n", serverUrl);
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

// ================= I2C SCANNER (chẩn đoán sensor) =================
void scanI2C() {
  Serial.println("[I2C] Scanning bus (SDA=5, SCL=6)...");
  int found = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.printf("  -> Tim thay thiet bi @ 0x%02X%s\n", addr,
                    addr == 0x57 ? "  <<< MAX30102" : "");
      found++;
    }
  }
  if (found == 0) {
    Serial.println("  (!) KHONG co thiet bi nao tren bus");
    Serial.println("      -> dut day / sai chan / mat nguon VIN / GND chua thong");
  } else {
    Serial.printf("[I2C] Tong %d thiet bi.\n", found);
  }
}

// ================= SETUP =================
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n============================================");
  Serial.printf("PPG Monitor v%s — XIAO ESP32-S3\n", FW_VERSION);
  Serial.println("============================================");
  printMemoryInfo();

  // Verify PSRAM enabled
  if (!psramFound()) {
    Serial.println("⚠️  PSRAM NOT FOUND — bật 'PSRAM: OPI PSRAM' trong Board Settings!");
  } else {
    Serial.printf("[PSRAM] OK, total %u KB\n", ESP.getPsramSize() / 1024);
  }

  // I2C — XIAO pin map
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  // LED
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // WiFi Manager
  initWiFiManager();

  // NTP (UTC)
  syncNTP();

  // MAX30102 — PPG sensor
  scanI2C();   // soi bus TRƯỚC khi init: thấy 0x57 = sensor sống; rỗng = dứt dây/nguồn
  debugPrint("INIT", "Initializing MAX30102 @ 0x57...");
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    debugPrint("INIT", "MAX30102 NOT FOUND — check SDA=5 SCL=6 wiring");
    // Tự quét lại + thử lại mỗi 3s (cắm lại dây là chạy, không cần reboot)
    while (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
      delay(3000);
      scanI2C();
      Serial.println("  Sensor missing! — kiem tra lai 4 day roi giu nguyen");
    }
  }
  debugPrint("INIT", "MAX30102 OK");

  particleSensor.setup(0x40, SAMPLE_AVG, LED_MODE_DUAL,
                       SENSOR_RATE_HZ, PULSE_WIDTH_US, ADC_RANGE);
  particleSensor.clearFIFO();

  // Batch buffer — try PSRAM first
  size_t batchBytes = BATCH_SIZE * sizeof(PPGSample);
  batch = (PPGSample*)heap_caps_malloc(batchBytes, MALLOC_CAP_SPIRAM);
  if (!batch) batch = (PPGSample*)malloc(batchBytes);
  if (!batch) {
    debugPrint("INIT", "Batch malloc failed — abort");
    while (true) delay(1000);
  }
  Serial.printf("[INIT] batch=%u bytes (%s)\n",
                (unsigned)batchBytes,
                psramFound() ? "PSRAM" : "SRAM");

  // Mutex
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
