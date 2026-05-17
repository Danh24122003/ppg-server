/*
==================================================
 PPG Optimized — XIAO ESP32S3 + MAX30102
 Render Server + WiFiManager + PSRAM
==================================================
 ✔ XIAO ESP32S3 (ESP32-S3R8: 512KB SRAM + 8MB PSRAM)
 ✔ TRUE 100 Hz IR sampling
 ✔ RED snapshot @ 50 Hz (giảm crosstalk)
 ✔ DUAL CALIBRATION: IR riêng → RED riêng
 ✔ WiFiManager Captive Portal (không hardcode WiFi)
 ✔ PSRAM-aware memory allocation
 ✔ Batch 500 mẫu gửi Render Server
 ✔ Retry upload 3 lần
 ✔ Drop tracking + Heap monitoring
 ✔ Partial flush khi nhấc tay
 ✔ NTP re-sync định kỳ
 ✔ Parse response hiển thị HR/SpO2/HRV
==================================================
 Version: 3.0.0
 Board: Seeed XIAO ESP32S3
 PSRAM: OPI PSRAM (bật trong Board Settings)
==================================================
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include "MAX30105.h"
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/queue.h>
#include <NTPClient.h>
#include <WiFiUdp.h>
#include <ArduinoJson.h>
#include "esp_timer.h"
#include <WiFiManager.h>
#include <Preferences.h>

// ================= FIRMWARE VERSION =================
#define FW_VERSION "3.0.0"

// ================= XIAO ESP32S3 PIN MAP =================
#define SDA_PIN 5 // XIAO ESP32S3 I2C SDA
#define SCL_PIN 6 // XIAO ESP32S3 I2C SCL
#define LED_PIN 21 // Built-in LED
#define BOOT_BTN 0 // Boot button (giữ 5s → reset WiFi)

// ================= SENSOR CONFIG =================
#define SENSOR_RATE_HZ 400
#define SAMPLE_AVG 3
#define LED_MODE 2 // Dual LED (IR + RED)
#define PULSE_WIDTH_US 411
#define ADC_RANGE 16384 // 16-bit resolution

#define RED_DIV 2 // RED sample mỗi 2 IR → 50 Hz
#define IR_MIN 5000
#define IR_MAX 262000

// Calibration targets
#define CALIB_THRESHOLD 8000 // Ngưỡng phát hiện ngón tay
#define IR_DC_LOW 80000 // IR DC target range
#define IR_DC_HIGH 140000
#define RED_DC_LOW 65000 // RED DC target range
#define RED_DC_HIGH 90000

// ================= SYSTEM CONFIG =================
#define BATCH_SIZE 500 // 5 giây @ 100Hz
#define PPG_QUEUE_SIZE 800 // Buffer ~8s
#define UPLOAD_QUEUE_SIZE 3
#define WIFI_RESET_HOLD_MS 5000 // Giữ BOOT 5s → reset WiFi
#define NTP_RESYNC_MS 60000 // Re-sync NTP mỗi 60s
#define STATS_INTERVAL_MS 5000 // In stats mỗi 5s
#define UPLOAD_RETRY 3 // Retry upload 3 lần
#define UPLOAD_TIMEOUT_MS 60000 // Render cold start timeout

// ================= STRUCTS =================
struct PPGSample {
 uint32_t ir;
 uint32_t red;
 uint64_t t;
};

struct UploadJob {
 char* payload;
 size_t len;
};

// ================= PSRAM JSON ALLOCATOR =================
struct PsramAllocator {
 void* allocate(size_t size) {
 if (psramFound()) return heap_caps_malloc(size, MALLOC_CAP_SPIRAM);
 return malloc(size);
 }
 void deallocate(void* ptr) {
 if (ptr) free(ptr);
 }
 void* reallocate(void* ptr, size_t new_size) {
 if (psramFound()) return heap_caps_realloc(ptr, new_size, MALLOC_CAP_SPIRAM);
 return realloc(ptr, new_size);
 }
};
using PsramJsonDocument = BasicJsonDocument<PsramAllocator>;

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
NTPClient timeClient(ntpUDP, "pool.ntp.org", 25200);
Preferences prefs;

QueueHandle_t ppgQueue, uploadQueue;

volatile bool calibrated = false;
volatile bool streamingEnabled = false;

uint8_t ledPower = 0x40;
uint8_t redPower = 0x40;
uint64_t timeOffsetMs = 0;

// Batch — allocated trên PSRAM trong setup()
PPGSample* batch = nullptr;
int batchIndex = 0;
uint64_t batchStart = 0;

// WiFiManager custom params — lưu vào Preferences
char serverUrl[128] = "https://ppg-server-6r55.onrender.com/api/ppg/upload";
char deviceId[32] = "esp32s3-001";

// Stats counters
volatile uint32_t samplesOut = 0;
volatile uint32_t dropsOut = 0;
volatile uint32_t batchesSent = 0;
volatile uint32_t uploadFails = 0;

// Debug counters
unsigned long lastDebugTime = 0;
unsigned long lastNtpSync = 0;
uint32_t redSampleCount = 0;
uint32_t redMinValue = 999999;
uint32_t redMaxValue = 0;
uint32_t redSumValue = 0;

// ================= HELPERS =================
inline uint64_t nowMs() {
 return esp_timer_get_time() / 1000ULL + timeOffsetMs;
}

void debugPrint(const char* stage, const char* msg) {
 Serial.printf("[%s] %s\n", stage, msg);
}

// Cấp phát bộ nhớ ưu tiên PSRAM
void* psramMalloc(size_t size) {
 if (psramFound()) return heap_caps_malloc(size, MALLOC_CAP_SPIRAM);
 return malloc(size);
}

// ================= ISO TIMESTAMP =================
String getISOTimestamp() {
 timeClient.update();
 unsigned long epoch = timeClient.getEpochTime();
 int hr = (epoch % 86400) / 3600;
 int mn = (epoch % 3600) / 60;
 int sec = epoch % 60;

 unsigned long days = epoch / 86400;
 int year = 1970;
 while (true) {
 int diy = (year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)) ? 366 : 365;
 if (days < (unsigned long)diy) break;
 days -= diy;
 year++;
 }
 int month = 1;
 int dim[] = {31,28,31,30,31,30,31,31,30,31,30,31};
 if (year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)) dim[1] = 29;
 while (days >= (unsigned long)dim[month - 1]) {
 days -= dim[month - 1];
 month++;
 }
 int day = days + 1;

 char buf[30];
 snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d.000Z",
 year, month, day, hr, mn, sec);
 return String(buf);
}

// ================= MEMORY INFO =================
void printMemoryInfo() {
 Serial.printf(" SRAM Free: %d KB | Min: %d KB\n",
 ESP.getFreeHeap() / 1024, ESP.getMinFreeHeap() / 1024);
 if (psramFound()) {
 Serial.printf(" PSRAM Free: %d KB / %d KB\n",
 ESP.getFreePsram() / 1024, ESP.getPsramSize() / 1024);
 }
}

// ================= WIFI MANAGER =================
void loadConfig() {
 prefs.begin("ppg", true); // read-only
 String url = prefs.getString("server_url", serverUrl);
 String did = prefs.getString("device_id", deviceId);
 url.toCharArray(serverUrl, sizeof(serverUrl));
 did.toCharArray(deviceId, sizeof(deviceId));
 prefs.end();
}

void saveConfig() {
 prefs.begin("ppg", false);
 prefs.putString("server_url", serverUrl);
 prefs.putString("device_id", deviceId);
 prefs.end();
}

void initWiFiManager() {
 debugPrint("WIFI", "Starting WiFiManager...");

 loadConfig();

 WiFiManager wm;
 wm.setConnectTimeout(15);
 wm.setConfigPortalTimeout(180); // Portal tự tắt sau 3 phút

 // Custom parameters cho Server URL và Device ID
 WiFiManagerParameter param_url("server_url", "Server URL", serverUrl, 128);
 WiFiManagerParameter param_did("device_id", "Device ID", deviceId, 32);
 wm.addParameter(&param_url);
 wm.addParameter(&param_did);

 // Kiểm tra nút BOOT — giữ 5s để reset WiFi
 pinMode(BOOT_BTN, INPUT_PULLUP);
 unsigned long pressStart = 0;
 if (digitalRead(BOOT_BTN) == LOW) {
 pressStart = millis();
 while (digitalRead(BOOT_BTN) == LOW && millis() - pressStart < WIFI_RESET_HOLD_MS) {
 delay(100);
 }
 if (millis() - pressStart >= WIFI_RESET_HOLD_MS) {
 debugPrint("WIFI", "BOOT held 5s -> Reset WiFi credentials!");
 wm.resetSettings();
 }
 }

 // autoConnect: nếu có credentials → connect, không → mở portal "PPG-Setup"
 bool connected = wm.autoConnect("PPG-Setup");

 if (connected) {
 debugPrint("WIFI", "Connected!");
 Serial.printf(" IP: %s\n", WiFi.localIP().toString().c_str());

 // Lưu custom params
 strncpy(serverUrl, param_url.getValue(), sizeof(serverUrl) - 1);
 strncpy(deviceId, param_did.getValue(), sizeof(deviceId) - 1);
 saveConfig();

 Serial.printf(" Server: %s\n", serverUrl);
 Serial.printf(" Device: %s\n", deviceId);
 } else {
 debugPrint("WIFI", "Failed to connect! Restarting...");
 ESP.restart();
 }

 WiFi.setSleep(false);
}

// ================= NTP =================
void syncNTP() {
 debugPrint("NTP", "Syncing...");
 timeClient.begin();
 int attempts = 0;
 while (!timeClient.update() && attempts < 10) {
 delay(500);
 attempts++;
 }
 if (attempts >= 10) {
 debugPrint("NTP", "Sync failed, using local time");
 } else {
 uint64_t ntpMs = (uint64_t)timeClient.getEpochTime() * 1000ULL;
 uint64_t espMs = esp_timer_get_time() / 1000ULL;
 timeOffsetMs = ntpMs - espMs;
 debugPrint("NTP", "Time synced!");
 }
 lastNtpSync = millis();
}

// ================= IR CALIBRATION =================
bool calibrateIR() {
 debugPrint("CALIB_IR", "Starting...");
 for (uint8_t p = 0x20; p <= 0x80; p += 0x08) {
 particleSensor.setPulseAmplitudeIR(p);
 particleSensor.setPulseAmplitudeRed(0);
 delay(300);
 particleSensor.clearFIFO();

 uint64_t sum = 0;
 int n = 0;
 unsigned long t0 = millis();
 while (n < 50 && millis() - t0 < 1000) {
 particleSensor.check();
 if (particleSensor.available()) {
 sum += particleSensor.getIR();
 particleSensor.nextSample();
 n++;
 }
 }
 if (n == 0) continue;
 uint32_t dc = sum / n;
 Serial.printf(" IR=0x%02X -> DC=%lu", p, dc);

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

// ================= RED CALIBRATION =================
bool calibrateRED() {
 debugPrint("CALIB_RED", "Starting...");
 uint8_t startP = (uint8_t)(ledPower * 0.85);
 for (uint8_t p = startP; p <= 0x7F; p += 0x08) {
 particleSensor.setPulseAmplitudeIR(0);
 particleSensor.setPulseAmplitudeRed(p);
 delay(300);
 particleSensor.clearFIFO();

 uint64_t sum = 0;
 int n = 0;
 unsigned long t0 = millis();
 while (n < 50 && millis() - t0 < 1000) {
 particleSensor.check();
 if (particleSensor.available()) {
 sum += particleSensor.getRed();
 particleSensor.nextSample();
 n++;
 }
 }
 if (n == 0) continue;
 uint32_t dc = sum / n;
 Serial.printf(" RED=0x%02X -> DC=%lu", p, dc);

 if (dc > RED_DC_LOW && dc < RED_DC_HIGH) {
 redPower = p;
 Serial.println(" OK");
 return true;
 }
 Serial.println(dc < RED_DC_LOW ? " (low)" : " (high)");
 }
 // Fallback
 if (redPower == 0x40) redPower = 0x60;
 debugPrint("CALIB_RED", "Using fallback");
 return false;
}

// ================= SAMPLING TASK (Core 0) =================
void sampleTask(void*) {
 debugPrint("TASK", "Sampling started");
 PPGSample s;
 uint32_t redSnap = 0;
 uint8_t redDiv = 0;

 while (true) {
 particleSensor.check();
 while (particleSensor.available()) {
 uint32_t ir = particleSensor.getIR();
 particleSensor.nextSample();

 // WAIT_FINGER
 if (sysState == WAIT_FINGER && ir > CALIB_THRESHOLD) {
 debugPrint("DETECT", "Finger detected");
 sysState = CALIBRATING_IR;
 continue;
 }

 // CALIBRATING_IR
 if (sysState == CALIBRATING_IR) {
 if (calibrateIR()) {
 sysState = CALIBRATING_RED;
 } else {
 sysState = WAIT_FINGER;
 }
 continue;
 }

 // CALIBRATING_RED
 if (sysState == CALIBRATING_RED) {
 calibrateRED();
 particleSensor.shutDown();
 delay(50);
 particleSensor.wakeUp();
 delay(50);
 particleSensor.setup(ledPower, SAMPLE_AVG, LED_MODE,
 SENSOR_RATE_HZ, PULSE_WIDTH_US, ADC_RANGE);
 particleSensor.setPulseAmplitudeIR(ledPower);
 particleSensor.setPulseAmplitudeRed(0);
 particleSensor.clearFIFO();

 calibrated = true;
 streamingEnabled = true;
 sysState = STREAMING;

 redSampleCount = 0;
 redMinValue = 999999;
 redMaxValue = 0;
 redSumValue = 0;
 lastDebugTime = millis();

 debugPrint("STREAM", "Started!");
 Serial.printf(" IR=0x%02X RED=0x%02X\n", ledPower, redPower);
 continue;
 }

 // STREAMING - finger removed
 if (sysState == STREAMING && ir < CALIB_THRESHOLD) {
 debugPrint("DETECT", "Finger removed");
 calibrated = false;
 streamingEnabled = false;
 sysState = WAIT_FINGER;
 // Partial flush — không mất dữ liệu cuối
 // batchIndex sẽ được xử lý bởi sendTask
 continue;
 }

 // STREAMING - process
 if (sysState == STREAMING) {
 if (ir < IR_MIN || ir > IR_MAX) continue;

 // RED snapshot mỗi RED_DIV mẫu
 if (++redDiv >= RED_DIV) {
 redDiv = 0;
 particleSensor.setPulseAmplitudeRed(redPower);
 delayMicroseconds(1200);

 uint32_t sum = 0;
 int cnt = 0;
 for (int i = 0; i < 5; i++) {
 particleSensor.check();
 if (particleSensor.available()) {
 sum += particleSensor.getRed();
 particleSensor.nextSample();
 cnt++;
 }
 delayMicroseconds(100);
 }
 particleSensor.setPulseAmplitudeRed(0);

 if (cnt > 0) {
 redSnap = sum / cnt;
 redSampleCount++;
 redSumValue += redSnap;
 if (redSnap < redMinValue) redMinValue = redSnap;
 if (redSnap > redMaxValue) redMaxValue = redSnap;
 }
 }

 s.ir = ir;
 s.red = redSnap;
 s.t = nowMs();

 if (xQueueSend(ppgQueue, &s, 0) != pdTRUE) {
 dropsOut++;
 } else {
 samplesOut++;
 }
 }
 }
 taskYIELD();
 }
}

// ================= SEND TASK (Core 1) =================
void sendBatch(int count); // forward declaration

void sendTask(void*) {
 debugPrint("TASK", "Send task started");
 PPGSample s;

 while (true) {
 if (!streamingEnabled || !calibrated) {
 // Partial flush: nếu vừa nhấc tay mà batch có dữ liệu
 if (batchIndex >= 50) { // Tối thiểu 50 mẫu (server yêu cầu)
 debugPrint("FLUSH", "Partial batch flush");
 sendBatch(batchIndex);
 }
 batchIndex = 0;
 vTaskDelay(pdMS_TO_TICKS(50));
 continue;
 }

 if (xQueueReceive(ppgQueue, &s, pdMS_TO_TICKS(100)) == pdTRUE) {
 if (batchIndex == 0) batchStart = s.t;
 batch[batchIndex++] = s;

 if (batchIndex >= BATCH_SIZE) {
 sendBatch(BATCH_SIZE);
 batchIndex = 0;
 }
 }
 }
}

void sendBatch(int count) {
 // Dùng PSRAM cho JSON document
 PsramJsonDocument doc(32768);

 doc["device_id"] = deviceId;
 doc["sample_rate"] = 100;
 doc["timestamp"] = getISOTimestamp();

 JsonArray irArr = doc.createNestedArray("ir_values");
 JsonArray redArr = doc.createNestedArray("red_values");

 for (int i = 0; i < count; i++) {
 irArr.add(batch[i].ir);
 redArr.add(batch[i].red);
 }

 size_t jsonLen = measureJson(doc);
 char* buf = (char*)psramMalloc(jsonLen + 1);
 if (!buf) {
 debugPrint("SEND", "malloc failed!");
 return;
 }
 serializeJson(doc, buf, jsonLen + 1);

 UploadJob job{buf, jsonLen};
 if (xQueueSend(uploadQueue, &job, pdMS_TO_TICKS(500)) != pdTRUE) {
 free(buf);
 debugPrint("SEND", "Upload queue full, dropping");
 } else {
 batchesSent++;
 }

 // Stats
 if (millis() - lastDebugTime > STATS_INTERVAL_MS) {
 printStats();
 lastDebugTime = millis();
 }
}

// ================= UPLOAD TASK (Core 1) =================
void uploadTask(void*) {
 debugPrint("TASK", "Upload task started");
 UploadJob job;

 while (true) {
 if (xQueueReceive(uploadQueue, &job, portMAX_DELAY)) {
 // WiFi check + reconnect
 if (WiFi.status() != WL_CONNECTED) {
 debugPrint("UPLOAD", "WiFi lost, reconnecting...");
 WiFi.reconnect();
 int wait = 0;
 while (WiFi.status() != WL_CONNECTED && wait < 20) {
 delay(500);
 wait++;
 }
 if (WiFi.status() != WL_CONNECTED) {
 debugPrint("UPLOAD", "Reconnect failed, dropping");
 free(job.payload);
 uploadFails++;
 continue;
 }
 }

 // Upload với retry
 HTTPClient http;
 http.begin(serverUrl);
 http.addHeader("Content-Type", "application/json");
 http.addHeader("X-Firmware-Version", FW_VERSION);
 http.setTimeout(UPLOAD_TIMEOUT_MS);

 bool success = false;
 int code = -1;

 for (int attempt = 0; attempt < UPLOAD_RETRY; attempt++) {
 Serial.printf(" Upload %zu bytes (attempt %d/%d)...\n",
 job.len, attempt + 1, UPLOAD_RETRY);
 code = http.POST((uint8_t*)job.payload, job.len);

 if (code == 200) {
 success = true;
 break;
 }
 if (attempt < UPLOAD_RETRY - 1) {
 vTaskDelay(pdMS_TO_TICKS(1000)); // Chờ 1s trước retry
 }
 }

 if (success) {
 // Parse response
 String response = http.getString();
 DynamicJsonDocument resDoc(4096);
 DeserializationError err = deserializeJson(resDoc, response);

 if (!err) {
 float hr = resDoc["heart_rate"];
 float sp = resDoc["spo2"];
 float hrv = resDoc["hrv_sdnn"];
 const char* q = resDoc["signal_quality"];
 const char* rid = resDoc["reading_id"];

 Serial.println("\n=== KET QUA DO ===");
 Serial.printf(" ID: %s\n", rid);
 Serial.printf(" Nhip tim: %.1f BPM\n", hr);
 Serial.printf(" SpO2: %.1f %%\n", sp);
 Serial.printf(" HRV: %.1f ms\n", hrv);
 Serial.printf(" Chat luong: %s\n", q);
 Serial.println("==================\n");
 }
 } else {
 uploadFails++;
 String errBody = http.getString();
 Serial.printf(" Upload FAILED: HTTP %d (total fails: %lu)\n", code, uploadFails);
 if (code == 400 || code == 422) {
 Serial.printf(" Server: %s\n", errBody.c_str());
 } else if (code == -1) {
 Serial.println(" Render cold start? Retry later.");
 }
 }

 http.end();
 free(job.payload);
 }
 }
}

// ================= STATS =================
void printStats() {
 Serial.println("\n=== STATS ===");
 Serial.printf(" Batches: %lu | Fails: %lu | Drops: %lu\n",
 batchesSent, uploadFails, dropsOut);

 if (redSampleCount > 0) {
 uint32_t redAvg = redSumValue / redSampleCount;
 Serial.printf(" RED -> Min:%lu Avg:%lu Max:%lu\n",
 redMinValue, redAvg, redMaxValue);
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
 Serial.printf("PPG Monitor v%s - XIAO ESP32S3\n", FW_VERSION);
 Serial.println("============================================");

 // Kiểm tra PSRAM
 if (psramFound()) {
 Serial.printf("PSRAM: %d KB\n", ESP.getPsramSize() / 1024);
 } else {
 Serial.println("PSRAM: NOT FOUND (check board settings!)");
 }
 printMemoryInfo();

 // I2C — XIAO ESP32S3 pins
 Wire.begin(SDA_PIN, SCL_PIN);
 Wire.setClock(400000);

 // LED
 pinMode(LED_PIN, OUTPUT);

 // WiFiManager (Captive Portal)
 initWiFiManager();

 // NTP
 syncNTP();

 // MAX30102
 debugPrint("INIT", "Initializing MAX30102...");
 if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
 debugPrint("INIT", "MAX30102 NOT FOUND!");
 while (true) {
 delay(1000);
 Serial.println("Check sensor connection!");
 }
 }
 debugPrint("INIT", "MAX30102 OK");

 particleSensor.setup(0x40, SAMPLE_AVG, LED_MODE,
 SENSOR_RATE_HZ, PULSE_WIDTH_US, ADC_RANGE);
 particleSensor.clearFIFO();

 // Cấp phát batch trên PSRAM
 batch = (PPGSample*)psramMalloc(BATCH_SIZE * sizeof(PPGSample));
 if (!batch) {
 debugPrint("INIT", "Batch malloc failed! Using SRAM fallback");
 batch = (PPGSample*)malloc(BATCH_SIZE * sizeof(PPGSample));
 }

 // Queues
 ppgQueue = xQueueCreate(PPG_QUEUE_SIZE, sizeof(PPGSample));
 uploadQueue = xQueueCreate(UPLOAD_QUEUE_SIZE, sizeof(UploadJob));

 // Tasks
 xTaskCreatePinnedToCore(sampleTask, "sample", 8192, NULL, 5, NULL, 0);
 xTaskCreatePinnedToCore(sendTask, "send", 16384, NULL, 3, NULL, 1);
 xTaskCreatePinnedToCore(uploadTask, "upload", 12288, NULL, 1, NULL, 1);

 Serial.println("\n============================================");
 debugPrint("READY", "Place finger on sensor");
 Serial.println("============================================\n");
}

// ================= LOOP =================
void loop() {
 // NTP re-sync định kỳ
 if (millis() - lastNtpSync > NTP_RESYNC_MS) {
 if (WiFi.status() == WL_CONNECTED) {
 timeClient.update();
 }
 lastNtpSync = millis();
 }

 // Kiểm tra nút BOOT — giữ 5s để reset WiFi
 if (digitalRead(BOOT_BTN) == LOW) {
 unsigned long ps = millis();
 while (digitalRead(BOOT_BTN) == LOW && millis() - ps < WIFI_RESET_HOLD_MS) {
 delay(100);
 }
 if (millis() - ps >= WIFI_RESET_HOLD_MS) {
 debugPrint("WIFI", "Resetting WiFi credentials...");
 WiFiManager wm;
 wm.resetSettings();
 ESP.restart();
 }
 }

 delay(100);
}
