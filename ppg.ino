#define PPG_PIN 34

const int avgWindow = 20;
int buffer[avgWindow];
int bufferIndex = 0;
long bufferSum = 0;

void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("ESP32_READY");

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  for (int i = 0; i < avgWindow; i++)
    buffer[i] = 0;
}

void loop() {
  int raw = analogRead(PPG_PIN);

  // DC removal (keep waveform shape)
  bufferSum -= buffer[bufferIndex];
  buffer[bufferIndex] = raw;
  bufferSum += raw;
  bufferIndex = (bufferIndex + 1) % avgWindow;

  int dcRemoved = raw - (bufferSum / avgWindow);

  // Send continuous filtered waveform
  Serial.println(dcRemoved);

  delay(10); // 100 Hz
}