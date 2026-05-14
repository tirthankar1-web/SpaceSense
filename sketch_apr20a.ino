#include <WiFi.h>
#include <HTTPClient.h>

const char* ssid = "ARJUN";
const char* password = "Cutebaby69";

String serverName = "https://public-space-detection-default-rtdb.firebaseio.com/data.json";

#define TRIG 5
#define ECHO 18

long duration;
float distance;

void setup() {
  Serial.begin(115200);

  pinMode(TRIG, OUTPUT);
  pinMode(ECHO, INPUT);

  WiFi.begin(ssid, password);

  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nConnected!");
}

float getDistance() {
  digitalWrite(TRIG, LOW);
  delayMicroseconds(2);

  digitalWrite(TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG, LOW);

  duration = pulseIn(ECHO, HIGH);
  distance = duration * 0.034 / 2;

  return distance;
}

void loop() {
  float d = getDistance();
  Serial.println(d);

  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;

    String jsonData = "{\"distance\": " + String(d) + "}";

    http.begin(serverName);
    http.addHeader("Content-Type", "application/json");

    int code = http.PUT(jsonData);

    Serial.print("Response: ");
    Serial.println(code);

    http.end();
  }

  delay(5000);
}