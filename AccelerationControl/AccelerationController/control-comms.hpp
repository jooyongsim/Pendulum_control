#pragma once

#include <Arduino.h>
#include <ArduinoJson.h>

class ControlComms {
 public:
  enum DebugLevel {
    DEBUG_NONE = 0,
    DEBUG_ERROR,
    DEBUG_WARN,
    DEBUG_INFO
  };

  enum StatusCode {
    OK = 0,
    RX_EMPTY,
    ERROR
  };

  int init(Stream &serial_port, DebugLevel debug_level = DEBUG_NONE) {
    stream_ = &serial_port;
    debug_ = debug_level;
    return OK;
  }

  void send_observation(
      int status,
      unsigned long timestamp,
      bool terminated,
      float *observation,
      size_t num_obs,
      uint8_t digits = 3) {
    stream_->print(F("{\"status\":"));
    stream_->print(status);
    stream_->print(F(",\"timestamp\":"));
    stream_->print(timestamp);
    stream_->print(F(",\"terminated\":"));
    stream_->print(terminated ? F("true") : F("false"));
    stream_->print(F(",\"observation\":["));
    for (size_t i = 0; i < num_obs; ++i) {
      stream_->print(observation[i], digits);
      if (i + 1 < num_obs) stream_->print(',');
    }
    stream_->println(F("]}"));
  }

  template <size_t num_actions>
  StatusCode receive_action(int *command, float *action_out) {
    if (stream_ == nullptr || stream_->available() <= 0) return RX_EMPTY;

    constexpr size_t capacity = JSON_OBJECT_SIZE(2) + JSON_ARRAY_SIZE(num_actions) + 64;
    StaticJsonDocument<capacity> doc;
    DeserializationError err = deserializeJson(doc, *stream_);

    if (err) {
      if (err == DeserializationError::EmptyInput ||
          err == DeserializationError::IncompleteInput ||
          err == DeserializationError::InvalidInput) {
        return RX_EMPTY;
      }
      return ERROR;
    }

    if (!doc.containsKey("command") || !doc.containsKey("action")) return ERROR;
    JsonArray vals = doc["action"].as<JsonArray>();
    if (vals.size() != num_actions) return ERROR;

    *command = doc["command"].as<int>();
    for (size_t i = 0; i < num_actions; ++i) {
      action_out[i] = vals[i].as<float>();
    }

    while (stream_->available() > 0 && isspace(stream_->peek())) stream_->read();
    return OK;
  }

 private:
  Stream *stream_ = nullptr;
  DebugLevel debug_ = DEBUG_NONE;
};
