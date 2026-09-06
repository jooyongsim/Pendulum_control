// Host-driven acceleration control for STEVAL-EDUKIT01 / NUCLEO-F401RE.
// The Jupyter host sends signed acceleration in pps^2. This firmware integrates
// acceleration to target step velocity and applies that velocity to the L6474.
// Inspired by the ACCEL_CONTROL/apply_acceleration path in the UCLA EDUKIT main.c.

#include <math.h>
#include "RotaryEncoder.h"
#include "L6474.h"
#include "control-comms.hpp"

const int LED_PIN = LED_BUILTIN;
const int ENC_A_PIN = D4;
const int ENC_B_PIN = D5;
const int STP_FLAG_IRQ_PIN = D2;
const int STP_STBY_RST_PIN = D8;
const int STP_DIR_PIN = D7;
const int STP_PWM_PIN = D9;
const int8_t STP_SPI_CS_PIN = D10;
const int8_t STP_SPI_MOSI_PIN = D11;
const int8_t STP_SPI_MISO_PIN = D12;
const int8_t STP_SPI_SCK_PIN = D13;

static const unsigned int BAUD_RATE = 500000;
static const ControlComms::DebugLevel CTRL_DEBUG = ControlComms::DEBUG_NONE;
static constexpr size_t NUM_ACTIONS = 1;
static constexpr size_t NUM_OBS = 7;

static const int STATUS_OK = 0;
static const int STATUS_MOVING = 1;
static const int STATUS_LIMIT = 2;
static const int STATUS_DRIVER_FAULT = 3;
static const int STATUS_TIMEOUT = 4;

static const int CMD_SET_HOME = 0;
static const int CMD_SET_ACCELERATION = 1;  // action[0] = signed pps^2
static const int CMD_HARD_STOP = 2;
static const int CMD_QUERY = 3;
static const int CMD_RESET_SAFETY = 4;
static const int CMD_ZERO_VELOCITY = 5;

const int ENC_STEPS_PER_ROTATION = 1200;
const int STP_STEPS_PER_ROTATION = 200;
static const unsigned int MICROSTEP_DIV = 16;

static const unsigned int MOTOR_TVAL_MA = 800;
static const unsigned int MOTOR_MAX_SPEED_PPS = 2000;
// L6474 run() cannot use a useful speed below its implementation minimum.
// Commands below this magnitude are represented as a stop/dead-zone.
static const unsigned int MOTOR_MIN_RUN_PPS = 30;

// Host acceleration limits. These are deliberately much lower than the
// 131071 pps^2 guard used in the original main.c. Increase only after testing.
static const float MAX_ACCEL_PPS2 = 12000.0f;
static const float MAX_DECEL_PPS2 = 12000.0f;
static const float ROTOR_SOFT_LIMIT_DEG = 90.0f;
static const unsigned long COMMAND_TIMEOUT_MS = 150;

L6474_init_t stepper_config = {
  6000, 6000, MOTOR_MAX_SPEED_PPS, MOTOR_MIN_RUN_PPS, MOTOR_TVAL_MA,
  L6474_OCD_TH_2250mA,
  L6474_CONFIG_OC_SD_ENABLE,
  L6474_CONFIG_EN_TQREG_TVAL_USED,
  L6474_STEP_SEL_1_16,
  L6474_SYNC_SEL_1_2,
  L6474_FAST_STEP_12us,
  L6474_TOFF_FAST_8us,
  3, 21,
  L6474_CONFIG_TOFF_044us,
  L6474_CONFIG_SR_320V_us,
  L6474_CONFIG_INT_16MHZ,
  L6474_ALARM_EN_OVERCURRENT |
  L6474_ALARM_EN_THERMAL_SHUTDOWN |
  L6474_ALARM_EN_THERMAL_WARNING |
  L6474_ALARM_EN_UNDERVOLTAGE |
  L6474_ALARM_EN_SW_TURN_ON |
  L6474_ALARM_EN_WRONG_NPERF_CMD
};

RotaryEncoder *encoder = nullptr;
SPIClass dev_spi(STP_SPI_MOSI_PIN, STP_SPI_MISO_PIN, STP_SPI_SCK_PIN);
L6474 *stepper = nullptr;
ControlComms ctrl;

volatile bool driver_flag_pending = false;
unsigned int last_l6474_status = 0;
bool safety_latched = false;
bool timeout_latched = false;
int commanded_direction = 0;  // -1, 0, +1

float accel_command_pps2 = 0.0f;
float target_velocity_pps = 0.0f;
unsigned long last_integrate_us = 0;
unsigned long last_host_command_ms = 0;

void stepperISR() { driver_flag_pending = true; }
void encoderISR() { encoder->tick(); }

float get_encoder_angle() {
  long pos = encoder->getPosition();
  pos %= ENC_STEPS_PER_ROTATION;
  if (pos < 0) pos += ENC_STEPS_PER_ROTATION;
  return (float)pos * 360.0f / (float)ENC_STEPS_PER_ROTATION;
}

float get_rotor_angle_signed() {
  const long steps_per_rev = (long)STP_STEPS_PER_ROTATION * MICROSTEP_DIV;
  return (float)stepper->get_position() * 360.0f / (float)steps_per_rev;
}

float get_measured_signed_speed() {
  float v = (float)stepper->get_speed();
  if (commanded_direction < 0) v = -v;
  if (commanded_direction == 0) v = 0.0f;
  return v;
}

void hard_stop_motor() {
  stepper->hard_stop();
  commanded_direction = 0;
}

void zero_velocity() {
  target_velocity_pps = 0.0f;
  accel_command_pps2 = 0.0f;
  hard_stop_motor();
}

void service_driver_flag() {
  if (!driver_flag_pending) return;
  noInterrupts();
  driver_flag_pending = false;
  interrupts();

  last_l6474_status = stepper->get_status();
  const unsigned int fault_mask = L6474_STATUS_OCD | L6474_STATUS_TH_SD | L6474_STATUS_UVLO;
  if ((last_l6474_status & fault_mask) != fault_mask) {
    safety_latched = true;
    zero_velocity();
  }
}

void apply_target_velocity(float signed_pps) {
  if (safety_latched || timeout_latched) {
    hard_stop_motor();
    return;
  }

  const float rotor = get_rotor_angle_signed();
  if ((rotor >= ROTOR_SOFT_LIMIT_DEG && signed_pps > 0.0f) ||
      (rotor <= -ROTOR_SOFT_LIMIT_DEG && signed_pps < 0.0f)) {
    safety_latched = true;
    zero_velocity();
    return;
  }

  if (signed_pps > (float)MOTOR_MAX_SPEED_PPS) signed_pps = MOTOR_MAX_SPEED_PPS;
  if (signed_pps < -(float)MOTOR_MAX_SPEED_PPS) signed_pps = -MOTOR_MAX_SPEED_PPS;

  const float mag = fabs(signed_pps);
  if (mag < (float)MOTOR_MIN_RUN_PPS) {
    hard_stop_motor();
    return;
  }

  const int new_dir = signed_pps >= 0.0f ? 1 : -1;
  const unsigned int pps = (unsigned int)lroundf(mag);

  // The original main.c changes DIR as integrated velocity crosses zero.
  // With the Arduino L6474 API, stop before reversing to avoid a phase jump.
  if (commanded_direction != 0 && new_dir != commanded_direction) {
    hard_stop_motor();
  }

  // During run(), changing max speed changes the generated STEP rate.
  stepper->set_max_speed(pps);
  if (commanded_direction == 0) {
    stepper->run(new_dir > 0 ? StepperMotor::FWD : StepperMotor::BWD);
    commanded_direction = new_dir;
  }
}

void integrate_acceleration() {
  const unsigned long now_us = micros();
  if (last_integrate_us == 0) {
    last_integrate_us = now_us;
    return;
  }

  unsigned long delta_us = now_us - last_integrate_us;
  last_integrate_us = now_us;

  // Avoid a giant integration jump after USB/serial stalls.
  if (delta_us > 20000UL) delta_us = 20000UL;
  const float dt = (float)delta_us * 1.0e-6f;

  float a = accel_command_pps2;
  const float old_v = target_velocity_pps;

  // Match the intent of the original main.c: positive acceleration increases
  // the signed velocity; negative acceleration reduces it and can reverse it.
  if (old_v >= 0.0f) {
    if (a > MAX_ACCEL_PPS2) a = MAX_ACCEL_PPS2;
    if (a < -MAX_DECEL_PPS2) a = -MAX_DECEL_PPS2;
  } else {
    if (a < -MAX_ACCEL_PPS2) a = -MAX_ACCEL_PPS2;
    if (a > MAX_DECEL_PPS2) a = MAX_DECEL_PPS2;
  }

  // Core EDUKIT relationship: v[k+1] = v[k] + a[k] Ts.
  target_velocity_pps += a * dt;
  if (target_velocity_pps > (float)MOTOR_MAX_SPEED_PPS) target_velocity_pps = MOTOR_MAX_SPEED_PPS;
  if (target_velocity_pps < -(float)MOTOR_MAX_SPEED_PPS) target_velocity_pps = -MOTOR_MAX_SPEED_PPS;

  apply_target_velocity(target_velocity_pps);
}

int host_status() {
  if (timeout_latched) return STATUS_TIMEOUT;
  if (safety_latched) {
    if (fabs(get_rotor_angle_signed()) >= ROTOR_SOFT_LIMIT_DEG) return STATUS_LIMIT;
    return STATUS_DRIVER_FAULT;
  }
  return commanded_direction == 0 ? STATUS_OK : STATUS_MOVING;
}

void send_state() {
  float observation[NUM_OBS];
  observation[0] = get_encoder_angle();
  observation[1] = get_rotor_angle_signed();
  observation[2] = get_measured_signed_speed();
  observation[3] = target_velocity_pps;
  observation[4] = accel_command_pps2;
  observation[5] = (float)last_l6474_status;
  observation[6] = (float)(millis() - last_host_command_ms);
  ctrl.send_observation(host_status(), millis(), safety_latched || timeout_latched, observation, NUM_OBS);
}

void setup() {
  pinMode(LED_PIN, OUTPUT);
  pinMode(ENC_A_PIN, INPUT_PULLUP);
  pinMode(ENC_B_PIN, INPUT_PULLUP);

  Serial.begin(BAUD_RATE);
  ctrl.init(Serial, CTRL_DEBUG);

  encoder = new RotaryEncoder(ENC_A_PIN, ENC_B_PIN, RotaryEncoder::LatchMode::TWO03);
  attachInterrupt(digitalPinToInterrupt(ENC_A_PIN), encoderISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_B_PIN), encoderISR, CHANGE);

  stepper = new L6474(STP_FLAG_IRQ_PIN, STP_STBY_RST_PIN, STP_DIR_PIN, STP_PWM_PIN,
                      STP_SPI_CS_PIN, &dev_spi);
  if (stepper->init(&stepper_config) != COMPONENT_OK) {
    while (1) {
      digitalWrite(LED_PIN, !digitalRead(LED_PIN));
      delay(150);
    }
  }

  stepper->attach_flag_irq(&stepperISR);
  stepper->enable_flag_irq();
  stepper->set_home();
  last_l6474_status = stepper->get_status();
  last_integrate_us = micros();
  last_host_command_ms = millis();
}

void loop() {
  service_driver_flag();

  if (!safety_latched && fabs(get_rotor_angle_signed()) > ROTOR_SOFT_LIMIT_DEG) {
    safety_latched = true;
    zero_velocity();
  }

  if (!timeout_latched && (millis() - last_host_command_ms) > COMMAND_TIMEOUT_MS) {
    timeout_latched = true;
    zero_velocity();
  }

  // Integration runs on the MCU, independent of serial transaction jitter.
  integrate_acceleration();

  int command = CMD_QUERY;
  float action[NUM_ACTIONS] = {0.0f};
  const ControlComms::StatusCode rx = ctrl.receive_action<NUM_ACTIONS>(&command, action);
  if (rx != ControlComms::OK) return;

  last_host_command_ms = millis();

  switch (command) {
    case CMD_SET_HOME:
      zero_velocity();
      stepper->set_home();
      safety_latched = false;
      timeout_latched = false;
      break;

    case CMD_SET_ACCELERATION:
      timeout_latched = false;
      accel_command_pps2 = action[0];
      break;

    case CMD_HARD_STOP:
      zero_velocity();
      break;

    case CMD_QUERY:
      timeout_latched = false;
      break;

    case CMD_RESET_SAFETY:
      zero_velocity();
      last_l6474_status = stepper->get_status();
      safety_latched = false;
      timeout_latched = false;
      break;

    case CMD_ZERO_VELOCITY:
      timeout_latched = false;
      zero_velocity();
      break;

    default:
      break;
  }

  send_state();
}
