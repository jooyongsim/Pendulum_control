// Swing-up variant of AccelerationControl/AccelerationController (kept separate so the
// original firmware and its limits stay unchanged for everyone else). Changes: rotor angle
// from the step counter re-anchored to ABS_POS while stopped (no SPI while moving), direct
// step-clock drive (no 100 ms stall at start/reversal), 24000 pps^2 / 4000 pps / 150 deg latch,
// fault cause bits in observation[5], CMD 6 = read ABS_POS while stopped.
// Host-driven acceleration control for STEVAL-EDUKIT01 / NUCLEO-F401RE.
// The Jupyter host sends signed acceleration in pps^2. This firmware integrates
// acceleration to target step velocity and applies that velocity to the L6474.
// Inspired by the ACCEL_CONTROL/apply_acceleration path in the UCLA EDUKIT main.c.

#include <math.h>
#include "RotaryEncoder.h"
#include "L6474.h"
#include "control-comms.hpp"

// L6474::run() hands the step rate to the library's DECELERATING -> STEADY -> ACCELERATING
// state machine, which only acts on step edges: the rate stayed at the 30 pps start value for
// three step periods (~100 ms) at every start and every reversal. The host already shapes the
// velocity, so drive the step clock directly. Rate changes are applied right after a step edge,
// as the library itself does, so a timer period is never shortened mid-cycle.
class L6474Direct : public L6474 {
 public:
  L6474Direct(uint8_t flag_irq, uint8_t standby_reset, uint8_t direction, uint8_t pwm_pin,
              uint8_t ssel, SPIClass *spi)
      : L6474(flag_irq, standby_reset, direction, pwm_pin, ssel, spi) {}

  void start_at(motorDir_t dir, uint16_t pps) {
    sync_steps();                            // bank the previous run before relativePos restarts
    L6474_SetDirection(dir);                 // only effective while INACTIVE
    lock_rate(pps);
    device_prm.commandExecuted = RUN_CMD;
    device_prm.accu = 0;
    noInterrupts();
    device_prm.relativePos = 0;
    last_rel_ = 0;
    interrupts();
    // HardStop() only stops the step clock; the bridges stay enabled, so the enable command
    // (an interrupt-blocking SPI transfer) is needed only after init or a safety reset.
    if (!enabled_) {
      L6474_CmdEnable();
      enabled_ = true;
    }
    L6474_PwmSetFreq(pps);
  }

  void set_rate(uint16_t pps) {
    lock_rate(pps);
    L6474_PwmSetFreq(pps);
  }

  uint32_t steps_taken() const { return device_prm.relativePos; }

  // Rotor position from the step clock itself: no SPI while the motor runs.
  int32_t position_steps() {
    sync_steps();
    return pos_steps_;
  }
  void zero_steps() {
    sync_steps();
    pos_steps_ = 0;
  }
  void require_enable() { enabled_ = false; }
  int32_t driver_abs_pos() { return L6474_GetPosition(); }   // SPI; call only while stopped

  // The library step counter drifts by tens of steps over a closed-loop run (every rate change
  // pauses and reprograms the step timer); ABS_POS counts the STEP pulses the driver received.
  // Re-anchor to it while stopped, when no step ISR can preempt the SPI transfer.
  void resync_from_driver() {
    sync_steps();
    pos_steps_ = L6474_GetPosition();
  }

 private:
  // STEADY with speed == maxSpeed: the step ISR leaves the rate alone.
  void lock_rate(uint16_t pps) {
    noInterrupts();
    device_prm.maxSpeed = pps;
    device_prm.speed = pps;
    device_prm.motionState = STEADY;
    interrupts();
  }

  void sync_steps() {
    noInterrupts();
    uint32_t rel = device_prm.relativePos;
    motorDir_t dir = device_prm.direction;
    interrupts();
    const int32_t d = (int32_t)(rel - last_rel_);
    last_rel_ = rel;
    pos_steps_ += (dir == FORWARD) ? d : -d;
  }

  int32_t pos_steps_ = 0;
  uint32_t last_rel_ = 0;
  bool enabled_ = false;
};

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
static const int CMD_READ_DRIVER_POS = 6;  // stopped only: observation[4] = SPI ABS_POS [deg] (diagnostic)

const int ENC_STEPS_PER_ROTATION = 1200;
const int STP_STEPS_PER_ROTATION = 200;
static const unsigned int MICROSTEP_DIV = 16;

static const unsigned int MOTOR_TVAL_MA = 800;
static const unsigned int MOTOR_MAX_SPEED_PPS = 4000;   // was 2000; raised for disturbance recovery
// L6474 run() cannot use a useful speed below its implementation minimum.
// Commands below this magnitude are represented as a stop/dead-zone.
static const unsigned int MOTOR_MIN_RUN_PPS = 30;

// Host acceleration limits. These are deliberately much lower than the
// 131071 pps^2 guard used in the original main.c. Increase only after testing.
static const float MAX_ACCEL_PPS2 = 24000.0f;   // was 12000; recovery is limited by arm acceleration
static const float MAX_DECEL_PPS2 = 24000.0f;
// Last-resort latch only. The host scripts keep their own, tighter arm limit (default 80 deg);
// widen that only after checking the encoder cable tolerates the extra travel.
static const float ROTOR_SOFT_LIMIT_DEG = 150.0f;
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
L6474Direct *stepper = nullptr;
ControlComms ctrl;

volatile bool driver_flag_pending = false;
volatile bool driver_spi_error = false;
bool driver_fault_flag = false;   // OCD / TH_SD / UVLO seen on a FLAG interrupt
bool report_driver_pos = false;
float driver_pos_deg = 0.0f;
unsigned int last_l6474_status = 0;
bool safety_latched = false;
bool timeout_latched = false;
int commanded_direction = 0;  // -1, 0, +1

float accel_command_pps2 = 0.0f;
// Rotor angle. get_position() is an L6474 SPI read that disables interrupts; doing it on every
// loop() pass dropped bytes of the 500000 baud host commands (~40% of queries went unanswered).
// The angle now comes from the library's own step counter (L6474Direct::position_steps), so no
// SPI runs while the motor moves. CMD_READ_DRIVER_POS compares it with ABS_POS while stopped.
float rotor_angle_deg = 0.0f;
uint32_t last_rate_step = 0;     // driver step count when the step rate was last changed
unsigned int last_rate_pps = 0;
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
  return (float)stepper->position_steps() * 360.0f / (float)steps_per_rev;   // step count, no SPI
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

// The L6474 library's default error handler calls exit(), which halts the MCU:
// the board then stops answering the host completely, which looks exactly like
// a dead serial link. It fires when an SPI transfer fails, and SPI transfers can
// be preempted by the step-clock ISR once the motor is running. Latch instead so
// the fault stays reportable and recoverable.
void driverErrorHandler(uint16_t error) {
  // Keep this minimal: no SPI, no printing -- it may run from an error context.
  driver_spi_error = true;
  safety_latched = true;
}

void service_driver_flag() {
  if (!driver_flag_pending) return;
  noInterrupts();
  driver_flag_pending = false;
  interrupts();

  // Reading STATUS clears the L6474 alarm flags, and a second FLAG edge used to overwrite the
  // faulty value with a healthy one -- the latch then looked like it had no cause. Once a fault
  // has latched, keep the first faulty status until RESET_SAFETY.
  const unsigned int st = stepper->get_status();
  if (!driver_fault_flag) last_l6474_status = st;
  const unsigned int fault_mask = L6474_STATUS_OCD | L6474_STATUS_TH_SD | L6474_STATUS_UVLO;
  if ((st & fault_mask) != fault_mask) {
    last_l6474_status = st;
    driver_fault_flag = true;
    safety_latched = true;
    zero_velocity();
  }
}

void apply_target_velocity(float signed_pps) {
  // This runs on every loop() pass. Only talk to the driver when the motor is actually
  // running: a hard_stop() SPI transfer on every idle pass blocks interrupts often enough
  // to drop bytes of the 500000 baud host commands (about 40% of queries went unanswered).
  if (safety_latched || timeout_latched) {
    if (commanded_direction != 0) hard_stop_motor();
    return;
  }

  const float rotor = rotor_angle_deg;
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
    if (commanded_direction != 0) hard_stop_motor();
    return;
  }

  const int new_dir = signed_pps >= 0.0f ? 1 : -1;
  const unsigned int pps = (unsigned int)lroundf(mag);

  // The original main.c changes DIR as integrated velocity crosses zero.
  // With the Arduino L6474 API, stop before reversing to avoid a phase jump.
  if (commanded_direction != 0 && new_dir != commanded_direction) {
    hard_stop_motor();
  }

  if (commanded_direction == 0) {
    stepper->start_at(new_dir > 0 ? FORWARD : BACKWARD, (uint16_t)pps);
    last_rate_step = stepper->steps_taken();
    last_rate_pps = pps;
    commanded_direction = new_dir;
  } else if (pps != last_rate_pps && stepper->steps_taken() != last_rate_step) {
    stepper->set_rate((uint16_t)pps);        // a step edge has passed since the last change
    last_rate_step = stepper->steps_taken();
    last_rate_pps = pps;
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
    if (fabs(rotor_angle_deg) >= ROTOR_SOFT_LIMIT_DEG) return STATUS_LIMIT;
    return STATUS_DRIVER_FAULT;
  }
  return commanded_direction == 0 ? STATUS_OK : STATUS_MOVING;
}

void send_state() {
  float observation[NUM_OBS];
  observation[0] = get_encoder_angle();
  observation[1] = rotor_angle_deg;
  observation[2] = get_measured_signed_speed();
  observation[3] = target_velocity_pps;
  observation[4] = report_driver_pos ? driver_pos_deg : accel_command_pps2;
  report_driver_pos = false;
  // Low 16 bits: last L6474 status. 0x10000: SPI error handler fired. 0x20000: FLAG fault latched.
  observation[5] = (float)(last_l6474_status | (driver_spi_error ? 0x10000UL : 0UL) |
                           (driver_fault_flag ? 0x20000UL : 0UL));
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

  stepper = new L6474Direct(STP_FLAG_IRQ_PIN, STP_STBY_RST_PIN, STP_DIR_PIN, STP_PWM_PIN,
                      STP_SPI_CS_PIN, &dev_spi);
  if (stepper->init(&stepper_config) != COMPONENT_OK) {
    while (1) {
      digitalWrite(LED_PIN, !digitalRead(LED_PIN));
      delay(150);
    }
  }

  stepper->attach_error_handler(&driverErrorHandler);
  stepper->attach_flag_irq(&stepperISR);
  stepper->enable_flag_irq();
  stepper->set_home();
  last_l6474_status = stepper->get_status();
  last_integrate_us = micros();
  last_host_command_ms = millis();
}

void loop() {
  service_driver_flag();

  rotor_angle_deg = get_rotor_angle_signed();   // exact step count every pass, no SPI
  if (!safety_latched && fabs(rotor_angle_deg) > ROTOR_SOFT_LIMIT_DEG) {
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
  if (commanded_direction == 0) {
    // Stopped, and the host is waiting for this reply, so nothing is arriving on the serial line.
    stepper->resync_from_driver();
    rotor_angle_deg = get_rotor_angle_signed();
  }

  switch (command) {
    case CMD_SET_HOME:
      zero_velocity();
      stepper->set_home();
      stepper->zero_steps();
      rotor_angle_deg = 0.0f;
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
      stepper->require_enable();          // an OCD/UVLO shutdown disables the bridges
      driver_spi_error = false;
      driver_fault_flag = false;
      safety_latched = false;
      timeout_latched = false;
      break;

    case CMD_READ_DRIVER_POS:
      if (commanded_direction == 0) {
        driver_pos_deg = (float)stepper->driver_abs_pos() * 360.0f / (float)(STP_STEPS_PER_ROTATION * MICROSTEP_DIV);
        report_driver_pos = true;
      }
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
