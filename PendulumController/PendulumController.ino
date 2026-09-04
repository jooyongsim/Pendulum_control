/**
 * STEVAL-EDUKIT01 rotary inverted pendulum serial controller.
 *
 * Designed for fast host-side PID experiments. The important addition over
 * the original example is CMD_SET_VELOCITY: the host sends signed pulse rate
 * rather than repeatedly stacking MOVE_BY commands.
 */

#include "RotaryEncoder.h"
#include "L6474.h"
#include "control-comms.hpp"

// Pins: Nucleo + X-NUCLEO-IHM01A1 / STEVAL-EDUKIT01 setup.
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
static constexpr size_t NUM_OBS = 4;

// Host-visible status.
static const int STATUS_OK = 0;
static const int STATUS_MOVING = 1;
static const int STATUS_LIMIT = 2;
static const int STATUS_DRIVER_FAULT = 3;

// Commands. IDs 0..3 remain compatible with the original project.
static const int CMD_SET_HOME = 0;
static const int CMD_MOVE_TO = 1;
static const int CMD_MOVE_BY = 2;
static const int CMD_SET_STEP_MODE = 3;
static const int CMD_SET_VELOCITY = 4;  // action = signed pps
static const int CMD_HARD_STOP = 5;
static const int CMD_QUERY = 6;
static const int CMD_RESET_SAFETY = 7;

const int ENC_STEPS_PER_ROTATION = 1200;
const int STP_STEPS_PER_ROTATION = 200;

// EDUKIT-oriented L6474 settings. Verify motor current rating before use.
static const unsigned int MOTOR_TVAL_MA = 800;
static const unsigned int MOTOR_MAX_SPEED_PPS = 2000;
static const unsigned int MOTOR_MIN_SPEED_PPS = 30;
static const unsigned int MOTOR_ACCEL_PPS2 = 6000;
static const unsigned int MOTOR_DECEL_PPS2 = 6000;

// Mechanical safety. Home should be set near the center of rotor travel.
static const float ROTOR_SOFT_LIMIT_DEG = 90.0f;

L6474_init_t stepper_config = {
  MOTOR_ACCEL_PPS2,
  MOTOR_DECEL_PPS2,
  MOTOR_MAX_SPEED_PPS,
  MOTOR_MIN_SPEED_PPS,
  MOTOR_TVAL_MA,
  L6474_OCD_TH_2250mA,
  L6474_CONFIG_OC_SD_ENABLE,
  L6474_CONFIG_EN_TQREG_TVAL_USED,
  L6474_STEP_SEL_1_16,
  L6474_SYNC_SEL_1_2,
  L6474_FAST_STEP_12us,
  L6474_TOFF_FAST_8us,
  3,
  21,
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
unsigned int div_per_step = 16;

volatile bool driver_flag_pending = false;
unsigned int last_l6474_status = 0;
bool safety_latched = false;
int commanded_direction = 0;  // -1 BWD, 0 stopped, +1 FWD

void stepperISR() {
  // Do not print or perform SPI transactions in this ISR. Text would corrupt
  // the host JSON stream, and SPI work is handled safely from loop().
  driver_flag_pending = true;
}

void encoderISR() {
  encoder->tick();
}

float get_encoder_angle() {
  long pos = encoder->getPosition();
  pos %= ENC_STEPS_PER_ROTATION;
  if (pos < 0) pos += ENC_STEPS_PER_ROTATION;
  return (float)pos * 360.0f / (float)ENC_STEPS_PER_ROTATION;
}

float get_rotor_angle_signed() {
  const long steps_per_rev = (long)STP_STEPS_PER_ROTATION * div_per_step;
  const long pos = stepper->get_position();
  return (float)pos * 360.0f / (float)steps_per_rev;
}

float get_signed_speed() {
  float speed = (float)stepper->get_speed();
  if (commanded_direction < 0) speed = -speed;
  if (commanded_direction == 0) speed = 0.0f;
  return speed;
}

void hard_stop_motor() {
  stepper->hard_stop();
  commanded_direction = 0;
}

void service_driver_flag() {
  if (!driver_flag_pending) return;
  noInterrupts();
  driver_flag_pending = false;
  interrupts();

  last_l6474_status = stepper->get_status();

  const unsigned int fault_mask =
      L6474_STATUS_OCD |
      L6474_STATUS_TH_SD |
      L6474_STATUS_UVLO;

  // L6474 fault bits are active-low in the status register.
  if ((last_l6474_status & fault_mask) != fault_mask) {
    safety_latched = true;
    hard_stop_motor();
  }
}

bool rotor_limit_exceeded_for_command(float signed_pps) {
  const float rotor = get_rotor_angle_signed();
  return (rotor >= ROTOR_SOFT_LIMIT_DEG && signed_pps > 0.0f) ||
         (rotor <= -ROTOR_SOFT_LIMIT_DEG && signed_pps < 0.0f);
}

void set_signed_velocity(float signed_pps) {
  if (safety_latched) {
    hard_stop_motor();
    return;
  }

  if (rotor_limit_exceeded_for_command(signed_pps)) {
    safety_latched = true;
    hard_stop_motor();
    return;
  }

  if (signed_pps > (float)MOTOR_MAX_SPEED_PPS) signed_pps = MOTOR_MAX_SPEED_PPS;
  if (signed_pps < -(float)MOTOR_MAX_SPEED_PPS) signed_pps = -MOTOR_MAX_SPEED_PPS;

  const float magnitude = fabs(signed_pps);
  if (magnitude < (float)MOTOR_MIN_SPEED_PPS) {
    hard_stop_motor();
    return;
  }

  const int new_direction = signed_pps >= 0.0f ? 1 : -1;
  const unsigned int target_pps = (unsigned int)magnitude;

  // Reversing an already-running step clock abruptly is a common source of
  // lost steps. Stop first, then restart in the opposite direction.
  if (commanded_direction != 0 && new_direction != commanded_direction) {
    hard_stop_motor();
  }

  stepper->set_max_speed(target_pps);

  if (commanded_direction == 0) {
    if (new_direction > 0) {
      stepper->run(StepperMotor::FWD);
    } else {
      stepper->run(StepperMotor::BWD);
    }
    commanded_direction = new_direction;
  }
}

void move_stepper_to(float deg) {
  if (safety_latched || stepper->get_device_state() != INACTIVE) return;
  if (deg > ROTOR_SOFT_LIMIT_DEG) deg = ROTOR_SOFT_LIMIT_DEG;
  if (deg < -ROTOR_SOFT_LIMIT_DEG) deg = -ROTOR_SOFT_LIMIT_DEG;
  const long steps = lroundf(deg * STP_STEPS_PER_ROTATION * div_per_step / 360.0f);
  stepper->go_to(steps);
}

void move_stepper_by(float deg) {
  if (safety_latched || stepper->get_device_state() != INACTIVE) return;

  const float current = get_rotor_angle_signed();
  float target = current + deg;
  if (target > ROTOR_SOFT_LIMIT_DEG) deg = ROTOR_SOFT_LIMIT_DEG - current;
  if (target < -ROTOR_SOFT_LIMIT_DEG) deg = -ROTOR_SOFT_LIMIT_DEG - current;

  long steps = lroundf(deg * STP_STEPS_PER_ROTATION * div_per_step / 360.0f);
  if (steps == 0) return;
  const StepperMotor::direction_t direction = steps > 0 ? StepperMotor::FWD : StepperMotor::BWD;
  if (steps < 0) steps = -steps;
  stepper->move(direction, steps);
}

void set_step_mode(int mode) {
  // L6474 STEP_MODE changes should only be made while inactive.
  if (stepper->get_device_state() != INACTIVE) hard_stop_motor();

  StepperMotor::step_mode_t step_mode = StepperMotor::STEP_MODE_1_16;
  unsigned int divisor = 16;
  switch (mode) {
    case 0: step_mode = StepperMotor::STEP_MODE_FULL; divisor = 1; break;
    case 1: step_mode = StepperMotor::STEP_MODE_HALF; divisor = 2; break;
    case 2: step_mode = StepperMotor::STEP_MODE_1_4; divisor = 4; break;
    case 3: step_mode = StepperMotor::STEP_MODE_1_8; divisor = 8; break;
    case 4: step_mode = StepperMotor::STEP_MODE_1_16; divisor = 16; break;
    default: return;
  }

  if (stepper->set_step_mode(step_mode)) {
    div_per_step = divisor;
    stepper->set_home();
  }
}

int host_status() {
  if (safety_latched) {
    const float rotor = get_rotor_angle_signed();
    if (fabs(rotor) >= ROTOR_SOFT_LIMIT_DEG) return STATUS_LIMIT;
    return STATUS_DRIVER_FAULT;
  }
  return stepper->get_device_state() == INACTIVE ? STATUS_OK : STATUS_MOVING;
}

void send_state() {
  float observation[NUM_OBS];
  observation[0] = get_encoder_angle();
  observation[1] = get_rotor_angle_signed();
  observation[2] = get_signed_speed();
  observation[3] = (float)last_l6474_status;
  ctrl.send_observation(host_status(), millis(), safety_latched, observation, NUM_OBS);
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

  stepper = new L6474(
      STP_FLAG_IRQ_PIN, STP_STBY_RST_PIN, STP_DIR_PIN, STP_PWM_PIN,
      STP_SPI_CS_PIN, &dev_spi);

  if (stepper->init(&stepper_config) != COMPONENT_OK) {
    // Do not enter the JSON protocol if motor initialization failed.
    while (1) {
      digitalWrite(LED_PIN, !digitalRead(LED_PIN));
      delay(150);
    }
  }

  stepper->attach_flag_irq(&stepperISR);
  stepper->enable_flag_irq();
  stepper->set_home();
  last_l6474_status = stepper->get_status();
}

void loop() {
  service_driver_flag();

  // Independent hardware-side rotor guard, even if the host stops responding.
  if (!safety_latched && fabs(get_rotor_angle_signed()) > ROTOR_SOFT_LIMIT_DEG) {
    safety_latched = true;
    hard_stop_motor();
  }

  int command = CMD_QUERY;
  float action[NUM_ACTIONS] = {0.0f};
  const ControlComms::StatusCode rx = ctrl.receive_action<NUM_ACTIONS>(&command, action);

  if (rx != ControlComms::OK) return;

  switch (command) {
    case CMD_SET_HOME:
      hard_stop_motor();
      stepper->set_home();
      safety_latched = false;
      break;

    case CMD_MOVE_TO:
      move_stepper_to(action[0]);
      break;

    case CMD_MOVE_BY:
      move_stepper_by(action[0]);
      break;

    case CMD_SET_STEP_MODE:
      set_step_mode((int)action[0]);
      break;

    case CMD_SET_VELOCITY:
      set_signed_velocity(action[0]);
      break;

    case CMD_HARD_STOP:
      hard_stop_motor();
      break;

    case CMD_QUERY:
      break;

    case CMD_RESET_SAFETY:
      hard_stop_motor();
      last_l6474_status = stepper->get_status();
      safety_latched = false;
      break;

    default:
      break;
  }

  send_state();
}
