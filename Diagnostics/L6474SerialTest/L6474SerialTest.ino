#include "L6474.h"

const int STP_FLAG_IRQ_PIN = D2;
const int STP_STBY_RST_PIN = D8;
const int STP_DIR_PIN = D7;
const int STP_PWM_PIN = D9;
const int8_t STP_SPI_CS_PIN = D10;
const int8_t STP_SPI_MOSI_PIN = D11;
const int8_t STP_SPI_MISO_PIN = D12;
const int8_t STP_SPI_SCK_PIN = D13;

static const int FULL_STEPS_PER_REV = 200;
static const int MICROSTEP_DIV = 16;

static const unsigned int MOTOR_TVAL_MA = 800;
static const unsigned int MOTOR_MAX_SPEED_PPS = 500;
static const unsigned int MOTOR_MIN_SPEED_PPS = 100;
static const unsigned int MOTOR_ACCEL_PPS2 = 1000;
static const unsigned int MOTOR_DECEL_PPS2 = 1000;

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

SPIClass dev_spi(STP_SPI_MOSI_PIN, STP_SPI_MISO_PIN, STP_SPI_SCK_PIN);
L6474 *stepper = nullptr;

long degreeToSteps(float deg) {
  const float stepsPerRev = (float)FULL_STEPS_PER_REV * (float)MICROSTEP_DIV;
  return lroundf(deg * stepsPerRev / 360.0f);
}

float stepsToDegree(long steps) {
  const float stepsPerRev = (float)FULL_STEPS_PER_REV * (float)MICROSTEP_DIV;
  return (float)steps * 360.0f / stepsPerRev;
}

void printStatus() {
  const long pos = stepper->get_position();
  const unsigned int status = stepper->get_status();

  Serial.print("STATUS pos_steps=");
  Serial.print(pos);
  Serial.print(" pos_deg=");
  Serial.print(stepsToDegree(pos), 3);
  Serial.print(" speed_pps=");
  Serial.print(stepper->get_speed());
  Serial.print(" state=");
  Serial.print((int)stepper->get_device_state());
  Serial.print(" l6474=0x");
  Serial.println(status, HEX);
}

void goToDegree(float targetDeg) {
  const long targetSteps = degreeToSteps(targetDeg);

  Serial.print("ACK GOTO target_deg=");
  Serial.print(targetDeg, 3);
  Serial.print(" target_steps=");
  Serial.println(targetSteps);

  stepper->go_to(targetSteps);
  stepper->wait_while_active();

  const long pos = stepper->get_position();
  const unsigned int status = stepper->get_status();

  Serial.print("DONE target_deg=");
  Serial.print(targetDeg, 3);
  Serial.print(" pos_steps=");
  Serial.print(pos);
  Serial.print(" pos_deg=");
  Serial.print(stepsToDegree(pos), 3);
  Serial.print(" l6474=0x");
  Serial.println(status, HEX);
}

void moveByDegree(float deltaDeg) {
  long steps = degreeToSteps(deltaDeg);

  if (steps == 0) {
    Serial.println("ERR requested movement is zero steps");
    return;
  }

  StepperMotor::direction_t dir;
  if (steps > 0) {
    dir = StepperMotor::FWD;
  } else {
    dir = StepperMotor::BWD;
    steps = -steps;
  }

  Serial.print("ACK MOVE delta_deg=");
  Serial.print(deltaDeg, 3);
  Serial.print(" steps=");
  Serial.println(steps);

  stepper->move(dir, steps);
  stepper->wait_while_active();

  const long pos = stepper->get_position();
  const unsigned int status = stepper->get_status();

  Serial.print("DONE delta_deg=");
  Serial.print(deltaDeg, 3);
  Serial.print(" pos_steps=");
  Serial.print(pos);
  Serial.print(" pos_deg=");
  Serial.print(stepsToDegree(pos), 3);
  Serial.print(" l6474=0x");
  Serial.println(status, HEX);
}

void handleCommand(String line) {
  line.trim();
  if (line.length() == 0) return;

  if (line == "HOME") {
    stepper->hard_stop();
    stepper->wait_while_active();
    stepper->set_home();
    Serial.println("DONE HOME");
    printStatus();
    return;
  }

  if (line == "STATUS") {
    printStatus();
    Serial.println("DONE STATUS");
    return;
  }

  if (line == "STOP") {
    stepper->hard_stop();
    stepper->wait_while_active();
    Serial.println("DONE STOP");
    printStatus();
    return;
  }

  if (line.startsWith("GOTO ")) {
    goToDegree(line.substring(5).toFloat());
    return;
  }

  if (line.startsWith("MOVE ")) {
    moveByDegree(line.substring(5).toFloat());
    return;
  }

  Serial.print("ERR unknown command: ");
  Serial.println(line);
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("========================================");
  Serial.println("L6474 serial diagnostic");
  Serial.println("200 steps/rev");
  Serial.println("1/16 microstep");
  Serial.println("TVAL = 800 mA");
  Serial.println("max = 500 pps");
  Serial.println("min = 100 pps");
  Serial.println("acc/dec = 1000 pps^2");
  Serial.println("========================================");

  stepper = new L6474(
      STP_FLAG_IRQ_PIN,
      STP_STBY_RST_PIN,
      STP_DIR_PIN,
      STP_PWM_PIN,
      STP_SPI_CS_PIN,
      &dev_spi);

  if (stepper->init(&stepper_config) != COMPONENT_OK) {
    Serial.println("FATAL L6474 initialization failed");
    while (1) delay(1000);
  }

  stepper->set_home();
  Serial.println("READY");
  printStatus();
}

void loop() {
  if (Serial.available() <= 0) return;
  String line = Serial.readStringUntil('\n');
  handleCommand(line);
}
