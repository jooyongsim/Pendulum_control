#include "L6474.h"

// Minimal L6474 / stepper diagnostic for:
//   NUCLEO-F401RE + X-NUCLEO-IHM01A1 + STEVAL-EDUKIT01 wiring
//
// Purpose:
//   Remove Python, serial command parsing, encoder interrupts, PID, and soft limits
//   from the test path. The motor should repeatedly move to +45 deg and -45 deg.
//
// Diagnostic configuration:
//   - Full step
//   - 200 full steps/rev motor
//   - 800 mA TVAL
//   - 500 pps max speed
//   - 100 pps min speed
//   - 1000 pps^2 acceleration/deceleration
//
// Serial Monitor: 115200 baud

const int STP_FLAG_IRQ_PIN = D2;
const int STP_STBY_RST_PIN = D8;
const int STP_DIR_PIN = D7;
const int STP_PWM_PIN = D9;
const int8_t STP_SPI_CS_PIN = D10;
const int8_t STP_SPI_MOSI_PIN = D11;
const int8_t STP_SPI_MISO_PIN = D12;
const int8_t STP_SPI_SCK_PIN = D13;

static const int FULL_STEPS_PER_REV = 200;
static const int TEST_ANGLE_DEG = 45;
static const int TEST_STEPS = FULL_STEPS_PER_REV * TEST_ANGLE_DEG / 360; // 25 steps
static const unsigned long HOLD_MS = 2000;

L6474_init_t stepper_config = {
  1000,                              // Acceleration [pps^2]
  1000,                              // Deceleration [pps^2]
  500,                               // Maximum speed [pps]
  100,                               // Minimum speed [pps]
  800,                               // Torque regulation current [mA]
  L6474_OCD_TH_2250mA,               // Over-current threshold
  L6474_CONFIG_OC_SD_ENABLE,
  L6474_CONFIG_EN_TQREG_TVAL_USED,
  L6474_STEP_SEL_1,                   // FULL STEP
  L6474_SYNC_SEL_1_2,
  L6474_FAST_STEP_12us,
  L6474_TOFF_FAST_8us,
  3,                                 // TON_MIN [us]
  21,                                // TOFF_MIN [us]
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

void printMotorState(const char *label) {
  const long pos = stepper->get_position();
  const unsigned int status = stepper->get_status();

  Serial.print(label);
  Serial.print("  position_steps=");
  Serial.print(pos);
  Serial.print("  approx_deg=");
  Serial.print((float)pos * 360.0f / (float)FULL_STEPS_PER_REV, 1);
  Serial.print("  status=0x");
  Serial.println(status, HEX);
}

void moveAndWait(long target_steps, const char *label) {
  Serial.println();
  Serial.print("Command: ");
  Serial.println(label);

  stepper->go_to(target_steps);
  stepper->wait_while_active();

  printMotorState("Stopped:");
  delay(HOLD_MS);
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("========================================");
  Serial.println("L6474 minimal stepper diagnostic");
  Serial.println("FULL STEP, +/-45 deg, 2 s hold");
  Serial.println("========================================");

  stepper = new L6474(
      STP_FLAG_IRQ_PIN,
      STP_STBY_RST_PIN,
      STP_DIR_PIN,
      STP_PWM_PIN,
      STP_SPI_CS_PIN,
      &dev_spi);

  Serial.println("Initializing L6474...");

  if (stepper->init(&stepper_config) != COMPONENT_OK) {
    Serial.println("ERROR: L6474 init failed");
    while (1) {
      delay(1000);
    }
  }

  Serial.println("L6474 init OK");

  // Current physical shaft position becomes 0 deg.
  stepper->set_home();
  printMotorState("Home:");

  delay(HOLD_MS);
}

void loop() {
  // +25 full steps = +45 deg from Home.
  moveAndWait(+TEST_STEPS, "+45 deg");

  // -25 full steps = -45 deg from Home.
  moveAndWait(-TEST_STEPS, "-45 deg");
}
