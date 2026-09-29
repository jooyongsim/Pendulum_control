/**
 * Swing-up + balance for the STEVAL-EDUKIT01, run ON THE BOARD.
 *
 * Why here and not on the host: the upright error doubles every ~94 ms, and one
 * serial round trip over the ST-Link VCP costs ~15 ms, so a host loop tops out
 * near 66 Hz and loses the pendulum right after catching it. Here the loop runs
 * at CONTROL_HZ with no link in the path; the host only starts/stops it, tunes
 * gains and logs.
 *
 * Angles
 *   theta  pendulum error from UPRIGHT [rad]; the encoder reads 0 hanging, so
 *          theta = wrap(encoder - 180 deg). Start with the pendulum hanging still.
 *   phi    rotor angle from home [rad]
 *
 * Control (input is pivot acceleration u [m/s^2]; the arm converts it)
 *   swing-up   u = PUMP_GAIN*(E - E_up)*sign(theta_dot*cos theta) - k_rot*phi - k_rot_d*phi_dot
 *   balance    u = -(k1*theta + k2*theta_dot + k3*phi + k4*phi_dot)
 *   both       alpha_ddot = u / ARM_RADIUS_M ;  v += alpha_ddot*dt ;  pps = SIGN*v*3200/2pi
 *
 * SIGN is the measured coupling: a positive pps command tips the pendulum the
 * other way (step_response.py, 6/6 steps agreed).
 *
 * Commands (action[0] unless noted)
 *   0 SET_HOME        rotor position = 0, clears the latch
 *   3 SET_STEP_MODE   0..4 (4 = 1/16)
 *   5 HARD_STOP       stop the motor and leave control mode
 *   6 QUERY           just read state
 *   7 RESET_SAFETY    clear the latch
 *   12 SET_HANG_REF   the rod is hanging NOW: use this as the reference
 *   14 RESET_GAINS    restore the compiled gain defaults (gains otherwise persist
 *                     across runs until the board is reset)
 *   13 SET_ACCEL      open-loop pivot acceleration [m/s^2] while idle; used by the
 *                     host to give the hanging rod a doublet for the vertical
 *                     calibration, and ignored in the control modes
 *   10 SET_MODE       0 idle, 1 balance only (hand it the pendulum), 2 swing-up + balance
 *   11 SET_GAIN       action = [index, value] -- see GAIN_* below
 *                     index 16 = observer pole [rad/s], 0 disables the observer
 *
 * Observation: [theta_deg, rotor_deg, v_cmd_pps, mode(3=balancing), u_accel,
 *               theta_dot_dps, upright_offset_deg]
 */

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
static constexpr size_t NUM_ACTIONS = 2;
static constexpr size_t NUM_OBS = 7;

// Host-visible status
static const int STATUS_OK = 0;
static const int STATUS_MOVING = 1;
static const int STATUS_LIMIT = 2;
static const int STATUS_DRIVER_FAULT = 3;

// Commands
static const int CMD_SET_HOME = 0;
static const int CMD_SET_STEP_MODE = 3;
static const int CMD_HARD_STOP = 5;
static const int CMD_QUERY = 6;
static const int CMD_RESET_SAFETY = 7;
static const int CMD_SET_MODE = 10;
static const int CMD_SET_GAIN = 11;
static const int CMD_SET_HANG_REF = 12;   // current position = hanging
static const int CMD_SET_ACCEL = 13;      // open-loop pivot accel [m/s^2], idle mode only
static const int CMD_RESET_GAINS = 14;    // back to the compiled defaults

// Modes
static const int MODE_IDLE = 0;
static const int MODE_BALANCE = 1;
static const int MODE_SWINGUP = 2;

// ---------------------------------------------------------------- mechanics
// From the free-swing identification (encoder_log_20260922_161218) and the
// measured arm radius. Change these if the rig changes.
static const float WN = 7.521f;            // rad/s, pendulum natural frequency
static const float SIGMA = 0.1446f;        // 1/s, viscous decay
static const float ARM_RADIUS_M = 0.120f;  // m, motor axis -> pendulum pivot
static const float L_EFF = 9.80665f / (7.521f * 7.521f);   // g/wn^2 [m]; input gain 1/L_eff
static const float COMMAND_SIGN = -1.0f;   // measured coupling direction

static const int ENC_STEPS_PER_ROTATION = 1200;
static const int STP_STEPS_PER_ROTATION = 200;

static const unsigned int MOTOR_TVAL_MA = 800;
static const unsigned int MOTOR_MAX_SPEED_PPS = 4000;
static const unsigned int MOTOR_MIN_SPEED_PPS = 30;
static const unsigned int MOTOR_ACCEL_PPS2 = 10000;
static const unsigned int MOTOR_DECEL_PPS2 = 10000;

static const float CONTROL_HZ = 500.0f;
static const float DT = 1.0f / CONTROL_HZ;
static const float TAU_D = 0.02f;          // derivative low-pass [s]
static const float ACCEL_MAX = 2.45f;      // m/s^2 at the pivot (measured 2026-09-22)
static const float PPS_PER_RAD_S = (STP_STEPS_PER_ROTATION * 16) / (2.0f * PI);
static const float ROTOR_LIMIT_DEG = 85.0f;
static const float ROTOR_ABORT_DEG = 200.0f;   // give up rather than wind the cable
static const float DEG = 180.0f / PI;

// ---------------------------------------------------------------- gains
enum {
  GAIN_K1 = 0, GAIN_K2, GAIN_K3, GAIN_K4,     // balance: theta, theta_dot, phi, phi_dot
  GAIN_PUMP, GAIN_KROT, GAIN_KROT_D,          // swing-up
  GAIN_CATCH_DEG, GAIN_CATCH_DPS, GAIN_GUARD_DEG,
  GAIN_CATCH_ARM_DEG,      // only catch with the arm inside this
  GAIN_BIAS_TAU,           // upright-offset adaptation time constant [s], 0 = off
  GAIN_PHIREF_TAU,         // how fast the arm reference decays to home [s]
  GAIN_UPRIGHT_OFFSET_DEG, // manual offset of the true vertical
  GAIN_TAU_D,              // derivative low-pass [s]; the encoder is 0.3 deg coarse,
                           // so one count at 500 Hz looks like 150 deg/s of noise
  GAIN_BIAS_DEADBAND_DEG,  // stop adapting while the arm is this close to home
  GAIN_KE,                 // >0 switches the pumping to energy-scaled: the push eases
                           // off near the target so the rod ARRIVES SLOWLY. With the
                           // plain law the energy error is ~-113 at the bottom, so the
                           // command saturates right up to the top and the rod flies
                           // past the catch window.
  GAIN_ROTOR_LIMIT_DEG,    // arm travel the controller may use; the pumping stalls
                           // whenever the arm sits on this limit
  GAIN_OBS_POLE,           // observer pole [rad/s]; 0 (default) = filtered difference.
                           // Opt-in: during a fast swing-up spin the estimate can lag
                           // the real rod, and the pumping sign follows the ESTIMATE.
  GAIN_COUNT
};
// Pole placement on the identified plant (wn_des 6, zeta 0.8, rotor poles -2, -2.6)
float gains[GAIN_COUNT] = {
  -25.372f, -3.075f, -0.397f, -0.459f,
  0.20f, 1.50f, 0.50f,
  8.0f, 90.0f, 25.0f,
  60.0f, 6.0f, 3.0f, 0.0f, 0.02f, 8.0f, 0.0f, 85.0f, 0.0f
};

// The encoder zero is wherever the rod happened to rest at reset, anywhere inside
// the stiction band, so "upright" is not exactly 180 deg away from it. The ratio
// k1/|k3| is ~64 deg of arm per deg of pendulum error here, so even 1 deg of bias
// parks the arm ~60 deg from home -- which is exactly what the first runs did.
// bias_rad is that offset: seeded from GAIN_UPRIGHT_OFFSET_DEG and then adapted
// while balancing, because an arm sitting away from its reference IS the error.
float bias_rad = 0.0f;
float phi_ref = 0.0f;

// The encoder counts from wherever it was at power-up, and a rod that went over
// the top during earlier runs leaves that zero half a turn out. Then "upright"
// lands on the hanging position and the controller tries to balance at the
// bottom. enc_zero_rad pins the reference to a known hanging rod instead.
float enc_zero_rad = 0.0f;
float open_loop_accel = 0.0f;   // m/s^2, idle-mode calibration drive

const float GAIN_DEFAULTS[GAIN_COUNT] = {
  -25.372f, -3.075f, -0.397f, -0.459f,
  0.20f, 1.50f, 0.50f,
  8.0f, 90.0f, 25.0f,
  60.0f, 6.0f, 3.0f, 0.0f, 0.02f, 8.0f, 0.0f, 85.0f, 0.0f
};

// ---------------------------------------------------------------- state
RotaryEncoder *encoder = nullptr;
SPIClass dev_spi(STP_SPI_MOSI_PIN, STP_SPI_MISO_PIN, STP_SPI_SCK_PIN);
L6474 *stepper = nullptr;
ControlComms ctrl;
unsigned int div_per_step = 16;

volatile bool driver_flag_pending = false;
unsigned int last_l6474_status = 0;
bool safety_latched = false;
bool balancing = false;
int mode = MODE_IDLE;

float theta = 0.0f, theta_dot = 0.0f;      // from upright [rad], [rad/s]
float phi = 0.0f, phi_dot = 0.0f;          // rotor [rad], [rad/s]
float theta_prev = 0.0f, phi_prev = 0.0f;
float v_cmd = 0.0f;                        // commanded rotor rate [rad/s]
float u_last = 0.0f;                       // last pivot acceleration [m/s^2]
int commanded_direction = 0;
unsigned long last_control_us = 0;

void stepperISR() { driver_flag_pending = true; }   // no SPI, no printing here
void encoderISR() { encoder->tick(); }

void driverErrorHandler(uint16_t error) {
  // The library's default handler calls exit(), which would halt the MCU and
  // take the serial link with it. Latch instead so the fault stays reportable.
  (void)error;
  safety_latched = true;
}

float wrap_pi(float a) {
  while (a > PI) a -= 2.0f * PI;
  while (a < -PI) a += 2.0f * PI;
  return a;
}

float encoder_angle_rad() {
  long pos = encoder->getPosition() % ENC_STEPS_PER_ROTATION;
  if (pos < 0) pos += ENC_STEPS_PER_ROTATION;
  return (float)pos * 2.0f * PI / (float)ENC_STEPS_PER_ROTATION;
}

// Physical rotor angle, as the driver counts it.
float rotor_angle_rad() {
  const long steps_per_rev = (long)STP_STEPS_PER_ROTATION * div_per_step;
  return (float)stepper->get_position() * 2.0f * PI / (float)steps_per_rev;
}

// The controller works in the frame where a POSITIVE command moves phi positive.
// COMMAND_SIGN is the measured coupling, so the physical angle must be flipped
// into that frame; using the raw angle makes the rotor-centring term push the
// arm outwards instead of home, which winds the cable up.
float rotor_angle_model_rad() {
  return COMMAND_SIGN * rotor_angle_rad();
}

void stop_motor() {
  stepper->hard_stop();
  commanded_direction = 0;
  v_cmd = 0.0f;
}

void service_driver_flag() {
  if (!driver_flag_pending) return;
  noInterrupts();
  driver_flag_pending = false;
  interrupts();
  last_l6474_status = stepper->get_status();
  const unsigned int fault_mask = L6474_STATUS_OCD | L6474_STATUS_TH_SD | L6474_STATUS_UVLO;
  if ((last_l6474_status & fault_mask) != fault_mask) {   // fault bits are active low
    safety_latched = true;
    stop_motor();
  }
}

void apply_velocity(float rad_s) {
  float pps = COMMAND_SIGN * rad_s * PPS_PER_RAD_S;
  if (pps > (float)MOTOR_MAX_SPEED_PPS) pps = MOTOR_MAX_SPEED_PPS;
  if (pps < -(float)MOTOR_MAX_SPEED_PPS) pps = -MOTOR_MAX_SPEED_PPS;

  if (fabsf(pps) < (float)MOTOR_MIN_SPEED_PPS) {
    if (commanded_direction != 0) stop_motor();
    return;
  }
  const int dir = pps >= 0.0f ? 1 : -1;
  if (commanded_direction != 0 && dir != commanded_direction) {
    // reversing a running step clock loses steps; stop first
    stepper->hard_stop();
    commanded_direction = 0;
  }
  stepper->set_max_speed((unsigned int)fabsf(pps));
  if (commanded_direction == 0) {
    stepper->run(dir > 0 ? StepperMotor::FWD : StepperMotor::BWD);
    commanded_direction = dir;
  }
}

// Energy measured so that upright at rest is WN^2 and hanging at rest is -WN^2.
float pendulum_energy() {
  return 0.5f * theta_dot * theta_dot + WN * WN * cosf(theta);
}

void control_step(float dt) {
  // --- state estimate -------------------------------------------------------
  const float th_meas = wrap_pi(encoder_angle_rad() - enc_zero_rad - PI - bias_rad);
  const float ph = rotor_angle_model_rad();
  const float tau_d = fmaxf(gains[GAIN_TAU_D], 1e-3f);
  const float a = tau_d / (tau_d + dt);
  phi_dot = a * phi_dot + (1.0f - a) * ((ph - phi_prev) / dt);
  phi = ph;
  phi_prev = ph;

  if (gains[GAIN_OBS_POLE] > 0.0f) {
    // Luenberger observer on the pendulum. Differentiating a 0.3 deg encoder at
    // 500 Hz turns a single count into 150 deg/s of noise, so the rate estimate
    // had to be filtered hard -- and that lag is what saturated the loop when the
    // gains were raised. The observer takes the rate from the MODEL, driven by
    // the acceleration we command, and uses the encoder only to correct it:
    //   theta'     = theta_dot + l1 (th_meas - theta)
    //   theta_dot' = wn^2 sin(theta) - 2 sigma theta_dot - (u/L_eff) cos(theta)
    //                + l2 (th_meas - theta)
    // with both observer poles at GAIN_OBS_POLE.
    const float pole = gains[GAIN_OBS_POLE];
    const float l1 = 2.0f * pole - 2.0f * SIGMA;
    const float l2 = pole * pole + WN * WN - 2.0f * SIGMA * l1;
    const float err = wrap_pi(th_meas - theta);
    const float acc = WN * WN * sinf(theta) - 2.0f * SIGMA * theta_dot
                      - (u_last / L_EFF) * cosf(theta);
    theta = wrap_pi(theta + (theta_dot + l1 * err) * dt);
    theta_dot += (acc + l2 * err) * dt;
  } else {
    theta_dot = a * theta_dot + (1.0f - a) * (wrap_pi(th_meas - theta_prev) / dt);
    theta = th_meas;
  }
  theta_prev = th_meas;

  if (safety_latched) {
    if (commanded_direction != 0) stop_motor();
    u_last = 0.0f;
    open_loop_accel = 0.0f;
    return;
  }
  if (mode == MODE_IDLE) {
    // Open-loop drive for the host's calibration doublet. The rotor guard below
    // does not run here, so the host keeps these short and zero-mean.
    if (open_loop_accel != 0.0f || commanded_direction != 0) {
      u_last = open_loop_accel;
      v_cmd += (open_loop_accel / ARM_RADIUS_M) * dt;
      const float v_max_idle = (float)MOTOR_MAX_SPEED_PPS / PPS_PER_RAD_S;
      if (v_cmd > v_max_idle) v_cmd = v_max_idle;
      if (v_cmd < -v_max_idle) v_cmd = -v_max_idle;
      if (fabsf(phi) * DEG > ROTOR_ABORT_DEG) { stop_motor(); open_loop_accel = 0.0f; }
      apply_velocity(v_cmd);
    }
    return;
  }

  // --- rotor travel guard ---------------------------------------------------
  // Hard stop well before the cable can wind up, whatever the controller wants.
  if (fabsf(phi) * DEG > ROTOR_ABORT_DEG) {
    mode = MODE_IDLE;
    stop_motor();
    u_last = 0.0f;
    return;
  }
  if (fabsf(phi) * DEG > gains[GAIN_ROTOR_LIMIT_DEG] && (phi * v_cmd) > 0.0f) {
    v_cmd = 0.0f;                       // do not drive further outwards
  }

  const float catch_rad = gains[GAIN_CATCH_DEG] / DEG;
  const float catch_rate = gains[GAIN_CATCH_DPS] / DEG;
  const float guard_rad = gains[GAIN_GUARD_DEG] / DEG;

  float u;
  const float catch_arm = gains[GAIN_CATCH_ARM_DEG] / DEG;
  const bool near_upright = fabsf(theta) < catch_rad && fabsf(theta_dot) < catch_rate
                            && fabsf(phi) < catch_arm;
  if (!balancing && near_upright) {
    balancing = true;
    phi_ref = phi;                       // catch where the arm happens to be ...
  } else if (balancing && fabsf(theta) > guard_rad) {
    balancing = false;                   // lost it; swing again
  }

  if (balancing) {
    // ... and walk that reference home, so the arm drifts back to centre instead
    // of being yanked there while the rod is still settling.
    phi_ref *= expf(-dt / fmaxf(gains[GAIN_PHIREF_TAU], 1e-3f));
    if (gains[GAIN_BIAS_TAU] > 0.0f) {
      // Slow upright-offset adaptation. At steady state the balancer holds
      //     phi - phi_ref = (k1/k3) * (bias - true_offset)
      // so a POSITIVE arm offset means the bias is too LARGE: subtract, do not add.
      // (Adding is what walked the offset from 1.35 to 3.11 deg and drove the arm
      // into its limit.)
      // Deadband: once the arm sits near its reference the offset is as good as
      // the encoder can tell, and further integration just wanders with friction.
      const float err = phi - phi_ref;
      if (fabsf(err) * DEG > gains[GAIN_BIAS_DEADBAND_DEG]) {
        const float gamma = fabsf(gains[GAIN_K3]) / (fabsf(gains[GAIN_K1]) * gains[GAIN_BIAS_TAU]);
        bias_rad -= gamma * err * dt;
      }
      const float lim = 5.0f / DEG;
      if (bias_rad > lim) bias_rad = lim;
      if (bias_rad < -lim) bias_rad = -lim;
    }
    u = -(gains[GAIN_K1] * theta + gains[GAIN_K2] * theta_dot
          + gains[GAIN_K3] * (phi - phi_ref) + gains[GAIN_K4] * phi_dot);
  } else if (mode == MODE_SWINGUP) {
    const float e_err = pendulum_energy() - WN * WN;      // <0 while short of upright
    const float s = (theta_dot * cosf(theta)) >= 0.0f ? 1.0f : -1.0f;
    if (gains[GAIN_KE] > 0.0f) {
      // Energy error normalised by the full hanging-to-upright gap (2 wn^2), so
      // the push fades as the rod approaches the top instead of saturating into it.
      float scale = gains[GAIN_KE] * e_err / (2.0f * WN * WN);
      if (scale > 1.0f) scale = 1.0f;
      if (scale < -1.0f) scale = -1.0f;
      u = ACCEL_MAX * scale * s;
    } else {
      u = gains[GAIN_PUMP] * e_err * s;
    }
    u -= gains[GAIN_KROT] * phi + gains[GAIN_KROT_D] * phi_dot;
  } else {
    // balance-only mode with the pendulum outside the window: wait, motor stopped
    if (commanded_direction != 0) stop_motor();
    u_last = 0.0f;
    return;
  }

  if (u > ACCEL_MAX) u = ACCEL_MAX;
  if (u < -ACCEL_MAX) u = -ACCEL_MAX;
  u_last = u;

  v_cmd += (u / ARM_RADIUS_M) * dt;                 // rotor rate integrator
  const float v_max = (float)MOTOR_MAX_SPEED_PPS / PPS_PER_RAD_S;
  if (v_cmd > v_max) v_cmd = v_max;
  if (v_cmd < -v_max) v_cmd = -v_max;
  apply_velocity(v_cmd);
}

int host_status() {
  if (safety_latched) return STATUS_DRIVER_FAULT;
  if (fabsf(phi) * DEG >= gains[GAIN_ROTOR_LIMIT_DEG]) return STATUS_LIMIT;  // phi is in the model frame
  return commanded_direction != 0 ? STATUS_MOVING : STATUS_OK;
}

void send_state() {
  float obs[NUM_OBS];
  obs[0] = theta * DEG;                    // error from upright
  obs[1] = rotor_angle_rad() * DEG;        // physical rotor angle
  obs[2] = COMMAND_SIGN * v_cmd * PPS_PER_RAD_S;
  obs[3] = (float)(balancing ? 3 : mode);   // 3 = balancing
  obs[4] = u_last;
  obs[5] = theta_dot * DEG;
  obs[6] = bias_rad * DEG;                 // live upright offset [deg]
  ctrl.send_observation(host_status(), millis(), safety_latched, obs, NUM_OBS);
}

void set_step_mode(int m) {
  if (stepper->get_device_state() != INACTIVE) stop_motor();
  StepperMotor::step_mode_t sm = StepperMotor::STEP_MODE_1_16;
  unsigned int divisor = 16;
  switch (m) {
    case 0: sm = StepperMotor::STEP_MODE_FULL; divisor = 1; break;
    case 1: sm = StepperMotor::STEP_MODE_HALF; divisor = 2; break;
    case 2: sm = StepperMotor::STEP_MODE_1_4; divisor = 4; break;
    case 3: sm = StepperMotor::STEP_MODE_1_8; divisor = 8; break;
    case 4: sm = StepperMotor::STEP_MODE_1_16; divisor = 16; break;
    default: return;
  }
  if (stepper->set_step_mode(sm)) {
    div_per_step = divisor;
    stepper->set_home();
  }
}

L6474_init_t stepper_config = {
  MOTOR_ACCEL_PPS2, MOTOR_DECEL_PPS2, MOTOR_MAX_SPEED_PPS, MOTOR_MIN_SPEED_PPS,
  MOTOR_TVAL_MA,
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
  L6474_ALARM_EN_OVERCURRENT | L6474_ALARM_EN_THERMAL_SHUTDOWN |
  L6474_ALARM_EN_THERMAL_WARNING | L6474_ALARM_EN_UNDERVOLTAGE |
  L6474_ALARM_EN_SW_TURN_ON | L6474_ALARM_EN_WRONG_NPERF_CMD
};

void setup() {
  pinMode(LED_PIN, OUTPUT);
  pinMode(ENC_A_PIN, INPUT_PULLUP);
  pinMode(ENC_B_PIN, INPUT_PULLUP);

  Serial.begin(BAUD_RATE);
  ctrl.init(Serial, ControlComms::DEBUG_NONE);

  encoder = new RotaryEncoder(ENC_A_PIN, ENC_B_PIN, RotaryEncoder::LatchMode::TWO03);

  stepper = new L6474(STP_FLAG_IRQ_PIN, STP_STBY_RST_PIN, STP_DIR_PIN, STP_PWM_PIN,
                      STP_SPI_CS_PIN, &dev_spi);
  stepper->attach_error_handler(&driverErrorHandler);   // before init(): see above

  if (stepper->init(&stepper_config) != COMPONENT_OK) {
    while (1) {                                          // fast blink = driver init failed
      digitalWrite(LED_PIN, !digitalRead(LED_PIN));
      delay(150);
    }
  }

  // Encoder interrupts only after init(), so a swinging pendulum cannot break
  // the SPI transfers inside it.
  attachInterrupt(digitalPinToInterrupt(ENC_A_PIN), encoderISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_B_PIN), encoderISR, CHANGE);

  stepper->attach_flag_irq(&stepperISR);
  stepper->enable_flag_irq();
  stepper->set_home();
  last_l6474_status = stepper->get_status();

  enc_zero_rad = encoder_angle_rad();     // boot: assume the rod hangs still
  theta_prev = wrap_pi(-PI);
  phi_prev = rotor_angle_model_rad();
  last_control_us = micros();
}

void loop() {
  service_driver_flag();

  // ---- fixed-rate control, independent of the host ------------------------
  const unsigned long now = micros();
  if ((unsigned long)(now - last_control_us) >= (unsigned long)(DT * 1e6f)) {
    const float dt = (now - last_control_us) * 1e-6f;
    last_control_us = now;
    control_step(dt);
  }

  // ---- host link ----------------------------------------------------------
  int command = CMD_QUERY;
  float action[NUM_ACTIONS] = {0.0f, 0.0f};
  if (ctrl.receive_action<NUM_ACTIONS>(&command, action) != ControlComms::OK) return;

  switch (command) {
    case CMD_SET_HOME:
      stop_motor();
      stepper->set_home();
      phi = 0.0f;
      phi_prev = 0.0f;
      phi_ref = 0.0f;
      phi_dot = 0.0f;
      safety_latched = false;
      break;

    case CMD_SET_STEP_MODE:
      set_step_mode((int)action[0]);
      break;

    case CMD_HARD_STOP:
      mode = MODE_IDLE;
      stop_motor();
      break;

    case CMD_QUERY:
      break;

    case CMD_RESET_SAFETY:
      stop_motor();
      last_l6474_status = stepper->get_status();
      safety_latched = false;
      break;

    case CMD_SET_MODE: {
      const int m = (int)action[0];
      if (m >= MODE_IDLE && m <= MODE_SWINGUP) {
        if (m != mode) {
          stop_motor();
          theta_dot = 0.0f;
          phi_dot = 0.0f;
          balancing = false;
          phi_ref = 0.0f;
          open_loop_accel = 0.0f;
          bias_rad = gains[GAIN_UPRIGHT_OFFSET_DEG] / DEG;
        }
        mode = m;
      }
      break;
    }

    case CMD_SET_HANG_REF:
      enc_zero_rad = encoder_angle_rad();
      theta_prev = wrap_pi(-PI - bias_rad);
      theta_dot = 0.0f;
      break;

    case CMD_SET_ACCEL:
      if (mode == MODE_IDLE) {
        open_loop_accel = action[0];
        if (action[0] == 0.0f) { stop_motor(); }
      }
      break;

    case CMD_RESET_GAINS:
      for (int i = 0; i < GAIN_COUNT; i++) gains[i] = GAIN_DEFAULTS[i];
      bias_rad = 0.0f;
      break;

    case CMD_SET_GAIN: {
      const int idx = (int)action[0];
      if (idx >= 0 && idx < GAIN_COUNT) gains[idx] = action[1];
      break;
    }

    default:
      break;
  }

  send_state();
}
