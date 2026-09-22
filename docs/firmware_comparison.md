# PendulumController와 SwingupController 비교

이 문서는 다음 두 펌웨어의 현재 구현을 비교한다.

- [PendulumController.ino](../PendulumController/PendulumController.ino)
- [SwingupController.ino](../swing-up/SwingupController/SwingupController.ino)

## 1. 가장 큰 차이: 위치·속도 명령과 가속도 명령

**PendulumController는 팔의 위치·속도를 명령하는 범용 실험용 펌웨어이고, SwingupController는 팔의 가속도를 명령하는 스윙업·직립 제어용 펌웨어다.**

두 펌웨어 모두 진자를 세우는 제어 법칙 자체는 PC의 Python 프로그램에서 계산한다. SwingupController만 업로드해도 자동으로 스윙업하는 구조는 아니다.

| 항목 | PendulumController | SwingupController |
|---|---|---|
| 주요 입력 | 위치 [deg], 속도 [pps] | 가속도 [pps²] |
| 속도 생성 | 지정한 속도로 라이브러리가 가감속 | 보드에서 `v ← v + a·dt` 계산 |
| 모터 구동 | `run()`, `set_max_speed()` | `L6474Direct`로 스텝 클록 직접 설정 |
| 최대 속도 | 4000 pps | 4000 pps |
| 가속도 | 라이브러리 가속·감속 설정 10000 pps² | 가속도 입력 제한 ±24000 pps² |
| 저속 영역 | 30 pps 미만 정지 | 30 pps 미만 정지 |
| 팔 위치 계산 | 드라이버 위치를 SPI로 읽음 | 구동 중 스텝 카운터 사용, 정지 중 드라이버 위치와 동기화 |
| 마이크로스텝 | 명령으로 변경 가능, 기본 1/16 | 1/16 고정 |
| 팔 각도 보호 | ±90° 설정은 있으나 현재 비활성화 | ±150° 범위 보호 활성화 |
| 통신 중단 보호 | 별도 타임아웃 없음 | 마지막 정상 수신 명령 후 150 ms 초과 시 정지 |
| 관측값 개수 | 4개 | 7개 |
| 시리얼 통신 속도 | 500000 baud | 500000 baud |

여기서 pps는 초당 스텝 펄스 수다. 기본 1/16 마이크로스텝에서는 모터 1회전이 3200펄스이므로 다음과 같이 변환한다.

```text
팔 각속도 [rad/s]   = 속도 [pps] × 2π / 3200
팔 각가속도 [rad/s²] = 가속도 [pps²] × 2π / 3200

4000 pps   ≈ 7.85 rad/s
24000 pps² ≈ 47.1 rad/s²
```

PendulumController에서 마이크로스텝을 변경하면 해당 분주비에 맞춰 변환해야 한다.

## 2. 제어 관점에서의 차이

### PendulumController: 속도를 지정한다

PC가 “팔을 +1000 pps 속도로 돌려라”라고 명령하면, 보드는 L6474 라이브러리의 가감속 동작을 이용해 목표 속도로 이동한다.

```text
Python의 속도 명령
    → set_signed_velocity()
    → set_max_speed() / run()
    → 라이브러리의 가감속
    → 모터 구동
```

절대 위치 이동 `MOVE_TO`와 상대 위치 이동 `MOVE_BY`도 지원한다. 두 위치 명령은 모터가 이미 동작 중이면 실행하지 않는다.

### SwingupController: 가속도를 지정한다

PC가 “팔에 +10000 pps² 가속도를 적용해라”라고 명령하면, 보드가 경과 시간을 이용해 목표 속도를 적분한다.

```text
Python의 가속도 명령
    → accel_command_pps2 저장
    → integrate_acceleration(): v ← v + a·dt
    → 속도 제한
    → apply_target_velocity()
    → 스텝 클록 설정
```

예를 들어 현재 속도가 0이고 +10000 pps²를 0.1초 동안 적용하면, 이상적인 목표 속도는 +1000 pps가 된다. 실제 구동에는 저속 정지 영역과 속도 제한이 적용된다.

가속도 적분은 보드의 `loop()`에서 수행한다. 시리얼 지연으로 한 번의 적분 시간이 지나치게 커지는 것을 막기 위해 적분에 사용하는 `dt`는 최대 20 ms로 제한한다.

스윙업 Python 코드의 모델 입력은 다음과 같다.

```text
u = φ̈  [rad/s²]
```

즉, 제어기가 계산하는 값이 팔 가속도이므로 SwingupController의 명령 방식과 직접 연결된다. 위치·속도 명령용 펌웨어로 바꾸면 제어기와 구동기 사이의 관계도 달라진다.

## 3. 모터 구동과 통신 처리의 차이

### 출발·방향 반전 처리

PendulumController는 라이브러리의 `run()`을 사용한다. SwingupController의 코드 주석에는 기존 라이브러리 구동에서 출발·반전 시 약 100 ms 지연이 발생했던 것이 변경 이유로 기록되어 있다.

SwingupController는 `L6474Direct` 클래스를 추가해 다음을 수행한다.

- 라이브러리 내부 속도 상태를 설정하고 스텝 클록 주파수를 직접 변경한다.
- 속도 변경은 이전 변경 이후 스텝이 발생했는지 확인한 뒤 적용한다.
- 방향 반전 시에는 먼저 정지한 뒤 반대 방향으로 시작한다.

스텝 클록 직접 제어는 라이브러리 내부 구조에 의존한다. 따라서 라이브러리 버전 호환성은 [스윙업 README](../swing-up/README.md)의 조건을 확인해야 한다.

### 팔 위치 읽기

PendulumController의 `get_rotor_angle_signed()`는 `stepper->get_position()`으로 드라이버 위치를 SPI로 읽는다.

SwingupController는 구동 중 소프트웨어 스텝 카운터를 사용하고, 정지 상태에서 호스트 명령을 받으면 드라이버의 ABS_POS와 다시 동기화한다. 반복적인 구동 중 위치 조회를 줄여 통신과 스텝 인터럽트에 미치는 영향을 줄이는 구조다.

“구동 중 SPI를 전혀 사용하지 않는다”는 의미는 아니다. 드라이버 오류 상태 확인 등에는 SPI 처리가 남아 있다.

**두 방식 모두 팔의 실제 축 운동을 별도 엔코더로 측정하는 것은 아니다.** 스텝 카운터와 ABS_POS는 펄스 기반 정보이므로, 실제 모터가 펄스를 따라가지 못하는 탈조를 직접 측정하는 값으로 해석하면 안 된다. 응답의 속도 역시 라이브러리의 구동 속도 정보다.

## 4. 명령 번호가 달라 서로 호환되지 않는다

| 명령 번호 | PendulumController | SwingupController |
|---:|---|---|
| 0 | `SET_HOME`: 홈 설정 | `SET_HOME`: 홈 설정 |
| 1 | `MOVE_TO`: 절대 위치 이동 [deg] | `SET_ACCELERATION`: 가속도 설정 [pps²] |
| 2 | `MOVE_BY`: 상대 위치 이동 [deg] | `HARD_STOP`: 정지 |
| 3 | `SET_STEP_MODE`: 마이크로스텝 변경 | `QUERY`: 상태 조회 |
| 4 | `SET_VELOCITY`: 속도 설정 [pps] | `RESET_SAFETY`: 보호 상태 초기화 |
| 5 | `HARD_STOP`: 정지 | `ZERO_VELOCITY`: 속도·가속도 0 및 정지 |
| 6 | `QUERY`: 상태 조회 | `READ_DRIVER_POS`: 정지 중 드라이버 위치 조회 |
| 7 | `RESET_SAFETY`: 보호 상태 초기화 | 정의 없음 |

예를 들어 기존 프로그램의 `command=2`는 상대 이동이지만, 스윙업 펌웨어에서는 정지다. 또한 `command=1`의 숫자는 한쪽에서는 각도이고 다른 쪽에서는 가속도다.

**펌웨어만 교체하고 기존 Python 프로그램을 그대로 실행하면 안 된다.** 명령 번호, 입력 단위, 관측값 형식을 함께 맞춰야 한다.

`SET_HOME`은 현재 팔 위치를 홈으로 지정하는 명령이다. 기존 홈 위치로 물리적으로 복귀시키는 명령은 아니다. 스윙업 Python 코드의 `home_arm()`이 기존 홈으로 복귀하는 동작을 따로 수행한다.

## 5. 응답 데이터 비교

배열 인덱스는 0부터 시작한다.

| 인덱스 | PendulumController | SwingupController |
|---:|---|---|
| 0 | 진자 엔코더 각도 [deg] | 진자 엔코더 각도 [deg] |
| 1 | 팔 각도 [deg] | 팔 각도 [deg] |
| 2 | 부호 있는 구동 속도 [pps] | 부호 있는 구동 속도 [pps] |
| 3 | 마지막 L6474 상태 | 적분한 목표 속도 [pps] |
| 4 | 없음 | 가속도 명령 [pps²], 또는 명령 6의 드라이버 위치 [deg] |
| 5 | 없음 | L6474 상태 및 추가 오류 비트 |
| 6 | 없음 | 마지막 정상 수신 명령 이후 경과 시간 [ms] |

공통으로 진자 엔코더 각도는 0 이상 360° 미만으로 반환한다. 직립 기준 각도로 변환하는 작업은 Python 쪽에서 수행한다.

SwingupController의 `observation[5]`는 다음 정보를 합친 값이다.

- 하위 16비트: 마지막 L6474 상태
- `0x10000`: SPI 오류 처리기 실행 여부
- `0x20000`: FLAG 오류 래치 여부

명령 6의 위치 조회는 정지 중일 때만 실행되며, 그 응답에서 `observation[4]`의 의미가 달라진다.

SwingupController는 최초 오류 상태를 보호 초기화 전까지 유지하도록 구현되어 있어, 후속 상태 읽기가 오류 원인을 덮어쓰는 것을 방지한다.

## 6. 보호 동작의 차이

### PendulumController의 현재 팔 제한은 꺼져 있다

```cpp
static const float ROTOR_SOFT_LIMIT_DEG = 90.0f;
static const bool ROTOR_LIMIT_ENABLED = false;
```

따라서 현재 코드에서는 위치 명령의 범위 제한과 팔 각도 초과 정지가 적용되지 않는다. “90°라는 상수가 있으므로 90°에서 멈춘다”고 해석하면 안 된다.

또한 호스트 통신 중단을 감지하는 타임아웃이 없다. 속도 구동 중 Python이 더 이상 명령을 보내지 않으면, 별도의 정지나 오류가 없는 한 기존 구동이 이어질 수 있다.

### SwingupController는 팔 제한과 통신 타임아웃을 적용한다

```cpp
static const float ROTOR_SOFT_LIMIT_DEG = 150.0f;
static const unsigned long COMMAND_TIMEOUT_MS = 150;
```

- 팔이 제한 밖으로 더 이동하려 하거나, 루프에서 절대각이 150°를 초과하면 보호 정지한다.
- 정상 수신 명령이 150 ms 동안 없으면 목표 속도와 가속도를 0으로 만들고 정지한다.
- 타임아웃 상태는 상태 조회나 새 가속도 명령 등 지정된 명령으로 해제할 수 있다.
- 드라이버 오류나 팔 범위 초과의 보호 래치는 별도로 관리한다.

스윙업 Python의 기본 팔 제한은 140°이며, 펌웨어의 150° 제한보다 먼저 정지하도록 되어 있다. 실제 허용 범위는 장비의 케이블과 기구 가동 범위에 맞춰 설정해야 한다.

두 펌웨어 모두 과전류·과열 셧다운·저전압 상태를 확인하고 보호 정지하는 코드를 포함한다.

## 7. Python 프로그램별 펌웨어 선택

아래 경로는 `Pendulum_control/`을 기준으로 한다.

| Python 프로그램 | 사용할 펌웨어 |
|---|---|
| `manual_keyboard_balance.py` | PendulumController |
| `diagnose_manual.py` | PendulumController |
| `measurement/log_free_swing.py` | PendulumController |
| `swing-up/accel_openloop_test.py` | SwingupController |
| `swing-up/balance_control.py` | SwingupController |
| `swing-up/swingup_control.py` | SwingupController |
| `simulator/rotary_swingup_realtime.py` | 펌웨어 불필요 |

PC 시뮬레이터는 펌웨어의 제한과 동작을 수치 모델로 모사한다. 실제 보드와 통신하지 않는다.

실제 장비에서 스윙업과 직립 균형을 실습하려면 **SwingupController.ino를 업로드하고 swing-up 폴더의 Python 프로그램을 실행한다.**

## 관련 소스

- [PendulumController 구현](../PendulumController/PendulumController.ino)
- [SwingupController 구현](../swing-up/SwingupController/SwingupController.ino)
- [스윙업 실행 안내](../swing-up/README.md)
- [스윙업 Python 제어기](../swing-up/swingup_control.py)
- [직립·감쇠 제어 및 통신 공통 코드](../swing-up/balance_control.py)
- [시뮬레이터 문서](../simulator/README.md)

이 비교는 저장소의 소스 코드를 기준으로 작성했다. 별도의 펌웨어 컴파일이나 실제 장비 구동 결과를 의미하지 않는다.


## 8. SwingupController 가속도 명령 처리

첫 번째 `SwingupController.ino`에서 받는 가속도는 **센서 측정값이 아니라, PC(Jupyter)가 보내는 모터 가속도 명령**입니다. 흐름은 **시리얼 수신 → 가속도 저장 → 속도로 적분 → 모터 구동**입니다.

### 1. 시리얼로 가속도 명령 수신

[414행 부근 (line 414)](<C:/Users/Sim/Dropbox/[Courses]/[고급딥러닝시스템과응용]/08_Claude/Pendulum_control/swing-up/SwingupController/SwingupController.ino:414>)에서 PC가 보낸 명령 종류와 입력값을 읽습니다.

```cpp
int command = CMD_QUERY;
float action[NUM_ACTIONS] = {0.0f};
const ControlComms::StatusCode rx =
    ctrl.receive_action<NUM_ACTIONS>(&command, action);
if (rx != ControlComms::OK) return;
```

- `command`: 실행할 명령 번호
- `action[0]`: PC가 보낸 가속도 값. 입력은 하나이므로 `NUM_ACTIONS = 1`입니다.
- 정상 수신되지 않으면 이번 `loop()`를 종료합니다.

통신은 `setup()`에서 `Serial`에 연결하며, 전송 속도는 **500000 baud**입니다.

### 2. 받은 값을 가속도 명령으로 저장

[435행 부근 (line 435)](<C:/Users/Sim/Dropbox/[Courses]/[고급딥러닝시스템과응용]/08_Claude/Pendulum_control/swing-up/SwingupController/SwingupController.ino:435>)이 실제로 가속도 입력을 반영하는 핵심 부분입니다.

```cpp
case CMD_SET_ACCELERATION:
  timeout_latched = false;
  accel_command_pps2 = action[0];
  break;
```

`CMD_SET_ACCELERATION`은 **명령 번호 1**입니다. 이 명령이 오면 `action[0]`을 `accel_command_pps2`에 저장합니다.

단위는 **pps²(스텝 펄스/초²)**입니다. 양수는 부호 있는 속도를 증가시키고, 음수는 감소시킵니다. 따라서 음수 입력을 계속 주면 양의 방향으로 움직이던 모터가 감속한 뒤 반대 방향으로 회전할 수 있습니다.

### 3. 가속도를 적분해 목표 속도 계산

[`integrate_acceleration()` (line 307)](<C:/Users/Sim/Dropbox/[Courses]/[고급딥러닝시스템과응용]/08_Claude/Pendulum_control/swing-up/SwingupController/SwingupController.ino:307>)에서 저장된 가속도로 속도를 갱신합니다.

```cpp
float a = accel_command_pps2;
// ... 가속도 제한 처리 ...

target_velocity_pps += a * dt;
apply_target_velocity(target_velocity_pps);
```

즉, 다음 식을 MCU에서 반복 계산합니다.

$$
v_{\text{새 값}} = v_{\text{기존 값}} + a\,\Delta t
$$

예를 들어 입력이 `1000 pps²`이고 경과 시간이 `0.01초`라면, 목표 속도는 `10 pps` 증가합니다. `apply_target_velocity()`는 계산된 속도에 따라 모터 방향과 스텝 펄스 주파수를 설정합니다.

여기서 알아둘 동작은 다음과 같습니다.

- 새 가속도 명령이 오기 전까지 **마지막으로 저장한 가속도**를 계속 사용합니다.
- **가속도 0은 정지 명령이 아닙니다.** 목표 속도를 유지합니다.
- 적용 가속도는 **±24000 pps²**, 목표 속도는 **±4000 pps**로 제한합니다.
- 정상 수신된 호스트 명령이 **150ms 넘게 없으면** 가속도와 목표 속도를 0으로 만들고 정지합니다.
- `loop()`에서 적분이 수신 처리보다 먼저 실행되므로, 새로 받은 가속도는 **다음 반복부터** 적분에 사용됩니다.
