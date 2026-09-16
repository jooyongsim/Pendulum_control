# Swing-up (STEVAL-EDUKIT01, 가속도 제어)

매달린 진자를 팔로 흔들어 올려 받아서 세우고, 넘어지면 다시 흔들어 올립니다.
같은 키트(NUCLEO-F401RE + X-NUCLEO-IHM01A1, 1200 CPR 엔코더) 기준입니다.

```
swing-up/
├── SwingupController/        스윙업용 펌웨어 (원래 AccelerationController는 그대로 둠)
├── swingup_control.py        스윙업 + 받기 + 균형 + 다시 흔들기
├── balance_control.py        흔들림 잡기 / 직립 균형 (swingup_control이 불러 씀)
└── accel_openloop_test.py    장비 점검: 모터 방향, 스텝 놓침, 입력 계수 b
```

모든 명령은 **저장소 최상위 폴더**(`Pendulum_control/`)에서 실행합니다. 결과 CSV는 `data/`에 저장됩니다.

## 필요한 것

- Python 3.9 이상, `numpy`, `pyserial` (`pip install -r requirements.txt`)
- Arduino IDE + STM32 코어 (확인: 2.10.1), 보드 `Nucleo-64` / `Nucleo F401RE`
- 라이브러리: **STM32duino X-NUCLEO-IHM01A1 1.0.1**, RotaryEncoder, ArduinoJson (확인: 7.4.2)
  - 펌웨어가 드라이버 라이브러리 내부(`device_prm` 등)를 직접 쓰므로 **IHM01A1 버전이 다르면 컴파일되지 않을 수 있습니다.**

## 장비별 첫 실행 (대당 약 5분)

1. **펌웨어 업로드**: `swing-up/SwingupController/SwingupController.ino`를 IDE에서 열고 **→ 업로드**.
   - 진자를 **매단 채 멈춘 상태**에서 올리세요. 보드가 켜질 때 진자 위치가 엔코더 0°가 됩니다.
   - 벌레 모양(Start Debugging) 버튼은 누르지 마세요. 디버거가 보드를 멈춥니다.
   - 업로드 후 **Serial Monitor를 닫으세요.**
2. **케이블 확인**: 모터 전원을 끄고 팔을 손으로 돌려, 케이블이 당기지 않는 범위를 확인합니다.
   140°가 안 되면 스윙업에 `--arm-limit`을 그보다 작게 주세요.
3. **개루프 테스트** (진자 매단 상태, 팔이 약 14° 움직였다 돌아옴)
   ```bash
   python3 swing-up/accel_openloop_test.py
   ```
   - `SAME as the Lec01 ...` → 부호 정상. **`OPPOSITE`면 스윙업을 돌리지 마세요** (모터 배선 반대).
   - `final` 팔 각도 1° 이내 → 스텝 놓침 없음
   - `r from fit` 110~130 mm 근처 → 입력 계수 정상
4. **흔들림 잡기** (진자 매단 상태): 3초 뒤 흔들림이 1° 안으로 줄면 정상
   ```bash
   python3 swing-up/balance_control.py --mode damp --excite
   ```
5. **스윙업**
   ```bash
   python3 swing-up/swingup_control.py --duration 40
   ```

## 실행

### 스윙업 (매달린 상태에서 시작)
```bash
python3 swing-up/swingup_control.py --duration 40
```
드라이버 점검 → 팔을 0°로 되돌림 → 진자가 멈추길 기다림 → 진자를 살짝 흔들어 진짜 수직 측정 →
**Enter** → 흔들기 → 받기 → 균형 → (넘어지면) 다시 흔들기. 끝나면 모터가 멈추고 진자가 쓰러집니다.

| 옵션 | 기본값 | 뜻 |
|---|---|---|
| `--duration` | 25 | 실행 시간 [s] |
| `--arm-limit` | 140 | 팔이 이 각도를 넘으면 정지 [deg] — 케이블 확인 후 조정 |
| `--catch` / `--catch-w` / `--catch-arm` | 12 / 4 / 60 | 받는 조건: 기울기 [deg], 각속도 [rad/s], 팔 위치 [deg] |
| `--pump` / `--kpa` / `--kda` | 30 / 3 / 5 | 흔들기 세기, 팔 위치·속도를 가운데로 잡는 세기 |
| `--b` | 0.730 | 입력 계수 3r/2l (개루프 테스트 값으로 바꿀 수 있음) |
| `--no-calibrate --upright-offset X` | | 시작 전 수직 측정을 건너뛰고 X [deg] 사용 |
| `--port /dev/ttyACM1` | 자동 | 여러 대를 한 컴퓨터에 꽂았을 때 필수 |
| `--yes` | | Enter 확인 생략 |

### 직립 균형 (손으로 세워서 시작)
```bash
python3 swing-up/balance_control.py --mode balance --poles="-6+6j,-6-6j,-1.5+1j,-1.5-1j" --duration 30 --arm-limit 80
```
`--poles=` 뒤에는 띄어쓰기 없이 `=`를 붙입니다 (값이 `-`로 시작하기 때문). ±3° 안에 0.2초 두면 `ENGAGED`가 뜨고, 그때 손을 뗍니다.

## 어떻게 동작하나

- **흔들기**: 에너지 `E = ½θ̇² + (3g/2l)(cosθ − 1)` (직립 정지 = 0, 매달림 ≈ −112).
  `dE/dt = b·θ̇·cosθ·φ̈`이므로 `sign(θ̇ cosθ)` 방향으로 팔을 가속해 에너지를 쌓고,
  팔 속도·위치를 가운데로 잡는 항으로 팔이 한쪽으로 흘러가지 않게 합니다. 출발 밀기는 +/− 대칭입니다.
- **받기**: |θ| < 12°, |θ̇| < 4 rad/s, |팔| < 60°일 때 4상태 극점 배치(−6±6j, −1.5±1j)로 전환.
  팔 기준은 받는 순간의 위치에서 시작해 3초 시상수로 0°에 돌아갑니다. 30° 넘게 기울면 다시 흔들기.
- **직립 기준**: 보드 리셋 때마다 엔코더 0°가 정지 마찰 범위(±2°) 안에서 바뀝니다.
  기준이 1.6°만 틀려도 팔이 약 90° 밀려나므로, 시작 전에 자유 흔들림의 중심을 재고
  균형 중에는 팔이 기준에서 벗어나는 방향으로 기준각을 천천히(6 s) 보정합니다.
- **모델** (Lec01 전체 비선형 식): `θ̈ = (3g/2l)sinθ + (3r/2l)cosθ·φ̈ + sinθcosθ·φ̇² − 감쇠`.
  우리 장비 식별값: ω_n = 7.499 rad/s, ζ = 0.0117, 쿨롱 2.7°/s, b = 0.730 (r ≈ 127 mm).

## SwingupController 펌웨어가 원래와 다른 점

| 문제 | 수정 |
|---|---|
| 명령이 약 40% 응답 없음 (SPI 읽기가 인터럽트를 막아 시리얼 바이트 손실) | 모터가 도는 동안 SPI 안 씀. 팔 위치는 스텝 카운터, 멈췄을 때 드라이버 ABS_POS로 재정렬 |
| 출발·방향 전환마다 약 100 ms 30 pps에 멈춤 (라이브러리 상태 전환이 스텝 3번 대기) | 스텝 클럭 주파수를 직접 설정 (`L6474Direct`) |
| 드라이버 잠김 원인이 안 보임 | `observation[5]` 상위 비트: 0x20000 = 드라이버 경보(과전류·과열·저전압) |
| 한계값 | 가속 24000 pps², 속도 4000 pps, 팔 잠금 150° (스크립트는 자체 팔 한계를 따로 둠) |
| 진단 | 명령 6: 정지 중 드라이버 ABS_POS 읽기 (`observation[4]`) |

## 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| `L6474 driver not healthy (0xFFFF)` | 모터 전원 없음. 어댑터 확인 후 다시 꽂고, 진자를 매단 채 보드 RESET |
| 첫 실행에 저전압 표시 | 전원을 막 넣어서 남은 표시. 스크립트가 두 번 읽어 지웁니다 |
| `Several ST-Link ports found` | 여러 대 연결됨 → `--port` 지정 |
| 포트가 안 생김 (`dmesg`에 `error -71`) | USB 구멍/케이블 불량. 다른 구멍에 꽂기 |
| `arm not following` | 드라이버·모터 전원 확인 |
| 받자마자 팔 한계 / 드라이버 잠김 | `--catch`를 좁히거나, 팔이 한쪽으로 치우친 채 시작하지 않았는지 확인 |
| `pendulum reads ... from hanging` | 진자를 매단 채 멈추고 보드 RESET (엔코더 0° 재설정) |

세게 흔들수록 모터 전류가 커집니다. 어댑터가 약하면 최대 가속에서 전원이 끊길 수 있습니다.
