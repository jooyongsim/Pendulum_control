# 제어 시뮬레이터 `pendulum_sim.py`

동정된 진자 모델 위에서 제어기를 시험하는 시뮬레이터의 레퍼런스.

`docs/pendulum_ivp_and_simulator.md` 와 `docs/pendulum_ivp_and_simulator_uniform_rod.md`
가 모델의 **유도**를 다룬다면, 이 문서는 시뮬레이터의 **인터페이스와 한계**를 다룬다.

- 코드: [`../pendulum_sim.py`](../pendulum_sim.py)
- 플랜트 파라미터 출처: [`../pendulum_model_id.py`](../pendulum_model_id.py),
  [`pendulum_model_identification.md`](pendulum_model_identification.md)

---

## 1. 무엇을 적분하는가

피벗이 가속도 $a$ 로 움직일 때의 진자 방정식을 그대로 적분한다.

$$\ddot{\theta} = \omega_n^{2}\sin\theta - 2\zeta\omega_n\dot{\theta}
- \frac{a}{L_{\text{eff}}}\cos\theta - F\,\mathrm{sgn}(\dot{\theta})$$

**각도 규약**: $\theta$ 는 **직립**에서 잰다. $\theta = 0$ 이 직립, $\theta = \pi$ 가 매달림이다.
펌웨어는 180°를 직립으로 보고하므로

$$\theta_{\text{rad}} = \frac{\pi}{180}\left(\phi_{\text{firmware}} - 180\right)$$

**입력 이득이 새 파라미터가 아니다.** 자유 진동에서 얻은 $L_{\text{eff}}$ 가 그대로 결정한다.

$$b = -\frac{m\,l_c}{J} = -\frac{1}{L_{\text{eff}}} = -5.748\ \frac{\mathrm{rad/s^{2}}}{\mathrm{m/s^{2}}}$$

이 값은 질량 분포 가정과 무관하다. 점질량으로 읽든 균일 봉으로 읽든 $J/(m l_c)$ 는
같은 $L_{\text{eff}}$ 이므로, **균일 봉 판을 위해 시뮬레이터를 고칠 필요가 없다.**

미동정 파라미터는 **팔 반지름 $r$ 하나뿐**이다. 로터 각가속도를 피벗 가속도로 바꾸는
$a = r\ddot\alpha$ 의 $r$ 은 직접 재서 `arm_radius_m` 에 넣어야 한다.

적분기는 **반음함수 오일러**(속도를 먼저 갱신하고 그 속도로 위치를 갱신)로, 진동계에서
에너지 드리프트가 작다. 기본 스텝은 $10^{-4}$ s.

---

## 2. API

### 2.1 `PendulumParams`

동정된 플랜트. 기본값이 곧 측정 결과다.

| 필드 | 기본값 | 의미 |
| --- | --- | --- |
| `wn` | 7.508 | 고유 진동수 [rad/s] |
| `zeta` | 0.010629 | 감쇠비 [무차원] |
| `coulomb_dps` | 3.149 | 건마찰에 의한 진폭 손실률 [deg/s] |
| `arm_radius_m` | 0.085 | **미동정** — 실제 치수로 교체할 것 [m] |

파생 속성 (읽기 전용):

| 속성 | 식 | 값 |
| --- | --- | --- |
| `sigma` | $\zeta\omega_n$ | 0.0798 1/s |
| `wd` | $\omega_n\sqrt{1-\zeta^2}$ | 7.5076 rad/s |
| `L_eff` | $g/\omega_n^2$ | 0.17397 m |
| `b` | $-1/L_{\text{eff}}$ | −5.748 |
| `coulomb_accel` | $c_A\,\omega_n^2 T/4$ | 건마찰 각가속도 [rad/s²] |

```python
p.physical(J)     # 측정한 J 로부터 c, m*l, m g l 을 해석
p.summary()       # 한 화면 요약 문자열
```

### 2.2 제어기

제어기는 매 제어 주기마다 `State` 를 받아 **피벗 가속도 명령 [m/s²]** 을 돌려준다.

```python
@dataclass
class State:
    theta: float      # rad, 직립 기준 (엔코더처럼 양자화되어 전달)
    omega: float      # rad/s
    alpha: float      # rad, 로터 각도
    alpha_dot: float  # rad/s
```

| 클래스 | 제어 법칙 |
| --- | --- |
| `Controller` | 베이스. 항상 0 — 자유 진동용 |
| `PID(kp, ki, kd, setpoint, u_max, tau_d)` | 미분 1차 필터 + 조건부 적분(anti-windup) |
| `StateFeedback(k1, k2, k3=0, k4=0, u_max)` | $u = -(k_1\theta + k_2\dot\theta + k_3\alpha + k_4\dot\alpha)$ |

`PID` 와 `StateFeedback` 는 등가다. $k_p = -k_1$, $k_d = -k_2$ 로 두면 같은 거동을 보인다
(검증 항목 7).

### 2.3 극점 배치

```python
k1, k2 = ps.place_poles(p, wn_des=12.0, zeta_des=0.8)          # 2상태
K      = ps.place_poles4(p, [-10+10j, -10-10j, -2+1j, -2-1j])  # 4상태
A, B   = ps.state_space(p)                                      # x = [th, th', al, al']
```

**4상태를 써야 하는 이유는 3절에 있다.** 2상태 게인 위에 $k_3, k_4$ 를 손으로 얹으면
발산한다.

### 2.4 `SimConfig`

하드웨어에서 성패를 가르는 요소들을 모사한다.

| 필드 | 기본값 | 의미 |
| --- | --- | --- |
| `duration` | 5.0 | 시뮬레이션 길이 [s] |
| `dt_plant` | 1e-4 | 적분 스텝 [s] |
| `control_hz` | 100.0 | 이산 제어 주기 [Hz] |
| `encoder_deg` | 0.3 | 각도 양자화 (1200 CPR). 0 이면 양자화 없음 |
| `accel_max` | ∞ | 피벗 가속도 포화 [m/s²] |
| `rotor_limit_deg` | ∞ | 로터 가동 한계. 한계에서 바깥쪽 명령은 0 이 된다 |
| `delay_steps` | 0 | 제어 출력 지연 [제어 스텝 수] |
| `linear` | False | True 면 직립 선형화 플랜트를 적분 (해석해 검증용) |
| `theta0_deg` | 5.0 | 초기 각도 [deg] |
| `omega0_dps` | 0.0 | 초기 각속도 [deg/s] |

### 2.5 `simulate()` 가 돌려주는 것

```python
log = ps.simulate(p, controller, cfg)
```

| 키 | 내용 |
| --- | --- |
| `t` | 시각 [s] |
| `theta`, `omega` | 플랜트 상태 [rad, rad/s] |
| `theta_deg` | 각도 [deg] |
| `theta_meas` | 제어기가 실제로 본 양자화 각도 [rad] |
| `u` | 적용된 명령 [m/s²] (포화·로터 한계 반영 후) |
| `alpha`, `alpha_deg` | 로터 각도 |
| `saturated` | 해당 스텝에서 명령이 포화되었는지 |
| `settled` | 마지막 0.5 s 동안 $\vert\theta\vert < 2°$ 이면 True |

### 2.6 보조 함수

```python
ps.free_swing(p, theta0_deg=20.0, duration=30.0)   # 제어 없이 매달림에서 놓기
ps.ivp_hanging(t, theta0, omega0, p)               # 매달림 해석해
ps.ivp_upright(t, theta0, omega0, p)               # 직립 해석해
```

---

## 3. 설계 워크플로

```python
import pendulum_sim as ps

p = ps.PendulumParams()
p.arm_radius_m = 0.085                       # 실측값으로 교체

K = ps.place_poles4(p, [-10+10j, -10-10j, -2+1j, -2-1j])
cfg = ps.SimConfig(duration=15.0, control_hz=100.0, encoder_deg=0.3,
                   accel_max=6.0, rotor_limit_deg=90.0, theta0_deg=5.0)
log = ps.simulate(p, ps.StateFeedback(*K, u_max=6.0), cfg)
print(log["settled"], log["theta_deg"][-1], log["alpha_deg"][-1])
```

### 각도만 되먹이면 로터가 흘러간다

$\theta$ 와 $\dot\theta$ 만 쓰는 2상태 설계는 진자를 잘 세우지만 로터 위치에 대해서는
아무 말도 하지 않는다. 15초 시뮬레이션:

| 설계 | 최종 $\vert\theta\vert$ | 로터 최대 변위 |
| --- | --- | --- |
| 2상태 (로터 되먹임 없음) | 0.150° | **2200°** (6바퀴) |
| 4상태 $[-8\pm8j,\ -3\pm j]$ | 1.085° | 28.3° |
| 4상태 $[-10\pm10j,\ -2\pm j]$ | 0.833° | **26.2°** |

2상태 설계는 펌웨어의 ±90° 한계에 즉시 걸린다.

> $k_3, k_4$ 를 2상태 게인 위에 **손으로 얹으면 발산한다.** $k_3=0.05,\ k_4=0.10$ 에서
> 로터가 29705° 까지 돌아간다. 로터를 한쪽으로 옮기려면 진자를 **반대쪽으로** 먼저
> 기울여야 하는 비최소위상 성질 때문이며, 그래서 4차로 한꺼번에 설계해야 한다.

---

## 4. 검증

| 검사 | 결과 |
| --- | --- |
| 선형 플랜트 적분 vs 직립 해석해 | 상대 오차 $2.0\times10^{-4}$ |
| 매달림 해석해 vs 직접 적분 | 최대 오차 $7.5\times10^{-4}$ 도 (20° 진폭) |
| 비선형 자유 진동의 주기 | 0.8384 s vs 동정 0.8369 s ($+0.19\%$) |
| 극점 배치 정확도 | 지정한 $\omega_n,\ \zeta$ 와 $10^{-6}$ 이내 |
| PID($k_p=-k_1$, $k_d=-k_2$) vs 상태 피드백 | 동일 거동 |
| 이동 중 E-stop | 정리 코드가 아니라 루프에서 즉시 정지 명령 |

---

## 5. 설계 여유

4상태 설계 $[-10\pm10j,\ -2\pm j]$, 로터 한계 ±90°, 초기 기울기 5° 기준:

| 조건 | 최종 $\vert\theta\vert$ | 로터 최대 | 성공 |
| --- | --- | --- | --- |
| 기준 (100 Hz) | 0.833° | 26.2° | 예 |
| 50 Hz 루프 | 0.849° | 23.4° | 예 |
| **20 Hz 루프** | 194° | 17216° | **아니오** |
| 엔코더 1.0° (거침) | 0.542° | 26.6° | 예 |
| 구동기 ±2 m/s² (약함) | 0.837° | 39.6° | 예 |
| 지연 1스텝 (10 ms) | 0.852° | 24.4° | 예 |
| **지연 3스텝 (30 ms)** | 3.85° | 33.8° | **아니오** |
| 초기 기울기 15° | 0.812° | **110.7°** | 예 (로터 한계 초과) |

---

## 6. 한계 — 특히 속도 신호

### 6.1 제어기가 **참 속도**를 받는다

`simulate()` 는 각도를 엔코더처럼 양자화해서 넘기지만, `State.omega` 와
`State.alpha_dot` 은 **플랜트의 참값을 그대로** 넘긴다. 실제 하드웨어에는 그런 신호가
없다. 0.3° 엔코더를 100 Hz 로 차분하면

$$\frac{0.3°}{0.01\ \mathrm{s}} = 30\ \mathrm{deg/s} = 0.524\ \mathrm{rad/s}$$

의 양자화 잡음이 생기고, $k_2 = -6.93$ 을 곱하면 **3.63 m/s² 의 명령 잡음**이다
(포화 한계가 6 m/s²).

측정값을 차분해서 쓰도록 바꿔 다시 돌려 보면:

| 제어기 / 조건 | 최종 $\vert\theta\vert$ | 성공 |
| --- | --- | --- |
| 참 속도 (5절 표가 쓴 것) | 0.833° | 예 |
| 차분 속도, 필터 없음 | 0.419° | 예 |
| 차분 속도, $\tau_d = 20$ ms | 0.850° | 예 |
| 차분 속도, 50 Hz, 필터 없음 | 0.691° | 예 |
| **차분 속도, 50 Hz, $\tau_d = 20$ ms** | **6.394°** | **아니오** |

**100 Hz 에서는 5절의 결론이 그대로 유지된다.** 그러나 **50 Hz 는 현실적인 속도 추정을
넣으면 실패한다** — 5절 표의 "50 Hz 예" 는 참 속도를 준 결과다. 100 Hz 권고는 여유가
아니라 필요조건에 가깝다.

> 재현: `analysis/` 에는 아직 이 실험이 없다. 필요하면
> `StateFeedback` 대신 각도만 받아 차분하는 제어기를 만들어 같은 `SimConfig` 로 돌리면 된다.

### 6.2 그 밖의 한계

- **진자만 모델링한다.** 로터는 $\ddot\alpha = u/r$ 로 운동학적으로만 적분되며, 스테퍼의
  가속 한계·탈조·L6474 의 속도 하한(`MOTOR_MIN_SPEED_PPS`)은 들어 있지 않다.
- **회전형 특유의 원심력·코리올리 항을 생략했다.** 소각도 밸런싱에서는 작지만 큰 스윙업
  궤적에는 부족하다.
- **$r$ 이 미측정이므로** $k_3, k_4$ 의 절대 크기는 $r$ 에 따라 달라진다. $k_1, k_2$ 는
  $r$ 과 무관하다.
- **마찰 비대칭을 모사하지 않는다.** 실측 진동은 양/음 피크가 17.4 % 비대칭이다
  ([pendulum_model_identification.md](pendulum_model_identification.md) 5절).
- **`settled` 판정은 느슨하다.** 마지막 0.5 s 동안 $\vert\theta\vert < 2°$ 이면 참이므로,
  한계 주기 진동(limit cycle)을 성공으로 볼 수 있다. 정밀한 판정이 필요하면 `theta_deg`
  를 직접 보는 편이 낫다.
