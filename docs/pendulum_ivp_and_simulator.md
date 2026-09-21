# 동정 파라미터의 두 가지 표현, 초기값 문제, 그리고 제어 시뮬레이터

`pendulum_model_identification.md`에서 측정으로 얻은 파라미터를 가지고

1. 표준형 $(\omega_n,\ \zeta)$ 와 물리형 $(c,\ J,\ g)$ 로 **각각** 정리하고,
2. 두 표현에서 **초기값 문제**를 푸는 과정을 비교하며,
3. 그 모델로 제어기를 시험할 수 있는 **시뮬레이터**를 설명한다.

- 시뮬레이터: `pendulum_sim.py`
- 동정 결과 출처: `pendulum_model_id.py`, `docs/pendulum_model_identification.md`
- 시뮬레이터 상세 문서: `docs/pendulum_simulator.md` (API, 설계 절차, 검증, 설계 여유, 한계)
- 실시간 대화형 시뮬레이터: `analysis/realtime_sim.py` — 벽시계 시간으로 돌려 보며 제어기를 끄거나 외란을 준다

---

## 1. 같은 방정식의 두 표현

피벗이 고정된 진자의 선형화 방정식은 물리 파라미터로 이렇게 쓴다.

$$J\ddot{\theta} + c\,\dot{\theta} + mgl\,\theta = 0$$

- $J$ : 피벗 기준 관성모멘트 $[\mathrm{kg\,m^2}]$
- $c$ : 점성 마찰 계수 $[\mathrm{N\,m\,s/rad}]$
- $m$ : 질량, $l$ : 피벗에서 질량중심까지 거리, $g$ : 중력가속도

양변을 $J$로 나누면

$$\ddot{\theta} + \frac{c}{J}\dot{\theta} + \frac{mgl}{J}\,\theta = 0$$

이고, 이를 표준형

$$\ddot{\theta} + 2\zeta\omega_n\dot{\theta} + \omega_n^2\theta = 0$$

과 항별로 맞추면 **변환 관계**가 나온다.

$$\boxed{\ \omega_n = \sqrt{\frac{mgl}{J}}, \qquad
\zeta = \frac{c}{2J\omega_n} = \frac{c}{2\sqrt{mgl\,J}}\ }$$

역방향은

$$\frac{mgl}{J} = \omega_n^2, \qquad \frac{c}{J} = 2\zeta\omega_n$$

### 1.1 측정값 대입

동정 결과 $\omega_n = 7.508\ \mathrm{rad/s}$, $\sigma \equiv \zeta\omega_n = 0.0798\ \mathrm{s^{-1}}$ 이므로

| 양 | 기호 | 값 |
| --- | --- | --- |
| 고유 진동수 | $\omega_n$ | $7.5080\ \mathrm{rad/s}$ |
| 감쇠비 | $\zeta = \sigma/\omega_n$ | $0.010629$ |
| 감쇠 고유 진동수 | $\omega_d = \omega_n\sqrt{1-\zeta^2}$ | $7.5076\ \mathrm{rad/s}$ |
| 감쇠율 | $\sigma = \zeta\omega_n$ | $0.0798\ \mathrm{s^{-1}}$ |

$\zeta$가 $0.011$로 매우 작아 $\omega_d/\omega_n = 0.99994$이다. **감쇠가 진동수를 사실상 바꾸지 않는다.**

물리형으로 옮기면

$$\frac{mgl}{J} = \omega_n^2 = 56.370\ \mathrm{s^{-2}}, \qquad
\frac{c}{J} = 2\zeta\omega_n = 0.1596\ \mathrm{s^{-1}}$$

$$L_{\text{eff}} \equiv \frac{g}{\omega_n^2} = \frac{J}{ml} = 0.17397\ \mathrm{m}$$

### 1.2 중요한 제약: 자유 진동은 비율만 준다

물리형에는 미지수가 $c,\ J,\ m,\ l$ 네 개지만, **해는 두 조합에만 의존한다.**

$$\frac{mgl}{J} \quad\text{와}\quad \frac{c}{J}$$

즉 $J$를 두 배로 하고 $c$와 $ml$도 동시에 두 배로 하면 **완전히 같은 파형**이 나온다.
자유 진동 데이터만으로는 개별 값을 분리할 수 없다.

$J$를 따로 측정하면 나머지가 즉시 따라온다.

$$c = 2\zeta\omega_n J = 0.1596\,J, \qquad ml = \frac{J}{L_{\text{eff}}} = 5.748\,J$$

예를 들어 $J = 5\times10^{-4}\ \mathrm{kg\,m^2}$ 라면

$$c = 7.98\times10^{-5}\ \mathrm{N\,m\,s/rad}, \qquad
ml = 2.87\times10^{-3}\ \mathrm{kg\,m}, \qquad
mgl = 2.82\times10^{-2}\ \mathrm{N\,m}$$

`pendulum_sim.py`의 `PendulumParams.physical(J)`가 이 계산을 한다.

---

## 2. 초기값 문제

초기 조건은 두 경우 모두 $\theta(0) = \theta_0$, $\dot\theta(0) = v_0$ 로 둔다.

### 2.1 매달림 평형점 — 표준형으로 풀기

$$\ddot{\theta} + 2\zeta\omega_n\dot{\theta} + \omega_n^2\theta = 0$$

$\theta = e^{st}$를 대입하면 미분 연산이 $s$의 다항식이 된다(특성방정식).

$$s^2 + 2\zeta\omega_n s + \omega_n^2 = 0
\quad\Longrightarrow\quad
s = -\zeta\omega_n \pm \omega_n\sqrt{\zeta^2 - 1}$$

$\zeta < 1$ 이므로 근은 켤레복소수다.

$$s = -\sigma \pm j\omega_d, \qquad \sigma = \zeta\omega_n,\quad \omega_d = \omega_n\sqrt{1-\zeta^2}$$

일반해 $\theta = e^{-\sigma t}(A\cos\omega_d t + B\sin\omega_d t)$ 에 초기 조건을 넣는다.

- $\theta(0) = A = \theta_0$
- $\dot\theta(0) = -\sigma A + \omega_d B = v_0 \;\Rightarrow\; B = \dfrac{v_0 + \sigma\theta_0}{\omega_d}$

$$\boxed{\ \theta(t) = e^{-\sigma t}\left[\theta_0\cos\omega_d t
+ \frac{v_0 + \sigma\theta_0}{\omega_d}\sin\omega_d t\right]}$$

진폭-위상 형태로도 쓸 수 있다.

$$\theta(t) = A_0\,e^{-\sigma t}\cos(\omega_d t - \phi), \qquad
A_0 = \sqrt{\theta_0^2 + \left(\frac{v_0+\sigma\theta_0}{\omega_d}\right)^2}, \quad
\tan\phi = \frac{v_0+\sigma\theta_0}{\omega_d\,\theta_0}$$

라플라스 변환으로 풀어도 같다. 변환하면

$$\Theta(s) = \frac{(s + 2\zeta\omega_n)\,\theta_0 + v_0}{s^2 + 2\zeta\omega_n s + \omega_n^2}$$

이고, 분모를 $(s+\sigma)^2 + \omega_d^2$ 로 완전제곱하면 위 시간 영역 해가 그대로 나온다.

**측정값 예시** ($\theta_0 = 20°$, $v_0 = 0$):

$$\theta(t) = 20°\,e^{-0.0798\,t}\left[\cos(7.5076\,t) + 0.01063\sin(7.5076\,t)\right]$$

$\sin$ 항의 계수 $\sigma/\omega_d = 0.0106$ 는 무시할 만하다. 감쇠가 약할 때
$\theta(t) \approx \theta_0 e^{-\sigma t}\cos\omega_d t$ 로 충분하다.

### 2.2 매달림 평형점 — 물리형으로 풀기

같은 문제를 물리 파라미터로 그대로 풀면 특성방정식이

$$J s^2 + c\,s + mgl = 0
\quad\Longrightarrow\quad
s = \frac{-c \pm \sqrt{c^2 - 4Jmgl}}{2J}$$

이고, 판별식이 음수인 부족감쇠 조건 $c^2 < 4Jmgl$ 에서

$$s = -\frac{c}{2J} \pm j\underbrace{\sqrt{\frac{mgl}{J} - \frac{c^2}{4J^2}}}_{\omega_d}$$

$$\boxed{\ \theta(t) = e^{-\frac{c}{2J}t}\left[\theta_0\cos\omega_d t
+ \frac{v_0 + \frac{c}{2J}\theta_0}{\omega_d}\sin\omega_d t\right]}$$

**두 해는 같은 식이다.** $\sigma \leftrightarrow c/2J$, $\omega_n^2 \leftrightarrow mgl/J$ 로
치환하면 정확히 겹친다. 부족감쇠 조건도 마찬가지로

$$c^2 < 4Jmgl \quad\Longleftrightarrow\quad \zeta < 1$$

차이는 **무엇을 아는가**에 있다.

| | 표준형 | 물리형 |
| --- | --- | --- |
| 파라미터 수 | 2개 $(\omega_n, \zeta)$ | 4개 $(c, J, m l, g)$ |
| 측정에서 직접 나오는가 | 예 — 주기와 포락선에서 | 아니오 — 비율만 |
| 설계에 바로 쓰이는가 | 예 — 극점이 곧 $-\sigma \pm j\omega_d$ | 아니오 |
| 물리적 해석 | 간접적 | 직접적 (재설계에 유용) |

실험으로 동정할 때는 표준형이, 진자를 물리적으로 바꿔가며 설계할 때는 물리형이 편하다.

### 2.3 직립 평형점 — 부호 하나가 바꾸는 것

$\theta$를 **직립**에서 재면 중력이 복원력이 아니라 **발산력**이 된다. 선형화에서
$\sin\theta$ 항의 부호가 뒤집히므로

$$J\ddot{\theta} + c\,\dot{\theta} - mgl\,\theta = 0
\qquad\Longleftrightarrow\qquad
\ddot{\theta} + 2\zeta\omega_n\dot{\theta} - \omega_n^2\theta = 0$$

특성방정식은

$$s^2 + 2\sigma s - \omega_n^2 = 0
\quad\Longrightarrow\quad
s = -\sigma \pm \sqrt{\sigma^2 + \omega_n^2}$$

판별식이 **항상 양수**이므로 근이 언제나 **실수**이고, 하나는 **양수**다.

$$s_1 = +7.4286\ \mathrm{s^{-1}}, \qquad s_2 = -7.5882\ \mathrm{s^{-1}}$$

해는 진동이 아니라 두 지수의 합이다.

$$\boxed{\ \theta(t) = C_1 e^{s_1 t} + C_2 e^{s_2 t}, \qquad
C_1 = \frac{v_0 - s_2\theta_0}{s_1 - s_2}, \quad C_2 = \theta_0 - C_1\ }$$

$\theta_0 = 5°$, $v_0 = 0$ 이면 $C_1 = 2.527°$, $C_2 = 2.473°$ 이므로

$$\theta(t) = 2.527°\,e^{7.4286\,t} + 2.473°\,e^{-7.5882\,t}$$

1초 뒤 $\theta \approx 4.3\times10^{3}\,\text{도}$. 즉 **진자는 이미 여러 바퀴 넘어가 있다.**

여기서 제어의 의미가 분명해진다. 해가 발산하지 않으려면 $C_1 = 0$, 즉

$$v_0 = s_2\,\theta_0 = -7.5882\,\theta_0$$

이어야 한다. $\theta_0 = 5°$ 라면 정확히 $-37.94\ \mathrm{deg/s}$ 로 직립 쪽을 향해 밀어야 한다.
이 조건을 만족하는 초기 조건의 집합은 상태 평면에서 **직선 하나**(안정 다양체)이고,
그 밖의 모든 초기 조건은 발산한다. 측도가 0인 집합이므로 **실제로는 절대 일어나지 않는다.**
되먹임이 하는 일은 이 직선을 평면 전체로 넓히는 것이다.

발산의 시간 척도는

$$\tau = \frac{1}{s_1} = 135\ \mathrm{ms}, \qquad
t_{2\times} = \frac{\ln 2}{s_1} = 93\ \mathrm{ms}$$

---

## 3. 제어 시뮬레이터

### 3.1 플랜트 모델

피벗이 가속도 $a\ [\mathrm{m/s^2}]$ 로 움직이면 관성력이 추가 토크를 만든다.

$$J\ddot{\theta} = mgl\sin\theta - m l\,a\cos\theta - c\,\dot{\theta} - \tau_{\text{Coulomb}}$$

$J$로 나누고 동정된 조합으로 바꾸면 시뮬레이터가 적분하는 식이 된다.

$$\ddot{\theta} = \omega_n^2\sin\theta - 2\zeta\omega_n\dot{\theta}
- \frac{a}{L_{\text{eff}}}\cos\theta - F\,\mathrm{sgn}(\dot{\theta})$$

여기서 눈여겨볼 점은 **입력 계수가 새 파라미터가 아니라는 것**이다.

$$b = -\frac{ml}{J} = -\frac{1}{L_{\text{eff}}} = -5.748\ \frac{\mathrm{rad/s^2}}{\mathrm{m/s^2}}$$

즉 자유 진동에서 얻은 $L_{\text{eff}}$ 가 입력 이득까지 결정한다. 추가 실험이 필요 없다.

**다만 팔 반지름 $r$ 은 동정되지 않았다.** 로터 각가속도를 피벗 가속도로 바꾸는
$a = r\,\ddot{\alpha}$ 의 $r$ 은 직접 재야 한다. 입력에 곱해지는 단순 배율이므로
`PendulumParams.arm_radius_m` 로 노출해 두었다.

각도 규약은 **직립이 $\theta = 0$**, 매달림이 $\theta = \pi$ 다. 펌웨어는 180°가 직립이므로

$$\theta_{\text{rad}} = \frac{\pi}{180}\left(\phi_{\text{firmware}} - 180\right)$$

### 3.2 무엇을 모사하는가

실제로 제어기의 성패를 가르는 요소들을 포함한다.

| 항목 | 설정 | 기본값 |
| --- | --- | --- |
| 제어 주기 | `control_hz` | 100 Hz |
| 엔코더 양자화 | `encoder_deg` | 0.3° (1200 CPR) |
| 구동기 포화 | `accel_max` | 제한 없음 |
| 로터 가동 한계 | `rotor_limit_deg` | 제한 없음 |
| 계산 지연 | `delay_steps` | 0 |
| 건마찰 | `coulomb_dps` | 3.149 deg/s |

적분은 반음함수 오일러(semi-implicit Euler)로, 진동계에서 에너지 드리프트가 작다.

### 3.3 사용법

```python
import pendulum_sim as ps

p = ps.PendulumParams()          # 동정된 기본값
print(p.summary())
p.arm_radius_m = 0.085           # 실제 치수로 교체할 것

# 2상태 설계: 진자만 세운다
k1, k2 = ps.place_poles(p, wn_des=12.0, zeta_des=0.8)

# 4상태 설계: 진자를 세우면서 로터도 제자리에 둔다
K = ps.place_poles4(p, [-10+10j, -10-10j, -2+1j, -2-1j])

cfg = ps.SimConfig(duration=15.0, control_hz=100.0, encoder_deg=0.3,
                   accel_max=6.0, rotor_limit_deg=90.0, theta0_deg=5.0)
log = ps.simulate(p, ps.StateFeedback(*K, u_max=6.0), cfg)
print(log["settled"], log["theta_deg"][-1], log["alpha_deg"][-1])
```

자유 진동 재현은 `ps.free_swing(p, theta0_deg=20.0)`, 해석해는
`ps.ivp_hanging(t, θ0, v0, p)` 와 `ps.ivp_upright(t, θ0, v0, p)` 로 얻는다.

### 3.4 검증

시뮬레이터는 다음을 통과한다.

| 검사 | 결과 |
| --- | --- |
| 선형 플랜트 적분 vs 직립 해석해 | 상대 오차 $2.0\times10^{-4}$ |
| 매달림 해석해 vs 직접 적분 | 최대 오차 $7.5\times10^{-4}$ 도 (20° 진폭) |
| 비선형 자유 진동의 주기 | $0.8384$ s vs 동정 $0.8369$ s ($+0.19\%$) |
| 극점 배치 정확도 | 지정한 $\omega_n,\ \zeta$ 와 $10^{-6}$ 이내 일치 |
| PID($k_p=-k_1$, $k_d=-k_2$) vs 상태 피드백 | 동일 거동 확인 |

---

## 4. 시뮬레이터가 알려준 것

### 4.1 각도만 되먹이면 로터가 흘러간다

$\theta$ 와 $\dot\theta$ 만 쓰는 2상태 설계는 진자를 잘 세우지만, 로터의 위치에 대해서는
아무 말도 하지 않는다. 15초 시뮬레이션 결과:

| 설계 | 최종 $\vert\theta\vert$ | 로터 최대 변위 |
| --- | --- | --- |
| 2상태 (로터 되먹임 없음) | 0.150° | **2200°** (6바퀴) |
| 4상태 $[-8\pm8j,\ -3\pm j]$ | 1.085° | 28.3° |
| 4상태 $[-10\pm10j,\ -2\pm j]$ | 0.833° | **26.2°** |
| 4상태 $[-12\pm6j,\ -1.5\pm j]$ | 0.802° | 38.0° |

2상태 설계는 펌웨어의 ±90° 한계에 즉시 걸린다. 4상태 설계는 ±38° 안에 머문다.

> $k_3,\ k_4$ 를 2상태 게인 위에 **손으로 얹으면 발산한다.** 실제로 시도해 보면
> $k_3 = 0.05,\ k_4 = 0.10$ 에서 로터가 29705° 까지 돌아간다. 로터를 한쪽으로 옮기려면
> 먼저 진자를 **반대쪽으로** 기울여야 하기 때문이며, 이 비최소위상 성질 때문에
> 4차 시스템으로 한꺼번에 설계해야 한다. `place_poles4()` 가 그 역할을 한다.

### 4.2 설계 여유

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
| 초기 기울기 15° | 0.812° | **110.7°** | 예(로터 한계 초과) |

> **주의 — 위 표는 제어기에 *참* 속도를 준 결과다.** `simulate()` 는 $\dot\theta$ 를
> 플랜트에서 그대로 넘기지만, 실제 하드웨어에는 그런 신호가 없고 0.3° 엔코더를 차분해야
> 한다. 차분 속도로 다시 돌리면 **100 Hz 는 그대로 유지되지만 50 Hz + 20 ms 필터는
> 실패한다(최종 6.394°)**. 즉 여기의 "50 Hz 예" 는 낙관적인 값이고, 100 Hz 권고는 여유가
> 아니라 필요조건에 가깝다. → `docs/pendulum_simulator.md` 6.1절

읽어낼 것:

- **샘플링 하한은 20 Hz와 50 Hz 사이**다. 50 Hz도 동작하지만 여유가 크지 않으니
  100 Hz를 권한다.
- **지연이 가장 치명적이다.** 30 ms면 실패한다. 500000 baud에서 왕복 통신은 1 ms 수준이라
  괜찮지만, 호스트 쪽 지터가 커지면 위험하다.
- 엔코더 분해능과 구동기 세기는 상대적으로 여유가 있다.
- **초기 기울기 15°에서는 로터가 110°까지 나간다.** ±90° 한계에서는 복구 가능한 각도가
  약 10° 이내라는 뜻이다. 그보다 크게 기울면 스윙업이 필요하다.

---

## 5. 한계

- 플랜트는 **진자만** 모델링한다. 로터는 $\ddot{\alpha} = u/r$ 로 운동학적으로만 적분되며,
  스테퍼의 가속 한계·탈조·L6474의 속도 하한은 들어 있지 않다.
- 회전형(Furuta) 특유의 원심력·코리올리 항을 생략했다. 소각도 밸런싱에서는 작지만,
  큰 스윙업 궤적에는 부족하다.
- $r$ 이 미측정이므로 $k_3,\ k_4$ 의 절대 크기는 $r$ 에 따라 달라진다. $k_1,\ k_2$ 는
  $r$ 과 무관하다.
- 마찰 비대칭(17.4%)은 모사하지 않는다. `pendulum_model_identification.md` 5절 참고.
