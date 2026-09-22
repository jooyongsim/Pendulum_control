# 회전형 진자 플랜트 — `deriv()` 한 줄 읽기

[`rotary_model.py`](rotary_model.py) 106–115 행, `Plant.deriv()` 안의 이 식이
시뮬레이터가 적분하는 물리의 전부다.

```python
dw = (A_G * math.sin(th) + self.b * math.cos(th) * acc
      + math.sin(th) * math.cos(th) * phid * phid - C_D * w)
if self.coulomb:
    dw -= F_C * math.tanh(w / 0.01)
```

> 이 문서의 수식은 `\text{}` 안에 한글을 넣지 않는다. KaTeX 는 라틴·그리스 글꼴만
> 갖고 있어서, 한글이 수식 안에 들어가면 뷰어의 대체 글꼴에 따라 깨져 보인다.
> 3계 미분처럼 빌드마다 지원이 갈리는 명령도 쓰지 않았다.

---

## 0. 어디에 있는 코드인가

[`rotary_balance_realtime.py`](rotary_balance_realtime.py) 에는 플랜트가 없다. 그 파일은
실행과 화면을 담당하고, 42 행에서 모델을 가져다 쓴다.

```python
plant = rm.Plant(math.radians(theta0_deg), coulomb=limits, motor_limits=limits)
return rm.Loop(plant, rm.Balancer(K), rm.Sensor(realistic), ...)
```

네 조각이 분리되어 있다.

| 클래스 | 역할 |
| --- | --- |
| `Plant` | 물리 + 모터 펌웨어의 제약 (이 문서) |
| `Balancer` | 제어 법칙 |
| `Sensor` | 호스트가 보는 값 (0.3° 양자화, 차분 속도) |
| `Loop` | 100 Hz 주기와 4 ms 지연 |

---

## 1. 상태

$$x = [\;\theta,\;\;\dot\theta,\;\;\phi,\;\;v\;]$$

| 기호 | 코드 | 뜻 |
| --- | --- | --- |
| $\theta$ | `th` | 진자 각도, **직립이 0**, 매달림이 $\pm\pi$ |
| $\dot\theta$ | `w` | 진자 각속도 |
| $\phi$ | `phi` | 팔 각도 |
| $v$ | `v` | **펌웨어의 목표 속도** |

네 번째가 낯설 수 있다. 호스트는 가속도를 명령하지만 펌웨어가 그것을 적분해 속도를
만들기 때문에, 모터가 이상적인 가속도원이 아니라는 사실이 상태에 들어가 있다.
`deriv()` 가 돌려주는 마지막 성분이 $\dot v = a$ 인 이유다.

---

## 2. 전체 식

$$\ddot{\theta} \;=\; \frac{3g}{2l}\sin\theta \;+\; b\cos\theta\,\ddot{\phi}
\;+\; \sin\theta\cos\theta\,\dot{\phi}^{2} \;-\; 2\zeta\omega_n\dot{\theta}
\;-\; F_c\,\mathrm{sgn}(\dot\theta)$$

동정된 상수는 다음과 같다.

| 코드 | 값 | 유래 |
| --- | --- | --- |
| `A_G` | $3g/2l = \omega_n^2 = 56.235\ \mathrm{s^{-2}}$ | 자유 진동, $l = 262$ mm |
| `b` | $3r/2l = 0.730$ | 개루프 시험, $r = 127$ mm |
| `C_D` | $2\zeta\omega_n = 0.1758\ \mathrm{s^{-1}}$ | 자유 진동, $\zeta = 0.01172$ |
| `F_C` | $0.5561\ \mathrm{rad/s^2}$ | 건마찰 2.705 deg/s |

---

## 3. 항별로

### 3.1 `A_G * math.sin(th)` — 중력

$$\frac{3g}{2l}\sin\theta \;=\; \omega_n^2\sin\theta \;=\; 56.235\,\sin\theta$$

$\theta = 0$ 이 **직립**이므로 부호가 $+$ 다. 기울어질수록 더 기울이는 쪽으로 작용하는
불안정 항이고, 매달린 진자의 $-\omega_n^2\sin\theta$ 와 정반대다. 균일 막대에서
$\omega_n^2 = 3g/2l$ 이라는 관계가 여기서 $l = 262$ mm 를 준다.

### 3.2 `self.b * math.cos(th) * acc` — 제어 입력

$$b\cos\theta\,\ddot\phi,\qquad b = \frac{3r}{2l} = 0.730$$

팔을 가속하면 피벗이 움직이고, 진자는 관성력을 받는다. **이 시뮬레이터에서 손댈 수 있는
유일한 양**이며 단위는 토크가 아니라 팔의 각가속도 $[\mathrm{rad/s^2}]$ 다.

$\cos\theta$ 가 결정적이다.

- $\theta \approx 0$ (직립) 에서 가장 강하다.
- $\theta = 90^\circ$ 에서 **정확히 0** 이다. 그 자세에서는 팔을 아무리 흔들어도 진자를
  세울 수 없다 — 제어 가능성이 사라지는 지점이다.

### 3.3 `math.sin(th) * math.cos(th) * phid * phid` — 원심력

$$\sin\theta\cos\theta\,\dot\phi^{2} \;=\; \tfrac{1}{2}\sin(2\theta)\,\dot\phi^{2}$$

팔이 도는 동안 생기는 항으로, **회전형(Furuta) 진자에만 있다.** 직선 레일 위의
카트-진자에는 없는 항이다. 유도는 8 절에 있다 — 다른 두 항과 달리 **계수가 붙지
않는 이유**도 거기서 나온다. $\dot\phi$ 의 제곱이라 팔이 느릴 때는 무시할 만하지만
스윙업처럼 팔이 빠르게 돌 때 커진다.

### 3.4 `- C_D * w` — 점성 감쇠

$$-\,2\zeta\omega_n\dot\theta \;=\; -0.1758\,\dot\theta$$

### 3.5 `- F_C * math.tanh(w / 0.01)` — 건마찰

$$-\,F_c\,\mathrm{sgn}(\dot\theta) \;\approx\; -\,0.5561\,\tanh\!\left(\frac{\dot\theta}{0.01}\right)$$

`sgn` 을 그대로 쓰면 $\dot\theta = 0$ 에서 불연속이라 RK4 가 흔들린다. 폭 0.01 rad/s
(= 0.57 deg/s) 의 `tanh` 로 부드럽게 만든 것이고, 이 폭이 모델 안의 스틱션 대역이다.

---

## 4. 항의 크기 비교

| 조건 | 중력 | 팔 입력 | 원심력 | 점성 | 건마찰 | 합 |
| --- | --- | --- | --- | --- | --- | --- |
| $\theta$=5°, $\dot\theta$=0.5, $\dot\phi$=1, $a$=10 | +4.90 | +7.27 | +0.09 | −0.09 | −0.56 | +11.62 |
| $\theta$=30°, $\dot\theta$=2, $\dot\phi$=3, $a$=20 | +28.12 | +12.64 | +3.90 | −0.35 | −0.56 | +43.75 |
| $\theta$=90°, $\dot\theta$=2, $\dot\phi$=5, $a$=20 | +56.24 | **0.00** | 0.00 | −0.35 | −0.56 | +55.33 |

단위는 모두 $\mathrm{rad/s^2}$.

밸런싱 영역(5°)에서는 **중력과 팔 입력이 같은 크기로 겨루고**, 원심력과 마찰은 곁가지다.
90° 에서는 $\cos\theta = 0$ 이라 팔 입력이 사라진다.

### 세울 수 있는 최대 기울기

$\theta$ 를 유지하려면 중력을 팔 가속도로 정확히 상쇄해야 한다.

$$\ddot\phi \;=\; \frac{\omega_n^2\sin\theta}{b\cos\theta}$$

$\theta = 5^\circ$ 에서 $6.74\ \mathrm{rad/s^2}$ 로, 펌웨어 한계 $47.1$ 의 14 % 다. 이 식이
한계를 넘는 각도를 풀면

$$\tan\theta \;=\; \frac{b\,\ddot\phi_{max}}{\omega_n^2}
\qquad\Longrightarrow\qquad \theta \approx 31.5^\circ$$

**31.5° 를 넘으면 밸런싱으로는 못 세운다.** 그 너머는 스윙업의 영역이다.

---

## 5. 설계에 쓰는 선형화

직립 근처에서 $\sin\theta \approx \theta$, $\cos\theta \approx 1$, $\dot\phi^2 \approx 0$ 이므로

$$\ddot\theta \;\approx\; \omega_n^2\,\theta \;+\; b\,u \;-\; 2\zeta\omega_n\,\dot\theta$$

파일 docstring 6 행의 식이 이것이고, 극점 배치는 이 모델 위에서 이루어진다.

| 설계 | 게인 $K$ | 결과 |
| --- | --- | --- |
| 2상태 | $[274.3,\ 22.8,\ 0,\ 0]$ | 진자는 서지만 팔이 흘러간다 |
| **4상태** | $[236.2,\ 26.5,\ -4.16,\ -4.55]$ | 극점 $-6\pm6j,\ -1.5\pm j$, 팔도 0 근처에 머문다 |

시뮬레이터가 적분하는 것은 어디까지나 2 절의 **비선형** 식이다. 선형 모델은 게인을
만들 때만 쓴다.

---

## 6. 물리 바깥의 제약

`Plant.step()` 은 RK4 로 적분하면서 실제 장비의 한계를 같이 건다. 이 시뮬레이터에서
"플랜트" 는 미분방정식만이 아니라 **모터 펌웨어까지 포함한 것**이다.

| 상수 | 값 | 뜻 |
| --- | --- | --- |
| `ACCEL_MAX` | 47.1 rad/s² | 명령 가속도 포화 |
| `SPEED_MAX` | 7.85 rad/s (450 deg/s) | 목표 속도 포화 |
| `SPEED_MIN` | 0.0589 rad/s (3.4 deg/s) | **이 아래면 모터가 아예 돌지 않는다** (L6474) |
| `ARM_LATCH` | 150° | 래치 정지 |
| `STEP_RAD` | 0.1125°/마이크로스텝 | 팔 각도의 분해능 |

`SPEED_MIN` 이 특히 중요하다. `running()` 이 이 판정을 하고, 모터가 멈춘 동안에는
`deriv()` 안에서 `acc` 와 `phid` 가 **둘 다 0 으로 바뀐다**. 미세 조정이 불가능한 구간이
생긴다는 뜻이라 밸런싱에서 실제로 문제가 되는 비선형이다.

`make(..., limits=False)` 로 끄면 이상적인 가속도원이 된다.

---

## 7. 매달린 진자 모델과의 대조

같은 폴더의 [`pid_pendulum_realtime.py`](pid_pendulum_realtime.py) 와는 전제가 다르다.

| | `pid_pendulum_realtime.py` | `rotary_model.py` 의 `Plant` |
| --- | --- | --- |
| $\theta = 0$ | 매달림 (안정) | **직립 (불안정)** |
| 중력 항 | $-\omega_n^2\sin\theta$ | $+\omega_n^2\sin\theta$ |
| 입력 | 토크 $\tau/J$ | **팔 가속도** $b\cos\theta\,\ddot\phi$ |
| 건마찰 | 없음 | 있음 ($F_c = 0.556$) |
| 원심력 | 없음 | 있음 |
| 모터 한계 | 포화만 | 가속·속도·최소속도·래치 |
| 상태 수 | 3 ($\theta,\dot\theta,\int e$) | 4 ($\theta,\dot\theta,\phi,v$) |

---

## 8. 부록 — 원심력 항은 어디서 오나

3.3 절의 $\sin\theta\cos\theta\,\dot\phi^2$ 를 유도한다. 다른 두 항과 달리 **계수가 붙지
않는 이유**까지 나온다.

### 8.1 기하 — 기울면 회전축에서 멀어진다

진자의 회전축은 팔 방향이므로, 진자는 팔의 접선 방향 평면에서 기울어진다. 피벗에서 거리
$s$ 인 막대 요소가 **수직 회전축**에서 떨어진 거리는

$$R(s)^2 = r^2 + (s\sin\theta)^2$$

직립($\theta = 0$)에서는 모든 요소가 축에서 정확히 $r$ 만큼 떨어져 있다. 기울면
$s\sin\theta$ 만큼 옆으로 나가므로 **반지름이 커진다.** 원심력이 $\theta$ 에 관여하는 것은
전적으로 이 때문이다.

### 8.2 유도 A — 원심 퍼텐셜

팔과 함께 도는 좌표계에서 원심력은 퍼텐셜로 쓸 수 있다.

$$V_{cf} = -\frac{1}{2}\dot\phi^2\!\int R^2\,dm
= -\frac{1}{2}\dot\phi^2\left(r^2 M + \sin^2\theta\!\int s^2\,dm\right)$$

$\int s^2\,dm = J_p$ 는 피벗 기준 관성모멘트이고 $r^2M$ 은 $\theta$ 와 무관하므로,
$\theta$ 에 작용하는 일반화 토크는

$$Q_\theta = -\frac{\partial V_{cf}}{\partial\theta} = J_p\,\dot\phi^2\sin\theta\cos\theta$$

운동방정식 $J_p\ddot\theta = \cdots + Q_\theta$ 를 $J_p$ 로 나누면

$$\ddot\theta = \cdots + \dot\phi^2\sin\theta\cos\theta$$

**계수가 정확히 1 인 이유가 여기 있다** — $J_p/J_p$ 다. 중력 항에는 $Mgl_c/J_p = A_G$,
입력 항에는 $rMl_c/J_p = b$ 라는 상수가 붙는데 원심력만 맨몸인 것이 처음에는 이상해
보이지만, 원심 토크 자체가 $J_p$ 에 비례하기 때문이다.

### 8.3 유도 B — 라그랑주 정공법

회전 좌표계를 빌리지 않고 관성계에서 해도 같은 결과가 나온다. 요소 속도의 제곱은

$$|v(s)|^2 = s^2\sin^2\theta\,\dot\phi^2 + r^2\dot\phi^2
+ 2rs\,\dot\phi\dot\theta\cos\theta + s^2\dot\theta^2$$

첫 항이 8.1 절의 "멀어진 만큼" 이다. 막대 전체로 적분하면

$$T = \frac{1}{2}\left(J_p\sin^2\theta\,\dot\phi^2 + Mr^2\dot\phi^2
+ 2rMl_c\,\dot\phi\dot\theta\cos\theta + J_p\dot\theta^2\right)$$

$\phi$ 를 주어진 입력으로 두고 $\theta$ 에 대해 오일러-라그랑주를 적용한다.

$$\frac{d}{dt}\frac{\partial T}{\partial\dot\theta}
= J_p\ddot\theta + rMl_c\left(\ddot\phi\cos\theta - \dot\phi\dot\theta\sin\theta\right)$$

$$\frac{\partial T}{\partial\theta}
= \frac{1}{2}J_p\dot\phi^2\sin 2\theta - rMl_c\,\dot\phi\dot\theta\sin\theta$$

$rMl_c\,\dot\phi\dot\theta\sin\theta$ **두 개가 정확히 상쇄된다.** Coriolis 항이 사라지는
것이고, 그래서 최종 식에 $\dot\phi\dot\theta$ 교차항이 남지 않는다 — 코드에도 없다.

$V = Mgl_c\cos\theta$ 까지 넣어 정리하면

$$J_p\ddot\theta = Mgl_c\sin\theta + J_p\dot\phi^2\sin\theta\cos\theta
- rMl_c\ddot\phi\cos\theta$$

$J_p$ 로 나누면 코드의 세 항이 그대로 나온다.

$$\ddot\theta = \frac{Mgl_c}{J_p}\sin\theta + \dot\phi^2\sin\theta\cos\theta
- \frac{rMl_c}{J_p}\ddot\phi\cos\theta$$

앞의 계수가 `A_G`, 뒤의 계수가 `b` 다. 균일 막대($J_p = ML^2/3$, $l_c = L/2$)를 넣으면
$A_G = 3g/2L$, $b = 3r/2L$ 이 된다.

### 8.4 수치 검증

막대를 20000 조각으로 나눠 $T$ 를 직접 쌓고, 오일러-라그랑주 미분을 **전부 수치로** 잡아
공식과 비교했다 ([`check_rotary_plant.py`](check_rotary_plant.py)).

| $\theta$ | $\dot\theta$ | $\dot\phi$ | $\ddot\phi$ | 수치 EL | 공식 | 차이 |
| --- | --- | --- | --- | --- | --- | --- |
| 5° | 0.5 | 1 | 10 | −2.284194 | −2.284194 | 4.2e−8 |
| 30° | 2 | 3 | 20 | 19.370643 | 19.370644 | −1.3e−6 |
| 60° | −1 | 4 | −15 | 61.104139 | 61.104143 | −3.5e−6 |
| 90° | 2 | 5 | 20 | 56.235008 | 56.235001 | 6.9e−6 |

차이는 유한차분의 오차이고, **차분 폭에 민감하다.** 여기 쓴 것은 모두 2계 차분이라
반올림 잡음이 $\varepsilon/h^2$ 로 들어온다. $h = 10^{-6}$ 이면 마지막 자리가 전부 잡음이라
일치가 $10^{-2}$ 수준까지 떨어지고, $h = 10^{-4}$ 에서 절단 오차와 반올림이 균형을 이뤄
위와 같이 $10^{-6}$ 까지 맞는다. 스크립트는 이 값을 기본으로 쓴다. 상수도 맞는다 — $Mgl_c/J_p = 56.2350 =$ `A_G`,
$rMl_c/J_p = 0.7300 =$ `b`, $J_p = ML^2/3$.

### 8.5 부호가 말하는 것

$\sin\theta\cos\theta$ 는 $0 < \theta < 90^\circ$ 에서 **양수**다. 중력 항과 같은 부호,
즉 **기울어진 쪽으로 더 밀어내는 불안정화 항**이다. 직관과도 맞는다 — 기울면 반지름이
커지고, 원심력은 반지름이 커지는 쪽으로 작용하니 더 기울게 만든다.

$\theta = 90^\circ$ 에서 0 이 되고, 그 너머에서는 부호가 바뀌어 되돌리는 방향이 된다.

$\dot\phi$ 의 **제곱**이라 밸런싱(팔이 거의 정지)에서는 곁가지다 — 4 절 표에서 5°,
$\dot\phi = 1$ 일 때 +0.09 로 중력 +4.90 의 2 % 다. 반면 스윙업에서 팔이 빠르게 돌면
30° 근처에서 +3.9, 더 빠르면 두 자리로 올라가 무시할 수 없다.

> **부호 규약**: 위 유도는 $-b\cos\theta\,\ddot\phi$ 를 주는데 코드는
> `+ b*cos(th)*acc` 다. $\phi$ 의 양의 방향을 반대로 잡은 것이고,
> `place2`/`place4` 가 같은 규약($\ddot\theta = A_G\theta + bu - C_D\dot\theta$)으로
> 게인을 만들기 때문에 내부적으로 일관된다.

---

## 관련 파일

- [`check_rotary_plant.py`](check_rotary_plant.py) — 8.4 절의 수치 검증과 이 문서의
  상수를 다시 계산하는 스크립트
- [`rotary_model.py`](rotary_model.py) — 이 문서가 설명하는 모델
- [`rotary_balance_realtime.py`](rotary_balance_realtime.py) — 밸런싱 실행·화면
- [`rotary_swingup_realtime.py`](rotary_swingup_realtime.py) — 같은 플랜트로 스윙업
- [`README.md`](README.md) — 파라미터 출처와 사용법
- [`pid_gain_design.md`](pid_gain_design.md) — 매달린 진자 쪽 PID 게인 유도
