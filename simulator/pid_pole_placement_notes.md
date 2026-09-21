# 진자 PID 설계 노트 — 극점 배치, α, 그리고 영점

[`pid_gain_design.md`](pid_gain_design.md)의 자매편. 게인 공식의 유도에서 출발해,
**목표 극점 자체를 어떻게 정하는가**, **적분기 극점 α의 원리와 트레이드오프**,
**PID가 만드는 영점의 정체**까지를 문답으로 파고든 기록이다.
대상 코드는 [`pid_pendulum_realtime.py`](pid_pendulum_realtime.py) 30–32행.

---

## 1. 이 세 숫자가 하는 일

PID에 적분기가 들어가면 닫힌 루프가 3차가 된다. 3차 다항식을 지정하려면 근이 세 개
필요한데, 그게 정확히 `wn_des`, `zeta_des`, `alpha` 세 숫자다:

$$\underbrace{\left(s^2 + 2\zeta_{des}\,\omega_{des}\,s + \omega_{des}^2\right)}_{\text{지배 복소극점 쌍}}\;\underbrace{\left(s + \alpha\right)}_{\text{적분기 극점}}$$

플랜트 $J\ddot\theta + c\dot\theta + k\theta = \tau$ 에 제어입력

$$\tau = K_p\,e + K_i\!\int\! e\,dt - K_d\,\dot\theta$$

를 넣고 한 번 미분하면 특성방정식이

$$J s^3 + (c + K_d)\,s^2 + (k + K_p)\,s + K_i = 0$$

이고, 위 목표 다항식과 계수를 맞추면 코드의 세 줄이 그대로 나온다:

$$K_d = J\left(2\zeta_{des}\,\omega_{des} + \alpha\right) - c$$

$$K_p = J\left(\omega_{des}^2 + 2\alpha\,\zeta_{des}\,\omega_{des}\right) - k$$

$$K_i = J\,\alpha\,\omega_{des}^2$$

> **핵심** — 끝의 $-c$, $-k$가 핵심이다. 플랜트가 이미 갖고 있는 감쇠와 중력 강성을
> 빼주는 것, 즉 아는 물리를 상쇄하고 원하는 동역학을 얹는 **모델 기반 설계**다.
> 그래서 $\omega_n$, $\zeta$ 동정값이 틀리면 실제 극점이 설계값에서 벗어난다.

검증하면 정확히 배치된다:

```
Kp = 222.165   Ki = 1152.0   Kd = 24.624
closed-loop poles: -8.4 ± 8.57j,  -8.0     ← 요청과 일치
```

| 파라미터 | 값 | 의미 |
|---|---|---|
| `wn_des = 12.0` | rad/s | 닫힌 루프 속도. 플랜트 고유값 7.5의 1.6배 → 정착 $\approx 4/(\zeta\omega) = 0.48\,\mathrm{s}$ |
| `zeta_des = 0.7` | — | 지배 극점 쌍의 감쇠비. 표준 2차 기준 오버슈트 4.6% |
| `alpha = 8.0` | rad/s | 적분기 실수극점 $-8$. 지배 극점(실수부 8.4)과 비슷한 속도 |

> **주의** — `--check`가 찍는 실제 오버슈트는 **19.7%** 로 4.6%와 많이 다르다.
> 두 가지 이유다. PID는 극점만이 아니라 영점도 만들고($K_p/K_d$ 비), $\alpha = 8$이
> 지배 극점 실수부 8.4와 거의 같아서 "느린 적분기" 가정이 성립하지 않는다.
> $\zeta_{des}$는 극점 배치용 파라미터일 뿐, 응답의 오버슈트를 그대로 뜻하지 않는다.
> (참고로 [`../analysis/design_pid.py`](../analysis/design_pid.py)의 직립 설계에서는
> α를 지배 극점의 1/3로 훨씬 느리게 잡았다.)

---

## 2. 목표 극점 자체는 어떻게 정했나

$-8.4 \pm 8.57j,\ -8$ 은 측정값($\zeta$, $\omega_n$)에서 계산되어 나온 게 아니라
**설계자가 "이런 응답을 원한다"고 먼저 정한 사양**이다. 측정값은 그 사양이 현실적인지
판단하는 기준과, 게인 공식의 $-c$, $-k$ 상쇄항으로만 쓰인다. 다섯 숫자처럼 보이지만
실은 설계 파라미터 세 개의 다른 표기다:

| 극점 | 어디서 왔나 | 근거 |
|---|---|---|
| $-8.4 \pm 8.57j$ | $-\zeta_{des}\omega_{des} \pm j\,\omega_{des}\sqrt{1-\zeta_{des}^2}$ | $(\omega_{des}=12,\ \zeta_{des}=0.7)$의 극좌표 표현일 뿐 |
| $\zeta_{des} = 0.7$ | 관례적 표준값 | 2차 기준 오버슈트 4.6%, 감쇠–정착시간의 고전적 절충. 극점 각도 45° |
| $\omega_{des} = 12$ | 플랜트 고유진동수 7.5의 **1.6배** | "플랜트보다 적당히 빠르게". 정착 $\approx 4/(\zeta\omega) = 0.48\,\mathrm{s}$ |
| $-8$ | α, 적분기 극점 | 지배 극점과 비슷한 속도 (강의 슬라이드 정합으로 추정 — 교과서 규칙과 반대) |

즉 측정된 $\omega_n = 7.5$는 "12를 골라도 되는가"의 **비교 기준**이었고, 측정된
$\zeta = 0.012$는 게인 공식의 $-c$ 항으로 들어가 상쇄될 뿐 목표 극점 선택에는
관여하지 않았다.

---

## 3. 일반적인 극점 선정 절차

표준 절차는 **시간 영역 사양 → 지배 2차 극점 쌍 → 제약 확인 → 나머지 극점** 순서다.

**① 사양에서 지배 쌍을 역산.** 2차 시스템 공식 두 개가 사양을 극점으로 변환해 준다:

$$M_p = \exp\!\left(\frac{-\pi\zeta}{\sqrt{1-\zeta^2}}\right) \;\Rightarrow\; \zeta \text{ 결정},\qquad
t_s \approx \frac{4}{\zeta\,\omega_n} \;\Rightarrow\; \omega_n \text{ 결정}$$

예를 들어 "오버슈트 5% 이하, 0.5초 안에 정착"이면 $\zeta = 0.7$,
$\omega_n = 4/(0.7 \times 0.5) \approx 11.4$ — 이 프로젝트의 12와 사실상 같은 계산이다.

**② 상한 제약으로 ω를 검증.** 빨리 갈수록 좋은 게 아니라, 다음 네 가지가 상한을 정한다:

- **액추에이터 포화** — 필요 토크가 대략 $\omega_{des}^2$에 비례한다($K_p \approx J\omega_{des}^2$).
  2배 빠르게 하면 4배의 토크. "20° 스텝 킥이 한계의 46%" 확인이 정확히 이 검증이다.
- **샘플링 속도** — 경험칙으로 $\omega_{des} \le \omega_s/10{\sim}20$. 여기선
  100 Hz(628 rad/s)라 12는 매우 여유.
- **모델 신뢰 대역** — 동정 모델이 맞는 범위를 넘어 극점을 두면 실제 극점이 설계에서
  벗어난다. 보통 플랜트 고유 동역학의 2~4배 이내가 안전. 1.6배는 보수적 선택.
- **센서 노이즈** — $\zeta_{plant} = 0.012$라 닫힌 루프 감쇠를 거의 전부 $K_d$가
  만들어야 하는데, $K_d$가 클수록 각속도 노이즈가 증폭된다.

**③ 나머지(비지배) 극점 배치.** 적분기 극점처럼 남는 실수극점은 지배 쌍이 응답을
지배하도록 배치한다. 교과서 규칙은 **지배 극점 실수부의 1/3~1/5 (느리게)** —
적분기는 정상상태만 담당하고 과도응답에 끼어들지 말라는 뜻이다. §7의 α 스윕이
그 이유를 정량적으로 보여준다.

**④ 다른 유파들.** 극점을 직접 찍는 대신:

- **표준 다항식 프로토타입** — ITAE 최적 다항식이나 Bessel 다항식 표에서 $\omega_0$
  하나만 골라 계수를 가져오는 방법. 3차면 ITAE 기준
  $s^3 + 1.75\,\omega_0 s^2 + 2.15\,\omega_0^2 s + \omega_0^3$.
- **LQR** — 극점 대신 가중치 $Q$, $R$을 고르면 최적화가 극점을 정해줌. 상태가 4개
  이상이면 극점을 직접 다 찍기 어려워 이쪽이 표준.
- **주파수 영역** — 교차 주파수(≈ 대역폭)와 위상 여유 60°를 목표로 잡는 방법.
  PM 60° ≈ ζ 0.6~0.7이라 결국 같은 곳에 도달한다.

> **요약** — 극점 세 개는 **사양(속도·감쇠·적분기 속도)의 번역**이지 측정의 결과가
> 아니다. 측정 $\zeta$, $\omega_n$의 역할은 두 가지뿐 — ① "플랜트의 몇 배 속도인가"라는
> 실현 가능성 잣대, ② 게인 공식에서 $-c$, $-k$로 상쇄되는 항. 이 설계는
> ζ = 0.7(관례), ω = 1.6×플랜트(적당히 빠르게 + 포화 검증)까지는 교과서 그대로이고,
> α = 8만 관례(1/3~1/5)에서 벗어난 선택이다.

---

## 4. 포화 한계 |u| ≤ 3·wn²

$u = \tau/J$라서 단위는 $\mathrm{rad/s^2}$이고, $\omega_n^2 = k/J = 56.24$는 막대를
수평(90°)으로 들고 있는 데 필요한 중력 토크다. 즉 이 한계는 **"중력 최대 토크의 3배"**
라는 뜻이다 — 장비 사양이 아니라 물리량에 스케일을 맞춘 편의적 선택이고, $J$를 몰라도
되게 한 정규화와 같은 사고방식이다.

| 자세 | 필요 토크 (rad/s²) | 한계 대비 |
|---|---|---|
| 20° 유지 | 19.2 | 11 % |
| 45° 유지 | 39.8 | 24 % |
| 90° (수평) | 56.2 | 33 % |

20° 스텝의 초기 킥은 $K_p \cdot e = 77.6$으로 한계의 46%라 포화되지 않는다. 실제로
포화가 걸리는 건 left/right 키로 200 deg/s 킥을 줬을 때 정도다. 포화 중에는 적분기가
얼도록 안티와인드업이 걸려 있다 — 54행의 한 줄이 그 역할이다:

```python
die = e if abs(tau_raw) < TAU_MAX else 0.0
```

---

## 5. α는 세 번째 극점이다

적분기가 상태를 하나 더하므로 닫힌 루프는 3차이고, 근이 셋 필요하다. `wn_des`와
`zeta_des`가 복소 쌍 $-8.4 \pm 8.57j$를 정하면, 남은 실수근 하나가 $\alpha$다:

$$\left(s^2 + 2\zeta_{des}\,\omega_{des}\,s + \omega_{des}^2\right)\,\underbrace{\left(s + \alpha\right)}_{\text{적분기 모드}}$$

게인에 들어가는 방식은 세 군데가 다르다:

$$K_i = J\,\alpha\,\omega_{des}^2 \;\;(\alpha\text{에 정비례}),\qquad K_p \ni 2\alpha\,\zeta_{des}\,\omega_{des},\qquad K_d \ni \alpha$$

시간 영역에서는 $e^{-\alpha t}$ 모드, 즉 시상수 $1/\alpha = 0.125\,\mathrm{s}$로
정상상태 오차가 지워진다.

---

## 6. 애초에 적분기가 왜 필요한가

20°를 유지하려면 중력 토크 $K\sin 20° = 19.2$를 누가 계속 내야 한다. PD만 있으면 그
일을 $K_p e$가 해야 하므로 **오차가 남아야만 토크가 나온다**:

| 구성 | 6초 뒤 각도 | 오차 |
|---|---|---|
| α = 0 (PD) | 12.23° | −7.77° |
| α = 2 | 20.000° | +0.000° |
| α = 8 | 20.000° | +0.000° |

건마찰($F_c = 0.556$)을 넣으면 PD의 오차는 −7.73°, 적분기가 있으면 ±0.03° 안이다.
α는 **"정상상태 오차를 얼마나 빨리 지울 것인가"**를 정하는 값이다.

---

## 7. α를 키우면

| α | $K_i$ | 영점 $-K_i/K_p$ | 오버슈트 | 2% 정착 |
|---|---|---|---|---|
| 1 | 144 | −1.38 | −0.1 % | 2.67 s |
| 2 | 288 | −2.37 | 0.0 % | 1.10 s |
| 4 | 576 | −3.72 | 8.5 % | 0.58 s |
| **8 (현재)** | **1152** | **−5.19** | **19.7 %** | **0.56 s** |
| 12 | 1728 | −5.97 | 23.9 % | 0.51 s |
| 20 | 2880 | −6.80 | 26.2 % | 0.46 s |

빨라지는 대가로 오버슈트가 단조 증가한다. 이유는 α가 **극점과 영점을 동시에 움직이는데
속도가 다르기 때문**이다. 영점은 $-K_i/K_p$에 있고, 극점은 $-\alpha$에 있는데 — 둘이
같아지는 지점이 있다:

$$-\frac{K_i}{K_p} = -\alpha \quad\Longleftrightarrow\quad \alpha = \frac{k}{2\zeta_{des}\,\omega_{des}} = \frac{56.24}{16.8} = 3.35\ \mathrm{rad/s}$$

| α | 적분기 극점 | 영점 | 영점/극점 |
|---|---|---|---|
| 1 | −1.00 | −1.38 | 1.38 — 영점이 더 빠름 |
| 3.35 | −3.35 | −3.35 | 1.00 — 상쇄 |
| 8 | −8.00 | −5.19 | 0.65 — 영점이 더 느림 |
| 20 | −20.0 | −6.80 | 0.34 |

- **α < 3.35**: 영점이 적분기 극점보다 빨라 서로 거의 상쇄된다. 오버슈트가 사라지는
  대신 느린 꼬리가 남는다(α = 1에서 정착 2.67초).
- **α > 3.35**: 영점이 뒤처지기 시작하고, 뒤처진 만큼 오버슈트가 커진다.

---

## 8. 현재 값 α = 8에 대한 관찰

지배 극점 실수부 8.4와 거의 같은 속도다. 교과서적인 "적분기는 지배 극점보다
느리게"(보통 1/3~1/5)와는 반대 방향이고, 실제로 그 대가가 19.7% 오버슈트다.

> **관찰** — 표에서 눈에 띄는 것은 α = 4와 α = 8의 정착 시간이 0.58초 대 0.56초로
> 사실상 같다는 점이다. 오버슈트는 8.5% 대 19.7%로 두 배 넘게 차이 나는데 말이다.
> 속도를 거의 못 얻으면서 오버슈트만 사는 구간이라, 이 플랜트에서는 **α ≈ 4가 더 나은
> 거래**로 보인다. 다만 이 값은 강의 슬라이드(Lec02 24–25쪽)와 맞춘 것일 수 있어
> 코드는 바꾸지 않았다.

참고로 직립 설계([`../docs/pendulum_pid_design.md`](../docs/pendulum_pid_design.md))에서는
$\omega_i$를 지배 극점의 1/3(4 vs 12)로 잡았다. 거기서는 기준값이 항상 0이라 영점이
문제되지 않고, 적분기가 지배 극점과 싸우지 않는 쪽이 중요했다.

---

## 9. 적분기 극점은 어떤 원리로 정하나 (KO)

**왜 극점이 "추가"되는가.** 적분기 $K_i\!\int\!e\,dt$는 컨트롤러 안에 새 상태(오차의
누적값)를 하나 만든다. 전달함수로는 $1/s$가 루프에 곱해지는 것이라 2차였던 닫힌 루프가
3차가 되고, 3차 다항식의 근은 셋이므로 지배 쌍을 정하고 나면 **남는 실수근 하나가
반드시 어딘가에 생긴다** — 그 위치가 설계자의 몫이고, 그게 $-\alpha$다. 물리적 의미는
$e^{-\alpha t}$ 모드, 즉 정상상태 오차가 지워지는 시상수 $1/\alpha$. 그러니 "무조건
크게"가 좋아 보이지만, 극점은 공짜로 오지 않는다: $K_i = J\alpha\omega_{des}^2$이므로
α를 움직이면 **영점 $-K_i/K_p$도 함께, 그러나 다른 속도로 움직인다**. 이 극점–영점
쌍(dipole)의 상대 위치가 규칙의 본질이다.

**"지배 극점보다 1/3~1/5 느리게"의 네 가지 이유:**

1. **역할 분리 (시간축 분리)** — P·D는 과도응답 담당, I는 정상상태 담당. 적분기가 지배
   극점만큼 빠르면 과도응답이 진행 중인데 끼어들어, 상승 구간에 쌓인 적분값이 목표 도달
   후에도 남아 넘어간다 — 오버슈트의 시간 영역 그림. 느리면 과도응답이 끝난 뒤의
   준정적 오차만 조용히 치운다.
2. **시간 영역: 극점–영점 근사 상쇄** — α가 느리면 극점 $-\alpha$와 영점 $-K_i/K_p$가
   가까워 기준입력 응답에서 거의 상쇄되고(잔차가 작음), 응답은 지배 쌍만으로 결정되어
   $\zeta_{des} = 0.7$이 약속한 4.6%가 실제로 나온다. α를 키우면 영점이 극점을 못
   따라가(α = 8일 때 극점 $-8$ vs 영점 $-5.19$) 상쇄가 깨지고, 지배 영역에 들어온
   영점이 미분성 선행 킥으로 오버슈트를 만든다.
3. **주파수 영역: 위상 여유 잠식** — PI의 위상 기여는 $-\arctan(\omega_z/\omega)$
   ($\omega_z = K_i/K_p$). 교차 주파수 $\omega_c$에서 잃는 위상은 $\omega_z = \omega_c/5$일
   때 $-11°$, $\omega_c/3$일 때 $-18°$, $\omega_c$와 같으면 $-45°$. 위상 여유 60°
   목표에서 10~20° 손실까지가 감내선이고, **그 감내선을 숫자로 옮긴 것이 1/3~1/5다.**
   위상 여유가 깎이면 유효 감쇠가 줄어 오버슈트가 커지니 ②의 주파수 영역 표현이다.
4. **반대쪽 제약: 무한정 느리게는 못 한다** — ②의 상쇄는 **기준입력 응답에만**
   적용된다. 부하 외란(중력 토크 오차, 마찰) 응답에는 그 영점이 없어 $e^{-\alpha t}$
   모드가 온전히 나타나고, α가 너무 느리면 외란 회복이 시상수 $1/\alpha$로
   굼뜬다(α = 1이면 정착 2.67초짜리 꼬리). 빠른 쪽으로는 $K_i \propto \alpha$라 포화 시
   와인드업도 커진다. 그래서 "지배 응답을 안 건드리는 한도 내에서 최대한 빠르게" →
   1/3~1/5.

**이 플랜트로 검산.** 규칙대로면 $\alpha \approx 8.4/3 \sim 8.4/5 = 1.7{\sim}2.8$.
§7의 스윕 표가 그대로 보여준다 — α = 2(규칙 구간): 오버슈트 0.0%, 정착 1.10 s로
$\zeta_{des}$의 약속이 지켜지고; α = 8(지배 극점과 동속): 오버슈트 19.7%, 대신 정착
0.56 s. [`../analysis/design_pid.py`](../analysis/design_pid.py)의 직립 설계가 1/3을
쓴 것이 교과서 규칙 그대로다.

---

## 10. How the Integrator Pole Is Chosen (EN)

**Why the integrator "adds" a pole.** The integral term $K_i\!\int\!e\,dt$ introduces
a new state inside the controller (the accumulated error). In transfer-function terms
it multiplies the loop by $1/s$, turning the second-order closed loop into a
third-order one. A cubic has three roots, so once the dominant pair is fixed,
**a leftover real root must land somewhere** — its location is the designer's choice,
and that is $-\alpha$. Physically it is the $e^{-\alpha t}$ mode: the time constant
$1/\alpha$ at which steady-state error is erased. "As fast as possible" looks
attractive, but the pole does not come free: since $K_i = J\alpha\omega_{des}^2$,
moving α also moves **the zero at $-K_i/K_p$, at a different speed**. The relative
position of this pole–zero pair (dipole) is what the rule is really about.

**Four reasons behind "place the integrator 1/3–1/5 slower than the dominant poles":**

1. **Separation of roles (time-scale separation)** — P and D own the transient; I owns
   steady state. An integrator as fast as the dominant poles interferes while the
   transient is still under way: charge accumulated during the rise remains after the
   target is reached and pushes past it — the time-domain picture of overshoot. A slow
   integrator quietly cleans up the quasi-static error after the transient is over.
2. **Time domain: near pole–zero cancellation** — with a slow α, the pole $-\alpha$
   and the zero $-K_i/K_p$ sit close together and nearly cancel in the reference
   response (small residue), so the response is shaped by the dominant pair alone and
   $\zeta_{des} = 0.7$ delivers its promised 4.6%. Increase α and the zero falls
   behind the pole (at α = 8: pole $-8$ vs zero $-5.19$); the cancellation breaks, and
   a zero inside the dominant region acts as an anticipatory derivative kick that
   creates overshoot.
3. **Frequency domain: phase-margin erosion** — the PI contributes phase
   $-\arctan(\omega_z/\omega)$ with $\omega_z = K_i/K_p$. The phase lost at the
   crossover frequency $\omega_c$ is $-11°$ when $\omega_z = \omega_c/5$, $-18°$ at
   $\omega_c/3$, and $-45°$ at $\omega_z = \omega_c$. With a 60° phase-margin target,
   a 10–20° loss is the tolerable band — **1/3–1/5 is that tolerance translated into a
   number.** Lost phase margin means less effective damping and more overshoot: the
   frequency-domain face of reason 2.
4. **The opposite constraint: it cannot be arbitrarily slow** — the cancellation in
   reason 2 applies **only to the reference response**. The load-disturbance response
   (gravity-torque error, friction) has no such zero, so the $e^{-\alpha t}$ mode
   appears in full; too slow an α means sluggish disturbance recovery with time
   constant $1/\alpha$ (a 2.67 s settling tail at α = 1). On the fast side,
   $K_i \propto \alpha$ makes windup worse under saturation. Hence: "as fast as
   possible without disturbing the dominant response" → 1/3–1/5.

**Checked against this plant.** The rule gives
$\alpha \approx 8.4/3 \sim 8.4/5 = 1.7{-}2.8$. The sweep table in §7 shows exactly
that — α = 2 (inside the rule band): 0.0% overshoot, 1.10 s settling, so
$\zeta_{des}$ keeps its promise; α = 8 (as fast as the dominant poles): 19.7%
overshoot in exchange for 0.56 s. The upright design in
[`../analysis/design_pid.py`](../analysis/design_pid.py) used exactly 1/3, per the
textbook.

---

## 11. PID가 영점을 만드는 이유 (KO)

**① 분자는 세 항의 통분에서 나온다.** 교과서 PID는 오차 $e$에 세 연산을 병렬로 건다.
라플라스 영역에서 적분은 $1/s$, 미분은 $s$이므로:

$$C(s) = K_p + \frac{K_i}{s} + K_d\,s = \frac{K_d\,s^2 + K_p\,s + K_i}{s}$$

즉 PID는 분모에 극점 하나($s = 0$, 적분기), 분자에 2차식 하나를 가진 전달함수다.
분자의 근이 영점이니, **게인 세 개를 고르는 순간 영점의 위치도 함께 정해진다** —
극점 배치의 부산물이다.

**② 영점은 피드백으로 못 옮긴다.** 닫힌 루프는

$$T(s) = \frac{C(s)\,G(s)}{1 + C(s)\,G(s)}$$

분모 $1 + CG$의 근인 극점은 게인으로 움직일 수 있지만, 분자에는 전방 경로의 영점이
그대로 살아남는다. **피드백은 극점을 옮기는 도구이지 영점을 옮기는 도구가 아니다.**
극점 셋을 원하는 곳에 놓는 데 성공해도 응답은 "극점 + 딸려온 영점"으로 결정된다.

**③ 영점이 응답에 하는 일: 미분 선행 킥.** 영점 없는 응답을 $y_0(t)$라 하고 같은
극점에 영점 $-z$를 붙이면:

$$T(s) = \left(1 + \frac{s}{z}\right)T_0(s) \;\;\Rightarrow\;\; y(t) = y_0(t) + \frac{1}{z}\,\dot y_0(t)$$

영점은 응답에 자기 자신의 미분을 $1/z$만큼 섞는다. $z$가 작을수록(원점에 가까울수록)
상승 중의 기울기가 크게 더해져 목표를 지나친다 — "영점이 지배 극점 영역에 들어오면
오버슈트"의 정확한 메커니즘이다.

**④ 이 코드에서는 영점이 하나뿐.** 이 시뮬레이터는 미분을 오차가 아니라 **측정값에**
건다:

$$\tau = K_p\,e + K_i\!\int\!e\,dt - K_d\,\dot\theta \qquad (\dot\theta = \text{측정 각속도},\ \dot e\ \text{아님})$$

기준입력이 상수인 동안 $\dot e = -\dot\theta$라 둘은 같지만, 스텝 순간에는 $\dot e$에
임펄스가 실린다(derivative kick). 그래서 실무 PID는 측정값 미분이 표준이고, 그 결과 —
기준입력 → 출력 경로의 분자는 $K_p s + K_i$ 뿐이라 영점은 **정확히
$-K_i/K_p = -1152/222.165 = -5.19$ 하나**이고, $K_d$ 항은 피드백 경로에서 극점 배치에만
참여한다. §7에서 영점을 $-K_i/K_p$로 쓴 것이 근사가 아니라 정확한 값이었던 이유다.
(교과서형 그대로면 판별식 $K_p^2 - 4K_dK_i = 49357 - 113467 < 0$이라 복소 영점 쌍이
됐을 것이다.)

**⑤ 숫자로 닫기.** 영점 $-5.19$는 지배 극점 실수부 $-8.4$보다 원점에 가깝다.
$1/z = 1/5.19$가 곱해진 미분 킥이 상승 구간에 실려, 극점 감쇠 $\zeta = 0.7$이 약속한
4.6% 대신 19.7%가 나온 것이다. α를 줄이면 $K_i = J\alpha\omega^2$가 줄어 영점이 원점
쪽으로 내려오고, 동시에 극점 $-\alpha$도 내려와 서로 상쇄한다는 것이 §7의 dipole
이야기다.

---

## 12. Why a PID Creates Zeros (EN)

**1. The numerator comes from combining three parallel terms.** A textbook PID applies
three operations to the error $e$ in parallel. In the Laplace domain integration is
$1/s$ and differentiation is $s$, so:

$$C(s) = K_p + \frac{K_i}{s} + K_d\,s = \frac{K_d\,s^2 + K_p\,s + K_i}{s}$$

A PID is therefore a transfer function with one pole ($s = 0$, the integrator) in the
denominator and a quadratic in the numerator. The numerator's roots are zeros, so
**the moment you pick three gains you have also placed the zeros** — a by-product of
pole placement.

**2. Feedback cannot move zeros.** The closed loop is

$$T(s) = \frac{C(s)\,G(s)}{1 + C(s)\,G(s)}$$

The poles — roots of $1 + CG$ — move with gain, but the forward-path zeros survive
untouched in the numerator. **Feedback is a tool for moving poles, not zeros.** Even
after placing all three poles exactly where you want them, the response is decided by
"the poles plus the zeros that came along".

**3. What a zero does to the response: an anticipatory derivative kick.** Let
$y_0(t)$ be the zero-free response, and attach a zero at $-z$ to the same poles:

$$T(s) = \left(1 + \frac{s}{z}\right)T_0(s) \;\;\Rightarrow\;\; y(t) = y_0(t) + \frac{1}{z}\,\dot y_0(t)$$

A zero mixes the response's own derivative into it, scaled by $1/z$. The smaller $z$
(the closer the zero to the origin), the more of the rising slope is added on top,
overshooting the target — the precise mechanism behind "a zero inside the
dominant-pole region causes overshoot".

**4. This code has only one zero.** The simulator differentiates the **measurement**,
not the error:

$$\tau = K_p\,e + K_i\!\int\!e\,dt - K_d\,\dot\theta \qquad (\dot\theta = \text{measured rate, not } \dot e)$$

While the reference is constant, $\dot e = -\dot\theta$ and the two are identical; but
at a step, $\dot e$ carries an impulse (derivative kick). Practical PIDs therefore
differentiate the measurement, and the structure changes as a result: the
reference-to-output numerator is just $K_p s + K_i$, so the zero is **exactly one, at
$-K_i/K_p = -1152/222.165 = -5.19$**, while the $K_d$ term moves to the feedback path
and participates only in pole placement. This is why §7 used $-K_i/K_p$ as the zero —
it was exact, not an approximation. (With the textbook form, the discriminant
$K_p^2 - 4K_dK_i = 49357 - 113467 < 0$ would have produced a complex pair of zeros
instead.)

**5. Closing with the numbers.** The zero at $-5.19$ sits closer to the origin than
the dominant poles' real part $-8.4$. The derivative kick scaled by $1/z = 1/5.19$
rides the rising edge, which is why the response overshoots 19.7% instead of the 4.6%
promised by the pole damping $\zeta = 0.7$. Reducing α lowers $K_i = J\alpha\omega^2$,
bringing the zero down toward the origin — and the pole $-\alpha$ comes down with it,
cancelling each other: the dipole story of §7.

---

## 부록 A. 극점 검증 스니펫

§1의 `closed-loop poles: -8.4 ± 8.57j, -8.0` 출력은 아래 검증 스크립트에서 나온 것이다.
게인 공식이 정말 요청한 극점을 만드는지, 특성다항식의 근을 수치로 재확인한다:

```python
import numpy as np
G, WN, ZETA = 9.81, 7.499, 0.01172      # 동정된 플랜트 (자유 진동 실험)
J, K, C = 1.0, WN**2, 2*ZETA*WN         # J=1 정규화: k=56.24, c=0.176
WD, ZD, AL = 12.0, 0.7, 8.0             # wn_des, zeta_des, alpha

KD = J*(2*ZD*WD + AL) - C               # 코드의 게인 공식 그대로
KP = J*(WD**2 + 2*AL*ZD*WD) - K
KI = J*AL*WD**2

print('closed-loop poles:', np.round(np.roots([J, C+KD, K+KP, KI]), 4))
print('requested        :', np.round(np.roots([1, 2*ZD*WD, WD**2]), 4), -AL)
```

```
closed-loop poles: [-8.4+8.5697j -8.4-8.5697j -8. +0.j]
requested        : [-8.4+8.5697j -8.4-8.5697j] -8.0
```

복소 쌍은 $-\zeta_{des}\omega_{des} \pm j\,\omega_{des}\sqrt{1-\zeta_{des}^2}
= -8.4 \pm 8.5697j$, 실수근은 $-\alpha = -8$. 게인이 특성다항식을
$(s^2 + 2\zeta\omega s + \omega^2)(s+\alpha)$와 같아지도록 역산된 것이므로 당연히
일치해야 하고, 일치한다.

## 부록 B. 용어: "동정(同定)"

**시스템 식별(system identification)**의 번역어로, 실험 데이터에서 모델 파라미터를
추정하는 일. 이 프로젝트에서는 자유 진동 실험
([`../docs/pendulum_model_identification.md`](../docs/pendulum_model_identification.md))의
감쇠 진동 파형에서 주기 → $\omega_n = 7.499\ \mathrm{rad/s}$, 로그 감소율 →
$\zeta = 0.01172$를 뽑았다. 모델 구조 $J\ddot\theta + c\dot\theta + k\theta = \tau$는
물리 법칙으로 미리 정하고 계수만 데이터로 채웠으므로 "동정한 모델"이라 부른다.
자유 진동 실험은 7.5 rad/s 근처의 거동만 보여주므로 그 대역을 크게 벗어난 곳에 닫힌
루프 극점을 두면 안 된다는 것(§3의 "모델 신뢰 대역")과, 게인 공식의 $-c$, $-k$ 상쇄가
동정값을 그대로 믿는 설계라 동정이 틀리면 실제 극점이 벗어난다는 것(§1)이 여기서
나온다.
