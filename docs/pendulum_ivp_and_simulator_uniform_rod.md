# 균일 막대 가정판 — 동정 파라미터, 초기값 문제, 제어 시뮬레이터

`pendulum_ivp_and_simulator.md` 와 같은 내용을, 진자가 **질량이 고르게 분포한 균일 막대**라는
가정 아래 다시 정리한 것이다. 즉 끝점에서 회전하는 길이 $L$, 질량 $m$ 의 막대이므로

$$\boxed{\ J = \tfrac{1}{3}mL^2, \qquad l_c = \tfrac{1}{2}L\ }$$

**이 가정이 바꾸는 것과 바꾸지 않는 것을 먼저 분명히 해 둔다.**

| | 바뀌는가 |
| --- | --- |
| $\omega_n,\ \zeta,\ \omega_d,\ \sigma$, 극점, 시상수 | **아니오** — 측정값 그대로 |
| 시뮬레이션 결과, 제어 게인, 4절의 모든 표 | **아니오** — 한 자리도 바뀌지 않는다 |
| $J,\ c,\ ml_c,\ mgl_c$ 의 **절대값** | **예** — 미정 스케일이 사라진다 |
| 무엇을 측정해야 하는가 | **예** — 관성모멘트 $J$ 대신 **질량 $m$** |
| 모델이 예측하는 **막대 길이** | **예** — $L = 261\ \mathrm{mm}$ 라는 검증 가능한 예측이 생긴다 |

원본 문서의 핵심 제약은 "자유 진동은 비율만 준다"였다. 균일 막대 가정은 **그 축퇴를 푼다.**

- 시뮬레이터: `pendulum_sim.py` (수정 불필요 — 3.2절)
- 동정 결과 출처: `pendulum_model_id.py`, `docs/pendulum_model_identification.md`
- 원본: `docs/pendulum_ivp_and_simulator.md`

---

## 0. 표기 주의

$l$ 이라는 기호가 두 뜻으로 쓰이므로 이 문서에서는 구분한다.

| 기호 | 뜻 | 값 |
| --- | --- | --- |
| $L$ | 막대 **전체 길이** (피벗 → 끝) | 260.95 mm |
| $l_c$ | 피벗 → **질량중심** 거리 $= L/2$ | 130.48 mm |
| $L_{\text{eff}}$ | 등가 단진자 길이 $= J/(ml_c)$ | 173.97 mm |

"$J = \tfrac13 ml^2$" 라고 쓸 때의 $l$ 은 **전체 길이** $L$ 이고,
$mgl\,\theta$ 항의 $l$ 은 **질량중심 거리** $l_c$ 다. 원본 문서는 후자를 $l$ 로 썼다.

균일 막대에서 세 길이의 관계는

$$L_{\text{eff}} = \frac{J}{m l_c} = \frac{\frac13 mL^2}{m\cdot\frac{L}{2}} = \frac{2}{3}L
\qquad\Longleftrightarrow\qquad
L = \frac{3}{2}L_{\text{eff}}$$

등가 단진자 길이는 막대 길이의 **2/3** 지점이다. 질량중심($L/2$)보다 **바깥쪽**이라는 점에 유의한다.

---

## 1. 같은 방정식의 세 표현

피벗이 고정된 진자의 선형화 방정식은

$$J\ddot{\theta} + c\,\dot{\theta} + mgl_c\,\theta = 0$$

양변을 $J$ 로 나누어 표준형과 맞추면

$$\ddot{\theta} + \underbrace{\frac{c}{J}}_{2\zeta\omega_n}\dot{\theta}
+ \underbrace{\frac{mgl_c}{J}}_{\omega_n^2}\,\theta = 0$$

### 1.1 균일 막대를 대입하면 $m$ 이 사라진다

$J = \frac13 mL^2$, $l_c = L/2$ 를 넣으면

$$\omega_n^2 = \frac{mgl_c}{J} = \frac{mg\cdot\frac{L}{2}}{\frac13 mL^2} = \frac{3g}{2L}$$

$$\boxed{\ \omega_n = \sqrt{\frac{3g}{2L}}\ }$$

**질량이 완전히 소거된다.** 중력 토크도 $\propto m$, 관성모멘트도 $\propto m$ 이므로
$\ddot\theta = \tau/J$ 에서 정확히 상쇄된다. 무거운 막대든 가벼운 막대든
**길이가 같으면 같은 주기로 흔들린다.**

감쇠 쪽은 $m$ 이 남는다.

$$2\zeta\omega_n = \frac{c}{J} = \frac{3c}{mL^2}
\qquad\Longrightarrow\qquad
\zeta = \frac{3c}{2mL^2\omega_n} = \frac{c}{m}\sqrt{\frac{3}{2gL^3}}\cdot\frac{3}{2}$$

세 표현을 나란히 두면 이렇다.

| | 파라미터 | 진동수를 주는 식 | 감쇠를 주는 식 |
| --- | --- | --- | --- |
| 표준형 | $\omega_n,\ \zeta$ | — | — |
| 물리형 (원본) | $c,\ J,\ ml_c$ | $\omega_n^2 = mgl_c/J$ | $2\zeta\omega_n = c/J$ |
| **균일 막대형** | $c,\ m,\ L$ | $\omega_n^2 = 3g/(2L)$ | $2\zeta\omega_n = 3c/(mL^2)$ |

### 1.2 측정값 대입

동정 결과 $\omega_n = 7.508\ \mathrm{rad/s}$, $\sigma \equiv \zeta\omega_n = 0.0798\ \mathrm{s^{-1}}$,
$g = 9.80665\ \mathrm{m/s^2}$ 이다. **이 값들은 원본과 동일하다.**

| 양 | 기호 | 값 |
| --- | --- | --- |
| 고유 진동수 | $\omega_n$ | $7.5080\ \mathrm{rad/s}$ |
| 감쇠비 | $\zeta = \sigma/\omega_n$ | $0.010629$ |
| 감쇠 고유 진동수 | $\omega_d = \omega_n\sqrt{1-\zeta^2}$ | $7.5076\ \mathrm{rad/s}$ |
| 감쇠율 | $\sigma = \zeta\omega_n$ | $0.0798\ \mathrm{s^{-1}}$ |
| 등가 단진자 길이 | $L_{\text{eff}} = g/\omega_n^2$ | $173.97\ \mathrm{mm}$ |

$\zeta = 0.011$ 로 매우 작아 $\omega_d/\omega_n = 0.99994$ 이다. 감쇠가 진동수를 사실상 바꾸지 않는다.

여기서부터가 원본과 갈라진다. $\omega_n^2 = 3g/(2L)$ 을 $L$ 에 대해 풀면

$$\boxed{\ L = \frac{3g}{2\omega_n^2} = \frac{3}{2}L_{\text{eff}}
= \frac{3 \times 9.80665}{2 \times 56.370} = 0.26095\ \mathrm{m} = \mathbf{260.95\ mm}\ }$$

$$l_c = \frac{L}{2} = 130.48\ \mathrm{mm}$$

### 1.3 축퇴가 풀린다 — 이 문서의 요점

원본 1.2절의 제약은 이랬다.

> 물리형에는 미지수가 $c,\ J,\ m,\ l$ 네 개지만 해는 $mgl/J$ 와 $c/J$ 두 조합에만 의존한다.
> $J$ 를 두 배로 하고 $c$ 와 $ml$ 도 동시에 두 배로 하면 완전히 같은 파형이 나온다.

균일 막대 가정은 미지수 네 개를 **$c,\ m,\ L$ 세 개로** 줄이고, 그중 **$L$ 을 자유 진동이 직접 결정한다.**

$$\omega_n \ \longrightarrow\ L = \frac{3g}{2\omega_n^2} \quad\text{(질량과 무관)}$$

남은 축퇴는 $m$ 방향 하나뿐이다. $m$ 을 두 배로 하면서 $c$ 도 두 배로 하면 같은 파형이 나온다.
**그런데 $m$ 은 저울로 재면 끝이다.**

| | 원본 (일반 물리형) | 이 문서 (균일 막대) |
| --- | --- | --- |
| 자유 진동이 주는 것 | 비율 $mgl_c/J$, $c/J$ 두 개 | 비율 두 개 **+ 길이 $L$** |
| 스케일을 고정하려면 | **관성모멘트 $J$ 를 측정** | **질량 $m$ 을 측정** |
| 측정 난이도 | 어려움 (비틀림 진자, 삼선 진자 등) | 쉬움 (전자저울 1초) |
| 모델이 하는 예측 | 없음 | $L = 261\ \mathrm{mm}$ — **검증 가능** |

이것이 균일 막대 가정의 실질적 이득이다. 어려운 측정 하나가 **쉬운 측정 하나 + 검증 하나**로 바뀐다.

### 1.4 가정의 검증 — 막대를 재 보면 된다

모델이 $L = 261\ \mathrm{mm}$ 를 예측하므로, **실제 막대 길이를 재서 맞춰 보면 가정이 타당한지 알 수 있다.**
일반적으로 질량 분포는 다음 비율로 드러난다.

$$\kappa \equiv \frac{L_{\text{eff}}}{L_{\text{measured}}} = \frac{J}{m\,l_c\,L_{\text{measured}}}$$

| $\kappa$ | 질량 분포 | 예측되는 $L_{\text{measured}}$ |
| --- | --- | --- |
| $2/3 = 0.667$ | **균일 막대** | $261\ \mathrm{mm}$ |
| $0.667 < \kappa < 1$ | 끝쪽으로 치우침 (끝단 추 등) | $174$–$261\ \mathrm{mm}$ |
| $1$ | 가벼운 봉 끝의 **점질량** | $174\ \mathrm{mm}$ |
| $\kappa < 2/3$ | 피벗 쪽으로 치우침 (뿌리 쪽 브래킷·엔코더 허브) | $> 261\ \mathrm{mm}$ |

읽는 법:

- 실측이 **261 mm에 가깝다** → 균일 막대 가정이 타당하다. 이 문서의 값을 그대로 쓴다.
- 실측이 **261 mm보다 짧다** → 질량이 끝쪽에 쏠려 있다. $J = \frac13 mL^2$ 은 $J$ 를 과대평가한다.
- 실측이 **261 mm보다 길다** → 피벗 근처 부품(엔코더 허브, 베어링 블록)의 질량이 무시할 수 없다.
  이 경우 막대만의 $m$ 을 재면 오차가 커진다.

**어긋났을 때의 보정은 가정을 버리는 쪽이 낫다.** 질량중심 $l_c$ 는 손가락으로 균형점을
찾으면 바로 재지므로, 분포 가정 없이

$$\boxed{\ J = m\,l_c\,L_{\text{eff}}\ }$$

로 쓰면 된다. $L_{\text{eff}} = 173.97\ \mathrm{mm}$ 는 **측정값**이고 $m,\ l_c$ 도 측정값이므로
이 식에는 어떤 가정도 들어 있지 않다. 균일 막대 가정은 $l_c = L/2$ 를 대신 넣어
"$l_c$ 조차 재지 않아도 되게" 해 주는 지름길일 뿐이다.

> $L_{\text{eff}} = 174\ \mathrm{mm}$ 는 **측정값**이고 $L = 261\ \mathrm{mm}$ 는 **가정의 귀결**이다.
> 둘을 혼동하지 않는다. 시뮬레이터가 쓰는 것은 언제나 전자다(3.2절).

### 1.5 질량 하나로 전부 결정된다

$L = 0.26095\ \mathrm{m}$ 를 고정하면 나머지는 $m$ 에 정비례한다.

$$J = \tfrac13 mL^2 = 2.2699\times10^{-2}\,m
\qquad
c = 2\sigma J = 3.6229\times10^{-3}\,m$$

$$m l_c = 0.13048\,m
\qquad
m g l_c = 1.27954\,m
\qquad
\tau_{\text{Coulomb}} = J F = 1.4713\times10^{-2}\,m$$

(단위는 각각 $\mathrm{kg\,m^2}$, $\mathrm{N\,m\,s/rad}$, $\mathrm{kg\,m}$, $\mathrm{N\,m}$;
$m$ 은 kg. 건마찰 각감속도는 $F = 0.648\ \mathrm{rad/s^2}$ 로 원본과 같다.)

| $m$ [g] | $J$ [kg m²] | $c$ [N m s/rad] | $m l_c$ [kg m] | $m g l_c$ [N m] | $\tau_{\text{Coulomb}}$ [N m] |
| --- | --- | --- | --- | --- | --- |
| 15 | $3.405\times10^{-4}$ | $5.434\times10^{-5}$ | $1.957\times10^{-3}$ | $1.919\times10^{-2}$ | $2.207\times10^{-4}$ |
| 20 | $4.540\times10^{-4}$ | $7.246\times10^{-5}$ | $2.610\times10^{-3}$ | $2.559\times10^{-2}$ | $2.943\times10^{-4}$ |
| 25 | $5.675\times10^{-4}$ | $9.057\times10^{-5}$ | $3.262\times10^{-3}$ | $3.199\times10^{-2}$ | $3.678\times10^{-4}$ |
| 30 | $6.810\times10^{-4}$ | $1.087\times10^{-4}$ | $3.914\times10^{-3}$ | $3.839\times10^{-2}$ | $4.414\times10^{-4}$ |
| 40 | $9.080\times10^{-4}$ | $1.449\times10^{-4}$ | $5.219\times10^{-3}$ | $5.118\times10^{-2}$ | $5.885\times10^{-4}$ |

원본의 예시($J = 5\times10^{-4}\ \mathrm{kg\,m^2}$)는 균일 막대 기준으로 $m = 22.0\ \mathrm{g}$ 에 해당한다.

**건마찰 토크가 이제 물리 단위로 나온다**는 점이 덤이다. $m = 20\ \mathrm{g}$ 이면
$\tau_{\text{Coulomb}} \approx 0.29\ \mathrm{mN\,m}$ 로, 베어링 사양과 직접 비교할 수 있다.

---

## 2. 초기값 문제

초기 조건은 두 경우 모두 $\theta(0) = \theta_0$, $\dot\theta(0) = v_0$ 로 둔다.

### 2.1 매달림 평형점 — 표준형으로 풀기

**원본과 완전히 동일하다.** 균일 막대 가정은 $\omega_n, \zeta$ 를 바꾸지 않기 때문이다.

$$s^2 + 2\zeta\omega_n s + \omega_n^2 = 0
\quad\Longrightarrow\quad
s = -\sigma \pm j\omega_d$$

$$\boxed{\ \theta(t) = e^{-\sigma t}\left[\theta_0\cos\omega_d t
+ \frac{v_0 + \sigma\theta_0}{\omega_d}\sin\omega_d t\right]}$$

**측정값 예시** ($\theta_0 = 20°$, $v_0 = 0$):

$$\theta(t) = 20°\,e^{-0.0798\,t}\left[\cos(7.5076\,t) + 0.01063\sin(7.5076\,t)\right]$$

### 2.2 매달림 평형점 — 균일 막대형으로 풀기

같은 문제를 $m, L, c$ 로 그대로 쓰면 특성방정식이

$$\tfrac13 mL^2\,s^2 + c\,s + \tfrac12 mgL = 0
\quad\Longrightarrow\quad
s = \frac{-c \pm \sqrt{c^2 - \tfrac23 m^2 g L^3}}{\tfrac23 mL^2}$$

판별식이 음수인 부족감쇠 조건은

$$\boxed{\ c^2 < \tfrac23\,m^2 g L^3\ }$$

이고, 이때

$$s = -\underbrace{\frac{3c}{2mL^2}}_{\sigma}
\ \pm\ j\underbrace{\sqrt{\frac{3g}{2L} - \frac{9c^2}{4m^2L^4}}}_{\omega_d}$$

$$\boxed{\ \theta(t) = e^{-\frac{3c}{2mL^2}t}\left[\theta_0\cos\omega_d t
+ \frac{v_0 + \frac{3c}{2mL^2}\theta_0}{\omega_d}\sin\omega_d t\right]}$$

**두 해는 같은 식이다.** $\sigma \leftrightarrow \dfrac{3c}{2mL^2}$,
$\omega_n^2 \leftrightarrow \dfrac{3g}{2L}$ 로 치환하면 정확히 겹친다.

부족감쇠 조건도 $c^2 < \frac23 m^2 gL^3 \Longleftrightarrow \zeta < 1$ 로 일치한다.
수치로 보면 $m = 0.020\ \mathrm{kg}$ 일 때 우변이 $4.65\times10^{-5}$,
실제 $c^2 = 5.25\times10^{-9}$ 이므로 조건을 **네 자릿수 여유로** 만족한다. 거의 무감쇠다.

세 표현의 성격 비교:

| | 표준형 | 물리형 (원본) | 균일 막대형 |
| --- | --- | --- | --- |
| 파라미터 수 | 2개 $(\omega_n, \zeta)$ | 4개 $(c, J, ml_c, g)$ | **3개** $(c, m, L)$ |
| 측정에서 직접 나오는가 | 예 | 아니오 — 비율만 | **$L$ 은 예, $m$ 은 저울로** |
| 설계에 바로 쓰이는가 | 예 | 아니오 | 아니오 |
| 물리적 해석 | 간접적 | 직접적 | **직접적 + 치수로 검증 가능** |

### 2.3 직립 평형점 — 부호 하나가 바꾸는 것

$\theta$ 를 직립에서 재면 중력이 복원력이 아니라 발산력이 된다.

$$\tfrac13 mL^2\ddot{\theta} + c\,\dot{\theta} - \tfrac12 mgL\,\theta = 0
\qquad\Longleftrightarrow\qquad
\ddot{\theta} + 2\sigma\dot{\theta} - \frac{3g}{2L}\,\theta = 0$$

$$s = -\sigma \pm \sqrt{\sigma^2 + \frac{3g}{2L}}$$

판별식이 **항상 양수**이므로 근이 언제나 실수이고 하나는 양수다. **수치는 원본과 같다.**

$$s_1 = +7.4286\ \mathrm{s^{-1}}, \qquad s_2 = -7.5882\ \mathrm{s^{-1}}$$

$$\boxed{\ \theta(t) = C_1 e^{s_1 t} + C_2 e^{s_2 t}, \qquad
C_1 = \frac{v_0 - s_2\theta_0}{s_1 - s_2}, \quad C_2 = \theta_0 - C_1\ }$$

$\theta_0 = 5°$, $v_0 = 0$ 이면 $C_1 = 2.527°$, $C_2 = 2.473°$ 이므로 1초 뒤
$\theta \approx 4.3\times10^{3}$ 도 — 진자는 이미 여러 바퀴 넘어가 있다.

발산하지 않으려면 $C_1 = 0$, 즉 $v_0 = s_2\theta_0 = -7.5882\,\theta_0$ 여야 한다.
$\theta_0 = 5°$ 라면 정확히 $-37.94\ \mathrm{deg/s}$. 이 조건을 만족하는 초기 조건의 집합은
상태 평면에서 **직선 하나**(안정 다양체)이고, 측도가 0이므로 실제로는 절대 일어나지 않는다.
되먹임이 하는 일은 이 직선을 평면 전체로 넓히는 것이다.

$$\tau = \frac{1}{s_1} = 135\ \mathrm{ms}, \qquad
t_{2\times} = \frac{\ln 2}{s_1} = 93\ \mathrm{ms}$$

### 2.4 길이가 난이도를 정한다

$\sigma \ll \omega_n$ 이므로 $s_1 \approx \omega_n = \sqrt{3g/(2L)}$ 이고, 따라서

$$\boxed{\ \tau \approx \frac{1}{\omega_n} = \sqrt{\frac{2L}{3g}} \ \propto\ \sqrt{L}\ }$$

**발산 시상수는 막대 길이의 제곱근에만 의존한다** — 질량, 재질, 단면과 무관하다.

| $L$ | $\tau = \sqrt{2L/3g}$ | 필요한 제어 주기 (대략 $\tau/10$) |
| --- | --- | --- |
| 261 mm (현재) | 133 ms | 13 ms → **100 Hz 권장** |
| 500 mm | 184 ms | 18 ms |
| 1000 mm | 261 ms | 26 ms |

시상수를 두 배로 늘리려면 막대를 **네 배** 길게 해야 한다. 손바닥 위에서 빗자루가
연필보다 훨씬 세우기 쉬운 이유가 이것이고, 동시에 **제어를 쉽게 만들려고 길이를 늘리는 것은
수익이 빠르게 줄어드는 전략**이라는 뜻이기도 하다.

---

## 3. 제어 시뮬레이터

### 3.1 플랜트 모델

피벗이 가속도 $a\ [\mathrm{m/s^2}]$ 로 움직이면 관성력이 추가 토크를 만든다.

$$\tfrac13 mL^2\,\ddot{\theta} = \tfrac12 mgL\sin\theta - \tfrac12 mL\,a\cos\theta
- c\,\dot{\theta} - \tau_{\text{Coulomb}}$$

양변을 $J = \frac13 mL^2$ 로 나누면 **$m$ 이 모든 항에서 소거된다.**

$$\boxed{\ \ddot{\theta} = \frac{3g}{2L}\sin\theta - 2\zeta\omega_n\dot{\theta}
- \frac{3a}{2L}\cos\theta - F\,\mathrm{sgn}(\dot{\theta})\ }$$

입력 계수가 새 파라미터가 아니라는 원본의 관찰이 여기서 더 선명해진다.

$$b = -\frac{ml_c}{J} = -\frac{3}{2L} = -\frac{1}{L_{\text{eff}}}
= -5.748\ \frac{\mathrm{rad/s^2}}{\mathrm{m/s^2}}$$

로터 각가속도로 바꾸면 $a = r\ddot{\alpha}$ 이므로 입력 계수는 $-\dfrac{3r}{2L}$ 이다.
**팔 반지름 $r$ 은 여전히 동정되지 않았다** — 직접 재야 하며 `PendulumParams.arm_radius_m` 로 노출돼 있다.

각도 규약은 원본과 같다. 직립이 $\theta = 0$, 매달림이 $\theta = \pi$ 이고

$$\theta_{\text{rad}} = \frac{\pi}{180}\left(\phi_{\text{firmware}} - 180\right)$$

### 3.2 시뮬레이터는 고칠 필요가 없다

중요한 점이라 따로 적는다. 시뮬레이터가 적분하는 식에 들어가는 값은

$$\omega_n,\qquad \zeta,\qquad L_{\text{eff}},\qquad F$$

네 개뿐이고, **넷 다 균일 막대 가정과 무관하게 측정에서 직접 나온다.**
$3g/(2L)$ 은 $\omega_n^2$ 과 같은 수이고 $3/(2L)$ 은 $1/L_{\text{eff}}$ 와 같은 수다.

$$\frac{3g}{2L} = \omega_n^2 = 56.370\ \mathrm{s^{-2}},
\qquad
\frac{3}{2L} = \frac{1}{L_{\text{eff}}} = 5.748\ \mathrm{m^{-1}}$$

따라서 **`pendulum_sim.py` 를 수정할 필요가 없고, 시뮬레이션 결과도 한 자리도 바뀌지 않는다.**
균일 막대 가정이 더해 주는 것은 시뮬레이터 밖의 정보 — $L$, $m$, 그리고 $J,\ c,\ \tau_{\text{Coulomb}}$ 의
절대값 — 이다. 이것들은 **모터 사양 선정, 베어링 마찰 비교, 실제 치수 검증**에 쓰인다.

### 3.3 사용법

기본 사용법은 원본과 동일하다.

```python
import pendulum_sim as ps

p = ps.PendulumParams()          # 동정된 기본값 (변경 없음)
print(p.summary())
p.arm_radius_m = 0.085           # 실제 치수로 교체할 것

# 2상태 설계: 진자만 세운다
k1, k2 = ps.place_poles(p, wn_des=12.0, zeta_des=0.8)

# 4상태 설계: 진자를 세우면서 로터도 제자리에 둔다
K = ps.place_poles4(p, [-10+10j, -10-10j, -2+1j, -2-1j])

cfg = ps.SimConfig(duration=15.0, control_hz=100.0, encoder_deg=0.3,
                   accel_max=6.0, rotor_limit_deg=90.0, theta0_deg=5.0)
log = ps.simulate(p, ps.StateFeedback(*K, u_max=6.0), cfg)
```

달라지는 부분은 **물리 파라미터를 뽑는 방법 하나**다. 원본은 측정한 $J$ 를 넣었지만,
균일 막대에서는 $J$ 를 $m$ 에서 계산한다. 기존 API 그대로 쓸 수 있다.

```python
L = 1.5 * p.L_eff                # 0.26095 m  — 균일 막대 가정의 귀결
m = 0.020                        # kg — 저울로 잰 값
phys = p.physical(m * L**2 / 3)  # J = (1/3) m L^2
# {'J': 4.540e-4, 'c': 7.246e-5, 'ml': 2.610e-3, 'mgl': 2.559e-2}
```

매번 쓰기 번거로우면 `PendulumParams` 에 다음을 얹어 두면 된다(현재 모듈에는 없다).

```python
@property
def rod_length(self):
    """Total rod length implied by the uniform-rod assumption [m]."""
    return 1.5 * self.L_eff

def uniform_rod(self, m):
    """Physical parameters for a uniform rod of mass m [kg] pivoted at one end."""
    L = self.rod_length
    out = self.physical(m * L ** 2 / 3.0)
    out.update(L=L, l_c=L / 2.0, m=m,
               tau_coulomb=out["J"] * self.coulomb_accel)
    return out
```

자유 진동 재현은 `ps.free_swing(p, theta0_deg=20.0)`, 해석해는
`ps.ivp_hanging(t, θ0, v0, p)` 와 `ps.ivp_upright(t, θ0, v0, p)` 로 얻는다 — 원본과 동일하다.

### 3.4 검증

시뮬레이터 자체의 검증 결과는 원본과 같다.

| 검사 | 결과 |
| --- | --- |
| 선형 플랜트 적분 vs 직립 해석해 | 상대 오차 $2.0\times10^{-4}$ |
| 매달림 해석해 vs 직접 적분 | 최대 오차 $7.5\times10^{-4}$ 도 (20° 진폭) |
| 비선형 자유 진동의 주기 | $0.8384$ s vs 동정 $0.8369$ s ($+0.19\%$) |
| 극점 배치 정확도 | 지정한 $\omega_n,\ \zeta$ 와 $10^{-6}$ 이내 일치 |
| PID($k_p=-k_1$, $k_d=-k_2$) vs 상태 피드백 | 동일 거동 확인 |

여기에 **균일 막대 가정 자체의 검증**이 하나 추가된다. 이것은 코드가 아니라 자로 하는 검사다.

| 검사 | 방법 | 판정 |
| --- | --- | --- |
| 막대 길이 | 피벗 축에서 끝까지 실측 | $261 \pm 10\ \mathrm{mm}$ 이면 가정 타당 |
| 질량중심 | 손가락으로 균형점 찾기 | $130\ \mathrm{mm}$ 부근이면 균일 |
| 질량 | 전자저울 | 1.5절 표에서 나머지 파라미터 확정 |

두 번째 검사가 특히 값싸고 유용하다. 균형점이 $L/2$ 에서 크게 벗어나면 균일 막대가 아니며,
그 경우 $l_c$ 를 실측값으로 바꾸고 $J = m\,l_c\,L_{\text{eff}}$ 로 잡는 편이 정확하다.

---

## 4. 시뮬레이터가 알려준 것

$\omega_n, \zeta, L_{\text{eff}}$ 가 그대로이므로 **원본 4절의 모든 수치가 그대로 유효하다.**
요점만 옮긴다.

### 4.1 각도만 되먹이면 로터가 흘러간다

| 설계 | 최종 $\vert\theta\vert$ | 로터 최대 변위 |
| --- | --- | --- |
| 2상태 (로터 되먹임 없음) | 0.150° | **2200°** (6바퀴) |
| 4상태 $[-10\pm10j,\ -2\pm j]$ | 0.833° | **26.2°** |

$k_3,\ k_4$ 를 2상태 게인 위에 손으로 얹으면 발산한다. 로터를 한쪽으로 옮기려면 진자를
**반대쪽으로** 먼저 기울여야 하는 비최소위상 성질 때문이며, 4차 시스템으로 한꺼번에
설계해야 한다. `place_poles4()` 가 그 역할을 한다.

### 4.2 설계 여유

4상태 설계 $[-10\pm10j,\ -2\pm j]$, 로터 한계 ±90°, 초기 기울기 5° 기준:

| 조건 | 성공 |
| --- | --- |
| 100 Hz / 50 Hz 루프 | 예 |
| **20 Hz 루프** | **아니오** |
| 지연 1스텝 (10 ms) | 예 |
| **지연 3스텝 (30 ms)** | **아니오** |
| 엔코더 1.0°, 구동기 ±2 m/s² | 예 |
| 초기 기울기 15° | 예 (로터 110° — 한계 초과) |

- **샘플링 하한은 20 Hz와 50 Hz 사이.** 2.4절의 $\tau = 133\ \mathrm{ms}$ 와 일관된다.
- **지연이 가장 치명적이다.** 30 ms면 실패한다.
- ±90° 로터 한계에서 복구 가능한 초기 기울기는 약 10° 이내다.

---

## 5. 한계

원본의 한계는 그대로 유지된다.

- 플랜트는 **진자만** 모델링한다. 로터는 $\ddot{\alpha} = u/r$ 로 운동학적으로만 적분되며,
  스테퍼의 가속 한계·탈조·L6474의 속도 하한은 들어 있지 않다.
- 회전형(Furuta) 특유의 원심력·코리올리 항을 생략했다. 소각도 밸런싱에서는 작지만,
  큰 스윙업 궤적에는 부족하다.
- $r$ 이 미측정이므로 $k_3,\ k_4$ 의 절대 크기는 $r$ 에 따라 달라진다.
  $k_1,\ k_2$ 는 $r$ 과 무관하다.
- 마찰 비대칭(17.4%)은 모사하지 않는다. `pendulum_model_identification.md` 5절 참고.

여기에 이 문서 고유의 한계가 하나 더해진다.

- **균일 막대 가정 자체가 검증되지 않았다.** $L = 261\ \mathrm{mm}$ 는 측정이 아니라
  가정의 귀결이고, 실제 진자는 균일 막대(261 mm)와 끝점 점질량(174 mm) **사이**에 있을 가능성이 높다.
  엔코더 허브·브래킷·체결부가 피벗 근처에 질량을 더하고, 끝단 마감이 반대로 작용한다.
  1.4절의 검사를 실제로 수행하기 전까지 $J,\ c,\ \tau_{\text{Coulomb}}$ 의 절대값은
  **수십 % 수준의 불확실성**을 갖는다고 보아야 한다.
- 다만 이 불확실성은 **제어 설계에 전파되지 않는다.** 시뮬레이터와 게인은 $\omega_n, \zeta,
  L_{\text{eff}}$ 만 쓰고 이들은 직접 측정값이기 때문이다(3.2절). 영향을 받는 것은
  모터 토크 사양 계산과 마찰 해석뿐이다.
