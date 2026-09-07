# 선형 진자 미분방정식의 지수해, pole 안정성, 그리고 PID 폐루프 응답

이 문서는 `simple_pendulum_free_response.md`의 선형화 모델을 한 단계 더 자세히 설명한다. 핵심은 다음 한 문장이다.

> 상수계수 선형 미분방정식에 \(e^{st}\)를 대입하면 미분 연산이 \(s\)의 다항식으로 바뀌며, 그 다항식의 근인 pole이 진동 주파수, 감쇠 속도, 안정성을 결정한다.

아래쪽 평형점 부근의 고정 피벗 진자를 먼저 다룬 뒤 PID 제어기가 추가될 때를 설명한다.

---

## 1. \(e^{st}\)인가, \(e^{(\sigma+j\omega)t}\)인가?

결론부터 말하면 둘은 같은 표현이다.

\[
\boxed{s=\sigma+j\omega},\qquad
\boxed{e^{st}=e^{(\sigma+j\omega)t}=e^{\sigma t}e^{j\omega t}}.
\]

Euler 공식

\[
e^{j\omega t}=\cos\omega t+j\sin\omega t
\]

을 적용하면

\[
e^{st}=e^{\sigma t}\big(\cos\omega t+j\sin\omega t\big).
\]

따라서

- \(e^{\sigma t}\): 진폭이 커지거나 작아지는 **포락선(envelope)**
- \(\cos\omega t\), \(\sin\omega t\): 각주파수 \(\omega\)의 **진동 성분**

이다. 실수 계수를 가진 물리계의 복소근은 항상 켤레쌍

\[
s_{1,2}=\sigma\pm j\omega
\]

으로 나타난다. 두 복소 지수해를 합치면 실제 측정 가능한 실수해

\[
\boxed{\theta(t)=e^{\sigma t}\left(A\cos\omega t+B\sin\omega t\right)}
\]

가 된다.

즉, 미분방정식을 풀 때는 일반적으로 \(e^{st}\)라고 놓는 것이 간단하고, 근을 구한 뒤 \(s=\sigma+j\omega\)로 분해하여 물리적 의미를 읽는다.

---

## 2. 왜 \(e^{st}\)를 대입하면 미분방정식이 풀리는가?

아래쪽 평형점 근처에서 점성 감쇠를 포함한 선형 진자 모델은

\[
J\ddot\theta+c\dot\theta+mgl_c\theta=0
\]

이다. \(J\)로 나누면

\[
\ddot\theta+\frac{c}{J}\dot\theta+rac{mgl_c}{J}\theta=0.
\]

해를

\[
\theta(t)=Ce^{st}
\]

라고 가정하면

\[
\dot\theta=sCe^{st},\qquad
\ddot\theta=s^2Ce^{st}.
\]

원래 방정식에 대입하여

\[
\left(s^2+\frac{c}{J}s+\frac{mgl_c}{J}\right)Ce^{st}=0
\]

을 얻는다. \(Ce^{st}\neq0\)인 비자명한 해를 원하므로

\[
\boxed{s^2+\frac{c}{J}s+\frac{mgl_c}{J}=0}
\]

이어야 한다. 이것이 **특성방정식(characteristic equation)** 이다.

지수함수는 미분해도 같은 함수에 상수 \(s\)만 곱해지는 고유함수이므로, 시간에 관한 미분방정식이 \(s\)에 관한 대수방정식으로 바뀐다. Laplace 변환에서도 미분 연산이 \(s\)의 곱으로 바뀌기 때문에 동일한 특성다항식이 분모에 나타난다.

---

## 3. \(\omega_n\), \(\zeta\), \(\sigma\), \(\omega_d\)의 관계

표준 2차 시스템을

\[
\boxed{\ddot\theta+2\zeta\omega_n\dot\theta+\omega_n^2\theta=0}
\]

로 쓴다. 진자 파라미터와 비교하면

\[
\boxed{\omega_n=\sqrt{\frac{mgl_c}{J}}},
\qquad
\boxed{\zeta=\frac{c}{2J\omega_n}}.
\]

또는 \(2\beta=c/J\)를 사용하면

\[
\beta=\zeta\omega_n.
\]

특성방정식은

\[
s^2+2\zeta\omega_ns+\omega_n^2=0
\]

이고 근은

\[
\boxed{s_{1,2}=-\zeta\omega_n
\pm\omega_n\sqrt{\zeta^2-1}}.
\]

### 3.1 부족감쇠: \(0<\zeta<1\)

\[
s_{1,2}=-\zeta\omega_n
\pm j\omega_n\sqrt{1-\zeta^2}.
\]

따라서

\[
\boxed{\sigma=-\zeta\omega_n},\qquad
\boxed{\omega_d=\omega_n\sqrt{1-\zeta^2}}.
\]

여기서

- \(\omega_n\): 감쇠가 없다고 가정한 고유 각주파수
- \(\omega_d\): 실제 감쇠 진동의 각주파수
- \(\sigma\): 지수 포락선의 증가/감소율

이다. 정지 상태에서 \(\theta_0\)만큼 들어 올렸다 놓으면

\[
\theta(t)=\theta_0e^{-\zeta\omega_nt}
\left[
\cos(\omega_dt)
+\frac{\zeta\omega_n}{\omega_d}\sin(\omega_dt)
\right].
\]

감쇠된 주기와 대표적인 정착시간은

\[
\boxed{T_d=\frac{2\pi}{\omega_d}},
\qquad
\boxed{T_s\approx\frac{4}{\zeta\omega_n}=\frac{4}{|\sigma|}}
\]

이다. 두 번째 식은 2% 정착 기준의 근사다. step 응답의 최대 overshoot 비율은 표준 2차 시스템에서

\[
\boxed{M_p=\exp\left(-\frac{\pi\zeta}{\sqrt{1-\zeta^2}}\right)}
\]

로 주어진다.

### 3.2 임계감쇠: \(\zeta=1\)

중근 \(s=-\omega_n\)을 가지며

\[
\theta(t)=(A+Bt)e^{-\omega_nt}.
\]

진동하지 않으면서 경계적으로 가장 빠른 이상적 2차 응답이다.

### 3.3 과감쇠: \(\zeta>1\)

두 pole이 서로 다른 음의 실수이다.

\[
s_{1,2}=-\zeta\omega_n
\pm\omega_n\sqrt{\zeta^2-1}.
\]

진동은 없지만, 원점에 더 가까운 느린 pole이 전체 정착시간을 지배하므로 임계감쇠보다 느려질 수 있다.

---

## 4. pole 실수부와 안정성

각 pole \(p_i\)에 대응하는 자유응답 성분은

\[
C_i e^{p_it}
\]

형태이다. \(p_i=\sigma_i+j\omega_i\)라면 진폭은 \(e^{\sigma_i t}\)에 비례한다.

| pole의 실수부 | 시간응답 | 연속시간 LTI 안정성 |
|---:|---|---|
| \(\sigma_i<0\) | 지수적으로 감소 | 해당 mode는 안정 |
| \(\sigma_i=0\) | 감소하지 않음 | 단순 허수축 pole이면 한계안정; 반복 pole이면 불안정 가능 |
| \(\sigma_i>0\) | 지수적으로 증가 | 불안정 |

전체 시스템이 점근적으로 안정하려면 **모든 폐루프 pole의 실수부가 음수**여야 한다.

Laplace 영역의 전달함수

\[
G(s)=\frac{N(s)}{D(s)}
\]

에서 pole은 \(D(s)=0\)의 근이다. 역 Laplace 변환을 부분분수로 전개하면

\[
y(t)=\sum_i C_i e^{p_it}
\]

가 되므로, Laplace의 pole 조건과 시간영역 지수해의 \(\sigma_i\) 조건은 같은 안정성 판정이다.

### 지배 pole과 반응속도

안정한 pole들 중 허수축, 즉 원점에 가장 가까운 pole을 **지배 pole(dominant pole)** 이라고 한다. 예를 들어

\[
p_1=-1,qquad p_2=-8,qquad p_3=-15
\]

이면 \(e^{-8t}\), \(e^{-15t}\) 성분은 빠르게 사라지고 장시간 응답은 거의 \(e^{-t}\)가 결정한다. 따라서 대략적인 정착시간은 지배 pole의 실수부 \(\sigma_{\rm dom}\)로

\[
T_s\approx\frac{4}{|\sigma_{\rm dom}|}
\]

처럼 읽을 수 있다.

복소 pole의 허수부는 진동 속도를, 실수부는 진동이 사라지는 속도를 결정한다. pole이 왼쪽으로 멀수록 일반적으로 빠르지만, 너무 빠른 pole을 요구하면 큰 제어입력, noise 증폭, actuator saturation과 모델 불확실성 문제가 생긴다.

---

## 5. 제어입력이 있는 선형 진자

아래쪽 평형점 부근에서 torque 또는 그에 비례하는 입력 \(u\)가 있으면 정규화된 식을

\[
\boxed{
\ddot\theta+2\beta\dot\theta+\omega_0^2\theta=b u
}
\]

로 쓸 수 있다. 직접 torque를 입력한다면 \(b=1/J\)이다. 영 초기조건에서 plant 전달함수는

\[
\boxed{P(s)=\frac{\Theta(s)}{U(s)}
=\frac{b}{s^2+2\beta s+\omega_0^2}}.
\]

제어기가 없을 때 분모는 기계 파라미터가 정한다. feedback 제어기는 폐루프 분모, 즉 특성방정식을 바꾸어 원하는 위치에 pole을 옮기는 역할을 한다.

---

## 6. PID 제어기를 추가하면 어떻게 바뀌는가?

오차와 PID를

\[
e=r-\theta,
\]

\[
u=K_pe+K_i\int e\,dt+K_d\dot e
\]

로 둔다. 제어기 전달함수는

\[
C(s)=K_p+\frac{K_i}{s}+K_ds
=\frac{K_ds^2+K_ps+K_i}{s}.
\]

단위 음의 feedback에서

\[
\frac{\Theta(s)}{R(s)}
=\frac{P(s)C(s)}{1+P(s)C(s)}
\]

이므로

\[
\boxed{
\frac{\Theta(s)}{R(s)}
=
\frac{b(K_ds^2+K_ps+K_i)}
{s^3+(2\beta+bK_d)s^2+(\omega_0^2+bK_p)s+bK_i}
}.
\]

폐루프 특성방정식은

\[
\boxed{
s^3+(2\beta+bK_d)s^2
+(\omega_0^2+bK_p)s+bK_i=0
}.
\]

PID의 적분 상태 하나가 추가되므로 plant가 2차여도 이상적인 폐루프 특성방정식은 3차가 된다.

3차 다항식을

\[
D(s)=s^3+a_2s^2+a_1s+a_0
\]

라고 쓰면 Routh--Hurwitz 조건에 의해 모든 pole이 열린 왼쪽 반평면에 있기 위한 필요충분조건은

\[
\boxed{a_2>0,\qquad a_1>0,\qquad a_0>0,\qquad a_2a_1>a_0}
\]

이다. 아래쪽 진자 PID에서는

\[
a_2=2\beta+bK_d,\quad
a_1=\omega_0^2+bK_p,\quad
a_0=bK_i
\]

를 대입한다. 단순히 모든 계수가 양수인 것만으로는 충분하지 않고 \(a_2a_1>a_0\)도 만족해야 한다. 실제 pole을 직접 계산하면 안정성과 함께 각 mode의 속도까지 확인할 수 있다.

### 각 gain이 계수에 미치는 직접적인 영향

- \(K_d\)는 \(s^2\) 계수에 들어가 유효 감쇠를 증가시키는 방향으로 작용한다.
- \(K_p\)는 \(s\) 계수에 들어가 유효 stiffness와 응답속도를 증가시키는 방향으로 작용한다.
- \(K_i\)는 상수항에 들어가 DC 오차를 제거하지만 세 번째 동적 mode를 만들고, 너무 크면 느린 진동이나 불안정을 만들 수 있다.

이는 직관적인 경향이며 gain 하나가 pole 하나에만 대응하는 것은 아니다. 실제로는 세 gain이 3개 pole의 위치를 함께 바꾼다.

> 주의: \(K_i=0\)인 PD 제어기는 처음부터 직접 대입하여
> \[
> s^2+(2\beta+bK_d)s+(\omega_0^2+bK_p)=0
> \]
> 로 다루는 것이 맞다. PID 식에 단순히 \(K_i=0\)을 넣으면 controller를 통분하면서 생긴 불필요한 \(s\) 인자가 나타나며, 이는 원점 pole-zero cancellation 대상이다.

---

## 7. 원하는 \(\omega_n\), \(\zeta\), 응답속도로 PID pole 배치하기

3차 폐루프에서 주된 응답을 원하는 2차 복소 pole 쌍으로 만들고, 나머지 한 pole을 더 빠른 실수 pole로 둘 수 있다.

원하는 지배 2차 성분을

\[
s^2+2\zeta_d\omega_{n,d}s+\omega_{n,d}^2
\]

로 두고, 세 번째 pole을 \(-\alpha\), \(\alpha>0\)에 둔다. 원하는 전체 특성다항식은

\[
\begin{aligned}
D_d(s)
&=(s^2+2\zeta_d\omega_{n,d}s+\omega_{n,d}^2)(s+\alpha)\\
&=s^3+(2\zeta_d\omega_{n,d}+\alpha)s^2\\
&\quad +(\omega_{n,d}^2+2\alpha\zeta_d\omega_{n,d})s
+\alpha\omega_{n,d}^2.
\end{aligned}
\]

이를 실제 폐루프 특성다항식과 계수 비교하면

\[
\boxed{
K_d=\frac{2\zeta_d\omega_{n,d}+\alpha-2\beta}{b}
}
\]

\[
\boxed{
K_p=\frac{\omega_{n,d}^2
+2\alpha\zeta_d\omega_{n,d}-\omega_0^2}{b}
}
\]

\[
\boxed{
K_i=\frac{\alpha\omega_{n,d}^2}{b}
}.
\]

보통 \(\alpha\)를 지배 pole 실수부 크기의 3~10배 정도로 두면 세 번째 mode가 비교적 빨리 사라지게 설계할 수 있다. 그러나 너무 큰 \(\alpha\)는 큰 gain과 제어입력을 요구하므로 actuator 한계 안에서 검증해야 한다.

원하는 2차 pole 쌍은

\[
p_{1,2}=-\zeta_d\omega_{n,d}
\pm j\omega_{n,d}\sqrt{1-\zeta_d^2}
\]

이다. 따라서

- \(\zeta_d\omega_{n,d}\)를 크게 하면 포락선이 더 빨리 감소한다.
- \(\omega_{n,d}\sqrt{1-\zeta_d^2}\)가 폐루프 진동 각주파수를 정한다.
- \(\zeta_d\)를 키우면 overshoot는 감소하지만 같은 \(\omega_{n,d}\)에서 진동 주기는 변한다.
- \(-\alpha\) mode가 충분히 빠르면 전체 3차 응답을 지배 2차 응답처럼 해석할 수 있다.

---

## 8. 시간응답을 mode별로 분해해서 보는 방법

PID 폐루프의 서로 다른 pole을 \(p_1,p_2,p_3\)라 하면, 자유응답 또는 입력응답의 과도성분은 부분분수 전개를 통해

\[
\boxed{
\theta_{\rm transient}(t)
=C_1e^{p_1t}+C_2e^{p_2t}+C_3e^{p_3t}
}
\]

처럼 분해된다. 복소 켤레 pole \(p_{1,2}=\sigma\pm j\omega_d\)는 합쳐서

\[
e^{\sigma t}\left(A\cos\omega_dt+B\sin\omega_dt\right)
\]

가 되고, 실수 pole \(p_3=-\alpha\)는

\[
C_3e^{-\alpha t}
\]

가 된다.

이 분해를 사용하면 3차 시스템도 다음처럼 읽을 수 있다.

1. 복소 pole 쌍의 실수부: 주된 진동이 사라지는 속도
2. 복소 pole 쌍의 허수부: 진동 주기
3. 추가 실수 pole: 적분기와 함께 생긴 별도 느린/빠른 mode
4. 허수축에 가장 가까운 pole: 최종 정착시간을 지배하는 mode
5. residue \(C_i\): 해당 mode가 실제 출력에 얼마나 강하게 나타나는지

pole이 느려도 대응 residue가 매우 작거나 zero에 의해 거의 상쇄되면 출력에서 잘 보이지 않을 수 있다. 반대로 모든 pole이 안정해도 폐루프 zero 때문에 undershoot나 큰 초기 변화가 생길 수 있다. 따라서 pole만으로 안정성과 주요 속도는 판단할 수 있지만, 전체 파형은 zero와 residue도 함께 확인해야 한다.

---

## 9. 위쪽 평형점의 inverted pendulum에서는 무엇이 달라지는가?

위쪽 평형점 주위의 작은 오차각을 \(\theta\)라 하면 중력 stiffness의 부호가 반대가 된다.

\[
\boxed{
\ddot\theta+2\beta\dot\theta-\omega_0^2\theta=b u
}.
\]

제어기가 없을 때 특성방정식은

\[
s^2+2\beta s-\omega_0^2=0.
\]

상수항이 음수이므로 두 실근 중 하나는 양수가 되고, \(e^{\sigma t}\)가 증가하여 open-loop가 불안정하다.

같은 음의 feedback PID를 적용하면 특성방정식은

\[
\boxed{
s^3+(2\beta+bK_d)s^2
+(bK_p-\omega_0^2)s+bK_i=0
}.
\]

아래쪽 평형점 식과 비교할 때 \(s\) 계수의 \(+\omega_0^2\)가 \(-\omega_0^2\)로 바뀐다. 원하는 3차 다항식의 계수를 \(a_2,a_1,a_0\)라 하면

\[
K_d=\frac{a_2-2\beta}{b},\qquad
K_p=\frac{a_1+\omega_0^2}{b},\qquad
K_i=\frac{a_0}{b}.
\]

따라서 upright stabilization에는 중력의 불안정한 negative stiffness를 이길 만큼 충분한 proportional action이 필요하다. 이 모델은 actuator가 직접 torque 또는 각가속도에 비례하는 입력을 준다는 이상화다. 현재 저장소의 stepper motor에는 속도 제한, 최소속도, 가속도 제한, 지연과 saturation이 있으므로 실제 gain은 이 이상 모델만으로 바로 적용하지 말고 시뮬레이션과 낮은 출력 실험으로 검증해야 한다.

---

## 10. 안정성 검사의 실제 순서

1. 동작점 주위에서 plant를 선형화한다. 아래쪽과 위쪽은 중력항 부호가 다르다.
2. controller를 포함한 폐루프 특성다항식을 구한다.
3. 모든 pole을 계산하고 \(\operatorname{Re}(p_i)<0\)인지 확인한다.
4. 지배 pole에서 예상 정착시간, 진동 주기, damping ratio를 읽는다.
5. 전달함수의 zero와 residue를 확인해 실제 파형에서 각 mode가 얼마나 보이는지 확인한다.
6. step/ramp/disturbance 응답을 수치적으로 시뮬레이션한다.
7. saturation, anti-windup, derivative filter, sampling delay, encoder 양자화 및 motor 동역학을 추가한다.
8. 실제 장치에서는 작은 gain과 제한된 명령부터 시험한다.

2차 시스템은 \(\omega_n\)과 \(\zeta\)로 직관적으로 요약할 수 있다. PID가 추가된 3차 이상 시스템은 먼저 전체 pole로 안정성을 판정하고, 충분히 빠른 비지배 pole이 있을 때만 지배 2차 pole 쌍의 \(\omega_{n,d}\), \(\zeta_d\)로 근사하여 반응속도와 overshoot를 설명하는 것이 정확하다.
