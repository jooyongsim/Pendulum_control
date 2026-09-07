# 고정 피벗 단진자의 자유응답: 선형 모델과 \(\sin\theta\) 비선형 모델

이 문서는 STEVAL-EDUKIT01 장치에서 **rotary arm/motor를 움직이지 않고**, 사람이 진자를 아래쪽 평형점에서 조금 들어 올린 뒤 정지 상태로 놓았을 때의 운동을 설명한다. 저장소의 `PendulumController.ino`가 보내는 `observation[0]`은 엔코더로 측정한 진자 각도이므로 실험과 직접 비교할 수 있다.

## 1. 모델, 좌표와 파라미터

아래쪽 평형점을 \(\theta=0\)으로 정하고, 놓는 순간을 \(t=0\)으로 둔다.

\[
\theta(0)=\theta_0,\qquad \dot\theta(0)=0.
\]

질량중심까지 거리 \(l_c\), 피벗에 대한 관성모멘트 \(J\), 점성 회전 감쇠계수 \(c\)를 쓰면

\[
J\ddot\theta+c\dot\theta+mgl_c\sin\theta=0.
\]

편의를 위해

\[
\omega_0^2=\frac{mgl_c}{J},\qquad 2\beta=\frac{c}{J},\qquad
\zeta=\frac{\beta}{\omega_0}
\]

로 정의한다. 따라서 비선형 모델은

\[
\boxed{\ddot\theta+2\beta\dot\theta+\omega_0^2\sin\theta=0}
\]

이다. 대표적인 두 물리 모델은 다음과 같다.

| 모델 | \(J\) | \(l_c\) | \(\omega_0\) |
|---|---:|---:|---:|
| 길이 \(L\)인 질량 없는 줄 끝의 점질량 | \(mL^2\) | \(L\) | \(\sqrt{g/L}\) |
| 길이 \(L\)인 균일 막대, 한쪽 끝 피벗 | \(mL^2/3\) | \(L/2\) | \(\sqrt{3g/(2L)}\) |

이 장치의 진자가 막대에 가깝다면 두 번째 식이 더 적절하다. 추가 추나 브래킷이 있으면 실제 \(J\)와 \(l_c\)를 사용하거나 측정 주기로부터 \(\omega_0\)를 추정한다.

엔코더 원시각 \(q\in[0,360^\circ)\)은 아래쪽 평형각 \(q_{\rm down}\)을 빼고 \([-180^\circ,180^\circ)\)로 감싸서 사용한다.

\[
\theta=\operatorname{wrap}_{[-\pi,\pi)}(q-q_{\rm down}).
\]

---

## 2. 감쇠를 고려하지 않는 경우

### 2.1 선형화 모델: 시간영역에서 직접 풀기

작은 각도에서 \(\sin\theta\approx\theta\)이므로

\[
\ddot\theta+\omega_0^2\theta=0.
\]

특성방정식 \(r^2+\omega_0^2=0\)의 근은 \(r=\pm j\omega_0\)이고,

\[
\theta(t)=A\cos(\omega_0t)+B\sin(\omega_0t).
\]

초기조건을 적용하면 \(A=\theta_0\), \(B=0\)이므로

\[
\boxed{\theta(t)=\theta_0\cos(\omega_0t)},\qquad
\boxed{T_{\rm lin}=\frac{2\pi}{\omega_0}}.
\]

### 2.2 선형화 모델: Laplace 변환으로 풀기

\(\mathcal L\{\ddot\theta\}=s^2\Theta-s\theta_0-\dot\theta(0)\)이므로

\[
(s^2+\omega_0^2)\Theta(s)=s\theta_0.
\]

따라서

\[
\Theta(s)=\theta_0\frac{s}{s^2+\omega_0^2}
\]

이고 역변환하면 같은 해 \(\theta(t)=\theta_0\cos(\omega_0t)\)를 얻는다. 자유응답에는 초기조건 항이 반드시 포함되어야 한다. 전달함수만 쓰고 초기조건을 0으로 놓으면 이 운동을 얻을 수 없다.

### 2.3 \(\sin\theta\) 비선형 모델: 해석해

\[
\ddot\theta+\omega_0^2\sin\theta=0.
\]

\(\dot\theta\)를 곱해 적분하면 에너지 보존식

\[
\frac12\dot\theta^2+\omega_0^2(1-\cos\theta)
=\omega_0^2(1-\cos\theta_0)
\]

을 얻는다. \(|\theta_0|<\pi\)인 진동 운동에서

\[
k=\sin\left(\frac{|\theta_0|}{2}\right)
\]

라 두면 Jacobi 타원함수로 쓴 정확한 해는

\[
\boxed{
\theta(t)=2\,\operatorname{sgn}(\theta_0)
\sin^{-1}\!\left[k\,\operatorname{cd}(\omega_0t,k)\right]
}
\]

이다. 여기서 \(\operatorname{cd}(u,k)=\operatorname{cn}(u,k)/\operatorname{dn}(u,k)\)이다. 정확한 주기는 제1종 완전 타원적분 \(K(k)\)를 사용하여

\[
\boxed{T_{\rm nl}=\frac{4K(k)}{\omega_0}}.
\]

작은 진폭에서는 \(k\to0\), \(K(k)\to\pi/2\)이므로 선형 주기로 수렴한다. 첫 보정은

\[
T_{\rm nl}\approx T_{\rm lin}
\left(1+\frac{\theta_0^2}{16}+\frac{11\theta_0^4}{3072}+\cdots\right)
\]

이며 각도는 radian이다. 진폭이 커질수록 실제 주기가 길어진다.

### 2.4 수치해

상태 \(x_1=\theta\), \(x_2=\dot\theta\)를 정의하면

\[
\dot x_1=x_2,\qquad \dot x_2=-\omega_0^2\sin x_1.
\]

이를 `scipy.integrate.solve_ivp`의 적응형 Runge--Kutta 방법으로 적분한다. 선형 모델은 두 번째 식의 \(\sin x_1\)만 \(x_1\)로 바꾸면 된다. 제공된 노트북은 닫힌형 해와 수치해의 최대 오차, 에너지 보존 및 진폭에 따른 주기 차이를 확인한다.

---

## 3. 점성 감쇠를 고려하는 경우

### 3.1 선형화 모델: 시간영역에서 직접 풀기

\[
\ddot\theta+2\beta\dot\theta+\omega_0^2\theta=0.
\]

특성근은

\[
r_{1,2}=-\beta\pm\sqrt{\beta^2-\omega_0^2}
\]

이다.

실험에서 흔한 부족감쇠 \(0<\zeta<1\)의 경우 \(\omega_d=\sqrt{\omega_0^2-\beta^2}\)이고, 정지 상태에서 놓으면

\[
\boxed{
\theta(t)=\theta_0e^{-\beta t}
\left[\cos(\omega_dt)+\frac{\beta}{\omega_d}\sin(\omega_dt)\right]
}.
\]

임계감쇠 \(\beta=\omega_0\)에서는

\[
\boxed{\theta(t)=\theta_0(1+\omega_0t)e^{-\omega_0t}}.
\]

과감쇠 \(\beta>\omega_0\)에서는 서로 다른 실근 \(r_1,r_2\)를 이용해

\[
\boxed{
\theta(t)=\theta_0\frac{-r_2e^{r_1t}+r_1e^{r_2t}}{r_1-r_2}
}.
\]

### 3.2 선형화 모델: Laplace 변환으로 풀기

초기조건을 포함하면

\[
(s^2+2\beta s+\omega_0^2)\Theta(s)
=(s+2\beta)\theta_0.
\]

즉,

\[
\boxed{
\Theta(s)=\theta_0\frac{s+2\beta}{s^2+2\beta s+\omega_0^2}
}.
\]

분모의 근에 따라 부분분수 전개하면 위의 부족/임계/과감쇠 시간응답이 각각 나온다.

### 3.3 \(\sin\theta\) 비선형 모델: 해석적으로 무엇이 가능한가

\[
\ddot\theta+2\beta\dot\theta+\omega_0^2\sin\theta=0.
\]

감쇠가 있으면

\[
\frac{d}{dt}
\left[\frac12\dot\theta^2+\omega_0^2(1-\cos\theta)\right]
=-2\beta\dot\theta^2\le0.
\]

따라서 에너지가 단조 감소한다는 정확한 해석 관계는 얻지만, 일반 초기조건에 대해 무감쇠 때와 같은 타원함수 닫힌형 해는 없다. 즉 **일반적인 감쇠 비선형 진자의 정확한 elementary/elliptic closed form을 제시할 수 없으며 수치적분이 기준해**이다.

약한 감쇠이고 각도가 작아지는 구간에서는 진폭 포락선을

\[
A(t)\approx A_0e^{-\beta t}
\]

로 두고, 순간 진폭에 따른 주기를

\[
T(A)\approx\frac{4K(\sin(A/2))}{\omega_0}
\]

로 보는 느린-진폭 근사가 유용하다. 이것은 근사해이지 정확한 해가 아니다. 작은 각도에서는 앞 절의 선형 감쇠 해가 가장 간단한 해석 근사다.

### 3.4 수치해

\[
\dot x_1=x_2,\qquad
\dot x_2=-2\beta x_2-\omega_0^2\sin x_1.
\]

`solve_ivp`로 적분하고, 같은 초기조건을 가진 선형 감쇠 해와 비교한다. 수치 품질은 허용오차를 줄였을 때 결과가 수렴하는지, 그리고 계산된 에너지의 변화가 \(-2\beta\dot\theta^2\)와 일치하는지로 검사할 수 있다.

---

## 4. 측정과 모델 비교 절차

1. rotary motor에는 속도/위치 명령을 보내지 않고 정지시킨다.
2. 진자를 완전히 늘어뜨린 상태에서 여러 샘플의 원형 평균으로 \(q_{\rm down}\)을 정한다.
3. 진자를 원하는 작은 각도만큼 손으로 들고 정지시킨다.
4. 기록을 시작한 뒤 놓는다. 손으로 미는 초기속도가 생기지 않도록 한다.
5. 각도를 unwrap하고 \(q_{\rm down}\)을 빼서 \(\theta(t)\)를 만든다.
6. 첫 두 양의 피크 사이 시간으로 주기를, 같은 부호 피크들의 로그 감쇠율로 \(\beta\)를 추정한다:

\[
\beta\approx\frac{1}{t_{n+1}-t_n}
\ln\left|\frac{\theta(t_n)}{\theta(t_{n+1})}\right|.
\]

7. 작은 진폭 데이터에는 선형 모델을, 큰 초기각 데이터에는 \(\sin\theta\) 모델을 비교한다.

엔코더 분해능은 `ENC_STEPS_PER_ROTATION = 1200`, 즉 \(0.3^\circ\)/count이다. 펌웨어의 USB serial 응답은 호스트 조회에 의해 생성되므로 샘플 간격은 완전히 균일하지 않다. 반드시 MCU timestamp 또는 host monotonic time을 저장하고, 미분이 필요하면 먼저 보간/필터링한다.

## 5. 제공 파일

- `simple_pendulum_simulation.ipynb`: 네 모델의 해석해/수치해 비교 및 CSV 측정 데이터와의 선택적 비교
- `encoder_release_experiment.ipynb`: 장치 연결, 아래쪽 영점 보정, 손으로 들어 올린 뒤 놓는 실험 기록, CSV 저장과 사후 plot
- `encoder_live_monitor.py`: 터미널에서 실행하는 실시간 plot + CSV 저장 버전

> 안전: 이 실험에서는 모터 명령을 보내지 않는다. 그래도 rotary arm이 자유롭게 움직이거나 기존 명령이 남아 있을 가능성에 대비해 전원을 넣기 전에 작업 공간을 비우고, 필요하면 모터 전원을 분리하거나 firmware의 hard stop을 먼저 실행한다.
