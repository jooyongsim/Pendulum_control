"""Deck content: Furuta pendulum — energy swing-up + LQR stabilisation.

This deck follows the torque/voltage-input Furuta rig (the Quanser-style
arrangement the design recipe assumes): the state is [theta, alpha, theta',
alpha'] with alpha = 0 upright, and the input is motor torque or voltage.

Our own rig is NOT that machine -- it is a stepper that takes an acceleration
command -- so the last slide states the difference and gives the gain our
acceleration-input model produces under the same Q and R. That number was
computed here, not quoted: see the note on slide 20.

Matrices are set as deck tables; the equation images go through matplotlib
mathtext, which has no matrix environment.
"""

FOOTER = "Sookmyung Women’s Univ. Autonomous Mechanical Systems Lab"

COVER = dict(
    title="Furuta 진자: Swing-up 과 LQR",
    subtitle="Energy Swing-up · LQR · Hysteresis Switching · Implementation",
    meta=["Rotary Inverted Pendulum 제어기 설계",
          "아래에서는 에너지로 올리고, 직립 근처에서는 상태 피드백으로 붙잡는다",
          "입력이 토크/전압인 장비 기준 — 우리 스테퍼 장비와의 차이는 마지막 장"],
)

EQ = {
    "state":   r"$x = [\ \theta\quad \alpha\quad \dot\theta\quad \dot\alpha\ ]^T"
               r"\ ,\qquad u = \tau$",
    "consts":  r"$a = J_r + m_pL_r^2,\quad b = m_pL_rl_p,\quad c = J_p,"
               r"\quad d = m_pgl_p,\quad \Delta = ac - b^2$",
    "eom":     r"$a\ddot\theta + b\ddot\alpha + B_r\dot\theta = \tau"
               r"\qquad\qquad b\ddot\theta + c\ddot\alpha + B_p\dot\alpha - d\alpha = 0$",
    "ss":      r"$\dot x = Ax + Bu$",
    "unstable": r"$A_{42} = \dfrac{ad}{\Delta} > 0"
                r"\qquad\Longrightarrow\qquad \mathrm{open\ loop\ unstable}$",
    "cost":    r"$J = \int_0^\infty \left( x^TQx + u^TRu \right) dt$",
    "qr":      r"$Q = \mathrm{diag}(5,\ 150,\ 1,\ 5)\ ,\qquad R = [\,0.5\,]"
               r"\ ,\qquad Q_{\alpha\alpha} \gg Q_{\theta\theta}$",
    "ulaw":    r"$\tau_{LQR} = -K\,[\ \theta - \theta_r\quad \alpha"
               r"\quad \dot\theta\quad \dot\alpha\ ]^T$",
    "ulaw2":   r"$\tau = -k_\theta(\theta - \theta_r) - k_\alpha\alpha"
               r" - k_{\dot\theta}\dot\theta - k_{\dot\alpha}\dot\alpha$",
    "energy":  r"$E = \frac{1}{2}J_p\dot\alpha^2 + m_pgl_p\cos\alpha"
               r"\ ,\qquad E_d = m_pgl_p\ ,\qquad e_E = E - E_d$",
    "swing":   r"$\ddot\theta_d = k_E(E - E_d)\,\dot\alpha\cos\alpha"
               r" - k_d\dot\theta - k_p\theta$",
    "edot":    r"$\dot E \simeq -b\,\dot\alpha\cos\alpha\,\ddot\theta"
               r"\qquad\Longrightarrow\qquad"
               r"\dot E \simeq -b\,k_E(E - E_d)\,\dot\alpha^2\cos^2\alpha$",
    "sat":     r"$\tau_{sw} = \mathrm{sat}\left[\ k_E(E - E_d)\dot\alpha\cos\alpha"
               r" - k_\theta\theta - k_{\dot\theta}\dot\theta\ \right]$",
    "switch":  r"$|\alpha| < 15^\circ\ \ \mathrm{and}\ \ |\dot\alpha| < 2\ \mathrm{rad/s}"
               r"\ \longrightarrow\ \mathrm{LQR}"
               r"\ ;\qquad |\alpha| > 25^\circ\ \longrightarrow\ \mathrm{swing\ up}$",
    "volt":    r"$\tau = K_uV_m - B_e\dot\theta\ ,"
               r"\qquad K_u = \dfrac{\eta K_gK_t}{R_m}\ ,"
               r"\qquad B_{r,\mathrm{eff}} = B_r + B_e$",
    "deriv":   r"$\dot\theta_f[k] = \beta\,\dot\theta_f[k-1]"
               r" + (1 - \beta)\dfrac{\theta[k] - \theta[k-1]}{T_s}$",
    "fs":      r"$f_s \geq 500\ \mathrm{Hz}\ ,\qquad"
               r"f_s = 1\ \mathrm{kHz}\ \ (T_s = 0.001\ \mathrm{s})\ \ \mathrm{preferred}$",
    "lqi":     r"$z = \int (\theta - \theta_r)\,dt\ ,\qquad"
               r"x_a = [\ \theta\quad \alpha\quad \dot\theta\quad \dot\alpha\quad z\ ]^T"
               r"\ ,\qquad u = -K_xx - K_iz$",
    "ours":    r"$\ddot\alpha = \dfrac{3g}{2l}\sin\alpha"
               r" + \dfrac{3r}{2l}\cos\alpha\ u\ ,\qquad \ddot\theta = u$",
}

SLIDES = [
    dict(
        n=2,
        title="1. Two Controllers, One Rig",
        lead="도립진자 제어기는 하나가 아니다 — 올리는 제어기와 세우는 제어기를 스위칭으로 묶는다",
        body=[
            ("term", "아래에 매달려 있을 때", "에너지 기반 swing-up 으로 흔들어 올린다"),
            ("term", "직립점 근처", "선형화 모델 위에서 LQR 로 붙잡는다"),
            ("term", "그 사이", "히스테리시스를 둔 switching — 경계에서 채터링을 막는다"),
        ],
        figure=("furuta_arch.jpg", "전체 구조. 이 강의는 왼쪽에서 오른쪽으로 따라간다"),
        red="이 강의의 부호 규약: 진자각 α = 0 이 직립, 팔각 θ = 0 이 기준 위치, 입력 u = τ 는 모터가 팔에 가하는 토크",
    ),
    dict(
        n=3,
        title="2. State and Input",
        lead="모든 설계가 이 네 개에서 출발한다",
        body=[("eq", "state")],
        table=dict(
            head=["기호", "뜻", "0 이 의미하는 것"],
            rows=[["θ", "rotary arm 각도 [rad]", "팔이 기준 위치"],
                  ["α", "pendulum 각도 [rad]", "직립 (매달림은 ±π)"],
                  ["θ̇", "arm 각속도", "팔 정지"],
                  ["α̇", "pendulum 각속도", "진자 정지"],
                  ["u = τ", "모터가 팔에 가하는 토크 [N·m]", "무구동"]],
        ),
        red="α = 0 을 직립으로 잡는 것이 핵심이다 — 부호를 반대로 잡으면 뒤에 나오는 swing-up 항의 부호까지 전부 뒤집힌다",
    ),
    dict(
        n=4,
        title="3. Linearised Model About Upright",
        lead="네 개의 합성 상수로 운동방정식을 정리한다",
        body=[("eq", "consts"), ("eq", "eom")],
        table=dict(
            head=["기호", "뜻"],
            rows=[["J_r", "rotary arm 측 등가 관성"],
                  ["J_p", "pendulum pivot 기준 관성"],
                  ["m_p, L_r, l_p", "진자 질량 · 팔 길이 · pivot에서 무게중심까지"],
                  ["B_r, B_p", "팔 · 진자의 점성 마찰"],
                  ["Δ = ac − b²", "두 식을 풀 때 공통으로 나오는 분모"]],
        ),
        red="b = m_p L_r l_p 가 팔과 진자를 이어 주는 항이다. b = 0 이면 두 식이 분리되어 팔로 진자를 세울 수 없다",
    ),
    dict(
        n=5,
        title="4. State-Space Form — and Why It Falls",
        lead="두 식을 θ̈, α̈ 에 대해 풀면 A, B 가 나온다",
        body=[("eq", "ss")],
        table=dict(
            head=["", "θ", "α", "θ̇", "α̇", "u"],
            rows=[["θ̇", "0", "0", "1", "0", "0"],
                  ["α̇", "0", "0", "0", "1", "0"],
                  ["θ̈", "0", "−bd/Δ", "−cB_r/Δ", "+bB_p/Δ", "c/Δ"],
                  ["α̈", "0", "+ad/Δ", "+bB_r/Δ", "−aB_p/Δ", "−b/Δ"]],
        ),
        tail_eq="unstable",
        red="A₄₂ 가 양수라는 것이 전부다 — 기울면 스스로 더 기운다. 이것은 안정한 시스템의 성능 개선이 아니라 불안정 플랜트의 안정화 문제다",
    ),
    dict(
        n=6,
        title="5. LQR — Writing Down the Cost",
        lead="이득을 찍는 대신 무엇을 얼마나 싫어하는지를 숫자로 적는다",
        body=[("eq", "cost"), ("eq", "qr")],
        table=dict(
            head=["", "무엇인가", "누가 정하나"],
            rows=[["A", "시스템 자체의 성질 (기울면 더 기운다)", "물리 / 동정"],
                  ["B", "입력이 상태에 주는 영향", "물리 / 동정"],
                  ["Q", "어떤 상태 오차를 싫어하는가", "설계자"],
                  ["R", "모터를 얼마나 아낄 것인가", "설계자"]],
        ),
        red="진자각 α 의 가중치를 가장 크게 잡는 것이 핵심이다. 팔은 좀 움직여도 되지만 진자는 넘어지면 끝이다",
    ),
    dict(
        n=7,
        title="6. MATLAB — Model and Controllability",
        lead="설계 전에 반드시 확인할 것: 입력 하나로 네 상태를 모두 움직일 수 있는가",
        body=[],
        table=dict(
            head=["MATLAB", "무엇을 하는가"],
            rows=[["a = Jr + mp*Lr^2;  b = mp*Lr*lp;", "합성 상수"],
                  ["c = Jp;  d = mp*g*lp;  Delta = a*c - b^2;", "합성 상수"],
                  ["A = [0 0 1 0; 0 0 0 1; ...];", "앞 장의 표를 그대로 옮긴다"],
                  ["B = [0; 0; c/Delta; -b/Delta];", "토크 입력"],
                  ["Co = ctrb(A,B);  rank(Co)", "4 가 나와야 한다"]],
        ),
        red="rank(ctrb) 가 4 미만이면 어떤 K 로도 그 모드를 움직일 수 없다 — 설계 이전의 문제이므로 여기서 걸러야 한다",
    ),
    dict(
        n=8,
        title="7. MATLAB — The Gain",
        lead="가중치를 정했으면 게인은 한 줄이다",
        body=[],
        table=dict(
            head=["MATLAB", "무엇을 하는가"],
            rows=[["Q = diag([5 150 1 5]);  R = 0.5;", "설계자의 선택"],
                  ["[K,S,P] = lqr(A,B,Q,R);", "K 는 게인, S 는 Riccati 해, P 는 폐루프 극"],
                  ["Acl = A - B*K;  eig(Acl)", "실수부가 모두 음수인지 검산"]],
        ),
        table2=dict(
            head=["확인할 것", "왜"],
            rows=[["k_α 가 가장 큰가", "진자를 세우는 항이 지배해야 정상이다"],
                  ["eig(A−BK) 실수부 < 0", "하나라도 양수면 그 K 는 버린다"],
                  ["|u| 가 구동기 한계 안인가", "LQR 은 한계를 스스로 지키지 않는다"]],
        ),
        red="eig(A − BK) 검산은 시뮬레이션을 돌리기 전에 끝난다 — 가장 싼 안전장치다",
    ),
    dict(
        n=9,
        title="8. The Control Law You Actually Run",
        lead="팔의 기준 위치 θ_r 를 빼 주는 것이 실제 구현의 차이다",
        body=[("eq", "ulaw"), ("eq", "ulaw2")],
        table=dict(
            head=["게인", "보는 상태", "역할"],
            rows=[["k_θ", "arm 위치", "팔을 기준 위치로 복귀"],
                  ["k_α", "pendulum 각도", "진자를 세우는 핵심 게인"],
                  ["k_θ̇", "arm 속도", "팔 움직임 damping"],
                  ["k_α̇", "pendulum 속도", "진자 흔들림 damping"]],
        ),
        red="α 에는 기준값을 빼지 않는다 — 직립이 곧 α = 0 이기 때문이다. θ 에만 θ_r 이 붙는다",
    ),
    dict(
        n=10,
        title="9. Swing-up — Aim at Energy, Not Position",
        lead="아래에 매달린 진자는 LQR 로 세울 수 없다. 위치가 아니라 에너지를 목표로 삼는다",
        body=[("eq", "energy")],
        table=dict(
            head=["상태", "E", "의미"],
            rows=[["직립 정지", "E = m_p g l_p = E_d", "목표 에너지"],
                  ["매달려 정지", "E = −m_p g l_p", "가장 낮은 상태"],
                  ["흔들리는 중", "운동 + 위치 에너지의 합", "e_E = E − E_d 를 0 으로"]],
        ),
        red="목표는 '진자를 위로 옮기는 것' 이 아니라 '에너지를 E_d 로 맞추는 것' 이다 — 에너지가 맞으면 진자는 직립점을 지나간다",
    ),
    dict(
        n=11,
        title="10. Why That Swing-up Law",
        lead="형태를 외우는 것이 아니라 에너지 변화율에서 나온다",
        body=[
            ("eq", "swing"),
            ("plain", "간략화된 동역학에서 에너지 변화율이 다음과 같기 때문이다:"),
            ("eq", "edot"),
        ],
        red="(E − E_d) 와 α̇²cos²α 의 곱이므로 부호가 항상 에너지 오차를 줄이는 쪽이다. 좌표 규약이 반대인 장비에서는 이 항의 부호를 뒤집어야 한다",
    ),
    dict(
        n=12,
        title="11. A Practical Swing-up Law",
        lead="토크를 직접 명령하는 장비라면 포화를 전제로 이렇게 시작한다",
        body=[("eq", "sat")],
        table=dict(
            head=["항", "없으면 무슨 일이 일어나는가"],
            rows=[["k_E(E − E_d)α̇cos α", "에너지를 올리는 본체 — 없으면 진자가 올라가지 않는다"],
                  ["−k_θθ", "팔이 한쪽으로 끝없이 돌아간다 (행정 제한이 있는 실기에서 치명적)"],
                  ["−k_θ̇θ̇", "팔이 과도하게 빨라진다"],
                  ["sat(·)", "명령이 모터 한계를 넘어 탈조·포화로 이어진다"]],
        ),
        red="−k_θθ 항이 실기에서 특히 중요하다. 시뮬레이터에서는 없어도 올라가지만 실제 장비는 팔 행정이 유한하다",
    ),
    dict(
        n=13,
        title="12. Switching with Hysteresis",
        lead="두 제어기를 하나의 경계로 나누면 그 경계에서 채터링이 생긴다",
        body=[("eq", "switch")],
        figure=("furuta_switch.jpg", "진입은 좁게, 탈출은 넓게 — 그 간격이 히스테리시스다"),
        red="각도만으로는 부족하다. |α| < 15° 라도 빠르게 지나가는 중이면 LQR 이 받자마자 놓치므로 |α̇| < 2 rad/s 를 함께 본다",
    ),
    dict(
        n=14,
        title="13. The Whole Controller",
        lead="매 주기 도는 코드의 골격",
        body=[],
        table=dict(
            head=["단계", "내용"],
            rows=[["① 모드 갱신", "balance_mode 가 아니면 진입 조건, 맞으면 탈출 조건만 본다"],
                  ["② LQR 분기", "x = [θ−θ_r; α; θ̇; α̇] 를 만들고 torque = −K*x"],
                  ["③ swing-up 분기", "E, E_d 계산 후 k_E·e_E·α̇·cos α − k_θθ − k_θ̇θ̇"],
                  ["④ 포화", "torque = min(max(torque, −τ_max), τ_max)"]],
        ),
        table2=dict(
            head=["구현 주의", ""],
            rows=[["balance_mode 는 persistent", "매 호출마다 초기화되면 히스테리시스가 사라진다"],
                  ["포화는 분기 밖에서 한 번", "두 모드 모두 같은 한계를 받아야 한다"]],
        ),
        red="모드 판정과 제어 계산을 한 덩어리로 쓰면 경계에서 디버깅이 불가능해진다 — 판정을 먼저, 계산을 나중에",
    ),
    dict(
        n=15,
        title="14. When the Input Is Motor Voltage",
        lead="실험 장비는 토크가 아니라 전압을 받는 경우가 많다",
        body=[("eq", "volt")],
        table=dict(
            head=["처리", "방법"],
            rows=[["토크/전압 게인", "B 의 c/Δ, −b/Δ 에 K_u 를 곱한다"],
                  ["역기전력 −B_e θ̇", "팔 마찰에 흡수: B_r,eff = B_r + B_e"],
                  ["결과", "LQR 출력 자체가 모터 전압 [V] 이 된다"]],
        ),
        red="이 처리를 빼먹으면 LQR 은 '토크' 를 내놓는데 드라이버는 '전압' 으로 읽는다 — 단위가 맞지 않아 게인이 통째로 어긋난다",
    ),
    dict(
        n=16,
        title="15. Tuning Q and R",
        lead="증상을 보고 가중치를 옮긴다",
        body=[],
        table=dict(
            head=["증상", "조정"],
            rows=[["진자가 직립점에서 너무 흔들림", "Q_α 증가"],
                  ["팔이 너무 크게 움직임", "Q_θ 증가"],
                  ["진자 속도가 너무 큼", "Q_α̇ 증가"],
                  ["팔이 빠르게 떨림", "Q_θ̇ 증가"],
                  ["모터 전압이 너무 큼", "R 증가"],
                  ["제어가 너무 느림", "R 감소"],
                  ["센서 잡음에 민감", "속도 가중치보다 추정기/필터를 먼저"]],
        ),
        table2=dict(
            head=["예", "Q", "R"],
            rows=[["기본", "diag(5, 150, 1, 5)", "0.5"],
                  ["더 공격적", "diag(10, 300, 2, 10)", "0.2"],
                  ["모터 보호", "diag(5, 150, 1, 5)", "1 ~ 5"]],
        ),
        red="Q 를 올리면 게인이 커지고 모터가 공격적으로, R 을 올리면 전체 게인이 작아진다. 상태마다 단위가 다르므로 숫자 자체로 중요도를 비교하면 안 된다",
    ),
    dict(
        n=17,
        title="16. Velocity from an Encoder",
        lead="엔코더는 각도만 준다. 단순 차분은 잡음이 심하다",
        body=[("eq", "deriv")],
        table=dict(
            head=["방법", "특징"],
            rows=[["단순 차분 (θ[k] − θ[k−1])/T_s", "구현은 쉽지만 잡음이 그대로 증폭"],
                  ["필터 미분, β = 0.9 ~ 0.98", "1 kHz 기준 출발점. 지연이 생긴다"],
                  ["관측기 / 칼만 필터", "모델을 쓰므로 잡음과 지연 모두 유리"]],
        ),
        red="β 를 키우면 잡음은 줄지만 지연이 커진다. 불안정 플랜트에서 지연은 곧 위상 여유 손실이므로 공짜가 아니다",
    ),
    dict(
        n=18,
        title="17. Sampling Rate and DLQR",
        lead="도립진자는 빠른 시스템이다",
        body=[("eq", "fs")],
        table=dict(
            head=["MATLAB", "무엇을 하는가"],
            rows=[["sysc = ss(A,B,eye(4),zeros(4,1));", "연속 모델"],
                  ["sysd = c2d(sysc, Ts);", "T_s = 0.001 s 로 이산화"],
                  ["Kd = dlqr(sysd.A, sysd.B, Q, R);", "이산 LQR"]],
        ),
        red="MCU / Simulink 구현이면 연속 LQR 을 그대로 쓰지 말고 처음부터 이산 설계하는 편이 낫다 — 샘플링 지연이 설계에 들어간다",
    ),
    dict(
        n=19,
        title="18. LQI — When Zero Is Not Zero",
        lead="마찰이나 센서 오프셋 때문에 팔이 정확히 0 으로 돌아오지 않는다면",
        body=[("eq", "lqi")],
        table=dict(
            head=["", "내용"],
            rows=[["추가하는 상태", "z = ∫(θ − θ_r)dt — 팔 각도 오차의 적분"],
                  ["확장 상태", "5차: [θ, α, θ̇, α̇, z]"],
                  ["제어법칙", "u = −K_x x − K_i z"],
                  ["순서", "LQR 로 먼저 안정화하고, 정상상태 오차가 남으면 그때 추가"]],
        ),
        red="첫 구현부터 LQI 로 가면 불안정한 플랜트에 적분기까지 얹는 셈이라 디버깅이 어려워진다 — 안정화 확인이 먼저다",
    ),
    dict(
        n=20,
        title="19. Our Rig Is a Different Machine",
        lead="이 강의는 입력이 토크/전압인 장비 기준이다. 우리 장비는 스테퍼라 입력이 다르다",
        body=[
            ("term", "우리 입력", "u = τ 가 아니라 u = θ̈_cmd — 드라이버가 받는 것은 STEP/DIR 펄스다"),
            ("eq", "ours"),
        ],
        table=dict(
            head=["", "이 강의의 장비", "우리 장비"],
            rows=[["입력", "토크 τ 또는 전압 V_m", "팔 가속도 θ̈_cmd"],
                  ["필요한 물성", "m_p, L_r, l_p, J_r, J_p, B_r, B_p", "3g/2l = 56.235, 3r/2l = 0.730 두 개"],
                  ["B 행렬", "c/Δ, −b/Δ 에 K_u", "θ̈ = u 라 훨씬 단순"],
                  ["게인 이식", "—", "토크용 K 를 그대로 쓸 수 없다"]],
        ),
        red="같은 Q = diag(5,150,1,5), R = 0.5 를 우리 가속도 입력 모델에 적용하면 K = [−3.16, 224.9, −3.76, 29.7] (ARE 잔차 1.3e−10, 폐루프 극 −8.45, −6.83, −1.42±1.03j)",
        note="우리 장비 수치는 Pendulum_control/simulator/rotary_model.py 의 동정값으로 직접 계산 · 질량과 관성모멘트는 동정에 사용되지 않았다",
    ),
    dict(
        n=21,
        title="20. One Slide to Remember",
        lead="구조와 순서",
        body=[
            ("term", "1. 모델", "직립 선형화 → A, B. 입력이 전압이면 K_u 를 B 에 반영"),
            ("term", "2. 가제어성", "rank(ctrb) = 4 확인. 여기서 걸리면 설계 이전의 문제"),
            ("term", "3. Q, R", "Q_α 를 가장 크게. diag(5,150,1,5), R = 0.5 에서 시작"),
            ("term", "4. K", "lqr(A,B,Q,R) → eig(A − BK) 로 검산"),
            ("term", "5. Swing-up", "에너지 법칙 + 팔 복귀 항 + 포화"),
            ("term", "6. Switching", "진입 15°/2 rad/s, 탈출 25°"),
            ("term", "7. 실장", "속도 필터, 1 kHz, DLQR, 필요하면 LQI"),
        ],
        red="게인 K 를 숫자로 확정하려면 m_p, L_r, l_p, J_r, J_p, B_r, B_p 와 입력이 토크인지 전압인지, 그리고 최대 전압이 필요하다",
    ),
]
