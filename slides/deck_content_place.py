"""Deck content: where the controller lives, how place4/place2 pick K, and LQR.

Built from the code that actually runs:
  Pendulum_control/simulator/rotary_model.py            place4, place2, Balancer
  Pendulum_control/simulator/rotary_balance_realtime.py the simulator that uses them
  Pendulum_control/swing-up/balance_control.py          the copy that runs on the rig

Every gain and pole in the tables was recomputed from those files, not quoted.
Matrices are set as deck tables: the equation images go through matplotlib
mathtext, which has no matrix environment.
"""

FOOTER = "Sookmyung Women’s Univ. Autonomous Mechanical Systems Lab"

COVER = dict(
    title="상태 피드백 이득은 어떻게 정하는가",
    subtitle="place4 · place2 · Ackermann · LQR",
    meta=["Rotary Inverted Pendulum · 우리 코드 읽기",
          "제어식은 u = −Kx 하나 — 차이는 K를 정하는 기준",
          "모든 이득과 극점은 현재 코드에서 다시 계산한 값"],
)

EQ = {
    "state":   r"$x = [\ \theta\quad \dot\theta\quad \phi\quad \dot\phi\ ]^T"
               r"\qquad u = \ddot\phi\ \ [\mathrm{rad/s^2}]$",
    "plant":   r"$\ddot\theta = a\theta - c\dot\theta + bu"
               r"\qquad\qquad \ddot\phi = u$",
    "ss":      r"$\dot x = Ax + Bu\ ,\qquad u = -Kx"
               r"\qquad\Longrightarrow\qquad \dot x = (A - BK)x$",
    "ctrb":    r"$\mathcal{C} = [\ B\quad AB\quad A^2B\quad A^3B\ ]"
               r"\ ,\qquad \mathrm{rank}\,\mathcal{C} = 4\ \ (\det \mathcal{C} = 1685)$",
    "poly":    r"$p(s) = \prod_{i=1}^{4}(s - p_i)"
               r" = s^4 + \alpha_3 s^3 + \alpha_2 s^2 + \alpha_1 s + \alpha_0$",
    "polyA":   r"$p(A) = A^4 + \alpha_3 A^3 + \alpha_2 A^2 + \alpha_1 A + \alpha_0 I$",
    "acker":   r"$K = e_4^T\,\mathcal{C}^{-1}p(A)\ ,"
               r"\qquad e_4 = [\ 0\quad 0\quad 0\quad 1\ ]^T$",
    "u4":      r"$u = -k_1\theta - k_2\dot\theta - k_3\phi - k_4\dot\phi$",
    "u2":      r"$u = -k_1\theta - k_2\dot\theta$",
    "closed2": r"$\ddot\theta + (c + bk_2)\,\dot\theta + (bk_1 - a)\,\theta = 0$",
    "target2": r"$\ddot\theta + 2\zeta\omega_n\dot\theta + \omega_n^2\theta = 0$",
    "gains2":  r"$k_1 = \dfrac{\omega_n^2 + a}{b}\ ,"
               r"\qquad\qquad k_2 = \dfrac{2\zeta\omega_n - c}{b}$",
    "cost":    r"$J = \int_0^\infty \left( x^TQx + u^TRu \right) dt$",
    "costdiag": r"$J = \int_0^\infty \left( q_\theta\theta^2 + q_{\dot\theta}\dot\theta^2"
                r" + q_\phi\phi^2 + q_{\dot\phi}\dot\phi^2 + r\,u^2 \right) dt$",
    "are":     r"$A^TP + PA - PBR^{-1}B^TP + Q = 0$",
    "kform":   r"$K = R^{-1}B^TP$",
}

SLIDES = [
    dict(
        n=2,
        title="1. Where the Controller Actually Is",
        lead="place4() 와 place2() 는 이득 K 를 계산하는 함수다. 제어는 매 주기 도는 u = −K·x 에서 일어난다",
        body=[],
        table=dict(
            head=["구분", "위치", "역할"],
            rows=[["시뮬 이득 생성", "rotary_balance_realtime.py:38", "K2, K4 = rm.place2(), rm.place4()"],
                  ["시뮬 제어기 연결", "rotary_balance_realtime.py:43", "rm.Balancer(K) 를 루프에 전달"],
                  ["이득 계산 본체", "rotary_model.py:70, 80", "place4(), place2()"],
                  ["시뮬 제어식", "rotary_model.py:177", "float(-self.K @ z)"],
                  ["실기 이득 계산", "balance_control.py:51, 124", "자체 place4(b, poles)"],
                  ["실기 제어식", "balance_control.py:193", "u = float(-K @ x)"]],
        ),
        red="실기는 194행에서 u 를 펄스 가속도로 바꿔 제한하고 199행에서 전송한다. damp 모드는 191행의 별도 식을 쓰고, 실기 파일에는 place2() 가 아예 없다",
    ),
    dict(
        n=3,
        title="2. The Model Both Functions Design On",
        lead="θ = 직립에서 기울어진 막대각, φ = 팔각, u = 팔의 각가속도 명령",
        body=[("eq", "state"), ("eq", "plant")],
        table=dict(
            head=["", "θ", "θ̇", "φ", "φ̇", "u"],
            rows=[["θ̇", "0", "1", "0", "0", "0"],
                  ["θ̈", "a", "−c", "0", "0", "b"],
                  ["φ̇", "0", "0", "0", "1", "0"],
                  ["φ̈", "0", "0", "0", "0", "1"]],
        ),
        table2=dict(
            head=["상수", "값", "의미"],
            rows=[["a = A_G = 3g/2l", "56.235 1/s²", "기울면 더 기운다 — 불안정의 원인"],
                  ["c = C_D = 2ζωn", "0.176 1/s", "동정된 점성 감쇠"],
                  ["b = B_U = 3r/2l", "0.730", "팔 가속이 막대에 주는 영향 (개루프 시험)"]],
        ),
        red="A 의 (2,1) 성분이 +56.2 로 양수인 것이 전부다. 이것 때문에 안정화가 필요하고, 두 함수는 그 K 를 서로 다른 방식으로 구한다",
    ),
    dict(
        n=4,
        title="3. What Both Functions Are For",
        lead="제어식을 넣으면 폐루프가 하나의 행렬로 정리된다",
        body=[
            ("eq", "ss"),
            ("term", "place4()", "네 상태를 모두 써서 A−BK 의 고유값을 지정한 위치에 놓는다"),
            ("term", "place2()", "막대 두 상태만 써서 표준 2차 응답에 맞춘다 (PD 형태)"),
            ("term", "LQR", "극을 찍지 않고 비용 J 를 최소화한다 — 극은 결과로 나온다"),
        ],
        red="세 방법 모두 결과물은 같은 모양의 K 하나다. 실행 코드만 보면 무엇으로 만든 K 인지 구별할 수 없다",
    ),
    dict(
        n=5,
        title="4. place4 — Step 1: Controllability",
        lead="입력 하나로 네 상태를 모두 움직일 수 있는가?",
        body=[("eq", "ctrb")],
        table=dict(
            head=["코드", "수식"],
            rows=[["np.linalg.matrix_power(A, i) @ B", "AⁱB (i = 0…3)"],
                  ["np.hstack([...])", "네 열을 가로로 붙여 𝒞 구성"],
                  ["if abs(det(ctrb)) < 1e−9: raise", "실기 파일에만 있는 방어 (balance_control.py:53)"]],
        ),
        red="Ackermann 공식은 𝒞 의 역행렬을 쓴다. 가역이 아니면 그 극점 배치는 애초에 불가능하다",
    ),
    dict(
        n=6,
        title="5. place4 — Step 2: The Desired Polynomial",
        lead="원하는 극을 근으로 갖는 다항식을 만들고, s 자리에 행렬 A 를 대입한다",
        body=[("eq", "poly"), ("eq", "polyA")],
        table=dict(
            head=["코드", "무엇인가"],
            rows=[["np.poly(poles)", "계수 [1, α₃, α₂, α₁, α₀] — 최고차부터"],
                  ["np.real(...)", "극이 켤레쌍이라 계수는 실수. 수치 잔여 허수부를 뗀다"],
                  ["phi = phi @ A + c * np.eye(4)", "Horner 방식: 0에서 시작해 ×A + 다음계수 반복"]],
        ),
        red="코드 변수 phi 는 팔 각도 φ 가 아니라 다항식 행렬 p(A) 다. 반복문의 c 도 감쇠계수 C_D 와 무관한 다항식 계수다 — 이름이 겹쳐 헷갈리기 쉽다",
    ),
    dict(
        n=7,
        title="6. place4 — Step 3: Ackermann",
        lead="마지막 한 줄이 공식 전체다",
        body=[("eq", "acker"), ("eq", "u4")],
        table=dict(
            head=["코드", "수식", "왜 이렇게"],
            rows=[["np.linalg.solve(ctrb.T, e₄)", "𝒞ᵀy = e₄ 를 푼 y", "yᵀ = e₄ᵀ𝒞⁻¹ 과 같다"],
                  ["... @ phi", "yᵀ p(A) = K", "역행렬을 만들지 않아 수치적으로 안전"]],
        ),
        red="결과는 막대의 균형과 팔의 원점 복귀를 한 번에 담당하는 K 하나다",
    ),
    dict(
        n=8,
        title="7. place2 — Coefficient Matching",
        lead="팔을 아예 빼고 막대만 본다. 뒤 두 이득을 0 으로 두는 것이 그 뜻이다",
        body=[
            ("eq", "u2"),
            ("plain", "막대 방정식에 대입하면 폐루프는"),
            ("eq", "closed2"),
            ("plain", "원하는 표준 2차 응답과 계수를 비교하면"),
            ("eq", "target2"),
            ("eq", "gains2"),
        ],
        red="k₁ 에서 a 를 더하는 것은 불안정한 +a 를 먼저 상쇄하고 그 위에 ωn² 을 새로 얹기 때문이고, k₂ 에서 c 를 빼는 것은 이미 있는 자연 감쇠만큼 덜 넣으면 되기 때문이다",
    ),
    dict(
        n=9,
        title="8. What place2 Costs",
        lead="인자 wn = 12, zeta = 0.7 은 원하는 폐루프 설정값이다. 동정값 WN = 7.499, ZETA = 0.01172 과 혼동하면 안 된다",
        body=[],
        figure=("place_runaway.jpg", "5° 에서 놓았을 때. 막대는 둘 다 서지만 팔은 place2 에서 돌아오지 않는다"),
        red="팔의 위치와 속도를 피드백하지 않으므로 4상태로 보면 극 두 개가 원점에 남는다 — 막대는 서고 팔은 계속 밀려난다",
    ),
    dict(
        n=10,
        title="9. The Numbers in the Current Code",
        lead="세 설계를 현재 코드로 다시 계산한 결과",
        body=[],
        table=dict(
            head=["제어기", "K = [k₁, k₂, k₃, k₄]", "폐루프 극"],
            rows=[["시뮬 place2()", "[274.295, 22.773, 0, 0]", "−8.4 ± 8.57j, 0, 0"],
                  ["시뮬 place4()", "[236.227, 26.537, −4.161, −4.548]", "−6 ± 6j, −1.5 ± 1j"],
                  ["실기 기본 극", "[365.031, 41.626, −11.381, −10.563]", "−8 ± 8j, −2 ± 1j"]],
        ),
        figure=("place_poles.jpg", "같은 플랜트, 다른 목표 극. 붉은 영역이 불안정 영역"),
        red="두 파일은 같은 4상태 설계법을 쓰지만 기본 목표 극이 달라 이득이 1.5 배 차이 난다 — 둘 중 하나는 오래된 값으로 보인다",
    ),
    dict(
        n=11,
        title="10. LQR — Same Law, Different Criterion",
        lead="LQR 도 u = −Kx 다. 바뀌는 것은 K 를 정하는 기준뿐이다",
        body=[],
        table=dict(
            head=["구분", "place4() — 극 배치", "LQR"],
            rows=[["설계자가 지정", "원하는 폐루프 극", "상태·입력의 비용 Q, R"],
                  ["설계 목표", "A−BK 의 고유값을 그 위치로", "누적 비용 J 를 최소화"],
                  ["조정 방식", "극을 옮겨 속도·감쇠 조절", "무엇을 얼마나 줄일지 조절"],
                  ["폐루프 극", "직접 지정", "최적화 결과로 결정"],
                  ["다입력(MIMO)", "K 가 무한히 많아 기준이 없음", "항상 유일한 최적해"]],
        ),
        red="극 배치는 제어 입력이 얼마나 필요한지를 비용으로 보지 않는다. 극을 왼쪽으로 옮길수록 이득이 커지지만 얼마나 커질지는 해보기 전에는 모른다",
    ),
    dict(
        n=12,
        title="11. LQR — Writing Down the Cost",
        lead="이득을 찍는 대신 무엇을 얼마나 싫어하는지 적는다",
        body=[
            ("eq", "cost"),
            ("plain", "Q 를 대각행렬로 두면 이 시스템의 비용은 다음과 같이 풀어 쓸 수 있다:"),
            ("eq", "costdiag"),
        ],
        table=dict(
            head=["가중치", "무엇에 벌점을 주는가"],
            rows=[["q_θ", "막대가 기울어지는 것"],
                  ["q_φ", "팔이 원점에서 벗어나는 것"],
                  ["q_θ̇, q_φ̇", "막대와 팔의 움직임"],
                  ["r", "큰 팔 가속도 명령"]],
        ),
        red="상태마다 단위와 크기가 다르므로 가중치 숫자만 보고 중요도를 비교하면 안 된다 — 같은 조건에서 상대적으로 키웠을 때의 경향으로만 읽어야 한다",
    ),
    dict(
        n=13,
        title="12. LQR — Riccati Gives the Gain",
        lead="비용을 최소화하는 조건이 대수 Riccati 방정식이고, 그 해 P 가 이득을 준다",
        body=[("eq", "are"), ("eq", "kform")],
        table=dict(
            head=["Q = diag(150, 5, 5, 1)", "K = [k₁, k₂, k₃, k₄]", "폐루프 극"],
            rows=[["R = 0.5", "[224.91, 29.72, −3.16, −3.76]", "−8.45, −6.82, −1.42 ± 1.03j"],
                  ["R = 5", "[187.91, 24.78, −1.00, −1.75]", "−7.78, −7.25, −0.74 ± 0.67j"],
                  ["q_φ 를 50 으로 (R = 0.5)", "[281.70, 37.19, −10.00, −7.43]", "−8.44, −6.84, −2.31 ± 2.10j"]],
        ),
        red="R 을 0.5 에서 5 로 키우면 전체 이득이 부드러워지고 팔 이득이 특히 많이 줄어든다. 극 배치에서는 이 조정을 극 위치로 번역해야 한다",
    ),
    dict(
        n=14,
        title="13. What Changes in the Code",
        lead="이득을 구하는 부분만 바뀌고 실시간 제어부는 그대로다",
        body=[
            ("term", "바뀌는 곳", "place4(poles) 대신 Riccati 를 풀어 K = R⁻¹BᵀP 를 얻는다"),
            ("term", "그대로인 곳", "u = float(-K @ x) — 매 주기 상태에 가중합을 적용할 뿐이다"),
            ("plain", "LQR 도 매 제어 주기마다 최적화를 다시 수행하지 않는다. 선형 모델에서는 K 를 미리 한 번 계산한다."),
        ],
        table=dict(
            head=["주의", "내용"],
            rows=[["이 환경에 scipy 가 없다", "solve_continuous_are 를 쓸 수 없다 — numpy 만으로 Hamiltonian 고유벡터 방식을 쓰면 된다"],
                  ["제약은 강제되지 않는다", "기본 LQR 은 가속도 한계·팔 한계를 직접 다루지 않는다"],
                  ["기존 안전장치 유지", "194행의 명령 포화, 212–223행의 중단 조건은 그대로 필요하다"]],
        ),
        red="LQR 의 '최적' 은 주어진 모델과 Q, R 에 대한 최적이라는 뜻이다 — 모델이 틀리면 최적도 틀린다",
    ),
    dict(
        n=15,
        title="14. One Slide to Remember",
        lead="세 방법, 하나의 제어식",
        body=[
            ("term", "제어식은 하나", "u = −Kx. place2 · place4 · LQR 은 모두 K 를 만드는 방법일 뿐이다"),
            ("term", "place2 는 막대만 본다", "표준 2차 응답에 맞추지만 팔이 돌아온다는 보장이 없다"),
            ("term", "place4 는 극을 찍는다", "네 상태를 다 보지만 제어 노력은 비용으로 따지지 않는다"),
            ("term", "LQR 은 비용을 적는다", "Q, R 을 정하면 Riccati 가 P 를, P 가 K 를 준다"),
        ],
        table=dict(
            head=["기억할 식", ""],
            rows=[["u = −Kx", "세 방법의 공통 결과"],
                  ["K = e₄ᵀ𝒞⁻¹p(A)", "Ackermann — 극 배치"],
                  ["k₁ = (ωn² + a)/b", "place2 — 계수 비교"],
                  ["K = R⁻¹BᵀP", "LQR — Riccati"]],
        ),
        note="근거 코드: Pendulum_control/simulator/rotary_model.py · rotary_balance_realtime.py · swing-up/balance_control.py · 모든 수치는 현재 코드에서 재계산",
    ),
]
