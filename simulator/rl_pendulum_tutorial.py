# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 강화학습(RL)으로 진자(pendulum) 각도 제어하기 — SAC Tutorial
#
# 이 노트북은 `pid_pendulum_realtime.py` 시뮬레이터의 **같은 플랜트(식별된 막대 진자)** 를
# PID 대신 **강화학습 에이전트(Soft Actor-Critic, SAC)** 로 제어하도록 학습시킵니다.
#
# | 구분 | PID (기존 시뮬레이터) | RL (이 튜토리얼) |
# |---|---|---|
# | 제어기 설계 | 극점 배치(pole placement)로 $K_p, K_i, K_d$ 계산 | 보상(reward)을 최대화하도록 신경망을 **데이터로 학습** |
# | 모델 필요 여부 | 선형화 모델 필요 | 시뮬레이터만 있으면 됨 (model-free) |
# | 출력 | $u = K_p e + K_i \int e - K_d \dot\theta$ | $u = \pi_\phi(s)$ (신경망 정책) |
#
# **목표 (task)**: 임의의 기준 각도 $\theta_{ref}$ 를 빠르고 정확하게 추종 (reference tracking),
# 외란(kick)이 들어와도 복귀.
#
# ### 목차
# 1. 환경 준비 & 시뮬레이터 불러오기
# 2. 플랜트 복습: 운동방정식과 RK4
# 3. RL 문제 정의: 상태·행동·보상 → Gym 스타일 `Env` 클래스
# 4. SAC 알고리즘 구현 (Replay buffer, Actor, Twin Critic, 자동 엔트로피 조절)
# 5. 학습 (training loop) & 학습 곡선
# 6. 평가: PID vs RL (계단 응답, 외란 응답, 다중 기준 추종)
# 7. 모델 저장 / 불러오기, 실시간 애니메이션
# 8. 실습 과제
#
# > **노트북 변환**: 이 파일은 jupytext의 *percent* 형식입니다 (`# %%` = 코드 셀, `# %% [markdown]` = 마크다운 셀).
# > ```
# > pip install jupytext
# > jupytext --to notebook rl_pendulum_tutorial.py      # -> rl_pendulum_tutorial.ipynb
# > ```
# > VS Code에서는 변환 없이도 각 `# %%` 셀을 `Run Cell`로 바로 실행할 수 있습니다.
# > 스크립트로 한 번에 실행: `python rl_pendulum_tutorial.py`

# %% [markdown]
# ## 1. 환경 준비
#
# 필요한 패키지: `numpy`, `matplotlib`, `torch` (CPU 버전으로 충분).
# ```
# pip install numpy matplotlib torch
# ```
# 이 파일은 `simulator/` 폴더에 있으며, 같은 폴더의 `pid_pendulum_realtime.py` 에서
# **플랜트 파라미터와 PID 제어기를 그대로 import** 합니다. 즉 RL과 PID가 *완전히 동일한 물리 모델* 위에서 비교됩니다.

# %%
import os
import sys
import time
import copy
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# 이 파일(또는 노트북)이 있는 폴더를 import 경로에 추가
try:
    HERE = Path(__file__).resolve().parent          # python 스크립트로 실행할 때
except NameError:
    HERE = Path.cwd()                                # Jupyter 노트북에서 실행할 때
sys.path.insert(0, str(HERE))

import pid_pendulum_realtime as sim                  # 기존 PID 시뮬레이터

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cpu")                         # 작은 MLP라 CPU가 오히려 빠름
torch.set_num_threads(1)

print(sim.header())

# %% [markdown]
# ## 2. 플랜트 복습
#
# 시뮬레이터의 운동방정식 (관성 $J$ 로 정규화, $J=1$):
#
# $$
# \ddot\theta = u - 2\zeta\omega_n\,\dot\theta - \omega_n^2 \sin\theta,
# \qquad \omega_n = 7.499\ \text{rad/s},\ \ \zeta = 0.01172
# $$
#
# - $\theta = 0$ 이 아래로 매달린 상태, $u = \tau/J$ [rad/s²] 가 제어 입력
# - 입력 포화: $|u| \le u_{max} = 3\omega_n^2 \approx 169$ rad/s²
# - 적분: RK4, $\Delta t = 2$ ms
#
# 감쇠비 $\zeta \approx 0.012$ 로 **감쇠가 매우 작아서**, 제어하지 않으면 한참 동안 흔들립니다.
# 또 기준각 $\theta_{ref}$ 에 멈춰 있으려면 중력을 상쇄하는 입력 $u_{ss} = \omega_n^2 \sin\theta_{ref}$ 가
# 계속 필요합니다 (PID는 적분항 $K_i$ 가 이 역할을 함).
#
# 기존 시뮬레이터의 `rk4_step` 은 PID가 내부에 묶여 있으므로, RL용으로는
# **입력 $u$ 를 외부에서 받는** 플랜트 함수를 따로 만듭니다 (파라미터는 `sim`에서 그대로 가져옴).

# %%
WN, ZETA = sim.WN, sim.ZETA
K, C, J = sim.K, sim.C, sim.J           # k/J = wn^2, c/J = 2 zeta wn
U_MAX = sim.TAU_MAX                      # 입력 포화 [rad/s^2]
DT_SIM = sim.DT                          # 물리 적분 스텝 0.002 s


def plant_deriv(th, w, u):
    """d/dt [theta, omega]  (시뮬레이터의 f()에서 PID 부분만 뺀 것)"""
    return w, (u - C * w - K * np.sin(th)) / J


def plant_rk4(th, w, u, dt=DT_SIM):
    """입력 u를 dt 동안 일정하게 유지(zero-order hold)하고 RK4로 한 스텝 적분"""
    k1 = plant_deriv(th, w, u)
    k2 = plant_deriv(th + dt / 2 * k1[0], w + dt / 2 * k1[1], u)
    k3 = plant_deriv(th + dt / 2 * k2[0], w + dt / 2 * k2[1], u)
    k4 = plant_deriv(th + dt * k3[0], w + dt * k3[1], u)
    th += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    w += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return th, w


# 검증: 입력 없이 20도에서 놓으면 약 T = 2*pi/wn = 0.84 s 주기로 거의 감쇠 없이 진동해야 함
th, w, TH = np.deg2rad(20), 0.0, []
for _ in range(int(3.0 / DT_SIM)):
    th, w = plant_rk4(th, w, 0.0)
    TH.append(np.rad2deg(th))
t = np.arange(len(TH)) * DT_SIM
plt.figure(figsize=(7, 2.5))
plt.plot(t, TH)
plt.title(f"free swing (u = 0): period ~ {2*np.pi/WN:.3f} s")
plt.xlabel("t [s]"); plt.ylabel("theta [deg]"); plt.grid(alpha=.3)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 3. RL 문제 정의 (MDP)
#
# 강화학습은 **에이전트 ↔ 환경** 의 반복입니다.
#
# ```
#        s_t (상태)                        a_t (행동)
#   환경 ──────────────▶  정책 π(a|s)  ──────────────▶ 환경
#    ▲                                                 │
#    └────────────── r_t (보상), s_{t+1} ◀──────────────┘
# ```
#
# 목표: 누적 할인 보상 $\;G_t = \sum_{k\ge 0} \gamma^k r_{t+k}\;$ 의 기댓값을 최대화하는 정책 $\pi$ 찾기.
#
# ### 3.1 제어 주기 (control period)
# 물리는 2 ms로 적분하지만, 에이전트는 **20 ms (50 Hz)** 마다 행동을 결정하고
# 그 사이 10번의 물리 스텝 동안 입력을 유지합니다 (실제 마이크로컨트롤러 제어 루프와 같은 구조).
# 제어 주기가 너무 짧으면 에피소드가 길어져 학습이 어려워지고, 너무 길면 제어 성능이 떨어집니다.
#
# ### 3.2 상태 (observation) $s_t$ — 6차원
# | 성분 | 이유 |
# |---|---|
# | $\sin\theta,\ \cos\theta$ | 각도를 연속적으로 표현 ($\pm\pi$ 불연속 제거) |
# | $\dot\theta / 10$ | 각속도 (신경망 입력 크기를 ~1 수준으로 스케일링) |
# | $5e,\ \ e = \mathrm{wrap}(\theta_{ref} - \theta)$ | 추종 오차 (PID의 P항에 해당), $[-\pi,\pi)$ 로 wrap. 1° 오차도 신경망이 잘 '느끼도록' 5배 확대 |
# | $\theta_{ref}$ | 기준각 |
# | $a_{t-1}$ | 직전 행동 → 입력 변화율 페널티를 계산할 수 있도록 (Markov 성질 유지) |
#
# > PID와 달리 적분 상태는 넣지 않았습니다. 아래 3.3의 feedforward가 정상상태 입력을 담당하기 때문입니다.
# > (대신 모델 오차가 있으면 정상상태 오차가 생길 수 있음 → 실습 과제 4)
#
# ### 3.3 행동 (action) $a_t \in [-1, 1]$ — **Residual RL (feedforward + 학습된 피드백)**
# 신경망 출력은 $[-1,1]$ 로 정규화하고, 실제 입력은
# $$
# u = \mathrm{sat}\big(\underbrace{\omega_n^2 \sin\theta_{ref}}_{\text{중력 보상 feedforward}} + \underbrace{a \cdot u_{max}}_{\text{RL 정책}}\big)
# $$
# **왜 feedforward를 더하나?** 처음엔 $u = a\cdot u_{max}$ 만으로 학습해 봤는데, 정상상태 오차가 2~8° 남았습니다.
# - 입력 범위가 $\pm 169$ rad/s² 로 매우 넓어서 신경망 출력이 **1%만 틀려도** $1.7$ rad/s² → 각도로 약 **2°** 오차
# - 오차 2°의 비용 $e^2 \approx 0.001$ 은 너무 작아 critic이 구별하기 어려움
#
# 이미 알고 있는 물리(평형 유지 입력 $\omega_n^2\sin\theta_{ref}$)를 더해 주면, 정책은 평형에서 **0만 출력하면** 되므로
# 정밀도 문제가 크게 줄어듭니다. 이처럼 *알고 있는 모델 + RL로 나머지(residual)를 학습* 하는 방식을
# **residual RL** 이라 하며 실제 로봇 제어에서 널리 쓰입니다. (`use_ff=False` 로 직접 비교해 보세요 → 실습 과제 1)
#
# ### 3.4 보상 (reward) — **가장 중요한 설계 요소**
# $$
# r_t = -\Big( e^2 \;+\; 0.002\,\dot\theta^2 \;+\; 0.01\,a_t^2 \;+\; 0.1\,(a_t - a_{t-1})^2 \Big)
# $$
# - $e^2$ : 기준각을 추종하라 (주 목표)
# - $\dot\theta^2$ : 불필요하게 흔들지 마라 (감쇠 역할, PID의 D항과 유사한 효과)
# - $a^2$ : 에너지를 아껴라
# - $(\Delta a)^2$ : 입력을 급격히 바꾸지 마라 (bang-bang/채터링 방지 → 실제 스테퍼 모터에 중요)
#
# 가중치를 바꾸면 **완전히 다른 제어기**가 학습됩니다. (실습 과제 1)
#
# ### 3.5 에피소드 구성 (다양한 상황 경험시키기)
# - 길이 4 s (= 200 스텝)
# - 초기 상태: $\theta_0 \sim U(-45°, 45°)$, $\dot\theta_0 \sim U(-2, 2)$ rad/s
# - 기준각: $\theta_{ref} \sim U(-60°, 60°)$, 에피소드 중간 (1.5~2.5 s) 에 **한 번 새 값으로 변경**
# - 확률 50%로 임의 시점에 **kick 외란** ($\dot\theta \mathrel{+}= \pm 200°/s$, 시뮬레이터의 ←/→ 키와 동일)
#
# 에피소드가 4 s에서 끝나는 것은 **시간 제한(truncation)** 이지 실패로 인한 **종료(termination)** 가 아닙니다.
# 따라서 TD target 계산 시 마지막 상태에서도 $\gamma V(s')$ 를 bootstrap 해야 합니다 (흔한 실수 포인트!).

# %%
def wrap(x):
    """각도를 [-pi, pi) 로 wrap (한 바퀴 돈 뒤의 오차가 2*pi 로 튀지 않도록)"""
    return (x + np.pi) % (2 * np.pi) - np.pi


class PendulumTrackEnv:
    """Gymnasium 스타일(reset/step) 진자 기준각 추종 환경"""

    obs_dim, act_dim = 6, 1

    def __init__(self, dt_ctrl=0.02, t_episode=4.0, ref_range_deg=60.0,
                 kick_prob=0.5, use_ff=True, seed=None):
        self.use_ff = use_ff                                 # 중력 보상 feedforward 사용 여부
        self.dt_ctrl = dt_ctrl
        self.n_sub = int(round(dt_ctrl / DT_SIM))           # 제어 1스텝당 물리 스텝 수 (=10)
        self.max_steps = int(round(t_episode / dt_ctrl))     # 200
        self.ref_range = np.deg2rad(ref_range_deg)
        self.kick_prob = kick_prob
        self.rng = np.random.default_rng(seed)

    # ---------- 보상 가중치 (실습 과제에서 바꿔 보세요) ----------
    W_E, W_W, W_A, W_DA = 1.0, 0.002, 0.01, 0.1

    def _obs(self):
        e = wrap(self.ref - self.th)
        return np.array([np.sin(self.th), np.cos(self.th), self.w / 10.0,
                         5 * e, self.ref, self.a_prev], dtype=np.float32)

    def reset(self, th0=None, w0=None, ref=None, ref_schedule=None, kicks=None):
        """평가용으로 초기상태/기준각 스케줄/kick을 직접 지정할 수도 있음.
        ref_schedule: [(t, ref_rad), ...],  kicks: [(t, d_omega), ...]"""
        r = self.rng
        self.th = r.uniform(-np.pi / 4, np.pi / 4) if th0 is None else th0
        self.w = r.uniform(-2, 2) if w0 is None else w0
        self.ref = r.uniform(-self.ref_range, self.ref_range) if ref is None else ref
        if ref_schedule is None:
            ref_schedule = [] if ref is not None else \
                [(r.uniform(1.5, 2.5), r.uniform(-self.ref_range, self.ref_range))]
        if kicks is None:
            kicks = [] if ref is not None or r.random() > self.kick_prob else \
                [(r.uniform(0.5, 3.5), r.choice([-1, 1]) * np.deg2rad(200))]
        self.ref_schedule = sorted(ref_schedule)
        self.kicks = sorted(kicks)
        self.a_prev, self.k, self.t = 0.0, 0, 0.0
        return self._obs()

    def step(self, action):
        a = float(np.clip(np.ravel(action)[0], -1.0, 1.0))   # 스칼라/배열 모두 허용
        u_ff = K * np.sin(self.ref) if self.use_ff else 0.0
        u = float(np.clip(u_ff + a * U_MAX, -U_MAX, U_MAX))     # residual RL
        # 예약된 기준각 변경 / kick 외란 적용
        while self.ref_schedule and self.t >= self.ref_schedule[0][0]:
            self.ref = self.ref_schedule.pop(0)[1]
        while self.kicks and self.t >= self.kicks[0][0]:
            self.w += self.kicks.pop(0)[1]
        # zero-order hold: n_sub 번의 RK4 물리 스텝
        for _ in range(self.n_sub):
            self.th, self.w = plant_rk4(self.th, self.w, u)
        self.t += self.dt_ctrl
        self.k += 1

        e = wrap(self.ref - self.th)
        cost = (self.W_E * e ** 2 + self.W_W * self.w ** 2
                + self.W_A * a ** 2 + self.W_DA * (a - self.a_prev) ** 2)
        self.a_prev = a
        terminated = False                         # 이 과제에는 '실패 종료'가 없음
        truncated = self.k >= self.max_steps       # 시간 제한
        return self._obs(), -cost, terminated, truncated, {"u": u, "e": e}


# 환경 동작 확인: 무작위 정책으로 한 에피소드
env = PendulumTrackEnv(seed=SEED)
obs, ret = env.reset(), 0.0
for _ in range(env.max_steps):
    obs, r, term, trunc, _ = env.step(np.random.uniform(-1, 1))
    ret += r
print("obs example:", np.round(obs, 3))
print(f"random policy return = {ret:.1f}   (학습 후 이 값이 얼마나 올라가는지 비교해 보세요)")

# %% [markdown]
# ### 3.6 비교 기준선 (baseline): 같은 환경에서 PID 돌려보기
# 시뮬레이터의 PID 게인 $K_p, K_i, K_d$ 를 그대로 사용해 **같은 보상 함수**로 점수를 매겨 봅니다.
# 공정한 비교를 위해 PID도 RL과 같은 50 Hz 샘플링 + zero-order hold 로 동작시킵니다
# (원래 시뮬레이터는 500 Hz 연속시간 PID). 적분기는 anti-windup 포함.

# %%
class PIDPolicy:
    """sim.pid_torque와 같은 법칙의 이산시간 PID. act(obs) -> a
    환경이 feedforward를 더하므로 그만큼 빼서, 최종 입력 u가 원래 PID와 같아지게 함"""

    def __init__(self, dt=0.02, use_ff=True):
        self.dt, self.ie, self.use_ff = dt, 0.0, use_ff

    def reset(self):
        self.ie = 0.0

    def act(self, obs):
        e, w = float(obs[3]) / 5, float(obs[2]) * 10.0
        u_raw = sim.KP * e + sim.KI * self.ie - sim.KD * w
        if abs(u_raw) < U_MAX:                  # anti-windup: 포화 중엔 적분 정지
            self.ie += e * self.dt
        u_ff = K * np.sin(float(obs[4])) if self.use_ff else 0.0
        return (np.clip(u_raw, -U_MAX, U_MAX) - u_ff) / U_MAX


def run_episode(env, policy_fn, reset_fn=None, **reset_kw):
    """policy_fn(obs)->a 로 한 에피소드 실행, 기록 반환"""
    if reset_fn: reset_fn()
    obs = env.reset(**reset_kw)
    log = {"t": [], "th": [], "ref": [], "u": [], "r": []}
    done = False
    while not done:
        a = policy_fn(obs)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        log["t"].append(env.t); log["th"].append(env.th); log["ref"].append(env.ref)
        log["u"].append(info["u"]); log["r"].append(r)
    return {k: np.array(v) for k, v in log.items()}


pid = PIDPolicy(dt=env.dt_ctrl)
log = run_episode(env, pid.act, pid.reset)
print(f"PID return on a random episode = {log['r'].sum():.1f}")

# %% [markdown]
# ## 4. Soft Actor-Critic (SAC)
#
# 연속 행동(continuous action) 제어에 가장 널리 쓰이는 **off-policy actor-critic** 알고리즘입니다
# (Haarnoja et al., 2018). 핵심 아이디어는 보상과 함께 **정책의 엔트로피(무작위성)** 도 최대화하는 것:
#
# $$
# J(\pi) = \mathbb{E}\Big[\sum_t \gamma^t \big( r_t + \alpha\, \mathcal{H}(\pi(\cdot|s_t)) \big)\Big]
# $$
#
# → 탐색(exploration)이 자연스럽게 유지되고, 학습이 안정적입니다.
#
# ### 구성 요소
# | 요소 | 역할 |
# |---|---|
# | **Replay buffer** $\mathcal{D}$ | 과거 경험 $(s,a,r,s',d)$ 저장 → 무작위 미니배치로 재사용 (off-policy, 데이터 효율적) |
# | **Actor** $\pi_\phi(a|s)$ | 가우시안 분포 $\mathcal{N}(\mu,\sigma)$ 에서 샘플 후 $\tanh$ 로 $[-1,1]$ 에 가둠 |
# | **Twin critics** $Q_{\theta_1}, Q_{\theta_2}$ | 행동가치 추정. 두 개 중 **작은 값** 사용 → Q 과대추정 방지 |
# | **Target critics** $Q_{\bar\theta}$ | critic의 느린 복사본 (Polyak 평균 $\bar\theta \leftarrow \tau\theta + (1-\tau)\bar\theta$) → 타깃 안정화 |
# | **Temperature** $\alpha$ | 엔트로피 가중치. 목표 엔트로피 $-\dim(\mathcal{A})$ 를 맞추도록 **자동 조절** |
#
# ### 손실 함수
# **Critic** (TD 학습):
# $$
# y = r + \gamma (1-d)\Big(\min_{i} Q_{\bar\theta_i}(s', \tilde a') - \alpha \log\pi_\phi(\tilde a'|s')\Big),\quad \tilde a' \sim \pi_\phi(\cdot|s')
# $$
# $$
# L_Q = \big(Q_{\theta_i}(s,a) - y\big)^2
# $$
# **Actor** (reparameterization trick: $\tilde a = \tanh(\mu + \sigma\epsilon),\ \epsilon\sim\mathcal{N}(0,1)$ 로 미분 가능):
# $$
# L_\pi = \mathbb{E}_{s}\Big[\alpha \log\pi_\phi(\tilde a|s) - \min_i Q_{\theta_i}(s, \tilde a)\Big]
# $$
# **Temperature**:
# $$
# L_\alpha = -\alpha\,\big(\log\pi_\phi(\tilde a|s) + \mathcal{H}_{target}\big)
# $$
#
# > 여기서 $d$ 는 **terminated** 만 의미합니다. 시간 제한(truncated)으로 끝난 경우 $d=0$ 으로 두고 bootstrap 합니다.
#
# > **목표 엔트로피**: SAC 논문의 기본값은 $-\dim(\mathcal{A}) = -1$ 인데, 이 경우 정책의 표준편차가 약 0.1
# > (= $\pm 17$ rad/s² 의 탐색 노이즈) 수준으로 유지되어 정밀한 각도 제어를 배우기 어렵습니다.
# > 여기서는 $-4$ 로 낮춰 학습 후반에 정책이 더 결정적(deterministic)이 되도록 했습니다.

# %% [markdown]
# ### 4.1 Replay buffer

# %%
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size=int(3e5)):
        self.s = np.zeros((size, obs_dim), np.float32)
        self.a = np.zeros((size, act_dim), np.float32)
        self.r = np.zeros((size, 1), np.float32)
        self.s2 = np.zeros((size, obs_dim), np.float32)
        self.d = np.zeros((size, 1), np.float32)
        self.ptr, self.n, self.size = 0, 0, size

    def add(self, s, a, r, s2, d):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
        self.ptr = (i + 1) % self.size                  # 가득 차면 오래된 것부터 덮어씀
        self.n = min(self.n + 1, self.size)

    def sample(self, batch):
        idx = np.random.randint(0, self.n, batch)
        to_t = lambda x: torch.as_tensor(x[idx], device=DEVICE)
        return to_t(self.s), to_t(self.a), to_t(self.r), to_t(self.s2), to_t(self.d)


# %% [markdown]
# ### 4.2 신경망: Actor (tanh-Gaussian 정책) 와 Critic (Q 함수)
#
# $\tanh$ 로 행동을 압축하면 확률밀도도 변수변환(change of variables)으로 보정해야 합니다:
# $$
# \log\pi(a|s) = \log\mathcal{N}(z;\mu,\sigma) - \sum \log(1 - \tanh^2 z),\qquad a=\tanh z
# $$
# 수치 안정성을 위해 $\log(1-\tanh^2 z) = 2(\log 2 - z - \text{softplus}(-2z))$ 형태를 사용합니다.

# %%
def mlp(inp, out, hidden=(128, 128)):
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -20, 2

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim)           # 출력: [mu, log_std]

    def forward(self, s, deterministic=False):
        mu, log_std = self.net(s).chunk(2, dim=-1)
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()
        z = mu if deterministic else mu + std * torch.randn_like(mu)   # reparameterization
        a = torch.tanh(z)
        # log pi(a|s): 가우시안 log-density + tanh 보정
        logp = (-0.5 * ((z - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1, keepdim=True)
        logp -= (2 * (np.log(2) - z - F.softplus(-2 * z))).sum(-1, keepdim=True)
        return a, logp


class Critic(nn.Module):
    """Twin Q-network: Q1(s,a), Q2(s,a)"""

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, 1)
        self.q2 = mlp(obs_dim + act_dim, 1)

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa), self.q2(sa)


# %% [markdown]
# ### 4.3 SAC 에이전트
# `update()` 한 번이 위 세 손실 함수에 대한 경사하강 1스텝입니다.

# %%
class SAC:
    def __init__(self, obs_dim, act_dim, gamma=0.98, tau=0.005, lr=1e-3):
        self.actor = Actor(obs_dim, act_dim).to(DEVICE)
        self.critic = Critic(obs_dim, act_dim).to(DEVICE)
        self.critic_targ = copy.deepcopy(self.critic)
        for p in self.critic_targ.parameters():
            p.requires_grad_(False)                   # target은 Polyak 평균으로만 갱신
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.target_entropy = -4.0 * act_dim          # 기본값 -dim(A) 보다 낮춤 (위 설명)
        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.a_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.gamma, self.tau = gamma, tau

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        s = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        a, _ = self.actor(s, deterministic)
        return a.cpu().numpy()[0]

    def update(self, buf, batch=256):
        s, a, r, s2, d = buf.sample(batch)

        # ---- (1) Critic: TD target y ----
        with torch.no_grad():
            a2, logp2 = self.actor(s2)
            q1_t, q2_t = self.critic_targ(s2, a2)
            y = r + self.gamma * (1 - d) * (torch.min(q1_t, q2_t) - self.alpha * logp2)
        q1, q2 = self.critic(s, a)
        loss_q = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.q_opt.zero_grad(); loss_q.backward(); self.q_opt.step()

        # ---- (2) Actor: maximize  min Q - alpha log pi ----
        a_new, logp = self.actor(s)
        q_new = torch.min(*self.critic(s, a_new))
        loss_pi = (self.alpha.detach() * logp - q_new).mean()
        self.pi_opt.zero_grad(); loss_pi.backward(); self.pi_opt.step()

        # ---- (3) Temperature alpha: 엔트로피를 목표값에 맞춤 ----
        loss_a = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.a_opt.zero_grad(); loss_a.backward(); self.a_opt.step()

        # ---- (4) Target network: Polyak averaging ----
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_targ.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)
        return loss_q.item(), loss_pi.item(), self.alpha.item()


# %% [markdown]
# ## 5. 학습 (Training loop)
#
# ```
# for step in 1..TOTAL_STEPS:
#     a ~ π(·|s)            (처음 START_STEPS 동안은 균일 무작위 행동 → 버퍼를 다양한 데이터로 채움)
#     s', r = env.step(a)
#     buffer.add(s, a, r, s', terminated)      ← truncated 는 저장하지 않음!
#     SAC.update(buffer)                         (매 스텝 1회)
#     주기적으로 결정적(deterministic) 정책을 고정된 평가 에피소드로 평가
# ```
#
# **하이퍼파라미터**
# - `gamma = 0.98` → 유효 시야(horizon) $\approx 1/(1-\gamma) = 50$ 스텝 = 1 s. 진자의 정착시간(~0.5 s)보다 길어야 함
# - `lr = 1e-3`, 은닉층 128×2 (작은 문제라 이 정도로 충분하고, CPU에서 빠름)
# - `TOTAL_STEPS = 30,000` (= 150 에피소드, 시뮬레이션 시간 10분). CPU 성능에 따라 약 5~15분 소요
#
# > 시간이 부족하면 `TOTAL_STEPS` 를 줄이거나, 아래 7절의 저장된 모델(`sac_pendulum.pt`)을 불러와 쓰세요.

# %%
TOTAL_STEPS = 30_000
START_STEPS = 2_000          # 초기 무작위 탐색 스텝
UPDATE_AFTER = 1_000         # 버퍼에 이만큼 쌓인 뒤부터 학습
BATCH = 256
EVAL_EVERY = 2_000

# 평가용 고정 시나리오 (매번 같은 조건에서 비교해야 학습 곡선이 의미 있음)
EVAL_SCENARIOS = [
    dict(th0=0.0, w0=0.0, ref=np.deg2rad(20)),                                      # 계단 0 -> 20도
    dict(th0=np.deg2rad(30), w0=0.0, ref=np.deg2rad(-40)),                           # 큰 역방향 계단
    dict(th0=0.0, w0=0.0, ref=np.deg2rad(40), kicks=[(2.0, np.deg2rad(200))]),       # 외란 복귀
    dict(th0=0.0, w0=0.0, ref=np.deg2rad(-10),
         ref_schedule=[(1.0, np.deg2rad(50)), (2.5, np.deg2rad(0))]),               # 다중 기준 변경
]


def evaluate(policy_fn, reset_fn=None):
    eval_env = PendulumTrackEnv()
    return np.mean([run_episode(eval_env, policy_fn, reset_fn, **sc)["r"].sum()
                    for sc in EVAL_SCENARIOS])


pid_score = evaluate(pid.act, pid.reset)
print(f"PID  eval score = {pid_score:.2f}")

# %%
env = PendulumTrackEnv(seed=SEED)
agent = SAC(env.obs_dim, env.act_dim)
buf = ReplayBuffer(env.obs_dim, env.act_dim)

hist = {"step": [], "eval": [], "ep_ret": [], "ep_step": [], "alpha": []}
obs, ep_ret, t0 = env.reset(), 0.0, time.time()

for step in range(1, TOTAL_STEPS + 1):
    # ----- 행동 선택 -----
    a = np.random.uniform(-1, 1, env.act_dim) if step <= START_STEPS else agent.act(obs)
    obs2, r, term, trunc, _ = env.step(a[0])
    buf.add(obs, a, r, obs2, float(term))      # d = terminated 만! (truncation은 bootstrap)
    obs, ep_ret = obs2, ep_ret + r

    if term or trunc:
        hist["ep_ret"].append(ep_ret); hist["ep_step"].append(step)
        obs, ep_ret = env.reset(), 0.0

    # ----- 학습 -----
    if step >= UPDATE_AFTER:
        lq, lpi, alpha = agent.update(buf, BATCH)

    # ----- 평가 -----
    if step % EVAL_EVERY == 0:
        score = evaluate(lambda o: agent.act(o, deterministic=True))
        hist["step"].append(step); hist["eval"].append(score); hist["alpha"].append(alpha)
        print(f"step {step:6d} | eval {score:8.2f} (PID {pid_score:.2f}) | "
              f"alpha {alpha:.4f} | Q-loss {lq:.3f} | {time.time()-t0:5.0f}s", flush=True)

# %% [markdown]
# ### 5.1 학습 곡선
# - 왼쪽: 학습 에피소드의 누적 보상 (탐색 노이즈 + 무작위 시나리오라 들쭉날쭉함)
# - 가운데: 고정 시나리오에서 결정적 정책의 평가 점수 vs PID 기준선
# - 오른쪽: 엔트로피 온도 $\alpha$ — 학습이 진행될수록 줄어들며 정책이 점점 결정적으로 변함

# %%
fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
ax[0].plot(hist["ep_step"], hist["ep_ret"], ".", alpha=.5)
ax[0].set_title("training episode return"); ax[0].set_xlabel("env step")
ax[1].plot(hist["step"], hist["eval"], "o-", label="SAC")
ax[1].axhline(pid_score, color="k", ls="--", label="PID")
ax[1].set_title("eval score (fixed scenarios)"); ax[1].set_xlabel("env step"); ax[1].legend()
ax[2].semilogy(hist["step"], hist["alpha"], "o-")
ax[2].set_title("entropy temperature alpha"); ax[2].set_xlabel("env step")
for a_ in ax: a_.grid(alpha=.3)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 6. 평가: PID vs RL
#
# 같은 시나리오를 두 제어기로 실행해 응답을 겹쳐 그립니다. 확인할 점:
# - **상승시간 / 오버슈트**: 어느 쪽이 더 빠르고 덜 튀는가?
# - **정상상태 오차**: PID는 적분항으로 0이 되지만, RL은 적분기 없이 feedforward + 학습된 피드백에 의존 → 작은 오차가 남을 수 있음
# - **입력 $u$ 모양**: 포화($\pm$169)를 얼마나 쓰는가, 채터링은 없는가?

# %%
rl_policy = lambda o: agent.act(o, deterministic=True)
titles = ["step 0 -> 20 deg", "step 30 -> -40 deg", "kick at t=2 s (ref 40)", "multi-step reference"]

fig, ax = plt.subplots(2, len(EVAL_SCENARIOS), figsize=(16, 6), sharex=True)
for j, sc in enumerate(EVAL_SCENARIOS):
    ev = PendulumTrackEnv()
    lp = run_episode(ev, pid.act, pid.reset, **copy.deepcopy(sc))
    lr = run_episode(ev, rl_policy, **copy.deepcopy(sc))
    ax[0, j].plot(lp["t"], np.rad2deg(lp["ref"]), "k--", lw=1, label="ref")
    ax[0, j].plot(lp["t"], np.rad2deg(lp["th"]), label=f"PID  ({lp['r'].sum():.1f})")
    ax[0, j].plot(lr["t"], np.rad2deg(lr["th"]), label=f"SAC  ({lr['r'].sum():.1f})")
    ax[1, j].plot(lp["t"], lp["u"], lw=1); ax[1, j].plot(lr["t"], lr["u"], lw=1)
    ax[0, j].set_title(titles[j]); ax[0, j].legend(fontsize=8)
    ax[1, j].set_xlabel("t [s]")
ax[0, 0].set_ylabel("theta [deg]"); ax[1, 0].set_ylabel("u [rad/s^2]")
for a_ in ax.flat: a_.grid(alpha=.3)
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 6.1 정량 지표 (계단 응답 0 → 20°)
# 원래 시뮬레이터의 `--check` 옵션과 같은 조건입니다.

# %%
def step_metrics(log, ref):
    th = log["th"]
    overshoot = 100 * (th.max() - ref) / ref
    idx = np.where(np.abs(th - ref) > 0.02 * abs(ref))[0]     # 2% 정착 기준
    t_settle = log["t"][idx[-1]] if len(idx) else 0.0
    ss_err = np.rad2deg(ref - th[-50:].mean())
    return overshoot, t_settle, ss_err


ref = np.deg2rad(20)
print(f"{'':6s} {'overshoot[%]':>13s} {'settle(2%)[s]':>14s} {'ss error[deg]':>14s} {'max|u|':>8s}")
for name, fn, rs in [("PID", pid.act, pid.reset), ("SAC", rl_policy, None)]:
    lg = run_episode(PendulumTrackEnv(), fn, rs, th0=0.0, w0=0.0, ref=ref)
    os_, ts, ss = step_metrics(lg, ref)
    print(f"{name:6s} {os_:13.1f} {ts:14.2f} {ss:14.3f} {np.abs(lg['u']).max():8.1f}")

# %% [markdown]
# ### 6.2 학습한 정책 들여다보기
# 정책은 결국 $u = u_{ff} + \pi(\theta, \dot\theta, \theta_{ref})$ 인 **비선형 상태 피드백** 입니다.
# $\theta_{ref}=20°$, $a_{t-1}=0$ 으로 고정하고 $(\theta, \dot\theta)$ 평면에서 RL 피드백 부분을 그려 보면
# PID 피드백 $K_p e - K_d\dot\theta$ (직선 경계) 와 비교할 수 있습니다.
# 또 평형점 ($\theta=\theta_{ref}, \dot\theta=0$) 에서 RL 출력이 0에 가까운지도 확인합니다.

# %%
ref = np.deg2rad(20)
TH, W = np.meshgrid(np.deg2rad(np.linspace(-40, 80, 121)), np.linspace(-10, 10, 101))
S_grid = np.stack([np.sin(TH), np.cos(TH), W / 10, 5 * (ref - TH), np.full_like(TH, ref), np.zeros_like(TH)], -1)
with torch.no_grad():
    A_grid = agent.actor(torch.as_tensor(S_grid.reshape(-1, 6), dtype=torch.float32), True)[0]
U_rl = A_grid.numpy().reshape(TH.shape) * U_MAX                        # RL 피드백 부분
U_pid = np.clip(sim.KP * (ref - TH) - sim.KD * W, -U_MAX, U_MAX)      # PID의 P+D 부분 (비교용)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for a_, U, name in [(ax[0], U_rl, "SAC residual  a * u_max"), (ax[1], U_pid, "PID  Kp e - Kd omega")]:
    cs = a_.contourf(np.rad2deg(TH), W, U, levels=np.linspace(-U_MAX, U_MAX, 21), cmap="RdBu_r")
    a_.contour(np.rad2deg(TH), W, U, levels=[0], colors="k", linewidths=1)   # u = 0 경계
    a_.axvline(20, color="k", ls="--", lw=1)
    a_.set_title(f"{name}   (ref = 20 deg)"); a_.set_xlabel("theta [deg]")
ax[0].set_ylabel("omega [rad/s]")
fig.colorbar(cs, ax=ax, label="u [rad/s^2]")
plt.show()

s_eq = np.array([np.sin(ref), np.cos(ref), 0, 0, ref, 0], np.float32)
print(f"RL feedback at equilibrium: {agent.act(s_eq, True)[0]*U_MAX:+.2f} rad/s^2  (0 이면 이상적)\n"
      f"feedforward wn^2 sin(ref) = {K*np.sin(ref):.2f} rad/s^2")

# %% [markdown]
# ## 7. 모델 저장 / 불러오기 & 실시간 애니메이션
#
# 학습된 actor 가중치만 있으면 제어기로 사용할 수 있습니다 (critic은 학습에만 필요).

# %%
MODEL_PATH = HERE / "sac_pendulum.pt"
torch.save(agent.actor.state_dict(), MODEL_PATH)
print("saved:", MODEL_PATH)

# 불러오기 예시
actor_loaded = Actor(PendulumTrackEnv.obs_dim, PendulumTrackEnv.act_dim)
actor_loaded.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
actor_loaded.eval()

# %% [markdown]
# ### 7.1 실시간 시뮬레이터에서 RL 제어기 돌려보기
# 원래 `pid_pendulum_realtime.py` 와 같은 키 조작으로 RL 정책을 테스트합니다.
#
# | 키 | 동작 |
# |---|---|
# | ↑ / ↓ | 기준각 ±5° |
# | ← / → | kick 외란 |
# | `c` | 제어기 전환 **RL ↔ PID** |
# | `r` | 리셋 |
#
# Jupyter에서는 별도 창이 필요하므로 `%matplotlib qt` (또는 `tk`) 를 먼저 실행하세요.
# 스크립트 실행 시에는 `RUN_REALTIME = True` 로 바꾸면 마지막에 창이 열립니다.

# %%
RUN_REALTIME = False


def run_realtime(actor):
    from matplotlib.animation import FuncAnimation
    ev = PendulumTrackEnv()
    ev.reset(th0=0.0, w0=0.0, ref=np.deg2rad(20))
    pid_rt = PIDPolicy(dt=ev.dt_ctrl)
    st = {"mode": "RL", "last": None, "T": [], "TH": [], "REF": []}

    def policy(o):
        if st["mode"] == "PID":
            return pid_rt.act(o)
        with torch.no_grad():
            return actor(torch.as_tensor(o).unsqueeze(0), True)[0].item()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.5))
    lim = 1.3 * sim.L
    a1.set_aspect("equal"); a1.set_xlim(-lim, lim); a1.set_ylim(-lim, 0.5 * lim)
    rod, = a1.plot([], [], lw=8, color="#E97820", solid_capstyle="round")
    refl, = a1.plot([], [], "--", color="#2774B8"); a1.plot([0], [0], "ko")
    info = a1.text(-lim * .95, .45 * lim, "", fontsize=9, va="top", family="monospace")
    l_th, = a2.plot([], [], color="#E97820", label="theta")
    l_rf, = a2.plot([], [], "--", color="#2774B8", label="ref")
    a2.set_ylim(-100, 100); a2.grid(alpha=.3); a2.legend(loc="upper right")
    a2.set_xlabel("t [s]"); a2.set_ylabel("deg")

    def on_key(e):
        if e.key == "up": ev.ref += np.deg2rad(5)
        elif e.key == "down": ev.ref -= np.deg2rad(5)
        elif e.key == "left": ev.w -= np.deg2rad(200)
        elif e.key == "right": ev.w += np.deg2rad(200)
        elif e.key == "c": st["mode"] = "PID" if st["mode"] == "RL" else "RL"; pid_rt.reset()
        elif e.key == "r":
            ev.reset(th0=0.0, w0=0.0, ref=ev.ref); st.update(T=[], TH=[], REF=[])
    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_):
        now = time.perf_counter()
        el = 0.0 if st["last"] is None else min(now - st["last"], 0.1)
        st["last"] = now
        for _ in range(int(el / ev.dt_ctrl)):
            ev.k = 0                              # 실시간 모드에서는 시간 제한 없음
            ev.step(policy(ev._obs()))
        th, rf = ev.th, ev.ref
        st["T"].append(ev.t); st["TH"].append(np.rad2deg(th)); st["REF"].append(np.rad2deg(rf))
        rod.set_data([0, sim.L * np.sin(th)], [0, -sim.L * np.cos(th)])
        refl.set_data([0, 1.1 * sim.L * np.sin(rf)], [0, -1.1 * sim.L * np.cos(rf)])
        info.set_text(f"[{st['mode']}] t={ev.t:6.2f}s\ntheta={np.rad2deg(th):7.2f}  ref={np.rad2deg(rf):6.1f}")
        t0_ = max(0.0, ev.t - 10); a2.set_xlim(t0_, t0_ + 10)
        l_th.set_data(st["T"], st["TH"]); l_rf.set_data(st["T"], st["REF"])
        return rod, refl, info, l_th, l_rf

    fig._ani = FuncAnimation(fig, update, interval=20, cache_frame_data=False)
    plt.tight_layout(); plt.show()


if RUN_REALTIME:
    run_realtime(actor_loaded)

# %% [markdown]
# ## 8. 정리 & 실습 과제
#
# ### 정리
# - RL은 **플랜트 모델 식 없이** 시뮬레이터와 보상 함수만으로 제어기를 얻습니다.
# - 대신 **보상 설계 · 상태 설계 · 학습 분포(초기조건/기준각 범위)** 가 제어 성능을 좌우합니다.
#   학습 때 보지 못한 영역(예: $|\theta_{ref}| > 60°$)에서는 성능이 보장되지 않습니다.
# - 순수 RL($u = a\,u_{max}$)은 넓은 입력 범위 때문에 정밀도가 부족했고, **알고 있는 물리(중력 보상)를 feedforward로 더한
#   residual RL** 로 PID 수준의 정상상태 정밀도를 얻었습니다. "모델을 버리는 것"이 아니라 "모델 + 학습"의 조합이 실용적입니다.
# - PID는 적분항으로 모델 오차에 강건하지만, RL 정책(적분기 없음)은 **시뮬레이터 모델**에 맞춘
#   feedforward/피드백에 의존하므로 실제 하드웨어와 모델이 다르면 정상상태 오차가 생길 수 있습니다 (sim-to-real gap).
#
# ### 실습 과제
# 1. **Feedforward 제거 (ablation)**: `PendulumTrackEnv(use_ff=False)`, `PIDPolicy(use_ff=False)` 로 바꿔 다시 학습하고
#    6.1의 정상상태 오차를 비교하세요. 왜 차이가 나는지 3.3의 설명과 연결해 보세요.
# 2. **보상 가중치 실험**: `PendulumTrackEnv.W_DA` 를 0으로 하면 입력 $u$ 가 어떻게 변하나요?
#    `W_W` 를 크게 하면 오버슈트/정착시간은? (학습 결과는 시드에 따라 달라지므로 시드 2~3개로 비교하는 습관을!)
# 3. **기준각 범위 밖 일반화**: `ref=np.deg2rad(80)` 로 평가해 보세요. PID와 비교하면?
#    `ref_range_deg` 를 넓혀 다시 학습하면?
# 4. **모델 오차 강건성 (sim-to-real)**: 평가 때만 플랜트의 `K` 를 ±20% 바꿔(`plant_deriv` 수정, feedforward는 그대로) 정상상태 오차를 비교하세요.
#    → 해결책 ① 상태에 오차 적분 $\int e\,dt$ 추가, ② 학습 중 매 에피소드 $\omega_n, \zeta$ 를 무작위화
#    (**domain randomization**). 둘 중 하나를 구현해 보세요.
# 5. **제어 주기**: `dt_ctrl` 를 0.01 s (100 Hz) 로 바꾸면 `gamma` 를 어떻게 조정해야 같은 시야(1 s)를 유지할까요? 학습 속도와 제어 성능은 어떻게 변하나요?
# 6. **Swing-up 도전**: 기준각을 180° (도립 상태)로 두고 $u_{max}$ 를 $0.5\,\omega_n^2$ 정도로 줄이면
#    한 번에 올릴 수 없어 **에너지를 모아 흔들어 올리는** 전략이 필요합니다. 보상과 상태를 어떻게 바꿔야 할까요?
#
# ### 참고 문헌
# - T. Haarnoja et al., *Soft Actor-Critic Algorithms and Applications*, arXiv:1812.05905, 2018
# - S. Fujimoto et al., *Addressing Function Approximation Error in Actor-Critic Methods (TD3)*, ICML 2018 — twin critic 아이디어
# - OpenAI Spinning Up — https://spinningup.openai.com/en/latest/algorithms/sac.html
