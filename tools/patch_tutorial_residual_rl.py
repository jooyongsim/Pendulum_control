"""One-off patch: switch simulator/rl_pendulum_tutorial.py to the residual-RL design
(gravity feedforward + 5x error scaling + target entropy -4), chosen from
tools/rl_reward_experiment.py results (ss error 0.33 deg vs 2-8 deg for pure RL)."""
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "simulator" / "rl_pendulum_tutorial.py"
s = p.read_text(encoding="utf-8")
R = []
R.append((r"""# | $e = \mathrm{wrap}(\theta_{ref} - \theta)$ | 추종 오차 (PID의 P항에 해당), $[-\pi,\pi)$ 로 wrap |
# | $\theta_{ref}$ | 기준각 → 정책이 중력 보상량 $\omega_n^2\sin\theta_{ref}$ 를 스스로 학습 |
# | $a_{t-1}$ | 직전 행동 → 입력 변화율 페널티를 계산할 수 있도록 (Markov 성질 유지) |
#
# > PID와 달리 적분 상태는 넣지 않았습니다. 정책이 $\theta_{ref}$ 를 보고 **feedforward 입력**을 직접 학습하기 때문입니다.
# > (대신 모델 오차가 있으면 정상상태 오차가 생길 수 있음 → 실습 과제 참고)
#
# ### 3.3 행동 (action) $a_t \in [-1, 1]$
# 신경망 출력은 $[-1,1]$ 로 정규화하고, 실제 입력은 $u = a \cdot u_{max}$.
""", r"""# | $5e,\ \ e = \mathrm{wrap}(\theta_{ref} - \theta)$ | 추종 오차 (PID의 P항에 해당), $[-\pi,\pi)$ 로 wrap. 1° 오차도 신경망이 잘 '느끼도록' 5배 확대 |
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
"""))
R.append(("""    def __init__(self, dt_ctrl=0.02, t_episode=4.0, ref_range_deg=60.0,
                 kick_prob=0.5, seed=None):
""", """    def __init__(self, dt_ctrl=0.02, t_episode=4.0, ref_range_deg=60.0,
                 kick_prob=0.5, use_ff=True, seed=None):
        self.use_ff = use_ff                                 # 중력 보상 feedforward 사용 여부
"""))
R.append(("                         e, self.ref, self.a_prev], dtype=np.float32)",
          "                         5 * e, self.ref, self.a_prev], dtype=np.float32)"))
R.append(("        u = a * U_MAX\n",
          "        u_ff = K * np.sin(self.ref) if self.use_ff else 0.0\n"
          "        u = float(np.clip(u_ff + a * U_MAX, -U_MAX, U_MAX))     # residual RL\n"))
R.append(('''    """sim.pid_torque와 같은 법칙의 이산시간 PID. act(obs) -> a in [-1,1]"""

    def __init__(self, dt=0.02):
        self.dt, self.ie = dt, 0.0''', '''    """sim.pid_torque와 같은 법칙의 이산시간 PID. act(obs) -> a
    환경이 feedforward를 더하므로 그만큼 빼서, 최종 입력 u가 원래 PID와 같아지게 함"""

    def __init__(self, dt=0.02, use_ff=True):
        self.dt, self.ie, self.use_ff = dt, 0.0, use_ff'''))
R.append(("        e, w = float(obs[3]), float(obs[2]) * 10.0",
          "        e, w = float(obs[3]) / 5, float(obs[2]) * 10.0"))
R.append(("        return np.clip(u_raw / U_MAX, -1, 1)",
          "        u_ff = K * np.sin(float(obs[4])) if self.use_ff else 0.0\n"
          "        return (np.clip(u_raw, -U_MAX, U_MAX) - u_ff) / U_MAX"))
R.append(("        self.target_entropy = -float(act_dim)",
          "        self.target_entropy = -4.0 * act_dim          # 기본값 -dim(A) 보다 낮춤 (위 설명)"))
R.append((r"""# > 여기서 $d$ 는 **terminated** 만 의미합니다. 시간 제한(truncated)으로 끝난 경우 $d=0$ 으로 두고 bootstrap 합니다.""",
          r"""# > 여기서 $d$ 는 **terminated** 만 의미합니다. 시간 제한(truncated)으로 끝난 경우 $d=0$ 으로 두고 bootstrap 합니다.
#
# > **목표 엔트로피**: SAC 논문의 기본값은 $-\dim(\mathcal{A}) = -1$ 인데, 이 경우 정책의 표준편차가 약 0.1
# > (= $\pm 17$ rad/s² 의 탐색 노이즈) 수준으로 유지되어 정밀한 각도 제어를 배우기 어렵습니다.
# > 여기서는 $-4$ 로 낮춰 학습 후반에 정책이 더 결정적(deterministic)이 되도록 했습니다."""))
R.append(("# - **정상상태 오차**: PID는 적분항으로 0이 되지만, RL은 적분기가 없음 → 작은 오차가 남을 수 있음",
          "# - **정상상태 오차**: PID는 적분항으로 0이 되지만, RL은 적분기 없이 feedforward + 학습된 피드백에 의존 → 작은 오차가 남을 수 있음"))
R.append((r"""# 정책은 결국 $u = \pi(\theta, \dot\theta, \theta_{ref})$ 인 **비선형 상태 피드백** 입니다.
# $\theta_{ref}=20°$, $a_{t-1}=0$ 으로 고정하고 $(\theta, \dot\theta)$ 평면에서 행동을 그려 보면
# PID의 선형 경계와 비교할 수 있습니다. 또 $\dot\theta=0$ 에서 $\theta=\theta_{ref}$ 일 때의 출력이
# 중력 보상량 $\omega_n^2\sin\theta_{ref}$ 와 얼마나 가까운지도 확인해 봅니다.""",
          r"""# 정책은 결국 $u = u_{ff} + \pi(\theta, \dot\theta, \theta_{ref})$ 인 **비선형 상태 피드백** 입니다.
# $\theta_{ref}=20°$, $a_{t-1}=0$ 으로 고정하고 $(\theta, \dot\theta)$ 평면에서 RL 피드백 부분을 그려 보면
# PID 피드백 $K_p e - K_d\dot\theta$ (직선 경계) 와 비교할 수 있습니다.
# 또 평형점 ($\theta=\theta_{ref}, \dot\theta=0$) 에서 RL 출력이 0에 가까운지도 확인합니다."""))
R.append(("""S_grid = np.stack([np.sin(TH), np.cos(TH), W / 10, ref - TH, np.full_like(TH, ref), np.zeros_like(TH)], -1)
with torch.no_grad():
    A_grid = agent.actor(torch.as_tensor(S_grid.reshape(-1, 6), dtype=torch.float32), True)[0]
U_grid = A_grid.numpy().reshape(TH.shape) * U_MAX

plt.figure(figsize=(6.5, 4.5))
cs = plt.contourf(np.rad2deg(TH), W, U_grid, levels=21, cmap="RdBu_r")
plt.colorbar(cs, label="u [rad/s^2]")
plt.axvline(20, color="k", ls="--", lw=1)
plt.xlabel("theta [deg]"); plt.ylabel("omega [rad/s]")
plt.title("learned policy u = pi(theta, omega | ref = 20 deg)")
plt.tight_layout(); plt.show()

a_ff = K * np.sin(ref) / U_MAX                     # 평형 유지에 필요한 정규화 행동
s_eq = np.array([np.sin(ref), np.cos(ref), 0, 0, ref, a_ff], np.float32)
print(f"policy output at equilibrium: u = {agent.act(s_eq, True)[0]*U_MAX:.2f}   "
      f"vs gravity compensation wn^2 sin(ref) = {K*np.sin(ref):.2f} rad/s^2")""",
          """S_grid = np.stack([np.sin(TH), np.cos(TH), W / 10, 5 * (ref - TH), np.full_like(TH, ref), np.zeros_like(TH)], -1)
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
print(f"RL feedback at equilibrium: {agent.act(s_eq, True)[0]*U_MAX:+.2f} rad/s^2  (0 이면 이상적)\\n"
      f"feedforward wn^2 sin(ref) = {K*np.sin(ref):.2f} rad/s^2")"""))
R.append((r"""# - PID는 적분항으로 모델 오차에 강건하지만, RL 정책(적분기 없음)은 **학습한 모델** 에 맞춘
#   feedforward에 의존하므로 실제 하드웨어와 모델이 다르면 정상상태 오차가 생길 수 있습니다 (sim-to-real gap).""",
          r"""# - 순수 RL($u = a\,u_{max}$)은 넓은 입력 범위 때문에 정밀도가 부족했고, **알고 있는 물리(중력 보상)를 feedforward로 더한
#   residual RL** 로 PID 수준의 정상상태 정밀도를 얻었습니다. "모델을 버리는 것"이 아니라 "모델 + 학습"의 조합이 실용적입니다.
# - PID는 적분항으로 모델 오차에 강건하지만, RL 정책(적분기 없음)은 **시뮬레이터 모델**에 맞춘
#   feedforward/피드백에 의존하므로 실제 하드웨어와 모델이 다르면 정상상태 오차가 생길 수 있습니다 (sim-to-real gap)."""))
R.append((r"""# 1. **보상 가중치 실험**: `PendulumTrackEnv.W_DA` 를 0으로 하면 입력 $u$ 가 어떻게 변하나요?
#    `W_W` 를 크게 하면 오버슈트/정착시간은?""",
          r"""# 1. **Feedforward 제거 (ablation)**: `PendulumTrackEnv(use_ff=False)`, `PIDPolicy(use_ff=False)` 로 바꿔 다시 학습하고
#    6.1의 정상상태 오차를 비교하세요. 왜 차이가 나는지 3.3의 설명과 연결해 보세요.
# 2. **보상 가중치 실험**: `PendulumTrackEnv.W_DA` 를 0으로 하면 입력 $u$ 가 어떻게 변하나요?
#    `W_W` 를 크게 하면 오버슈트/정착시간은? (학습 결과는 시드에 따라 달라지므로 시드 2~3개로 비교하는 습관을!)"""))
R.append(("# 2. **기준각 범위 밖 일반화**", "# 3. **기준각 범위 밖 일반화**"))
R.append(("# 3. **모델 오차 강건성 (sim-to-real)**: 평가 때만 `K` 를 ±20% 바꿔(`plant_deriv` 수정) 정상상태 오차를 비교하세요.",
          "# 4. **모델 오차 강건성 (sim-to-real)**: 평가 때만 플랜트의 `K` 를 ±20% 바꿔(`plant_deriv` 수정, feedforward는 그대로) 정상상태 오차를 비교하세요."))
R.append(("# 4. **제어 주기**", "# 5. **제어 주기**"))
R.append(("# 5. **Swing-up 도전**", "# 6. **Swing-up 도전**"))

for a, b in R:
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("patched", p)
