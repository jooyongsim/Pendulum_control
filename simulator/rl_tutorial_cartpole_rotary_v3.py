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
# # 모방학습과 강화학습 실습 — 기존 제어기 흉내 내기부터 PPO 까지 (CartPole · 회전형 도립진자)
#
# 이 튜토리얼은 **우리 실험 장비 (STEVAL-EDUKIT01 회전형 도립진자)** 의 동정된 모델 위에서, 신경망 제어기를 학습하는
# 여러 방법을 처음부터 차례로 익히도록 구성했습니다.
#
# | Part | 주제 | 문제 | 방법 | 배우는 것 |
# |---|---|---|---|---|
# | 0 | 시뮬레이터 & **모방학습** | 회전형 진자 | **BC, DAgger** (장비 제어기를 teacher 로) | 지도학습으로 제어기 흉내 내기, covariate shift |
# | I | 강화학습 기초 | — | MDP, 가치함수, 벨만 방정식 | 보상과 시행착오로 배우는 원리 |
# | II | 고전 벤치마크 | **CartPole** | **REINFORCE**, **DQN** | 정책 기반 vs 가치 기반 |
# | III | 실제 장비 | **회전형 진자** 균형·스윙업 | **SAC** + DQN·REINFORCE 비교, sim-to-real | 연속 행동, 보상 설계, 강건성 |
# | IV | 더 비교하기 | 회전형 진자 | **TD3, ARS** (결정적 정책), σ 설계, **PPO** | 확률적 vs 결정적 정책, 표준 알고리즘 PPO |
#
# ### 목차
# 0. 실행 환경 준비
# - **[Part 0]** A. 회전형 진자의 물리 모델과 시뮬레이터 환경 / **B. 모방학습: BC 와 DAgger**
# 1. **[Part I]** 강화학습 기초 — 에이전트·환경·보상, MDP, 리턴, 가치함수, 벨만 방정식, 알고리즘 지도
# 2. **[Part II]** CartPole 환경 직접 만들기
# 3. 정책 경사(Policy Gradient) — REINFORCE
# 4. 가치 기반(Value-based) — DQN
# 5. CartPole 결과 비교 & 애니메이션
# 6. **[Part III]** 회전형 진자 RL 환경 다시 보기
# 7. 연속 행동을 위한 SAC (Soft Actor-Critic)
# 8. 과제 A — 직립 균형(balance) 학습
# 9. 과제 B — 스윙업(swing-up) + 균형 학습
# 10. 같은 문제를 DQN, REINFORCE 로 — 세 알고리즘 비교
# 11. Sim-to-real 검증: 실제 장비 조건의 시뮬레이터에서 돌리기
# 12. 애니메이션 & 실시간 시뮬레이터
# 13. **[Part IV]** 확률적(Gaussian) 정책 vs 결정적(deterministic) 정책 — TD3, ARS, σ 를 어떻게 둘까
# 14. **PPO**
# 15. 전체 비교 — 모방학습부터 PPO 까지 (+ 결과 파일 내려받기)
# 16. 정리 & 실습 과제
#
# > **학습 결과 저장**
# > - 모든 학습의 **학습 기록** (에피소드 리턴, 평가 점수, 학습 시간) 을 모델 옆에 `.json` 으로 저장합니다.
# >   그래서 저장된 모델을 불러와도 **학습 곡선을 다시 그릴 수 있습니다.**
# > - **학습 곡선 (training curve)** 을 PNG 로 저장합니다: `rl_results/figures/`
# > - 최종 모델의 동작을 **데모 영상 (MP4, ffmpeg 가 없으면 GIF)** 으로 저장합니다: `rl_results/videos/`
# > - 마지막 15.2 절에서 결과 전체를 zip 으로 묶어 내려받을 수 있습니다 (Colab).
#
# > **노트북 변환**: 이 파일은 jupytext *percent* 형식입니다 (`# %%` 코드 셀, `# %% [markdown]` 마크다운 셀).
# > 같은 폴더의 `rl_tutorial_cartpole_rotary_v3.ipynb` 가 이 파일을 변환한 노트북입니다.
# > VS Code 에서는 `.py` 그대로 각 셀을 `Run Cell` 로 실행할 수도 있습니다.

# %% [markdown]
# ## 0. 실행 환경 준비
#
# 필요한 패키지: `numpy`, `matplotlib`, `torch` — **Google Colab 에는 모두 설치되어 있으므로 바로 실행하면 됩니다.**
# 로컬 PC 에서는 `pip install numpy matplotlib torch` (CPU 버전으로 충분).
#
# **Colab 에서 실행할 때**
# 1. 위의 *Open in Colab* 배지를 누르거나, Colab 에서 `파일 → 노트북 업로드` 로 이 `.ipynb` 를 엽니다.
# 2. (권장) `런타임 → 런타임 유형 변경 → T4 GPU` — Part III 의 SAC 학습이 빨라집니다. CPU 로도 동작합니다.
# 3. 위에서부터 차례로 셀을 실행합니다. Part III 에 필요한 장비 모델 `rotary_model.py` 는 GitHub 에서 **자동으로 내려받습니다**.
# 4. 모델·학습 기록·그림·영상은 `rl_results/` (Colab: `/content/rl_results/`) 에 저장되며 **런타임이 끊기면 사라집니다.**
#    15.2 절에서 zip 으로 묶어 내려받으세요.
#
# ```
# rl_results/
# ├── models/    *.pt (신경망 가중치) + *.json (학습 기록: 에피소드 리턴, 평가 점수, 학습 시간)
# ├── figures/   *.png (학습 곡선, 응답 그래프)
# └── videos/    *.mp4 또는 *.gif (동작 영상)
# ```
#
# **학습 시간** (대략): REINFORCE 1 분, DQN 1~2 분, 균형 SAC 5 분, 스윙업 SAC 20~40 분 (GPU 런타임이면 더 빠름),
# 10 절 DQN·REINFORCE 비교 (4 개 모델) 약 10~15 분,
# B 절 BC·DAgger 약 2 분, 13 절 TD3 약 25 분 · ARS 약 4 분 · REINFORCE σ 변형 약 15 분, 14 절 PPO (4 개 모델) 약 60~90 분.
# **전체를 처음부터 학습하면 3 시간 이상** 걸리므로, 수업에서는 `USE_PRETRAINED = True` 로 미리 학습된 모델을 쓰고
# 관심 있는 것만 `RETRAIN` 하는 것을 권합니다.
# - `RETRAIN = False` 이면 **저장된 모델이 있을 때 학습을 건너뛰고 불러옵니다.** 셀을 다시 실행해도 처음부터 학습하지 않습니다.
# - `USE_PRETRAINED = True` 이면 모델이 없을 때 GitHub 에 올려 둔 **미리 학습된 모델** 을 내려받아 씁니다.
#   수업 시간이 부족할 때 (특히 스윙업) 사용하세요.

# %%
import copy
import json
import math
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    HERE = Path(__file__).resolve().parent          # python 스크립트로 실행할 때
except NameError:
    HERE = Path.cwd()                                # Jupyter / Colab 에서 실행할 때 (Colab: /content)
sys.path.insert(0, str(HERE))

REPO_RAW = "https://raw.githubusercontent.com/jooyongsim/Pendulum_control/main/simulator"
RESULTS_DIR = HERE / "rl_results"
MODEL_DIR, FIG_DIR, VIDEO_DIR = RESULTS_DIR / "models", RESULTS_DIR / "figures", RESULTS_DIR / "videos"
for d in (MODEL_DIR, FIG_DIR, VIDEO_DIR):
    d.mkdir(parents=True, exist_ok=True)
RETRAIN = False                 # True: 저장된 모델이 있어도 다시 학습
USE_PRETRAINED = False          # True: 모델이 없으면 GitHub 의 미리 학습된 모델을 내려받음

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)        # 작은 MLP 는 단일 스레드가 가장 빠름
IN_NOTEBOOK = "ipykernel" in sys.modules
plt.rcParams["animation.embed_limit"] = 60      # 노트북 애니메이션 최대 크기 [MB]


def have_model(path):
    """학습을 건너뛰어도 되는가? (저장된 모델이 있거나, 미리 학습된 모델을 내려받았으면 True)"""
    if RETRAIN:
        return False
    if not path.exists() and USE_PRETRAINED:
        for p in (path, path.with_suffix(".json")):          # 모델 + 학습 기록
            try:
                urllib.request.urlretrieve(f"{REPO_RAW}/rl_results/models/{p.name}", p)
                print("downloaded pretrained", p.name)
            except Exception as e:
                print(f"pretrained {p.name} not available ({e})")
    return path.exists()


def _to_builtin(x):
    """json 으로 저장할 수 있게 numpy 값 -> 파이썬 기본형"""
    if isinstance(x, dict):
        return {k: _to_builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_builtin(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def load_or_train(path, model, train_fn, map_location="cpu"):
    """저장된 모델이 있으면 불러오고(학습 기록 .json 도 함께), 없으면 train_fn() 으로 학습 후 둘 다 저장.
    train_fn() -> (학습된 nn.Module, hist dict).  반환: hist (기록 파일이 없으면 None)"""
    hist_path = path.with_suffix(".json")
    if have_model(path):
        model.load_state_dict(torch.load(path, map_location=map_location))
        hist = json.loads(hist_path.read_text()) if hist_path.exists() else None
        print(f"loaded {path.name}" + ("  (+ training history)" if hist else "  (no training history file)"))
        return hist
    trained, hist = train_fn()
    if trained is not model:
        model.load_state_dict(trained.state_dict())
    torch.save(model.state_dict(), path)
    hist_path.write_text(json.dumps(_to_builtin(hist)))
    print(f"saved {path.name} + {hist_path.name}")
    return hist


def save_figure(fig, name):
    """그림을 rl_results/figures/<name>.png 로 저장"""
    p = FIG_DIR / f"{name}.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    print("figure ->", p.relative_to(RESULTS_DIR.parent))


def moving_average(x, n):
    x = np.asarray(x, float)
    if len(x) < n:
        return x
    return np.convolve(x, np.ones(n) / n, mode="valid")


# 영상 저장: ffmpeg 가 있으면 MP4, 없으면 GIF (Colab 에는 ffmpeg 가 기본 설치되어 있음)
try:
    import imageio_ffmpeg                                     # pip install imageio-ffmpeg 이면 이 ffmpeg 사용
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass
from matplotlib.animation import FFMpegWriter, PillowWriter
VIDEO_EXT = "mp4" if FFMpegWriter.isAvailable() else "gif"


def mlp(inp, out, hidden=(128, 128)):
    """완전연결 신경망: inp -> hidden... -> out  (은닉층 활성함수 ReLU)"""
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


def show_animation(ani, name):
    """애니메이션을 rl_results/videos/<name>.mp4 (또는 .gif) 로 저장하고, 노트북이면 그 영상을 보여줌"""
    fps = ani._fps
    path = VIDEO_DIR / f"{name}.{VIDEO_EXT}"
    writer = FFMpegWriter(fps=fps, bitrate=1800) if VIDEO_EXT == "mp4" else PillowWriter(fps=fps)
    ani.save(path, writer=writer)
    print(f"video  -> {path.relative_to(RESULTS_DIR.parent)}  ({path.stat().st_size / 1e6:.1f} MB)")
    if IN_NOTEBOOK:
        from IPython.display import Image, Video, display
        display(Video(str(path), embed=True, html_attributes="controls loop autoplay muted")
                if VIDEO_EXT == "mp4" else Image(filename=str(path)))
        plt.close(ani._fig)
    else:
        plt.show()


print("torch", torch.__version__, "| results ->", RESULTS_DIR, "| video format:", VIDEO_EXT)

# %% [markdown]
# ---
# # Part 0. 회전형 진자 시뮬레이터와 모방학습
#
# 강화학습을 배우기 전에, 이 튜토리얼 전체에서 쓸 **우리 장비의 시뮬레이터** 를 먼저 만들고 (A 절),
# 장비에서 이미 쓰고 있는 제어기를 **흉내 내는 신경망** 을 학습해 봅니다 (B 절 — 모방학습).
#
# ## A. 우리 장비의 물리 모델과 시뮬레이터 환경
#
# ### A.1 장비와 좌표
# STEVAL-EDUKIT01: 스테퍼 모터가 **수평 팔(arm)** 을 돌리고, 팔 끝에 **막대(pendulum)** 가 자유롭게 매달려 있습니다
# (Furuta pendulum 이라고도 함).
#
# | 기호 | 뜻 | 범위 |
# |---|---|---|
# | $\theta$ | 막대 각도, **직립 = 0**, 매달림 = $\pm\pi$ | $[-\pi, \pi)$ |
# | $\phi$ | 팔 각도 (모터 축) | 케이블 때문에 $\lvert\phi\rvert \le 140°$ |
# | $u = \ddot\phi$ | **팔 각가속도 명령** [rad/s²] — 호스트가 모터 펌웨어에 보내는 값 | $\lvert u\rvert \le 47.1$ |
#
# ### A.2 운동방정식 (동정된 모델, `rotary_model.py`)
# $$
# \ddot\theta = \underbrace{\omega_n^2\sin\theta}_{\text{gravity}}
#             + \underbrace{b\cos\theta\,\ddot\phi}_{\text{arm input}}
#             + \underbrace{\sin\theta\cos\theta\,\dot\phi^2}_{\text{centrifugal}}
#             - \underbrace{2\zeta\omega_n\dot\theta}_{\text{viscous}}
#             - \underbrace{F_c\,\mathrm{sgn}(\dot\theta)}_{\text{dry friction}}
# $$
#
# | 상수 | 값 | 출처 |
# |---|---|---|
# | $\omega_n^2 = 3g/2l$ | 56.2 s⁻² ($\omega_n$ = 7.499 rad/s, $l$ = 262 mm) | 자유 진동 실험 |
# | $\zeta$ | 0.0117 | 자유 진동 실험 |
# | $b = 3r/2l$ | 0.730 ($r$ = 127 mm) | 개루프 실험 |
# | $F_c$ | 0.556 rad/s² | 건마찰 (진폭 감소율 2.7 °/s) |
#
# **모터(펌웨어)는 이상적인 가속도원이 아닙니다** — 이 제약도 시뮬레이터에 들어 있습니다.
# - 가속도 $\lvert\ddot\phi\rvert \le 47.1$ rad/s², 속도 $\lvert\dot\phi\rvert \le 7.85$ rad/s
# - 속도가 0.059 rad/s (30 pps) 미만이면 모터가 돌지 않음 (L6474 최소 속도)
#
# ### A.3 CartPole 과 무엇이 다른가?
# | | CartPole | Rotary pendulum |
# |---|---|---|
# | 입력 | 힘 ±10 N (이산 2개) | 팔 가속도 $[-47, 47]$ rad/s² (**연속**) |
# | 제한 | 레일 ±2.4 m | 팔 ±140° (넘으면 케이블 손상 → 비상정지) |
# | 입력이 듣는 정도 | 항상 비슷 | $b\cos\theta$ — **$\theta = 90°$ 에서 0** (수평이면 제어 불가) |
# | 직립에서 버틸 수 있는 최대 기울기 | — | $\tan\theta = b\,u_{max}/\omega_n^2$ → **약 31.5°** |
# | 과제 | 균형만 | **균형** + 아래에서 **흔들어 올리기 (swing-up)** |
#
# 매달린 상태에서는 가속도 한계(47 rad/s²) 때문에 한 번에 올릴 수 없습니다 ($\omega_n^2 / b = 77$ rad/s² 가 필요).
# **팔을 좌우로 흔들어 에너지를 모아야** 하며, 이런 *비선형·다단계 전략* 을 사람이 설계하지 않고 RL 이 스스로 찾는 것이
# 이 튜토리얼 후반 (Part III) 의 목표입니다. 그 전에 B 절에서는 사람이 설계한 제어기를 신경망이 흉내 내게 해 봅니다.

# %% [markdown]
# ### A.4 환경 설계
#
# > 아래의 **보상(reward)·종료 조건** 은 강화학습 (Part I 이후) 에서 쓰는 부분입니다.
# > B 절의 모방학습은 보상을 쓰지 않으므로, 처음 읽을 때는 관측·행동·제어 주기만 보고 넘어가도 됩니다.
#
# **관측 (observation)** — 5차원, 물리 상태 그대로 (회전 각도만 표준적인 방법으로 표현)
# $$
# o = \big[\cos\theta,\ \sin\theta,\ \dot\theta,\ \phi,\ \dot\phi\big]
# $$
# - 막대는 계속 **회전** 할 수 있어서 $\theta = +179°$ 와 $-179°$ 가 사실 같은 자세입니다. 숫자 $\theta$ 를 그대로 주면
#   신경망 입장에서 가장 가까운 두 자세가 가장 먼 값이 됩니다. 그래서 회전 각도는 $(\cos\theta, \sin\theta)$ 로 표현합니다
#   (Gymnasium `Pendulum-v1`, `Acrobot` 등에서도 쓰는 표준 방법).
# - 팔 각도 $\phi$ 는 ±140° 안에서만 움직이므로 그대로 씁니다.
# - 4개 상태 $(\theta,\dot\theta,\phi,\dot\phi)$ 가 모두 들어 있으므로 마르코프 성질을 만족합니다.
#
# **행동 (action)** — $a \in [-1, 1]$ 를 $u = a \cdot u_{max}$ ($u_{max}$ = 47.1 rad/s²) 로 변환.
# 연속 행동 알고리즘은 보통 행동 범위를 $[-1,1]$ 로 정규화해서 다룹니다.
#
# **제어 주기** — 물리는 1 ms RK4 로 적분하고, 에이전트는 **20 ms (50 Hz)** 마다 행동을 정해 그 사이 입력을 유지합니다
# (zero-order hold). 실제 장비 루프는 100 Hz 이므로 장비에서는 같은 명령을 두 주기 동안 보내면 됩니다 (11 절).
# 제어 주기가 길수록 한 에피소드의 스텝 수가 줄어 학습이 빨라지지만, 너무 길면 직립 균형이 어려워집니다.
#
# **보상 (reward)**
# $$
# r = \underbrace{\tfrac{1}{2}(1+\cos\theta)}_{\text{위로 갈수록 } 0 \to 1}
#   - 0.1\left(\frac{\phi}{140°}\right)^2 - 0.01\,a^2,
# \qquad \text{팔 한계 초과 시 } -50 \text{ 후 종료}
# $$
# - 첫 항: 직립이면 1, 수평이면 0.5, 매달리면 0. *얼마나 위에 있는가* 를 연속적으로 알려 주는 **dense reward** 입니다.
#   "직립하면 +1, 아니면 0" 같은 **sparse reward** 로는 무작위 탐색으로 직립에 도달할 확률이 거의 없어 학습이 시작되지 않습니다.
# - 둘째 항: 팔을 중앙 근처에 두라 (CartPole 에서 수레가 흘러가던 문제를 미리 막음).
# - 셋째 항: 불필요하게 큰 가속도를 쓰지 마라.
#
# > **보상 설계의 함정 — 종료가 이득이 되면 안 된다.** 만약 보상이 대부분 음수라면 (예: $r = -\theta^2$),
# > 매달린 상태에서 계속 음수를 받느니 **일부러 팔을 한계까지 돌려 에피소드를 끝내는 것** 이 리턴이 더 큽니다.
# > 그래서 평상시 보상은 0 이상이 되게 하고, 종료(실패)에는 큰 벌점을 줍니다.
#
# **두 과제**
#
# | | 과제 A: balance | 과제 B: swing-up |
# |---|---|---|
# | 초기 상태 | 직립 근처 $\lvert\theta_0\rvert \le 11°$ | 70%: 매달림 근처, 30%: 무작위 각도 |
# | 종료 | 팔 한계 또는 $\lvert\theta\rvert > 30°$ (넘어짐) | 팔 한계 |
# | 길이 | 5 s (250 스텝) | 10 s (500 스텝) |
#
# swing-up 에서 **30% 를 무작위 각도에서 시작** 하는 이유: 처음부터 매달린 상태에서만 시작하면, 에이전트가 직립 근처 상태를
# 경험하기까지 오래 걸립니다. 다양한 시작점을 주면 "위에서 버티는 법" 을 일찍 배워 학습이 빨라집니다
# (초기 상태 분포도 설계 변수입니다).

# %%
try:
    import rotary_model as rm
except ImportError:          # Colab 등에서 파일이 없으면 GitHub 에서 내려받기
    urllib.request.urlretrieve(f"{REPO_RAW}/rotary_model.py", HERE / "rotary_model.py")
    import rotary_model as rm

print(f"wn^2 = {rm.A_G:.2f} 1/s^2, b = {rm.B_U}, zeta = {rm.ZETA}, Fc = {rm.F_C:.3f} rad/s^2")
print(f"motor: |u| <= {rm.ACCEL_MAX:.1f} rad/s^2, |phi_dot| <= {rm.SPEED_MAX:.2f} rad/s, "
      f"stops below {rm.SPEED_MIN:.3f} rad/s")


def wrap(a):
    """각도를 [-pi, pi) 로"""
    return (a + math.pi) % (2 * math.pi) - math.pi


class RotaryPendulumEnv:
    """우리 장비(STEVAL-EDUKIT01)의 동정 모델 위에서 동작하는 RL 환경 (Gymnasium 스타일).

    obs    = [cos th, sin th, th_dot, phi, phi_dot]      (th: 직립 = 0)
    action = a in [-1, 1]  ->  팔 가속도 u = a * ACCEL_MAX
    task   = "balance" (직립 근처에서 시작) | "swingup" (매달린 상태에서 시작)
    """

    obs_dim, act_dim = 5, 1
    DT_CTRL = 0.02                               # 제어 주기 20 ms (50 Hz)
    ARM_LIM = math.radians(140)
    FALL = math.radians(30)                      # balance 과제의 넘어짐 판정

    def __init__(self, task="balance", seed=None, b=rm.B_U, friction=1.0):
        self.task = task
        self.rng = np.random.default_rng(seed)
        self.max_steps = int((5.0 if task == "balance" else 10.0) / self.DT_CTRL)
        self.n_sub = int(round(self.DT_CTRL / rm.DT))       # 제어 1 스텝 = 물리 20 스텝 (1 ms RK4)
        self.b, self.fc = b, rm.F_C * friction              # 모델 오차 실험용 (11 절)

    # ---------------- 물리: rotary_model.Plant.deriv 와 같은 식 ----------------
    def _deriv(self, th, w, v, u):
        running = abs(v) >= rm.SPEED_MIN                    # 너무 느리면 모터가 안 돎
        acc = u if running else 0.0
        phi_dot = v if running else 0.0
        s, c = math.sin(th), math.cos(th)
        w_dot = (rm.A_G * s + self.b * c * acc + s * c * phi_dot ** 2
                 - rm.C_D * w - self.fc * math.tanh(w / 0.01))
        return w, w_dot, phi_dot, u                         # d/dt [th, w, phi, v]

    def _obs(self):
        th, w, phi, v = self.x
        phi_dot = v if abs(v) >= rm.SPEED_MIN else 0.0
        return np.array([math.cos(th), math.sin(th), w, phi, phi_dot], np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        r, o = self.rng, (options or {})
        if self.task == "balance":
            th0 = r.uniform(-0.2, 0.2)
        elif r.random() < 0.7:
            th0 = math.pi + r.uniform(-0.1, 0.1)
        else:
            th0 = r.uniform(-math.pi, math.pi)
        self.x = [o.get("theta0", th0), o.get("theta_dot0", r.uniform(-0.5, 0.5)),
                  o.get("phi0", r.uniform(-0.2, 0.2)), 0.0]   # v (모터 속도) 는 0 에서 시작
        self.k = 0
        return self._obs(), {}

    def step(self, action):
        a = float(np.clip(np.ravel(action)[0], -1.0, 1.0))
        u = a * rm.ACCEL_MAX
        th, w, phi, v = self.x
        h, d = rm.DT, self._deriv
        for _ in range(self.n_sub):                          # RK4, 입력은 20 ms 동안 유지
            k1 = d(th, w, v, u)
            k2 = d(th + h / 2 * k1[0], w + h / 2 * k1[1], v + h / 2 * k1[3], u)
            k3 = d(th + h / 2 * k2[0], w + h / 2 * k2[1], v + h / 2 * k2[3], u)
            k4 = d(th + h * k3[0], w + h * k3[1], v + h * k3[3], u)
            th += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            w += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            phi += h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
            v += h / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
            v = max(-rm.SPEED_MAX, min(rm.SPEED_MAX, v))     # 펌웨어 속도 제한
        th = wrap(th)
        self.x = [th, w, phi, v]
        self.k += 1

        reward = 0.5 * (1 + math.cos(th)) - 0.1 * (phi / self.ARM_LIM) ** 2 - 0.01 * a ** 2
        terminated = abs(phi) > self.ARM_LIM or (self.task == "balance" and abs(th) > self.FALL)
        if terminated:
            reward -= 50.0
        truncated = self.k >= self.max_steps
        return self._obs(), reward, terminated, truncated, {"u": u}


# 환경 확인: 입력 없이 직립 5° 에서 놓으면 넘어지고, 매달린 쪽으로 가서 진동해야 함
env = RotaryPendulumEnv("swingup")
obs, _ = env.reset(options=dict(theta0=math.radians(5), theta_dot0=0.0, phi0=0.0))
TH = []
for _ in range(150):
    obs, r, term, trunc, _ = env.step(0.0)
    TH.append(math.degrees(env.x[0]))
plt.figure(figsize=(7, 2.5))
plt.plot(np.arange(len(TH)) * env.DT_CTRL, TH)
plt.xlabel("t [s]"); plt.ylabel("theta from upright [deg]"); plt.title("free fall from 5 deg (u = 0)")
plt.grid(alpha=.3); plt.tight_layout(); plt.show()


# %% [markdown]
# ### A.5 실행·평가 도구
# 이 튜토리얼의 모든 방법(모방학습, REINFORCE, DQN, SAC, TD3, ARS, PPO)을 **같은 방식으로** 평가하고 그리기 위한 함수들입니다.
# 평가 대상은 `act(obs, deterministic=True)` 메서드만 있으면 됩니다.
# - `eval_balance`: 직립 근처 4 가지 초기 기울기 (5, −8, 11, −3°) 에서 5 초. 만점 ≈ 250. `falls` = 넘어지거나 팔 한계를 넘은 횟수
# - `eval_swingup`: 매달린 상태 3 가지 (180, 177, 186°) 에서 10 초. 만점 ≈ 480. `t_up` = 처음 직립 ±10° 에 들어간 시각, `hold` = 마지막 2 초 동안 직립 비율

# %%
def rollout(policy, task, theta0, seconds=None, theta_dot0=0.0, phi0=0.0, env_kw=None):
    """결정적 정책으로 한 에피소드 실행 후 궤적 반환. policy(obs) -> a"""
    env = RotaryPendulumEnv(task, **(env_kw or {}))
    if seconds:
        env.max_steps = int(seconds / env.DT_CTRL)
    obs, _ = env.reset(options=dict(theta0=theta0, theta_dot0=theta_dot0, phi0=phi0))
    log = {k: [] for k in ("t", "th", "w", "phi", "u", "r")}
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs))
        done = term or trunc
        for k, v in zip(("t", "th", "w", "phi", "u", "r"),
                        (env.k * env.DT_CTRL, env.x[0], env.x[1], env.x[2], info["u"], r)):
            log[k].append(v)
    log = {k: np.array(v) for k, v in log.items()}
    log["terminated"] = term
    return log


def plot_rotary(logs, title, fig_name=None):
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for name, lg in logs.items():
        ax[0].plot(lg["t"], np.degrees(lg["th"]), label=name + ("  (terminated)" if lg["terminated"] else ""))
        ax[1].plot(lg["t"], np.degrees(lg["phi"]))
        ax[2].plot(lg["t"], lg["u"], lw=1)
    ax[0].set_ylabel("theta from up [deg]"); ax[0].legend(fontsize=8, loc="upper right")
    ax[1].set_ylabel("arm phi [deg]")
    for s in (1, -1):
        ax[1].axhline(s * 140, color="r", ls=":", lw=1)
    ax[2].set_ylabel("u [rad/s^2]"); ax[2].set_xlabel("t [s]")
    for a_ in ax:
        a_.grid(alpha=.3)
    ax[0].set_title(title); plt.tight_layout()
    if fig_name:
        save_figure(fig, fig_name)
    plt.show()


def animate_rotary(lg, title="", fps=20):
    from matplotlib.animation import FuncAnimation
    fig = plt.figure(figsize=(9, 4.5), dpi=72)
    ax_s = fig.add_axes([0.03, 0.05, 0.45, 0.85])
    ax_t = fig.add_axes([0.52, 0.45, 0.22, 0.45])
    ax_p = fig.add_axes([0.55, 0.10, 0.42, 0.28])
    L = rm.L
    ax_s.set_aspect("equal"); ax_s.set_xlim(-1.3 * L, 1.3 * L); ax_s.set_ylim(-1.3 * L, 1.3 * L)
    ax_s.add_patch(plt.Circle((0, 0), L, fill=False, color="0.85", ls="--"))
    ax_s.plot([0], [0], "ko"); ax_s.set_title(title, fontsize=10)
    rod, = ax_s.plot([], [], lw=7, color="#E97820", solid_capstyle="round")
    txt = ax_s.text(-1.25 * L, 1.15 * L, "", family="monospace", fontsize=8)
    ax_t.set_aspect("equal"); ax_t.set_xlim(-1.2, 1.2); ax_t.set_ylim(-1.2, 1.2)
    ax_t.set_xticks([]); ax_t.set_yticks([]); ax_t.set_title("arm (top view)", fontsize=8)
    ax_t.add_patch(plt.Circle((0, 0), 1, fill=False, color="0.7"))
    for s in (1, -1):
        ax_t.plot([0, s * math.sin(math.radians(140))], [0, math.cos(math.radians(140))], "r:", lw=.8)
    arm, = ax_t.plot([], [], lw=4, color="#2774B8")
    ax_p.plot(lg["t"], np.degrees(lg["th"]), color="#E97820", lw=1)
    ax_p.set_ylim(-190, 190); ax_p.set_yticks([-180, 0, 180]); ax_p.grid(alpha=.3)
    ax_p.set_xlabel("t [s]"); ax_p.set_ylabel("theta"); cursor = ax_p.axvline(0, color="k", lw=.8)

    def update(i):
        th, phi = lg["th"][i], lg["phi"][i]
        rod.set_data([0, L * math.sin(th)], [0, L * math.cos(th)])
        rod.set_color("#2E8B57" if abs(th) < math.radians(10) else "#E97820")
        arm.set_data([0, math.sin(phi)], [0, math.cos(phi)])
        cursor.set_xdata([lg["t"][i]] * 2)
        txt.set_text(f"t={lg['t'][i]:5.2f}s theta={math.degrees(th):6.1f} arm={math.degrees(phi):6.1f}")
        return rod, arm, cursor, txt

    dt = lg["t"][1] - lg["t"][0]
    stride = max(1, round(1 / (dt * fps)))                    # 실시간 속도로 재생되도록 프레임 간격 결정
    ani = FuncAnimation(fig, update, frames=range(0, len(lg["t"]), stride), interval=1000 * dt * stride)
    ani._fig, ani._fps = fig, round(1 / (dt * stride))
    return ani


BALANCE_STARTS = [math.radians(d) for d in (5, -8, 11, -3)]


def eval_balance(agent):
    logs = [rollout(lambda o: agent.act(o, True), "balance", th0) for th0 in BALANCE_STARTS]
    falls = sum(lg["terminated"] for lg in logs)
    return np.mean([lg["r"].sum() for lg in logs]), f"falls {falls}/4"


SWINGUP_STARTS = [math.pi, math.pi - 0.05, math.pi + 0.1]


def swingup_metrics(lg, up_deg=10.0):
    up = np.abs(lg["th"]) < math.radians(up_deg)
    t_up = lg["t"][np.argmax(up)] if up.any() else None
    n_last = int(2.0 / RotaryPendulumEnv.DT_CTRL)
    hold = up[-n_last:].mean() if not lg["terminated"] else 0.0
    return t_up, hold


def eval_swingup(agent):
    logs = [rollout(lambda o: agent.act(o, True), "swingup", th0) for th0 in SWINGUP_STARTS]
    m = [swingup_metrics(lg) for lg in logs]
    t_up = " ".join("--" if t is None else f"{t:.1f}" for t, _ in m)
    return np.mean([lg["r"].sum() for lg in logs]), f"t_up {t_up:>13s} | hold {np.mean([h for _, h in m]):.2f}"


# %% [markdown]
# ---
# ## B. 모방학습 (Imitation Learning) — 기존 제어기를 "따라 배우기"
#
# 강화학습에 들어가기 전에, 더 단순한 방법부터 봅니다. 우리 장비에는 이미 **잘 동작하는 제어기** 가 있습니다
# (`rotary_model.py`, 장비의 `swing-up/swingup_control.py` 와 같은 법칙). 이 제어기를 **선생님(teacher, expert)** 으로 두고,
# 신경망이 그 행동을 흉내 내도록 **지도학습** 으로 가르칠 수 있습니다. 이것을 **모방학습 (imitation learning)** 이라 합니다.
#
# | | 모방학습 (BC, DAgger) | 강화학습 (Part I 이후) |
# |---|---|---|
# | 필요한 것 | **시연(demonstration)** — 잘하는 선생님의 (상태, 행동) | **보상 함수** + 시행착오 |
# | 학습 방법 | 지도학습 (회귀) | 보상을 최대화하는 최적화 |
# | 성능의 상한 | 대체로 **선생님 수준** | 선생님보다 나아질 수 있음 |
# | 데이터 양 | 적음 | 많음 |
#
# ### B.1 선생님: 에너지 스윙업 + 극점 배치 균형 제어기
# **스윙업 (에너지 펌핑)** — 진자의 에너지 $E = \tfrac12\dot\theta^2 + \omega_n^2(\cos\theta - 1)$ (직립 정지 = 0, 매달림 = −112) 를
# 목표값까지 올립니다. 팔 가속도가 에너지를 바꾸는 비율은 $\dot E \approx b\,\dot\theta\cos\theta\,u$ 이므로
# $$
# u = u_{pump}\cdot\mathrm{clip}\big(k_E(E_{target} - E),\,0,\,1\big)\cdot\mathrm{sgn}(\dot\theta\cos\theta) \;-\; k_{da}\dot\phi \;-\; k_{pa}\phi
# $$
# ($u_{pump}$ = 30 rad/s², $k_E$ = 0.3, $k_{da}$ = 5, $k_{pa}$ = 3 — 장비 값. 마지막 두 항은 팔을 가운데로 붙잡아 둠)
#
# **균형 (4-상태 극점 배치)** — 직립 근처 ($\lvert\theta\rvert < 20°$, $\lvert\dot\theta\rvert < 4$ rad/s) 에서는
# $$
# u = -K\,[\theta,\ \dot\theta,\ \phi,\ \dot\phi]^\top, \qquad K = \texttt{rotary\_model.place4()}\ \ (\text{poles } -6\pm6j,\ -1.5\pm1j)
# $$
#
# > 장비의 실제 제어기에는 상태 기계(swing → catch → balance, 넘어지면 다시 swing), 시작 킥, 엔코더 영점 보정이 더 있습니다.
# > 여기서는 **현재 상태만 보고 행동을 정하는** 버전으로 단순화했습니다. 모방학습의 선생님은 *어떤 상태에서든* 정답 행동을
# > 알려 줄 수 있어야 하기 때문입니다 (B.3 의 DAgger 에 꼭 필요).
#
# 먼저 선생님이 우리 RL 환경 (50 Hz) 에서 잘 동작하는지 확인합니다.

# %%
K_TEACHER = rm.place4()                      # 장비의 4-상태 극점 배치 이득 (poles -6±6j, -1.5±1j)
TP = rm.SWING                                # 장비 스윙업 파라미터 (pump, ke, etarget, kpa, kda, ...)
TEACHER_CATCH_DEG = 20.0                     # 이 각도 안 (그리고 |theta'| < 4 rad/s) 이면 균형 제어


def teacher_action(obs):
    """장비의 에너지 스윙업 + 극점 배치 균형 제어기 (상태만 보고 결정하는 버전). obs -> a in [-1, 1]"""
    c, s, w, phi, phid = (float(v) for v in obs)
    th = math.atan2(s, c)
    if abs(th) < math.radians(TEACHER_CATCH_DEG) and abs(w) < TP["catch_w"]:
        u = -float(K_TEACHER @ np.array([th, w, phi, phid]))          # 균형: u = -K x
    else:
        E = 0.5 * w * w + rm.A_G * (c - 1.0)                          # 에너지 (직립 정지 = 0)
        scale = min(1.0, max(0.0, TP["ke"] * (TP["etarget"] - E)))
        sgn = 1.0 if w * c > 0 else -1.0                              # dE/dt ~ b theta' cos(theta) u
        u = TP["pump"] * scale * sgn - TP["kda"] * phid - TP["kpa"] * phi
    return np.array([np.clip(u / rm.ACCEL_MAX, -1.0, 1.0)], np.float32)


class TeacherAgent:
    def act(self, obs, deterministic=True):
        return teacher_action(obs)

teacher = TeacherAgent()
print("teacher balance :", eval_balance(teacher))
print("teacher swing-up:", eval_swingup(teacher))
plot_rotary({f"theta0 = {math.degrees(t):.0f} deg": rollout(lambda o: teacher.act(o), "swingup", t)
             for t in SWINGUP_STARTS}, "teacher: energy swing-up + pole-placement balance", "rotary_teacher_swingup")
show_animation(animate_rotary(rollout(lambda o: teacher.act(o), "swingup", math.pi), "teacher (rig controller)"),
               "rotary_swingup_teacher")

# %% [markdown]
# ### B.2 Behavior Cloning (BC) — 시연을 지도학습으로 회귀
# 1. 선생님이 환경을 직접 움직이며 에피소드 40 개 (= 상태 2 만 개) 의 **(관측 $o$, 선생님 행동 $a^*$)** 쌍을 모읍니다.
# 2. 신경망 $\pi_\theta(o) = \tanh(f_\theta(o))$ 가 $a^*$ 를 맞히도록 **평균제곱오차** 를 최소화합니다.
# $$
# \min_\theta\ \frac{1}{N}\sum_i \big(\pi_\theta(o_i) - a^*_i\big)^2
# $$
# 보상도, 시행착오도 없습니다. 그냥 회귀 문제입니다.
#
# ### B.3 BC 의 문제: covariate shift (분포 이동) 와 DAgger
# BC 로 학습한 학생은 **선생님이 방문한 상태에서만** 정답을 배웠습니다. 학생이 실제로 제어를 하면 아주 작은 오차가 생기고,
# 그 오차 때문에 선생님이 **한 번도 가 보지 않은 상태** 로 들어갑니다. 거기서는 학생이 배운 적이 없으니 오차가 더 커지고 …
# 오차가 시간에 따라 **눈덩이처럼 커집니다** (Ross et al., 2011: 오차가 시간의 제곱에 비례해 누적).
#
# **DAgger (Dataset Aggregation)** 의 아이디어: *학생이 실제로 가는 상태* 에서 선생님에게 정답을 물어보자.
# ```
# D <- 선생님 시연 (BC 데이터)
# pi <- D 로 BC 학습
# 반복 (iteration = 1..8):
#     학생 pi 로 환경을 움직임 (에피소드 10 개)       <- 학생이 실제로 방문하는 상태
#     방문한 모든 상태 o 에 대해 선생님 행동 a*(o) 를 레이블로 붙임
#     D <- D ∪ {(o, a*(o))}                             <- 데이터 누적
#     pi <- D 전체로 다시 학습
# ```
# 선생님이 *임의의 상태에서* 정답을 줄 수 있어야 한다는 것이 조건입니다 (사람 시연이라면 이게 어렵습니다).

# %%
class BCPolicy(nn.Module):
    """obs -> a = tanh(net(obs))  (결정적 정책, 지도학습으로 teacher 행동을 회귀)"""

    def __init__(self, obs_dim=5, act_dim=1, hidden=(128, 128)):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, hidden)

    def forward(self, s):
        return torch.tanh(self.net(s))

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        return self(torch.as_tensor(obs, dtype=torch.float32)).numpy()


def collect(task, n_episodes, act_fn, seed):
    """act_fn 으로 환경을 움직이며 방문한 관측과 그 관측에서의 teacher 행동(레이블)을 모음"""
    env = RotaryPendulumEnv(task, seed=seed)
    O, A, n = [], [], 0
    for _ in range(n_episodes):
        (obs, _), done = env.reset(), False
        while not done:
            O.append(obs); A.append(teacher_action(obs))
            obs, r, term, trunc, _ = env.step(act_fn(obs))
            done = term or trunc
            n += 1
    return np.array(O, np.float32), np.array(A, np.float32)


def fit_bc(pi, O, A, epochs=30, lr=1e-3, batch=256):
    """평균제곱오차(MSE) 회귀. 반환: epoch 별 loss"""
    opt = torch.optim.Adam(pi.parameters(), lr=lr)
    O_t, A_t = torch.as_tensor(O), torch.as_tensor(A)
    losses = []
    for _ in range(epochs):
        perm = torch.randperm(len(O_t))
        tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            loss = F.mse_loss(pi(O_t[idx]), A_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        losses.append(tot / len(O_t))
    return losses


def train_bc_dagger(n_demo=40, dagger_iters=8, n_dagger=10, seed=SEED, epochs=30):
    """1) teacher 시연으로 BC  2) DAgger: 학생이 방문한 상태를 teacher 가 레이블 -> 데이터에 추가 -> 재학습"""
    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time()
    O, A = collect("swingup", n_demo, teacher_action, seed)          # teacher 가 직접 움직인 궤적
    pi = BCPolicy()
    hist = {"iter": [], "n_data": [], "loss": [], "eval_swingup": [], "eval_balance": []}

    def log(it, loss):
        hist["iter"].append(it); hist["n_data"].append(len(O)); hist["loss"].append(loss)
        sw, msg_sw = eval_swingup(pi); bal, msg_bal = eval_balance(pi)
        hist["eval_swingup"].append(sw); hist["eval_balance"].append(bal)
        print(f"iter {it:2d} | data {len(O):6d} | mse {loss:.4f} | swing-up {sw:6.1f} ({msg_sw}) | "
              f"balance {bal:6.1f} ({msg_bal}) | {time.time() - t0:5.0f}s", flush=True)

    loss = fit_bc(pi, O, A, epochs)[-1]
    log(0, loss)
    bc_state = copy.deepcopy(pi.state_dict())                        # 순수 BC 결과 (iter 0) 보관
    for it in range(1, dagger_iters + 1):
        O2, A2 = collect("swingup", n_dagger, lambda o: pi.act(o), seed + it)   # 학생이 움직임
        O, A = np.concatenate([O, O2]), np.concatenate([A, A2])
        pi = BCPolicy()                                              # 모은 데이터 전체로 처음부터 재학습
        loss = fit_bc(pi, O, A, epochs)[-1]
        log(it, loss)
    hist["time"] = time.time() - t0
    bc = BCPolicy(); bc.load_state_dict(bc_state)
    return bc, pi, hist

bc_policy, dagger_policy = BCPolicy(), BCPolicy()
p_bc, p_dag = MODEL_DIR / "rotary_bc.pt", MODEL_DIR / "rotary_dagger.pt"
if have_model(p_bc) and have_model(p_dag):
    bc_policy.load_state_dict(torch.load(p_bc)); dagger_policy.load_state_dict(torch.load(p_dag))
    il_hist = json.loads(p_dag.with_suffix(".json").read_text()) if p_dag.with_suffix(".json").exists() else None
    print("loaded", p_bc.name, p_dag.name)
else:
    bc_trained, dag_trained, il_hist = train_bc_dagger()
    bc_policy.load_state_dict(bc_trained.state_dict()); dagger_policy.load_state_dict(dag_trained.state_dict())
    for p, m in ((p_bc, bc_policy), (p_dag, dagger_policy)):
        torch.save(m.state_dict(), p)
        p.with_suffix(".json").write_text(json.dumps(_to_builtin(il_hist)))

il_results = {}                       # 15 절 전체 비교에서 사용
for name, ag in (("teacher", teacher), ("BC", bc_policy), ("DAgger", dagger_policy)):
    il_results[name] = {task: ev(ag) for task, ev in (("balance", eval_balance), ("swingup", eval_swingup))}
    print(f"{name:<8} balance {il_results[name]['balance'][0]:6.1f} ({il_results[name]['balance'][1]}) | "
          f"swing-up {il_results[name]['swingup'][0]:6.1f} ({il_results[name]['swingup'][1]})")

# %% [markdown]
# ### B.4 학습 곡선: DAgger 반복에 따른 성능
# 가로축은 DAgger 반복 횟수 (0 = 순수 BC). 평가 점수와 함께 **누적 데이터 수** 와 **학습 오차 (MSE)** 를 봅니다.
# 학습 오차는 계속 작은데 (회귀는 잘 되고 있음) 제어 성능은 처음에 나쁩니다 — **"훈련 데이터에서 정확" ≠ "제어를 잘함"** 이
# covariate shift 의 핵심입니다.

# %%
if il_hist is not None:
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    tb, ts = il_results["teacher"]["balance"][0], il_results["teacher"]["swingup"][0]
    ax[0].plot(il_hist["iter"], il_hist["eval_swingup"], "o-", color="#E97820", label="student")
    ax[0].axhline(ts, color="k", ls="--", lw=1, label="teacher"); ax[0].set_title("swing-up eval score")
    ax[1].plot(il_hist["iter"], il_hist["eval_balance"], "o-", color="#2E8B57", label="student")
    ax[1].axhline(tb, color="k", ls="--", lw=1, label="teacher"); ax[1].set_title("balance eval score")
    ax[2].plot(il_hist["iter"], il_hist["loss"], "o-", color="0.3"); ax[2].set_title("training MSE (imitation error)")
    ax2 = ax[2].twinx(); ax2.plot(il_hist["iter"], np.array(il_hist["n_data"]) / 1000, "s:", color="#2774B8")
    ax2.set_ylabel("dataset size [k samples]", color="#2774B8")
    for a_ in ax:
        a_.set_xlabel("DAgger iteration (0 = plain BC)"); a_.grid(alpha=.3)
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    fig.suptitle(f"Behavior cloning -> DAgger   (total {il_hist['time']:.0f} s)")
    plt.tight_layout(); save_figure(fig, "rotary_bc_dagger_curve"); plt.show()

logs = {name: rollout(lambda o, ag=ag: ag.act(o), "swingup", math.pi)
        for name, ag in (("teacher", teacher), ("BC", bc_policy), ("DAgger", dagger_policy))}
plot_rotary(logs, "swing-up from hanging: teacher vs BC vs DAgger", "rotary_bc_dagger_response")
for name, ag in (("BC", bc_policy), ("DAgger", dagger_policy)):
    show_animation(animate_rotary(rollout(lambda o, ag=ag: ag.act(o), "swingup", math.pi), f"{name} student"),
                   f"rotary_swingup_{name.lower()}")

# %% [markdown]
# ### B.5 해석
#
# 저자가 시드 0 으로 실행한 결과입니다 (시드에 따라 조금씩 다릅니다).
#
# | | 학습 데이터 | 스윙업 | balance |
# |---|---|---|---|
# | 선생님 (장비 제어기) | — | 2.6 s 에 올리고 **100% 유지** (414 점) | **0/4** 넘어짐 (249 점) |
# | **BC** | 선생님 시연 2 만 개 | 2.7 s 에 올라가지만 **유지 9%** (252 점) | **4/4 넘어짐** |
# | **DAgger** (8 회) | 레이블 5.5 만 개, 약 2 분 | 2.5 s 에 올리고 **100% 유지** (412 점) | **0/4** 넘어짐 (249 점) |
#
# - **BC 는 회귀 오차(MSE 0.006)가 매우 작은데도 제어에 실패했습니다.** 학습 데이터에서는 선생님을 잘 흉내 내지만,
#   실제로 제어를 하면 조금씩 어긋나 선생님이 가 본 적 없는 상태로 들어가고, 거기서부터 무너집니다 — covariate shift.
#   특히 balance 는 "직립 근처에서 거의 멈춰 있는" 상태에서 시작하는데, 선생님의 스윙업 시연에는 그런 상태가 거의 없습니다.
# - **DAgger 는 약 5 번의 반복으로 선생님 수준에 도달** 했습니다. 늘어난 데이터는 "학생이 실제로 가는 상태" 에 대한 정답이므로,
#   데이터 양 자체보다 **어떤 상태의 데이터인가** 가 중요하다는 것을 보여 줍니다. (2~3 회차에 성능이 잠시 떨어진 것은 학생이 새로운
#   실패 상태를 찾아낸 직후라서입니다 — 그 상태들에 정답이 붙으면서 다시 좋아집니다.)
# - DAgger 학생은 **선생님과 거의 같은 성능** 입니다 (2.5 s vs 2.6 s). 선생님을 *흉내* 냈으니 선생님보다 나아질 이유가 없습니다.
# - 학습에 쓴 것은 보상이 아니라 **선생님의 행동 레이블** 뿐이고, 전체 학습이 2 분이면 끝납니다. 강화학습(SAC 스윙업 20 만 스텝)과 비교해 보세요.
#
# **그럼 왜 강화학습이 필요할까?**
# - 모방학습은 **선생님이 있어야** 합니다. 좋은 제어기를 설계하기 어려운 문제(복잡한 로봇, 접촉이 많은 조작)에서는 선생님이 없습니다.
# - 모방학습의 성능은 대체로 **선생님 수준이 상한** 입니다. 강화학습은 보상만 주면 선생님보다 나은 해를 찾을 수 있습니다
#   (9 절의 SAC 는 같은 장비 제약에서 스윙업을 약 1.0 초에 끝냅니다 — 선생님은 약 2.6 초).
# - 실무에서는 둘을 **조합** 합니다: 모방학습으로 초기 정책을 만들고 강화학습으로 다듬기 (실습 과제).
#
# 이제 보상과 시행착오로 배우는 강화학습으로 넘어갑니다.


# %% [markdown]
# ---
# # Part I. 강화학습 기초
#
# ## 1. 강화학습이란?
#
# 지도학습(supervised learning)은 *정답 레이블* 이 주어진 데이터로 배웁니다.
# 강화학습은 정답이 없습니다. 대신 **에이전트(agent)** 가 **환경(environment)** 안에서 행동해 보고,
# 그 결과로 받는 **보상(reward)** 이라는 *점수* 만으로 "어떻게 행동해야 점수를 많이 받는지" 를 스스로 찾아냅니다.
#
# ```
#              상태 s_t, 보상 r_t
#     ┌───────────────────────────────┐
#     │                               │
#     ▼                               │
#  ┌──────────┐   행동 a_t   ┌────────┴────┐
#  │ 에이전트 │ ───────────▶ │    환경     │
#  │  π(a|s)  │              │ (진자 물리) │
#  └──────────┘              └─────────────┘
# ```
#
# 제어공학 용어로 바꾸면: **에이전트 = 제어기**, **환경 = 플랜트(+센서, 구동기)**, **행동 = 제어 입력 $u$**,
# **상태 = 측정값/상태변수 $x$**, **보상 = (음의) 비용 함수**. 즉 강화학습은
# *"모델 식을 몰라도, 시뮬레이터(또는 실험)와 비용 함수만으로 최적 제어기를 데이터로 찾는 방법"* 입니다.
#
# ### 1.1 마르코프 결정 과정 (MDP)
# 강화학습 문제는 보통 MDP $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$ 로 정식화합니다.
#
# | 기호 | 이름 | CartPole 예 |
# |---|---|---|
# | $\mathcal{S}$ | 상태 공간 | $(x, \dot x, \theta, \dot\theta)$ |
# | $\mathcal{A}$ | 행동 공간 | {왼쪽으로 밀기, 오른쪽으로 밀기} |
# | $P(s'\mid s,a)$ | 상태 전이 (환경의 물리) | 운동방정식을 0.02 s 적분 |
# | $r(s,a)$ | 보상 | 쓰러지지 않으면 매 스텝 +1 |
# | $\gamma\in[0,1)$ | 할인율 | 0.99 |
#
# **마르코프 성질**: 다음 상태는 *현재 상태와 행동* 에만 의존하고 과거 이력에는 의존하지 않는다.
# → 상태를 설계할 때 "미래를 예측하는 데 필요한 정보를 모두 담았는가?" 를 늘 확인해야 합니다
# (예: 각도만 주고 각속도를 빼면 마르코프가 아님).
#
# ### 1.2 리턴(return) 과 할인율
# 에이전트가 최대화하려는 것은 *당장의 보상* 이 아니라 **앞으로 받을 보상의 합** 입니다.
# $$
# G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots = \sum_{k=0}^{\infty}\gamma^k r_{t+k}
# $$
# - $\gamma$ 가 1에 가까울수록 먼 미래까지 고려합니다. **유효 시야(horizon)** $\approx 1/(1-\gamma)$ 스텝.
#   예: $\gamma=0.99$, 제어 주기 0.01 s → 100 스텝 = **1 초 앞까지** 내다보는 제어기.
# - 제어 문제에서는 시스템의 정착시간보다 시야가 길어야 의미 있는 제어기가 학습됩니다.
#
# ### 1.3 정책과 가치함수
# - **정책(policy)** $\pi(a\mid s)$: 상태 $s$ 에서 행동 $a$ 를 고를 확률 (= 제어 법칙). 결정적 정책이면 $a=\pi(s)$.
# - **상태가치 함수** $V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$: 상태 $s$ 에서 정책 $\pi$ 를 따랐을 때 기대 리턴.
# - **행동가치 함수** $Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid s_t=s, a_t=a]$: 지금 $a$ 를 하고 이후 $\pi$ 를 따를 때 기대 리턴.
# - **어드밴티지** $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$: 행동 $a$ 가 *평균보다 얼마나 나은가*.
#
# ### 1.4 벨만 방정식 (Bellman equation)
# 리턴의 재귀 구조 $G_t = r_t + \gamma G_{t+1}$ 에서 바로 나옵니다.
# $$
# Q^\pi(s,a) = \mathbb{E}\big[\,r + \gamma\, Q^\pi(s', a')\,\big], \quad a'\sim\pi(\cdot\mid s')
# $$
# 최적 정책의 Q 함수 $Q^*$ 는 **벨만 최적 방정식** 을 만족합니다.
# $$
# Q^*(s,a) = \mathbb{E}\big[\,r + \gamma \max_{a'} Q^*(s', a')\,\big]
# $$
# $Q^*$ 만 알면 최적 정책은 $\pi^*(s) = \arg\max_a Q^*(s,a)$ 로 바로 얻어집니다 → **DQN 의 아이디어**.
#
# > 제어공학과의 연결: 선형 시스템 + 이차 비용이면 $V^*(x) = -x^\top P x$ 이고, 벨만 방정식이
# > **리카티(Riccati) 방정식** 이 됩니다. 즉 LQR 은 "모델을 알 때 벨만 방정식을 닫힌 형태로 푼 것" 이고,
# > 강화학습은 "모델 없이 데이터로 근사해서 푼 것" 입니다.
#
# ### 1.5 탐색과 활용 (exploration vs exploitation)
# 지금까지 가장 좋았던 행동만 하면(활용) 더 좋은 행동을 영영 못 찾고, 무작위로만 하면(탐색) 배운 것을 못 씁니다.
# 각 알고리즘은 이 균형을 다르게 잡습니다: DQN 은 $\varepsilon$-greedy, REINFORCE·SAC 는 *확률적 정책* 자체.
#
# ### 1.6 알고리즘 지도 (이 튜토리얼에서 다루는 것)
#
# | 분류 | 무엇을 학습? | 데이터 사용 | 행동 공간 | 대표 알고리즘 |
# |---|---|---|---|---|
# | 정책 기반 (policy-based) | 정책 $\pi_\theta$ 를 직접 | **on-policy** (현재 정책의 데이터만, 쓰고 버림) | 이산·연속 | **REINFORCE**, A2C, PPO |
# | 가치 기반 (value-based) | $Q(s,a)$ → $\arg\max$ | **off-policy** (과거 데이터 재사용: replay buffer) | **이산만** | **DQN**, Double DQN |
# | Actor-Critic | 정책(actor) + 가치(critic) | on/off-policy 모두 있음 | 연속에 강함 | DDPG, TD3, **SAC** |
#
# - **on-policy**: 데이터 효율은 낮지만 단순하고 안정적인 이론 / **off-policy**: 한 번 모은 데이터를 여러 번 재사용 → 데이터 효율 높음.
# - DQN 은 $\max_a Q$ 를 계산해야 하므로 행동이 몇 개뿐일 때(이산) 적합합니다.
#   진자의 *연속적인* 팔 가속도를 다루려면 Part III 의 actor-critic(SAC) 이 필요합니다.

# %% [markdown]
# ---
# # Part II. CartPole
#
# ## 2. CartPole 환경 직접 만들기
#
# 마찰 없는 레일 위의 수레(cart)에 막대(pole)가 힌지로 달려 있고, 수레를 좌/우로 **10 N 으로 밀어서** 막대를 세워 둡니다.
# 강화학습의 "Hello world" 로 쓰이는 고전 문제입니다 (Barto, Sutton & Anderson, 1983).
# 물리 식과 규칙은 RL 교재·벤치마크에서 쓰는 **표준 CartPole** 과 같고, 시뮬레이터는 이 노트북에서 **직접 구현** 합니다.
# 라이브러리 없이 직접 만들어 보면 *환경(environment)이 무엇인지* — 결국 "상태를 받아 물리를 한 스텝 적분하고 보상을 돌려주는 함수" —
# 가 명확해지고, Part III 에서 우리 장비의 시뮬레이터를 같은 방식으로 감쌀 수 있습니다.
#
# ### 2.1 운동방정식
# 수레 질량 $m_c=1$ kg, 막대 질량 $m_p=0.1$ kg, 막대 반길이 $l=0.5$ m, $g=9.8$, 힘 $F=\pm10$ N.
# $\theta=0$ 이 **직립**(위쪽) 입니다.
# $$
# \ddot\theta = \frac{g\sin\theta - \cos\theta\cdot\dfrac{F + m_p l\dot\theta^2\sin\theta}{m_c+m_p}}
#                    {l\left(\dfrac{4}{3} - \dfrac{m_p\cos^2\theta}{m_c+m_p}\right)},
# \qquad
# \ddot x = \frac{F + m_p l\,(\dot\theta^2\sin\theta - \ddot\theta\cos\theta)}{m_c+m_p}
# $$
# 표준 CartPole 과 같게 **오일러 적분** 을 $\Delta t = 0.02$ s 로 합니다.
#
# ### 2.2 MDP 정의
# | 항목 | 내용 |
# |---|---|
# | 상태 $s$ | $(x,\ \dot x,\ \theta,\ \dot\theta)$ — 물리 상태 그대로, 4차원 연속 |
# | 행동 $a$ | 0 = 왼쪽으로 $-10$ N, 1 = 오른쪽으로 $+10$ N — **이산 2개** (bang-bang) |
# | 보상 $r$ | 매 스텝 **+1** (살아 있는 시간 = 점수) |
# | 종료 (terminated) | $\lvert x\rvert > 2.4$ m (레일 이탈) 또는 $\lvert\theta\rvert > 12°$ (넘어짐) → **실패** |
# | 잘림 (truncated) | 500 스텝(10 초) 도달 → **시간 제한**. 만점 = 500 |
# | 초기 상태 | 네 변수 모두 $U(-0.05, 0.05)$ |
#
# ### 2.3 환경 인터페이스: `reset()` / `step()`
# 이 튜토리얼의 모든 환경(CartPole, 회전형 진자)은 RL 분야의 사실상 표준인 **Gymnasium 스타일 인터페이스** 를 따릅니다
# (gymnasium 패키지를 설치할 필요는 없고, 메서드 형태만 맞춘 것입니다).
#
# ```python
# obs, info = env.reset()                                   # 새 에피소드 시작
# obs, reward, terminated, truncated, info = env.step(a)    # 행동 a 를 한 제어주기 적용
# ```
# 이렇게 맞춰 두면 **학습 알고리즘과 환경이 분리** 됩니다. 같은 학습 코드에 환경만 바꿔 끼울 수 있고,
# 나중에 Stable-Baselines3 같은 라이브러리에도 그대로 연결할 수 있습니다.
#
# ### 2.4 terminated vs truncated — 흔한 실수
# 둘 다 에피소드를 끝내지만 의미가 다릅니다.
# - **terminated** (실패): 그 뒤로 받을 보상이 정말 0 → 가치 $V(s')=0$.
# - **truncated** (시간 제한): 물리적으로는 계속 갈 수 있었는데 우리가 끊은 것 → 미래 가치는 여전히 존재.
#   벨만 타깃에서 $\gamma V(s')$ 를 **계속 더해야(bootstrap)** 합니다. 이를 0 으로 두면, 에이전트가 시간 제한 직전 상태를
#   "나쁜 상태" 로 오해합니다.

# %%
class CartPoleEnv:
    """표준 CartPole 시뮬레이터 (Gymnasium 스타일 인터페이스).
    reset() -> obs, info      step(a) -> obs, reward, terminated, truncated, info"""

    obs_dim, n_actions = 4, 2
    G, M_CART, M_POLE, L_HALF, FORCE, DT = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    X_LIM, TH_LIM, MAX_STEPS = 2.4, math.radians(12), 500

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.push = 0.0                        # 외란 실험용: 다음 스텝에 더할 힘 [N]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        state = (options or {}).get("state")   # 초기 상태를 직접 지정할 때
        self.s = self.rng.uniform(-0.05, 0.05, 4) if state is None else np.array(state, float)
        self.k = 0
        return self.s.astype(np.float32), {}

    def step(self, action):
        x, x_dot, th, th_dot = self.s
        force = (self.FORCE if action == 1 else -self.FORCE) + self.push
        self.push = 0.0
        m_tot = self.M_CART + self.M_POLE
        pml = self.M_POLE * self.L_HALF
        c, s = math.cos(th), math.sin(th)
        tmp = (force + pml * th_dot ** 2 * s) / m_tot
        th_acc = (self.G * s - c * tmp) / (self.L_HALF * (4 / 3 - self.M_POLE * c ** 2 / m_tot))
        x_acc = tmp - pml * th_acc * c / m_tot
        # 오일러 적분 (표준 CartPole 과 같은 순서)
        x, x_dot = x + self.DT * x_dot, x_dot + self.DT * x_acc
        th, th_dot = th + self.DT * th_dot, th_dot + self.DT * th_acc
        self.s = np.array([x, x_dot, th, th_dot])
        self.k += 1
        terminated = bool(abs(x) > self.X_LIM or abs(th) > self.TH_LIM)
        truncated = self.k >= self.MAX_STEPS
        return self.s.astype(np.float32), 1.0, terminated, truncated, {}


def run_cartpole(policy, n=10, seed=123, record=False):
    """policy(obs)->action 으로 n 에피소드 실행. 평균 점수 (record=True 면 마지막 에피소드 궤적도) 반환"""
    env = CartPoleEnv(seed)
    scores, traj = [], []
    for _ in range(n):
        obs, info = env.reset()
        done, score, traj = False, 0.0, [env.s.copy()]
        while not done:
            obs, r, term, trunc, info = env.step(policy(obs))
            score += r
            done = term or trunc
            traj.append(env.s.copy())
        scores.append(score)
    return (np.mean(scores), np.array(traj)) if record else np.mean(scores)


random_policy = lambda obs: np.random.randint(2)
print(f"무작위 정책 평균 점수: {run_cartpole(random_policy, n=100):.1f} / 500")

# %% [markdown]
# 무작위로 밀면 평균 20 스텝(0.4 초) 정도 버팁니다. 이것이 학습의 출발점입니다.

# %% [markdown]
# ### 2.5 애니메이션 도구
# 학습 전후의 행동을 눈으로 보기 위한 함수입니다. `traj` 는 매 스텝의 $(x,\dot x,\theta,\dot\theta)$ 배열입니다.

# %%
def animate_cartpole(traj, title="", stride=2):
    from matplotlib.animation import FuncAnimation
    fig, ax = plt.subplots(figsize=(7, 3), dpi=72)
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-0.3, 1.3); ax.set_aspect("equal")
    ax.axhline(0, color="0.6", lw=1)
    for xl in (-2.4, 2.4):
        ax.axvline(xl, color="r", ls=":", lw=1)
    cart = plt.Rectangle((0, 0), 0.4, 0.2, color="#2774B8")
    ax.add_patch(cart)
    pole, = ax.plot([], [], lw=5, color="#E97820", solid_capstyle="round")
    txt = ax.text(-2.5, 1.15, "", family="monospace", fontsize=9)
    ax.set_title(title)
    L_DRAW = 2 * CartPoleEnv.L_HALF

    def update(i):
        x, _, th, _ = traj[i]
        cart.set_xy((x - 0.2, -0.1))
        pole.set_data([x, x + L_DRAW * math.sin(th)], [0.1, 0.1 + L_DRAW * math.cos(th)])
        txt.set_text(f"step {i:3d}  x={x:+.2f} m  theta={math.degrees(th):+5.1f} deg")
        return cart, pole, txt

    ani = FuncAnimation(fig, update, frames=range(0, len(traj), stride),
                        interval=1000 * CartPoleEnv.DT * stride, blit=False)
    ani._fig, ani._fps = fig, round(1 / (CartPoleEnv.DT * stride))
    return ani


_, traj_random = run_cartpole(random_policy, n=1, record=True)
show_animation(animate_cartpole(traj_random, "random policy"), "cartpole_random")

# %% [markdown]
# ---
# ## 3. 정책 경사 (Policy Gradient) — REINFORCE
#
# ### 3.1 아이디어
# 정책을 신경망 $\pi_\theta(a\mid s)$ 로 두고, 기대 리턴 $J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}[G_0]$ 를
# **경사 상승법** 으로 직접 키웁니다. 문제는 $J$ 가 환경의 물리(미분 불가)를 거쳐 계산된다는 점인데,
# **로그 미분 트릭** $\nabla_\theta \pi = \pi\,\nabla_\theta\log\pi$ 를 쓰면 환경을 미분하지 않고도 기울기를 얻습니다.
#
# **정책 경사 정리 (policy gradient theorem)**
# $$
# \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\Big[\sum_t \nabla_\theta \log\pi_\theta(a_t\mid s_t)\; G_t\Big]
# $$
# 직관: **리턴이 컸던 행동의 확률은 올리고, 작았던 행동의 확률은 내린다.**
# $\log\pi$ 의 기울기 방향으로, 그 행동이 얼마나 좋았는지($G_t$) 만큼 움직입니다.
#
# ### 3.2 알고리즘 (Williams, 1992)
# ```
# repeat:
#     현재 정책으로 에피소드 하나를 끝까지 실행 → (s_t, a_t, r_t) 기록
#     각 시점의 reward-to-go  G_t = r_t + γ r_{t+1} + ...  계산
#     G_t 를 정규화 (평균 0, 표준편차 1)      ← 분산 감소용 baseline
#     loss = - Σ_t log π(a_t|s_t) · G_t       (경사 '상승' 을 하려고 부호 반대로 해서 최소화)
#     경사하강 1 스텝, 데이터는 버림         ← on-policy
# ```
# - **reward-to-go**: 시점 $t$ 의 행동은 과거 보상에 영향을 줄 수 없으므로 $t$ 이후 보상만 씁니다.
# - **baseline**: $G_t$ 에서 상수(여기서는 평균)를 빼도 기울기의 기댓값은 그대로이고 **분산만 줄어듭니다**.
#   정규화하면 "이번 에피소드에서 평균보다 나았던 행동만 강화" 하는 효과가 생깁니다.
#
# ### 3.3 정책 신경망
# 출력은 행동별 **로짓(logit)** 2개 → softmax → 범주형(Categorical) 분포. 학습 중에는 이 분포에서 *샘플링* 하므로
# 탐색이 자동으로 됩니다. 평가 때는 확률이 가장 큰 행동(argmax)을 고릅니다.

# %%
class PolicyNet(nn.Module):
    def __init__(self, obs_dim=4, n_actions=2):
        super().__init__()
        self.net = mlp(obs_dim, n_actions, hidden=(64, 64))

    def dist(self, obs):
        return torch.distributions.Categorical(logits=self.net(obs))

    @torch.no_grad()
    def greedy(self, obs):
        return int(self.net(torch.as_tensor(obs)).argmax())


def train_reinforce(n_episodes=600, gamma=0.99, lr=2e-3, eval_every=25, seed=SEED):
    torch.manual_seed(seed)
    env = CartPoleEnv(seed)
    pi = PolicyNet()
    opt = torch.optim.Adam(pi.parameters(), lr=lr)
    hist = {"ep": [], "ret": [], "eval_ep": [], "eval": []}
    best, best_state, t0 = -1, None, time.time()
    for ep in range(1, n_episodes + 1):
        # (1) 현재 정책으로 에피소드 1개 수집
        (obs, _), done, logps, rewards = env.reset(), False, [], []
        while not done:
            d = pi.dist(torch.as_tensor(obs))
            a = d.sample()
            logps.append(d.log_prob(a))
            obs, r, term, trunc, _ = env.step(int(a))
            rewards.append(r)
            done = term or trunc
        # (2) reward-to-go 계산 (뒤에서부터 누적)
        G, returns = 0.0, []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.append(G)
        returns = torch.tensor(returns[::-1])
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)      # baseline + 스케일 정규화
        # (3) 정책 경사 1 스텝
        loss = -(torch.stack(logps) * returns).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        hist["ep"].append(ep); hist["ret"].append(len(rewards))
        # (4) 주기적 평가 (탐색 없이 greedy) + 최고 성능 모델 보관
        if ep % eval_every == 0:
            score = run_cartpole(pi.greedy)
            hist["eval_ep"].append(ep); hist["eval"].append(score)
            if score >= best:
                best, best_state = score, copy.deepcopy(pi.state_dict())
            print(f"episode {ep:4d} | train return(last 25) {np.mean(hist['ret'][-25:]):6.1f} | "
                  f"eval {score:6.1f} | {time.time() - t0:5.1f}s", flush=True)
    pi.load_state_dict(best_state)
    hist["time"] = time.time() - t0
    return pi, hist


pg_policy = PolicyNet()
pg_hist = load_or_train(MODEL_DIR / "cartpole_reinforce.pt", pg_policy, train_reinforce)
print(f"REINFORCE (best checkpoint) 평가 점수: {run_cartpole(pg_policy.greedy, n=20):.1f} / 500")

# %% [markdown]
# ### 3.4 학습 곡선 읽기
# - 회색 점: 학습 에피소드의 길이(= 리턴). 확률적 정책으로 탐색 중이라 들쭉날쭉합니다.
# - 주황 선: 10 개 고정 시드 에피소드에서 greedy 정책의 평균 점수.
#
# REINFORCE 는 500 점에 도달했다가 **갑자기 무너지는 현상** 이 자주 보입니다. 원인:
# 1. **높은 분산**: 에피소드 1 개로 추정한 기울기는 잡음이 매우 큽니다.
# 2. **on-policy**: 정책이 나빠지면 나쁜 데이터만 모이고, 그 데이터로 다시 학습 → 회복이 느립니다.
# 3. **스텝 크기**: 한 번의 큰 업데이트가 좋은 정책을 망가뜨릴 수 있습니다 → 이 문제를 해결한 것이 **PPO** (clipping 으로 정책 변화량 제한).
#
# 그래서 실무에서는 **평가 점수가 가장 높았던 체크포인트를 저장** 합니다 (위 코드의 `best_state`).
#
# **학습 곡선 그리는 법** — 강화학습의 학습 곡선은 보통 두 가지를 같이 봅니다.
# 1. **학습 에피소드 리턴** (회색 점) + **이동 평균** (검은 선): 탐색 중인 정책이 실제로 받은 보상. 잡음이 커서 이동 평균으로 추세를 봅니다.
# 2. **평가 점수** (주황): 탐색을 끈 정책을 고정된 시나리오에서 돌린 점수. "지금 정책이 얼마나 좋은가" 의 공정한 척도입니다.
#
# 가로축은 **환경 스텝 수** 로 두는 것이 표준입니다. 에피소드 길이가 다르면 에피소드 수로는 데이터 양을 비교할 수 없기 때문입니다.
# 그림은 `rl_results/figures/` 에 저장됩니다.

# %%
def plot_training_curve(hist, title, name, x="step", x_eval="eval_step", xlabel="environment step",
                        max_score=None, ma=20, extra=None):
    """학습 곡선: 왼쪽 = 학습 에피소드 리턴 (+이동 평균), 오른쪽 = 평가 점수. extra=(key, label) 이면 세 번째 칸."""
    if hist is None:
        print(f"{title}: 학습 기록이 없습니다 (기록 파일 없이 모델만 불러옴)")
        return
    n = 3 if extra else 2
    fig, ax = plt.subplots(1, n, figsize=(5.5 * n, 3.4))
    xs, ret = np.asarray(hist[x]), np.asarray(hist["ret"], float)
    ax[0].plot(xs, ret, ".", color="0.65", ms=3, label="training episode")
    if len(ret) >= ma:
        ax[0].plot(xs[ma - 1:], moving_average(ret, ma), "k", lw=1.5, label=f"moving average ({ma})")
    ax[0].set_title("training episode return"); ax[0].set_ylabel("return")
    ax[1].plot(hist[x_eval], hist["eval"], "o-", color="#E97820", ms=4)
    best = int(np.argmax(hist["eval"]))
    ax[1].plot(hist[x_eval][best], hist["eval"][best], "*", color="r", ms=14, label="best (saved)")
    ax[1].set_title("evaluation score (deterministic policy)")
    for a_ in ax[:2]:
        if max_score is not None:
            a_.axhline(max_score, color="k", ls="--", lw=1)
        a_.legend(fontsize=8, loc="lower right")
    if extra:
        ax[2].semilogy(hist[x_eval], hist[extra[0]], "o-", ms=4); ax[2].set_title(extra[1])
    for a_ in ax:
        a_.set_xlabel(xlabel); a_.grid(alpha=.3)
    t = hist.get("time")
    fig.suptitle(title + (f"   (training time {t / 60:.1f} min)" if t else ""))
    plt.tight_layout()
    save_figure(fig, name)
    plt.show()


plot_training_curve(pg_hist, "REINFORCE on CartPole", "cartpole_reinforce_curve", x="ep", x_eval="eval_ep",
                    xlabel="episode", max_score=500)

# %% [markdown]
# ---
# ## 4. 가치 기반 (Value-based) — DQN
#
# ### 4.1 Q-learning
# 벨만 최적 방정식 $Q^*(s,a) = \mathbb{E}[r + \gamma\max_{a'}Q^*(s',a')]$ 을 **반복 대입** 으로 풉니다.
# 경험 $(s,a,r,s')$ 하나를 볼 때마다
# $$
# Q(s,a) \leftarrow Q(s,a) + \alpha\big(\underbrace{r + \gamma\max_{a'}Q(s',a')}_{\text{TD target } y} - Q(s,a)\big)
# $$
# $y - Q(s,a)$ 를 **TD 오차(temporal-difference error)** 라 합니다. 상태가 연속이라 표(table)로 저장할 수 없으니,
# $Q$ 를 신경망 $Q_\theta(s,\cdot)$ (입력: 상태, 출력: 행동별 Q 값 2개) 로 근사한 것이 **DQN** (Mnih et al., 2015) 입니다.
#
# ### 4.2 신경망 + Q-learning 을 안정화하는 장치
# 그냥 합치면 발산하기 쉽습니다. DQN 의 핵심 기법:
#
# | 기법 | 문제 | 해결 |
# |---|---|---|
# | **Experience replay** | 연속된 샘플은 서로 강하게 상관 → SGD 가정 위반 | 경험을 버퍼에 쌓고 **무작위 미니배치** 로 학습 (off-policy, 데이터 재사용) |
# | **Target network** $Q_{\bar\theta}$ | 타깃 $y$ 가 학습 중인 $Q_\theta$ 에 의존 → "움직이는 과녁" | 타깃은 느리게 따라오는 복사본으로 계산: $\bar\theta \leftarrow \tau\theta + (1-\tau)\bar\theta$ |
# | **Double DQN** | $\max$ 연산이 잡음까지 골라서 Q 를 **과대추정** | 행동 선택은 $Q_\theta$, 평가는 $Q_{\bar\theta}$: $y = r + \gamma Q_{\bar\theta}(s', \arg\max_{a'} Q_\theta(s',a'))$ |
# | **Huber loss** | 큰 TD 오차가 큰 기울기 → 불안정 | 오차가 크면 L1 처럼 동작 |
# | **$\varepsilon$-greedy** | 탐색 | 확률 $\varepsilon$ 로 무작위 행동, $\varepsilon$: 1 → 0.05 로 선형 감소 |
#
# ### 4.3 알고리즘
# ```
# for step = 1 .. TOTAL:
#     ε-greedy 로 행동 a 선택, 환경 1 스텝 → (s, a, r, s', terminated) 를 버퍼에 저장
#     버퍼에서 미니배치 샘플 → TD target y (Double DQN) → Huber(Q(s,a), y) 최소화
#     target network 를 Polyak 평균으로 조금씩 갱신
# ```
# REINFORCE 는 *에피소드 단위* 로 학습하지만, DQN 은 **매 스텝** 학습하고 데이터를 **재사용** 합니다.

# %%
class ReplayBuffer:
    """고정 크기 원형 버퍼. 가득 차면 가장 오래된 경험부터 덮어씀"""

    def __init__(self, obs_dim, act_dim, size, act_dtype=np.float32):
        self.s = np.zeros((size, obs_dim), np.float32)
        self.a = np.zeros((size, act_dim), act_dtype)
        self.r = np.zeros((size, 1), np.float32)
        self.s2 = np.zeros((size, obs_dim), np.float32)
        self.d = np.zeros((size, 1), np.float32)
        self.ptr, self.n, self.size = 0, 0, size

    def add(self, s, a, r, s2, d):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
        self.ptr = (i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, batch):
        idx = np.random.randint(0, self.n, batch)
        return tuple(torch.as_tensor(x[idx]) for x in (self.s, self.a, self.r, self.s2, self.d))


class QNet(nn.Module):
    def __init__(self, obs_dim=4, n_actions=2):
        super().__init__()
        self.net = mlp(obs_dim, n_actions, hidden=(128, 128))

    def forward(self, s):
        return self.net(s)

    @torch.no_grad()
    def greedy(self, obs):
        return int(self.net(torch.as_tensor(obs)).argmax())


def train_dqn(total_steps=50_000, gamma=0.99, lr=5e-4, batch=128, buffer_size=50_000,
              eps_decay_steps=10_000, eps_end=0.05, learn_start=1_000, tau=0.005,
              eval_every=2_500, seed=SEED):
    torch.manual_seed(seed); np.random.seed(seed)
    env = CartPoleEnv(seed)
    q = QNet()
    q_targ = copy.deepcopy(q)
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buf = ReplayBuffer(4, 1, buffer_size, act_dtype=np.int64)
    hist = {"step": [], "ret": [], "eval_step": [], "eval": []}
    (obs, _), ep_len, best, best_state, t0 = env.reset(), 0, -1, None, time.time()
    for step in range(1, total_steps + 1):
        # (1) ε-greedy 행동
        eps = max(eps_end, 1.0 - step / eps_decay_steps)
        a = np.random.randint(2) if np.random.rand() < eps else q.greedy(obs)
        obs2, r, term, trunc, _ = env.step(a)
        buf.add(obs, a, r, obs2, float(term))             # d = terminated 만! (truncated 는 bootstrap)
        obs, ep_len = obs2, ep_len + 1
        if term or trunc:
            hist["step"].append(step); hist["ret"].append(ep_len)
            (obs, _), ep_len = env.reset(), 0
        # (2) 미니배치 학습
        if step >= learn_start:
            s, a_b, r_b, s2, d = buf.sample(batch)
            with torch.no_grad():
                a2 = q(s2).argmax(1, keepdim=True)                         # Double DQN: 선택은 온라인망
                y = r_b + gamma * (1 - d) * q_targ(s2).gather(1, a2)      #             평가는 타깃망
            q_sa = q(s).gather(1, a_b)
            loss = F.smooth_l1_loss(q_sa, y)                               # Huber loss
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()
            with torch.no_grad():                                          # (3) target network (Polyak)
                for p, pt in zip(q.parameters(), q_targ.parameters()):
                    pt.mul_(1 - tau).add_(tau * p)
        # (4) 평가 + 최고 모델 보관
        if step % eval_every == 0:
            score = run_cartpole(q.greedy)
            hist["eval_step"].append(step); hist["eval"].append(score)
            if score >= best:
                best, best_state = score, copy.deepcopy(q.state_dict())
            print(f"step {step:6d} | eps {eps:.2f} | eval {score:6.1f} | {time.time() - t0:5.1f}s", flush=True)
    q.load_state_dict(best_state)
    hist["time"] = time.time() - t0
    return q, hist


dqn_q = QNet()
dqn_hist = load_or_train(MODEL_DIR / "cartpole_dqn.pt", dqn_q, train_dqn)
print(f"DQN (best checkpoint) 평가 점수: {run_cartpole(dqn_q.greedy, n=20):.1f} / 500")

# %%
plot_training_curve(dqn_hist, "DQN on CartPole", "cartpole_dqn_curve", max_score=500)

# %% [markdown]
# ### 4.4 학습 곡선 읽기
# - 처음 1 만 스텝은 $\varepsilon$ 이 커서 거의 무작위 → 점수가 낮은 것이 정상입니다.
# - DQN 도 500 점 근처에서 오르내립니다. CartPole 은 **성공하면 매 스텝 보상이 똑같이 +1** 이라
#   버퍼가 "잘 버티는" 데이터로만 채워지면 실패 상황을 잊어버리는(catastrophic forgetting) 경향이 있습니다.
# - **x 축 단위 비교**: REINFORCE 600 에피소드는 약 15 만 환경 스텝, DQN 은 5 만 스텝. off-policy 재사용 덕분에
#   DQN 의 **데이터 효율** 이 더 좋습니다 (대신 매 스텝 신경망 업데이트를 하므로 계산량은 더 많음).
#
# ### 4.5 학습된 Q 함수 들여다보기
# $\theta$–$\dot\theta$ 평면에서 ($x=\dot x=0$) 어떤 행동을 고르는지와 $\max_a Q$ (상태가치) 를 그려 봅니다.
# 경계선은 대략 $\theta + c\,\dot\theta = 0$ 꼴 — **막대가 넘어지려는 쪽으로 수레를 미는** PD 제어기와 비슷한 구조를
# 데이터만으로 찾아냈음을 확인할 수 있습니다.

# %%
TH, THD = np.meshgrid(np.linspace(-0.25, 0.25, 101), np.linspace(-2, 2, 101))
S_grid = np.stack([np.zeros_like(TH), np.zeros_like(TH), TH, THD], -1).reshape(-1, 4)
with torch.no_grad():
    Qg = dqn_q(torch.as_tensor(S_grid, dtype=torch.float32)).numpy()
fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
ax[0].contourf(np.degrees(TH), THD, Qg.argmax(1).reshape(TH.shape), levels=[-.5, .5, 1.5], colors=["#2774B8", "#E97820"])
ax[0].set_title("greedy action  (blue = push left, orange = push right)")
cs = ax[1].contourf(np.degrees(TH), THD, Qg.max(1).reshape(TH.shape), 20, cmap="viridis")
fig.colorbar(cs, ax=ax[1]); ax[1].set_title("V(s) = max_a Q(s,a)")
for a_ in ax:
    a_.axvline(12, color="r", ls=":"); a_.axvline(-12, color="r", ls=":"); a_.set_xlabel("theta [deg]")
ax[0].set_ylabel("theta_dot [rad/s]")
plt.tight_layout(); plt.show()

# %% [markdown]
# ---
# ## 5. CartPole 결과 비교 & 애니메이션
#
# 두 정책을 같은 조건에서 비교하고, **학습 때 없던 외란** (2 초에 수레를 30 N 으로 한 번 더 밀기) 에도 버티는지 확인합니다.
# 학습 분포 밖의 상황에 대한 **일반화(generalization)/강건성** 은 강화학습 제어기를 실제로 쓸 때 가장 중요한 질문입니다.

# %%
def cartpole_trace(policy, push_at=None, push_force=30.0, seed=7):
    env = CartPoleEnv(seed)
    (obs, _), done = env.reset(), False
    traj = [env.s.copy()]
    while not done:
        if push_at is not None and env.k == int(push_at / env.DT):
            env.push = push_force
        obs, r, term, trunc, _ = env.step(policy(obs))
        done = term or trunc
        traj.append(env.s.copy())
    return np.array(traj)


fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
for name, pol, col in [("REINFORCE", pg_policy.greedy, "#7B3FA0"), ("DQN", dqn_q.greedy, "#2E8B57")]:
    for push, ls in [(None, "-"), (2.0, "--")]:
        tr = cartpole_trace(pol, push)
        t = np.arange(len(tr)) * CartPoleEnv.DT
        lab = f"{name}{' + push' if push else ''} ({len(tr) - 1} steps)"
        ax[0].plot(t, np.degrees(tr[:, 2]), ls, color=col, label=lab)
        ax[1].plot(t, tr[:, 0], ls, color=col)
ax[0].set_ylabel("theta [deg]"); ax[1].set_ylabel("cart x [m]")
ax[1].axhline(2.4, color="r", ls=":"); ax[1].axhline(-2.4, color="r", ls=":")
for a_ in ax:
    a_.set_xlabel("t [s]"); a_.grid(alpha=.3); a_.axvline(2.0, color="0.7", lw=.8)
ax[0].legend(fontsize=8); plt.tight_layout(); plt.show()

# %%
show_animation(animate_cartpole(cartpole_trace(dqn_q.greedy, push_at=2.0), "DQN, push at t = 2 s"),
               "cartpole_dqn_push")
show_animation(animate_cartpole(cartpole_trace(pg_policy.greedy, push_at=2.0), "REINFORCE, push at t = 2 s"),
               "cartpole_reinforce_push")

# %% [markdown]
# ### 5.1 관찰할 점
# - 두 정책 모두 막대는 세우지만 **수레 위치 $x$ 가 천천히 흘러가는(drift)** 경우가 많습니다. 보상이 "살아 있기" 뿐이라
#   $x=0$ 으로 돌아올 이유가 없기 때문입니다. 500 스텝 안에 레일 끝에 닿지만 않으면 만점이니까요.
#   → **보상은 우리가 원하는 것을 정확히 표현해야 한다** (실습 과제 1).
# - 행동이 ±10 N 두 개뿐이라 bang-bang 으로 떨립니다(chattering). 실제 모터라면 곤란 → Part III 의 **연속 행동** 으로.

# %% [markdown]
# ---
# # Part III. 회전형 도립진자 (Rotary Inverted Pendulum)
#
# ## 6. 회전형 진자 RL 환경 다시 보기
# Part 0 의 A 절에서 만든 `RotaryPendulumEnv` 를 그대로 씁니다. 이제 강화학습의 언어로 정리하면:
#
# | MDP 요소 | 내용 |
# |---|---|
# | 상태 / 관측 | $o = [\cos\theta, \sin\theta, \dot\theta, \phi, \dot\phi]$ (물리 상태 그대로) |
# | 행동 | $a\in[-1,1]$ → 팔 가속도 $u = a \times 47.1$ rad/s², 20 ms 동안 유지 |
# | 보상 | $r = \tfrac12(1+\cos\theta) - 0.1(\phi/140°)^2 - 0.01a^2$, 팔 한계 초과 시 −50 |
# | 종료 | 팔 ±140° 초과 (balance 과제는 $\lvert\theta\rvert > 30°$ 도) |
# | 잘림 | balance 5 s (250 스텝), swing-up 10 s (500 스텝) |
#
# 보상 설계의 이유 (dense reward, "종료가 이득이 되면 안 된다") 는 A.4 절을 다시 읽어 보세요.
# 모방학습(B 절)과 달리 이제는 **선생님 없이** 이 보상만으로 제어기를 찾습니다.


# %% [markdown]
# ---
# ## 7. 연속 행동을 위한 SAC (Soft Actor-Critic)
#
# DQN 은 $\max_a Q(s,a)$ 를 행동 몇 개에 대해 직접 계산했습니다. 행동이 연속이면 이 최대화가 그 자체로 최적화 문제가 됩니다.
# 해결책: 최대화를 대신해 줄 **정책 신경망(actor)** 을 따로 두고, **Q 신경망(critic)** 이 actor 를 가르치게 합니다 → **actor-critic**.
#
# | | 역할 | CartPole 에서의 대응 |
# |---|---|---|
# | Critic $Q_\psi(s,a)$ | 행동의 가치 평가 (벨만 방정식으로 학습) | DQN 의 Q 네트워크 |
# | Actor $\pi_\phi(a\mid s)$ | $Q$ 가 크게 나오는 행동을 출력하도록 학습 | REINFORCE 의 정책 네트워크 |
#
# **SAC** (Haarnoja et al., 2018) 는 여기에 **엔트로피 최대화** 를 더한 off-policy actor-critic 으로,
# 연속 제어에서 가장 널리 쓰이는 알고리즘 중 하나입니다.
# $$
# J(\pi) = \mathbb{E}\Big[\sum_t \gamma^t\big(r_t + \alpha\,\mathcal{H}(\pi(\cdot\mid s_t))\big)\Big]
# $$
# "보상을 최대화하되 **가능한 한 무작위로**" → 탐색이 오래 유지되고, 한 가지 해법에 너무 일찍 갇히지 않습니다.
#
# ### 7.1 구성 요소 (DQN·REINFORCE 에서 배운 것의 조합)
# | 요소 | 내용 | 어디서 왔나 |
# |---|---|---|
# | Replay buffer | 과거 경험 재사용 (off-policy) | DQN |
# | **Twin critics** $Q_{\psi_1}, Q_{\psi_2}$ | 두 개 중 **작은 값** 사용 → 과대추정 방지 | Double DQN 의 아이디어 (TD3) |
# | Target critics | Polyak 평균으로 천천히 따라오는 복사본 | DQN |
# | **확률적 actor** | $a = \tanh(\mu_\phi(s) + \sigma_\phi(s)\,\epsilon)$, $\epsilon\sim\mathcal{N}(0,1)$ | REINFORCE 의 확률적 정책 |
# | **Reparameterization** | 무작위성을 $\epsilon$ 으로 빼내어 $a$ 를 $\phi$ 에 대해 미분 가능하게 | SAC/VAE |
# | **자동 온도 조절** $\alpha$ | 정책 엔트로피가 목표값 $\bar{\mathcal{H}}$ 를 유지하도록 $\alpha$ 자체를 학습 | SAC v2 |
#
# ### 7.2 손실 함수
# **Critic** — 엔트로피 항이 들어간 벨만 타깃:
# $$
# y = r + \gamma(1-d)\Big(\min_{i=1,2} Q_{\bar\psi_i}(s', \tilde a') - \alpha\log\pi_\phi(\tilde a'\mid s')\Big),
# \quad \tilde a'\sim\pi_\phi(\cdot\mid s'), \qquad
# L_Q = \sum_i \big(Q_{\psi_i}(s,a) - y\big)^2
# $$
# **Actor** — $Q$ 는 크게, 정책은 무작위하게:
# $$
# L_\pi = \mathbb{E}_s\Big[\alpha\log\pi_\phi(\tilde a\mid s) - \min_i Q_{\psi_i}(s,\tilde a)\Big]
# $$
# **Temperature**:
# $$
# L_\alpha = -\alpha\,\big(\log\pi_\phi(\tilde a\mid s) + \bar{\mathcal{H}}\big), \qquad \bar{\mathcal{H}} = -\dim(\mathcal{A}) = -1
# $$
#
# ### 7.3 tanh 로 행동을 $[-1,1]$ 에 가둘 때의 확률 보정
# $a = \tanh(z)$, $z\sim\mathcal{N}(\mu,\sigma)$ 이면 변수변환 공식으로
# $$
# \log\pi(a\mid s) = \log\mathcal{N}(z;\mu,\sigma) - \log\big(1-\tanh^2 z\big)
# $$
# 수치 안정성을 위해 $\log(1-\tanh^2 z) = 2\big(\log 2 - z - \mathrm{softplus}(-2z)\big)$ 를 씁니다.
#
# ### 7.4 장치(device)
# Colab 에서 **GPU 런타임** (런타임 → 런타임 유형 변경 → T4 GPU) 을 켜면 신경망 업데이트가 빨라집니다. 아래 코드는 GPU 가
# 있으면 자동으로 사용합니다.

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("SAC device:", DEVICE)


class Actor(nn.Module):
    """tanh-Gaussian 정책: 출력 [mu, log_std] -> a = tanh(mu + std * eps)"""
    LOG_STD_MIN, LOG_STD_MAX = -20, 2

    def __init__(self, obs_dim, act_dim, hidden=(128, 128)):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim, hidden)

    def forward(self, s, deterministic=False):
        mu, log_std = self.net(s).chunk(2, dim=-1)
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()
        z = mu if deterministic else mu + std * torch.randn_like(mu)        # reparameterization
        a = torch.tanh(z)
        logp = (-0.5 * ((z - mu) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
        logp -= (2 * (math.log(2) - z - F.softplus(-2 * z))).sum(-1, keepdim=True)   # tanh 보정
        return a, logp


class Critic(nn.Module):
    """Twin Q 네트워크: (s, a) -> Q1, Q2"""

    def __init__(self, obs_dim, act_dim, hidden=(128, 128)):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, 1, hidden)
        self.q2 = mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa), self.q2(sa)


class SAC:
    def __init__(self, obs_dim, act_dim, gamma=0.98, tau=0.005, lr=3e-4, hidden=(128, 128)):
        self.actor = Actor(obs_dim, act_dim, hidden).to(DEVICE)
        self.critic = Critic(obs_dim, act_dim, hidden).to(DEVICE)
        self.critic_targ = copy.deepcopy(self.critic)
        for p in self.critic_targ.parameters():
            p.requires_grad_(False)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.target_entropy = -float(act_dim)
        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.a_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.gamma, self.tau = gamma, tau

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        s = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        return self.actor(s, deterministic)[0].cpu().numpy()[0]

    def update(self, buf, batch=256):
        s, a, r, s2, d = (x.to(DEVICE) for x in buf.sample(batch))
        alpha = self.log_alpha.exp().detach()
        # (1) critic
        with torch.no_grad():
            a2, logp2 = self.actor(s2)
            y = r + self.gamma * (1 - d) * (torch.min(*self.critic_targ(s2, a2)) - alpha * logp2)
        q1, q2 = self.critic(s, a)
        loss_q = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.q_opt.zero_grad(); loss_q.backward(); self.q_opt.step()
        # (2) actor
        a_new, logp = self.actor(s)
        loss_pi = (alpha * logp - torch.min(*self.critic(s, a_new))).mean()
        self.pi_opt.zero_grad(); loss_pi.backward(); self.pi_opt.step()
        # (3) temperature
        loss_a = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.a_opt.zero_grad(); loss_a.backward(); self.a_opt.step()
        # (4) target critic (Polyak)
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_targ.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)
        return loss_q.item(), alpha.item()


# %% [markdown]
# ### 7.5 학습 루프 (환경에 상관없이 재사용)
# CartPole 의 DQN 루프와 구조가 같습니다. 차이는 (1) 행동이 연속이고 (2) 처음 `start_steps` 동안 균일 무작위 행동으로
# 버퍼를 채운다는 점뿐입니다. 평가 함수 `evaluate(agent)` 를 인자로 받아 **최고 성능 체크포인트** 를 보관합니다.

# %%
def train_sac(env, agent, total_steps, evaluate, start_steps=5_000, learn_start=1_000,
              batch=256, eval_every=5_000, buffer_size=300_000):
    buf = ReplayBuffer(env.obs_dim, env.act_dim, buffer_size)
    hist = {"step": [], "ret": [], "eval_step": [], "eval": [], "alpha": []}
    (obs, _), ep_ret, best, best_state, t0, alpha = env.reset(), 0.0, -np.inf, None, time.time(), 1.0
    for step in range(1, total_steps + 1):
        a = np.random.uniform(-1, 1, env.act_dim) if step <= start_steps else agent.act(obs)
        obs2, r, term, trunc, _ = env.step(a)
        buf.add(obs, a, r, obs2, float(term))           # truncated 는 bootstrap
        obs, ep_ret = obs2, ep_ret + r
        if term or trunc:
            hist["step"].append(step); hist["ret"].append(ep_ret)
            (obs, _), ep_ret = env.reset(), 0.0
        if step >= learn_start:
            _, alpha = agent.update(buf, batch)
        if step % eval_every == 0:
            score, msg = evaluate(agent)
            hist["eval_step"].append(step); hist["eval"].append(score); hist["alpha"].append(alpha)
            if score >= best:
                best, best_state = score, copy.deepcopy(agent.actor.state_dict())
            print(f"step {step:7d} | eval {score:7.1f} | {msg} | alpha {alpha:.3f} | "
                  f"{time.time() - t0:6.0f}s", flush=True)
    agent.actor.load_state_dict(best_state)
    hist["time"] = time.time() - t0
    return hist


def plot_sac_hist(hist, title, name):
    plot_training_curve(hist, title, name, extra=("alpha", "entropy temperature alpha"))


# %% [markdown]
# ---
# ## 8. 과제 A — 직립 균형 (balance)
#
# 직립 근처($\lvert\theta_0\rvert \le 11°$)에서 시작해 5 초 동안 넘어지지 않고(30° 이내), 팔도 한계 안에 두는 것이 목표입니다.
# 평가는 고정된 4 가지 초기 기울기에서의 리턴 평균입니다 (만점 ≈ 250).

# %%
BALANCE_STEPS = 30_000          # 약 10 분 분량의 시뮬레이션 경험 (CPU 로 3~5 분)
balance_agent = SAC(RotaryPendulumEnv.obs_dim, RotaryPendulumEnv.act_dim)


def _train_balance():
    torch.manual_seed(SEED); np.random.seed(SEED)
    hist = train_sac(RotaryPendulumEnv("balance", seed=SEED), balance_agent,
                     total_steps=BALANCE_STEPS, evaluate=eval_balance)
    return balance_agent.actor, hist


balance_hist = load_or_train(MODEL_DIR / "rotary_balance_sac.pt", balance_agent.actor, _train_balance,
                             map_location=DEVICE)
print("balance eval:", eval_balance(balance_agent))

# %%
plot_sac_hist(balance_hist, "SAC - rotary balance", "rotary_balance_sac_curve")
balance_policy = lambda o: balance_agent.act(o, True)
plot_rotary({f"theta0 = {math.degrees(t):+.0f} deg": rollout(balance_policy, "balance", t, seconds=8)
             for t in BALANCE_STARTS}, "learned balance controller (8 s, trained on 5 s episodes)",
            "rotary_balance_sac_response")

# %% [markdown]
# ### 8.1 관찰할 점
# - 학습 에피소드는 5 초인데 8 초 동안도 버티는가? (정책은 시간을 모릅니다. 상태만 보고 행동하므로 시간 제한과 무관하게 동작해야 정상)
# - 팔 각도 $\phi$ 가 0 으로 돌아오는가, 아니면 한쪽으로 흘러가는가? (보상의 $\phi^2$ 항의 효과)
# - 입력 $u$ 가 ±47 에 붙어 있는가(bang-bang), 부드러운가?
# - 학습된 정책은 결국 $u = \pi(\theta,\dot\theta,\phi,\dot\phi)$ 인 **비선형 상태 피드백** 입니다.
#   직립 근처에서 선형화하면 $u \approx -Kx$ 꼴이 되는데, 그 $K$ 를 아래처럼 수치 미분으로 뽑아 볼 수 있습니다.

# %%
def policy_gain(policy, eps=1e-3):
    """직립 평형점에서 u = pi(x) 를 수치 미분 -> 선형 상태 피드백 이득 K (u ~ -K x)"""
    def u_of(x):
        th, w, phi, phid = x
        return policy(np.array([math.cos(th), math.sin(th), w, phi, phid], np.float32)) * rm.ACCEL_MAX
    x0 = np.zeros(4)
    return np.array([-(u_of(x0 + eps * e) - u_of(x0 - eps * e)) / (2 * eps) for e in np.eye(4)])


K_rl = policy_gain(lambda o: balance_agent.act(o, True)[0])
print("RL policy, linearised at upright:  K = [" + ", ".join(f"{k:7.2f}" for k in K_rl) + "]   (theta, theta', phi, phi')")
print(f"u(upright, at rest) = {balance_agent.act(np.array([1, 0, 0, 0, 0], np.float32), True)[0] * rm.ACCEL_MAX:+.3f} rad/s^2")

# %% [markdown]
# 두 번째 출력도 확인하세요. 극점 배치/LQR 제어기는 평형점에서 정확히 $u=0$ 이지만, **RL 정책은 0 이 아닐 수 있습니다**
# (저자가 학습한 정책은 약 +23 rad/s²). 정책이 평형점에 *정확히* 머무는 상태를 거의 경험하지 않았고, 평형점 주변에서
# 작게 흔들리며(limit cycle) 균형을 잡아도 보상은 거의 같기 때문입니다. 위 그래프에서 $u$ 가 0 근처에서 계속 떨리는지 보세요.
# 이득도 매우 클 수 있습니다 — 신경망은 "매끄러운 선형 제어기" 일 이유가 없습니다.
#
# 부호를 해석해 보세요. $\theta$ 에 대한 이득이 양수이면 "막대가 + 쪽으로 기울면 팔을 + 쪽으로 가속해 받친다" 는 뜻입니다
# (CartPole 에서 넘어지는 쪽으로 수레를 미는 것과 같은 원리). 이 값들을 `rotary_model.place4()` 로 극점 배치해서 구한
# 이득과 비교해 보는 것도 좋은 연습입니다 (실습 과제 3).

# %% [markdown]
# ---
# ## 9. 과제 B — 스윙업(swing-up) + 균형
#
# 이제 **매달린 상태($\theta = 180°$)에서 시작** 합니다. 정책은 하나뿐인데, 그 하나가
# 1. 팔을 좌우로 흔들어 에너지를 모으고 (펌핑),
# 2. 막대가 위에 도달하는 순간 속도를 죽이며 받아내고 (catch),
# 3. 그 뒤로는 균형을 유지하며 팔을 가운데로 되돌리는
#
# **세 단계 전략을 스스로** 찾아야 합니다. 기존 제어 방식이라면 에너지 기반 스윙업 + 상태 전환 + LQR 균형을 사람이 따로 설계하고
# 전환 조건을 튜닝해야 하는 문제입니다.
#
# 학습이 균형 과제보다 훨씬 오래 걸립니다. 학습 로그에서 `t_up` (처음 직립 ±10° 에 들어간 시각) 이 먼저 생기고,
# 그 뒤에 `hold` (마지막 2 초 동안 직립 유지 비율) 가 올라가는 순서를 확인해 보세요 — **"올리기" 를 먼저 배우고 "버티기" 는 나중에**
# 배웁니다.
#
# ### 9.1 학습 중에 보게 될 현상: "빙빙 돌리기" 국소 최적해 (local optimum)
# 학습 중간(대략 2~10 만 스텝)의 정책을 보면, 막대를 위로 올리기는 하는데 **붙잡지 않고 계속 한 방향으로 돌립니다.**
# 보상 $\tfrac12(1+\cos\theta)$ 기준으로 계산해 보면:
#
# | 전략 | 스텝당 평균 보상 |
# |---|---|
# | 매달려 있기 | 0 |
# | **계속 돌리기** (위를 천천히 지나감) | **약 0.6** |
# | 위에서 붙잡아 균형 | 1.0 |
#
# 돌리기만 해도 보상의 60% 를 얻으므로, 이 전략이 꽤 안정적인 **국소 최적해** 가 됩니다. 붙잡기는 정밀한 타이밍이 필요해서
# 탐색으로 우연히 성공하기 어렵고, 성공 경험이 쌓여 critic 이 그 가치를 알아채기까지 시간이 걸립니다.
# 그래서 학습 곡선이 한동안 평평하다가 어느 순간 다시 올라갑니다. **평평한 구간에서 학습을 멈추면 안 되는 이유** 입니다.
#
# 이 정체를 줄이려고 해 볼 수 있는 것들 (실습 과제 7):
# - 직립 근처에서만 큰 보상을 주는 **뾰족한 보상** (예: $e^{-(\theta/0.3)^2}$ 를 더함) — 돌리기의 이득이 줄어듦
# - 직립 근처 상태에서 시작하는 에피소드 비율을 늘리기 — 붙잡기 연습 기회가 늘어남
# - 직립 근처의 $\dot\theta^2$ 벌점 — 위를 빠르게 지나가는 것을 억제
#
# 단, 실제로 실험해 보면 이런 수정이 항상 빨라지는 것은 아닙니다 (예: 속도 벌점이 너무 크면 **올리기 자체** 를 배우지 못함).
# 보상 설계는 *직관 → 실험 → 수정* 의 반복입니다.

# %%
SWINGUP_STEPS = 200_000         # 약 67 분 분량의 경험. CPU 로 20~40 분 -> 시간이 없으면 USE_PRETRAINED = True
swingup_agent = SAC(RotaryPendulumEnv.obs_dim, RotaryPendulumEnv.act_dim)


def _train_swingup():
    torch.manual_seed(SEED); np.random.seed(SEED)
    hist = train_sac(RotaryPendulumEnv("swingup", seed=SEED), swingup_agent,
                     total_steps=SWINGUP_STEPS, evaluate=eval_swingup)
    return swingup_agent.actor, hist


swingup_hist = load_or_train(MODEL_DIR / "rotary_swingup_sac.pt", swingup_agent.actor, _train_swingup,
                             map_location=DEVICE)
print("swing-up eval:", eval_swingup(swingup_agent))

# %%
plot_sac_hist(swingup_hist, "SAC - rotary swing-up", "rotary_swingup_sac_curve")
swingup_policy = lambda o: swingup_agent.act(o, True)
sw_logs = {f"theta0 = {math.degrees(t):.0f} deg": rollout(swingup_policy, "swingup", t) for t in SWINGUP_STARTS}
plot_rotary(sw_logs, "learned swing-up + balance", "rotary_swingup_sac_response")
for name, lg in sw_logs.items():
    t_up, hold = swingup_metrics(lg)
    print(f"{name}: first upright at {t_up} s, upright for {100 * hold:.0f}% of the last 2 s, "
          f"max |arm| = {np.degrees(np.abs(lg['phi']).max()):.0f} deg")

# %% [markdown]
# ### 9.2 에너지로 해석하기
# 진자의 (질량으로 정규화한) 에너지 $E = \tfrac{1}{2}\dot\theta^2 + \omega_n^2(\cos\theta - 1)$ 는 직립 정지에서 0,
# 매달린 정지에서 $-2\omega_n^2 \approx -112$ 입니다. 팔 입력이 에너지를 바꾸는 비율은
# $\dot E \approx b\,\dot\theta\cos\theta\,\ddot\phi$ — 즉 **$\dot\theta\cos\theta$ 와 같은 부호로 팔을 가속하면 에너지가 늘어납니다.**
# 학습된 정책의 $u$ 와 $\dot\theta\cos\theta$ 의 부호가 스윙업 동안 얼마나 일치하는지 그려 보면,
# 에너지 기반 스윙업 법칙($u \propto \mathrm{sgn}(\dot\theta\cos\theta)$, Åström & Furuta 2000)을 RL 이 스스로 재발견했는지 확인할 수 있습니다.

# %%
lg = sw_logs[next(iter(sw_logs))]
E = 0.5 * lg["w"] ** 2 + rm.A_G * (np.cos(lg["th"]) - 1)
pump = np.sign(lg["w"] * np.cos(lg["th"]))
fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax[0].plot(lg["t"], E); ax[0].axhline(0, color="g", lw=.8); ax[0].axhline(-2 * rm.A_G, color="0.6", ls=":")
ax[0].set_ylabel("energy E")
ax[1].plot(lg["t"], lg["u"] / rm.ACCEL_MAX, label="u / u_max (RL)")
ax[1].plot(lg["t"], pump, "--", lw=1, label="sign(theta' cos theta)  (energy pumping law)")
ax[1].set_ylabel("normalised"); ax[1].set_xlabel("t [s]"); ax[1].legend(fontsize=8, loc="lower right")
for a_ in ax:
    a_.grid(alpha=.3)
plt.tight_layout(); plt.show()
t_up = swingup_metrics(lg)[0] or lg["t"][-1]
mask = (lg["t"] < t_up) & (np.abs(lg["u"]) > 1.0)
print(f"swing 단계에서 u 와 sign(theta' cos theta) 의 부호 일치율: {100 * np.mean(np.sign(lg['u'][mask]) == pump[mask]):.0f}%")

# %% [markdown]
# ---
# ## 10. 같은 문제를 DQN, REINFORCE 로 풀어 보기 — 알고리즘 비교
#
# Part II 의 두 알고리즘을 회전형 진자에 그대로 가져오면 어떻게 될까요? **환경, 보상, 평가는 SAC 와 완전히 같게** 두고
# 알고리즘만 바꿉니다. 학습 결과는 9 절 SAC 와 같은 평가 함수(`eval_balance`, `eval_swingup`) 로 비교합니다.
#
# | | 바꿔야 하는 것 | 예상되는 문제 |
# |---|---|---|
# | **DQN** | 연속 행동 $u\in[-47, 47]$ 을 **몇 단계로 이산화** | 단계가 거칠면 정밀한 균형이 어렵다 (떨림) |
# | **REINFORCE** | 정책 출력을 softmax 대신 **가우시안** $\mathcal{N}(\mu(s), \sigma)$ 로 | on-policy + 높은 분산 → 데이터가 매우 많이 필요 |
#
# ### 10.1 DQN — 행동 이산화
# 행동을 $a \in \{-1, -0.5, 0, +0.5, +1\}$ 다섯 단계로 나눕니다 ($u = a\cdot 47.1$). 그러면 Q 네트워크는
# **입력 5 (관측) → 출력 5 (각 행동의 Q 값)** 이 되어 CartPole 의 DQN 과 같은 구조가 됩니다.
#
# 환경을 고치지 않고 **래퍼(wrapper)** 로 감쌉니다: 래퍼는 이산 행동 번호를 받아 연속 행동으로 바꿔 원래 환경에 넘깁니다.
# (Gymnasium 의 `ActionWrapper` 와 같은 아이디어 — 환경과 알고리즘 사이의 "어댑터")
#
# 학습 루프는 4 절 `train_dqn` 과 같습니다. 달라진 점은 환경·평가 함수를 인자로 받는 것, 그리고 $\varepsilon$ 감소 기간을
# **전체 학습 스텝의 30%** 로 둔 것뿐입니다 (고정값으로 두면 짧은 학습에서는 끝날 때까지 탐색만 하게 됩니다).

# %%
DQN_LEVELS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])


class DiscreteActions:
    """연속 행동 환경을 이산 행동 환경으로: step(k) -> env.step(levels[k])"""

    def __init__(self, env, levels=DQN_LEVELS):
        self.env, self.levels = env, levels
        self.obs_dim, self.n_actions = env.obs_dim, len(levels)

    def reset(self, **kw):
        return self.env.reset(**kw)

    def step(self, k):
        return self.env.step(self.levels[int(k)])


class DQNAgent:
    """평가 함수(eval_balance, eval_swingup, rollout)가 쓰는 act(obs, deterministic) 인터페이스를 맞춰 줌"""

    def __init__(self, obs_dim, levels=DQN_LEVELS, hidden=(128, 128)):
        self.levels = levels
        self.q = mlp(obs_dim, len(levels), hidden)

    @torch.no_grad()
    def greedy(self, obs):
        return int(self.q(torch.as_tensor(obs, dtype=torch.float32)).argmax())

    def act(self, obs, deterministic=True):
        return np.array([self.levels[self.greedy(obs)]])


def train_dqn_env(env, agent, total_steps, evaluate, gamma=0.98, lr=5e-4, batch=256, buffer_size=300_000,
                  eps_decay_frac=0.3, eps_end=0.05, learn_start=1_000, tau=0.005, eval_every=5_000):
    q, q_targ = agent.q, copy.deepcopy(agent.q)
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buf = ReplayBuffer(env.obs_dim, 1, buffer_size, act_dtype=np.int64)
    hist = {"step": [], "ret": [], "eval_step": [], "eval": []}
    (obs, _), ep_ret, best, best_state, t0 = env.reset(), 0.0, -np.inf, None, time.time()
    for step in range(1, total_steps + 1):
        eps = max(eps_end, 1.0 - step / (eps_decay_frac * total_steps))    # 전체의 30% 동안 1 -> 0.05
        k = np.random.randint(env.n_actions) if np.random.rand() < eps else agent.greedy(obs)
        obs2, r, term, trunc, _ = env.step(k)
        buf.add(obs, k, r, obs2, float(term))
        obs, ep_ret = obs2, ep_ret + r
        if term or trunc:
            hist["step"].append(step); hist["ret"].append(ep_ret)
            (obs, _), ep_ret = env.reset(), 0.0
        if step >= learn_start:
            s, a_b, r_b, s2, d = buf.sample(batch)
            with torch.no_grad():
                a2 = q(s2).argmax(1, keepdim=True)                         # Double DQN
                y = r_b + gamma * (1 - d) * q_targ(s2).gather(1, a2)
            loss = F.smooth_l1_loss(q(s).gather(1, a_b), y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()
            with torch.no_grad():
                for p, pt in zip(q.parameters(), q_targ.parameters()):
                    pt.mul_(1 - tau).add_(tau * p)
        if step % eval_every == 0:
            score, msg = evaluate(agent)
            hist["eval_step"].append(step); hist["eval"].append(score)
            if score >= best:
                best, best_state = score, copy.deepcopy(q.state_dict())
            print(f"step {step:7d} | eps {eps:.2f} | eval {score:7.1f} | {msg} | {time.time() - t0:6.0f}s", flush=True)
    q.load_state_dict(best_state)
    hist["time"] = time.time() - t0
    return hist


# %% [markdown]
# **과제 A (balance) 와 과제 B (swing-up) 를 DQN 으로 학습**
# - swing-up 은 SAC 와 같은 20 만 스텝.
# - balance 는 SAC 의 3 만 스텝으로는 학습되지 않아서 (직접 해 보세요) **10 만 스텝** 을 줍니다.

# %%
DQN_BALANCE_STEPS, DQN_SWINGUP_STEPS = 100_000, SWINGUP_STEPS
algo_results = {}            # 비교표용: {(algo, task): dict(score, steps, time)}

for task, steps, evaluate in [("balance", DQN_BALANCE_STEPS, eval_balance),
                              ("swingup", DQN_SWINGUP_STEPS, eval_swingup)]:
    agent = DQNAgent(RotaryPendulumEnv.obs_dim)

    def _train(agent=agent, task=task, steps=steps, evaluate=evaluate):
        torch.manual_seed(SEED); np.random.seed(SEED)
        print(f"--- DQN {task} ---")
        return agent.q, train_dqn_env(DiscreteActions(RotaryPendulumEnv(task, seed=SEED)), agent, steps, evaluate)

    hist = load_or_train(MODEL_DIR / f"rotary_{task}_dqn.pt", agent.q, _train)
    score, msg = evaluate(agent)
    algo_results[("DQN", task)] = dict(agent=agent, score=score, msg=msg, steps=steps, hist=hist,
                                       time=None if hist is None else hist["time"])
    print(f"DQN {task}: eval {score:.1f} | {msg}")
    plot_training_curve(hist, f"DQN - rotary {task}", f"rotary_{task}_dqn_curve")
dqn_balance_agent, dqn_swingup_agent = algo_results[("DQN", "balance")]["agent"], algo_results[("DQN", "swingup")]["agent"]

# %% [markdown]
# ### 10.2 REINFORCE — 가우시안 정책
# 연속 행동에서는 정책을 **가우시안 분포** 로 둡니다.
# $$
# \pi_\theta(a\mid s) = \mathcal{N}\big(a;\ \mu_\theta(s),\ \sigma^2\big),
# \qquad \log\pi_\theta(a\mid s) = -\frac{(a-\mu_\theta(s))^2}{2\sigma^2} - \log\sigma - \tfrac12\log 2\pi
# $$
# - 신경망은 평균 $\mu_\theta(s)$ 만 출력하고, $\log\sigma$ 는 상태와 무관한 **학습 가능한 파라미터 하나** 로 둡니다 (PPO 등에서도 쓰는 표준 방식).
# - 샘플한 $a$ 는 환경 안에서 $[-1,1]$ 로 잘립니다. 평가 때는 평균 $\mu$ 를 씁니다.
# - 학습 알고리즘은 3 절과 **완전히 같습니다** — 바뀐 것은 `dist()` 가 Categorical 대신 Normal 을 돌려준다는 것뿐입니다.
#   이것이 정책 경사의 장점입니다: 행동이 이산이든 연속이든 $\nabla\log\pi$ 만 계산할 수 있으면 됩니다.
#
# REINFORCE 는 매 스텝 신경망을 업데이트하지 않으므로 **스텝당 계산은 매우 싸지만**, 에피소드를 쓰고 버리므로 **스텝은 훨씬 많이** 필요합니다.
# 그래서 SAC 보다 많은 환경 스텝(balance 50 만, swing-up 100 만)을 주고, 대신 걸린 **시간** 도 함께 비교합니다.
#
# 분산을 줄이려고 한 번의 업데이트에 **에피소드 여러 개(batch)** 를 모아 씁니다 (3 절은 1 개).

# %%
class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim=1, hidden=(64, 64), log_std0=-0.5):
        super().__init__()
        self.mu = mlp(obs_dim, act_dim, hidden)
        self.log_std = nn.Parameter(torch.full((act_dim,), log_std0))

    def dist(self, obs):
        return torch.distributions.Normal(self.mu(obs), self.log_std.exp())

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        s = torch.as_tensor(obs, dtype=torch.float32)
        a = self.mu(s) if deterministic else self.dist(s).sample()
        return a.clamp(-1, 1).numpy()


def train_reinforce_env(env, pi, total_steps, evaluate, gamma=0.98, lr=1e-3, episodes_per_update=8,
                        eval_every=50_000):
    opt = torch.optim.Adam(pi.parameters(), lr=lr)
    hist = {"step": [], "ret": [], "eval_step": [], "eval": []}
    step, next_eval, best, best_state, t0 = 0, eval_every, -np.inf, None, time.time()
    while step < total_steps:
        obs_b, act_b, ret_b = [], [], []
        for _ in range(episodes_per_update):                  # (1) 현재 정책으로 에피소드 여러 개 수집
            (obs, _), done, O, A, R = env.reset(), False, [], [], []
            while not done:
                with torch.no_grad():
                    a = pi.dist(torch.as_tensor(obs)).sample().numpy()
                O.append(obs); A.append(a)
                obs, r, term, trunc, _ = env.step(a)
                R.append(r); done = term or trunc
            G, Gs = 0.0, []                                    # (2) reward-to-go
            for r in reversed(R):
                G = r + gamma * G
                Gs.append(G)
            obs_b += O; act_b += A; ret_b += Gs[::-1]
            step += len(R)
            hist["step"].append(step); hist["ret"].append(sum(R))
        ret = torch.tensor(ret_b, dtype=torch.float32)
        ret = (ret - ret.mean()) / (ret.std() + 1e-8)          # baseline + 정규화 (배치 전체)
        logp = pi.dist(torch.as_tensor(np.array(obs_b))).log_prob(torch.as_tensor(np.array(act_b))).sum(-1)
        loss = -(logp * ret).mean()                            # (3) 정책 경사
        opt.zero_grad(); loss.backward(); opt.step()
        if step >= next_eval:                                  # (4) 평가 + 최고 모델 보관
            next_eval += eval_every
            score, msg = evaluate(pi)
            hist["eval_step"].append(step); hist["eval"].append(score)
            if score >= best:
                best, best_state = score, copy.deepcopy(pi.state_dict())
            print(f"step {step:8d} | sigma {pi.dist(torch.as_tensor(np.array(obs_b))).stddev.mean().item():.3f} | eval {score:7.1f} | {msg} | "
                  f"{time.time() - t0:6.0f}s", flush=True)
    pi.load_state_dict(best_state)
    hist["time"] = time.time() - t0
    return hist


PG_BALANCE_STEPS, PG_SWINGUP_STEPS = 500_000, 1_000_000

for task, steps, evaluate in [("balance", PG_BALANCE_STEPS, eval_balance),
                              ("swingup", PG_SWINGUP_STEPS, eval_swingup)]:
    pi = GaussianPolicy(RotaryPendulumEnv.obs_dim)

    def _train(pi=pi, task=task, steps=steps, evaluate=evaluate):
        torch.manual_seed(SEED); np.random.seed(SEED)
        print(f"--- REINFORCE {task} ---")
        return pi, train_reinforce_env(RotaryPendulumEnv(task, seed=SEED), pi, steps, evaluate)

    hist = load_or_train(MODEL_DIR / f"rotary_{task}_reinforce.pt", pi, _train)
    score, msg = evaluate(pi)
    algo_results[("REINFORCE", task)] = dict(agent=pi, score=score, msg=msg, steps=steps, hist=hist,
                                             time=None if hist is None else hist["time"])
    print(f"REINFORCE {task}: eval {score:.1f} | {msg}")
    plot_training_curve(hist, f"REINFORCE - rotary {task}", f"rotary_{task}_reinforce_curve", ma=50)
pg_balance_agent, pg_swingup_agent = (algo_results[("REINFORCE", "balance")]["agent"],
                                      algo_results[("REINFORCE", "swingup")]["agent"])

# %% [markdown]
# ### 10.3 세 알고리즘 비교
# 같은 환경·같은 평가 시나리오에서의 점수입니다. (balance 만점 ≈ 250, swing-up 만점 ≈ 480)

# %%
algo_results[("SAC", "balance")] = dict(agent=balance_agent, steps=BALANCE_STEPS, hist=balance_hist,
                                        time=(balance_hist or {}).get("time"))
algo_results[("SAC", "swingup")] = dict(agent=swingup_agent, steps=SWINGUP_STEPS, hist=swingup_hist,
                                        time=(swingup_hist or {}).get("time"))
for key, ev in [(("SAC", "balance"), eval_balance), (("SAC", "swingup"), eval_swingup)]:
    algo_results[key]["score"], algo_results[key]["msg"] = ev(algo_results[key]["agent"])

print(f"{'algorithm':<10} {'task':<8} {'env steps':>10} {'train time':>11}  {'eval':>6}  detail")
for algo in ("SAC", "DQN", "REINFORCE"):
    for task in ("balance", "swingup"):
        r = algo_results[(algo, task)]
        tt = "--" if r["time"] is None else f"{r['time'] / 60:.1f} min"
        print(f"{algo:<10} {task:<8} {r['steps']:>10,} {tt:>11}  {r['score']:6.1f}  {r['msg']}")

# %%
fig, ax = plt.subplots(2, 2, figsize=(13, 6), sharex="col")
for j, (task, th0, secs) in enumerate([("balance", math.radians(8), 5.0), ("swingup", math.pi, 10.0)]):
    for algo, col in [("SAC", "#E97820"), ("DQN", "#2E8B57"), ("REINFORCE", "#7B3FA0")]:
        agent = algo_results[(algo, task)]["agent"]
        lg = rollout(lambda o: agent.act(o, True), task, th0, seconds=secs)
        ax[0, j].plot(lg["t"], np.degrees(lg["th"]), color=col, label=algo + (" (terminated)" if lg["terminated"] else ""))
        ax[1, j].plot(lg["t"], lg["u"], color=col, lw=0.8)
    ax[0, j].set_title(f"{task}: theta0 = {math.degrees(th0):.0f} deg"); ax[0, j].legend(fontsize=8)
    ax[1, j].set_xlabel("t [s]")
ax[0, 0].set_ylabel("theta from up [deg]"); ax[1, 0].set_ylabel("u [rad/s^2]")
for a_ in ax.flat:
    a_.grid(alpha=.3)
plt.tight_layout(); save_figure(fig, "rotary_algorithms_response"); plt.show()

# %% [markdown]
# **세 알고리즘의 학습 곡선 비교** — 같은 평가 점수를 환경 스텝 수 기준으로 그립니다.
# 가로축이 **로그 스케일** 인 것에 주의하세요 (REINFORCE 는 다른 알고리즘보다 스텝을 10 배 이상 씁니다).
# 학습 시간은 PC, 다른 프로그램, 절전 등에 따라 크게 달라지므로 비교의 기준으로는 **환경 스텝 수** 를 씁니다.
# (위 표의 `train time` 은 실제 기록된 시간이라 참고만 하세요. 같은 조건에서 잰 계산 비용은 15.1 절에 있습니다.)

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
for i, task in enumerate(("balance", "swingup")):
    for algo, col in [("SAC", "#E97820"), ("DQN", "#2E8B57"), ("REINFORCE", "#7B3FA0")]:
        h = algo_results[(algo, task)].get("hist")
        if h is None:
            continue
        ax[i].semilogx(np.asarray(h["eval_step"], float), h["eval"], "o-", color=col, ms=3, label=algo)
    ax[i].set_title(f"{task}: eval score"); ax[i].set_xlabel("environment steps (log)")
for a_ in ax.flat:
    a_.grid(alpha=.3, which="both"); a_.legend(fontsize=8)
fig.suptitle("rotary pendulum: SAC vs DQN vs REINFORCE")
plt.tight_layout(); save_figure(fig, "rotary_algorithms_curves"); plt.show()

# %% [markdown]
# ### 10.4 해석
# 결과를 보며 다음을 확인해 보세요 (아래 수치는 저자가 시드 0 으로 실행한 결과이며, 시드에 따라 달라집니다).
#
# | 알고리즘 | 과제 | 환경 스텝 | 학습 시간 (CPU 1 스레드, 측정 기반) | 결과 |
# |---|---|---|---|---|
# | SAC | balance | 3 만 | 약 2 분 | 4 개 시나리오 모두 성공 |
# | SAC | swing-up | 20 만 | 약 12 분 | 1.0 s 에 올리고 100% 유지 |
# | DQN (5 단계) | balance | 10 만 | 약 2 분 | 0/4 넘어짐 (다른 실행에서는 2/4 — 학습이 불안정) |
# | DQN (5 단계) | swing-up | 20 만 | 약 4 분 | 1.1 s 에 올리고 100% 유지 |
# | REINFORCE | balance | 50 만 | 약 1 분 | 2/4 넘어짐 |
# | REINFORCE | swing-up | 100 만 | 약 2 분 | 2.5 s 에 위로 가지만 **붙잡지 못함** (유지 9%) |
#
# **1. DQN 은 swing-up 을 잘한다.** 행동 5 단계만으로 SAC 와 같은 결과를 **더 짧은 시간** 에 얻었습니다.
# 스윙업의 핵심인 에너지 펌핑은 원래 "최대 가속으로 좌우로 흔들기(bang-bang)" 라서 거친 이산 행동과 잘 맞고,
# DQN 은 신경망이 하나뿐이라 업데이트가 SAC(actor + critic 2 개) 보다 쌉니다.
#
# **2. DQN 은 정밀한 균형이 약하다.** balance 에는 SAC 보다 3 배 이상 많은 스텝이 필요했고, 학습 중 점수가 성공·실패를 오갔습니다.
# 3 단계 $\{-47, 0, +47\}$ 로는 10 만 스텝을 줘도 **한 번도** 균형을 잡지 못했습니다 — 작은 보정 입력을 낼 수 없기 때문입니다.
# 연속 행동이 필요한 이유를 직접 보여 주는 결과입니다.
#
# **3. REINFORCE 는 스텝당 계산이 싸지만 데이터가 훨씬 많이 필요하다.** 50 만~100 만 스텝을 썼지만 시간은 1~2 분입니다
# (매 스텝 신경망 업데이트가 없으므로). 그런데 swing-up 에서는 SAC 학습 중간에 보았던 **"빙빙 돌리기" 국소 최적해** 에서
# 빠져나오지 못했습니다. 붙잡기에 우연히 성공한 경험이 생겨도 on-policy 라서 **한 번 쓰고 버리기** 때문입니다.
# SAC·DQN 은 replay buffer 로 그 드문 성공 경험을 계속 다시 학습합니다.
#
# **4. 점수가 같아도 제어기는 다르다.** 위 그래프의 입력 $u$ 를 보세요. DQN 은 ±47 / ±23.5 사이를 계속 오가고(채터링),
# 11 절의 장비 조건 실험에서 DQN 스윙업은 **팔을 135° 까지** 쓰다가 지연이 12 ms 로 늘면 팔 한계(140°)를 넘었습니다.
# SAC 는 107° 까지만 쓰고 같은 조건을 통과했습니다. 시뮬레이션 점수만으로 실제 장비용 제어기를 고르면 안 됩니다.
#
# **5. "환경 스텝" 과 "계산 시간" 은 다른 비용이다.** 시뮬레이터에서는 스텝이 싸서 REINFORCE 처럼 데이터를 많이 쓰는 방법도 쓸 만합니다.
# 하지만 **실제 장비에서 학습** 한다면 1 스텝 = 실제 시간 20 ms 이므로, 100 만 스텝은 5.5 시간의 실험입니다.
# 실제 로봇 학습에서 off-policy 알고리즘(SAC 등)이 선호되는 이유입니다.

# %% [markdown]
# ---
# ## 11. Sim-to-real 검증: 실제 장비 조건에서 돌리기
#
# 학습에 쓴 환경은 **이상적** 입니다: 상태를 정확히 알고, 지연이 없고, 명령이 즉시 적용됩니다.
# 실제 장비는 다릅니다. `rotary_model.Loop` 는 장비의 호스트 루프를 그대로 흉내 냅니다.
#
# | 실제 장비 조건 | 학습 환경 | `rotary_model.Loop` |
# |---|---|---|
# | 제어 루프 | 50 Hz | **100 Hz** (정책은 2 주기마다 갱신) |
# | 측정 → 명령 지연 | 0 | **4 ms** |
# | 막대 각도 | 정확 | **0.3° 엔코더 양자화** |
# | 각속도 $\dot\theta$ | 정확 | 양자화된 각도의 **필터링된 차분** |
# | 팔 각도 | 정확 | 마이크로스텝 단위 |
# | 팔 150° | — | 펌웨어가 모터 정지 (latch) |
#
# 학습된 정책을 이 루프에 그대로 넣어 봅니다. `Loop` 는 100 Hz 마다 `controller(z, t)` 를 호출하며
# $z = [\theta, \dot\theta, \phi, \dot\phi]$ (센서가 본 값) 을 넘겨줍니다. 이를 관측 벡터로 바꿔 주는 어댑터만 있으면 됩니다.

# %%
class RLController:
    """학습된 정책을 rotary_model.Loop 의 제어기로 쓰기 위한 어댑터.
    Loop 는 100 Hz 로 호출하므로 hold=2 주기마다 정책을 다시 계산 (학습 때의 50 Hz 와 맞춤)."""

    def __init__(self, policy, hold=2):
        self.policy, self.hold = policy, hold
        self.n, self.u = 0, 0.0

    def __call__(self, z, t):
        if self.n % self.hold == 0:
            th, w, phi, phid = z
            obs = np.array([math.cos(th), math.sin(th), w, phi, phid], np.float32)
            self.u = float(np.clip(self.policy(obs), -1, 1)) * rm.ACCEL_MAX
        self.n += 1
        return self.u


def run_on_rig_model(policy, theta0=math.pi, seconds=10.0, realistic=True, b=rm.B_U,
                     friction=1.0, delay=rm.DELAY, kicks=()):
    """rotary_model 의 장비 모델(Plant + Sensor + Loop)에서 정책 실행. kicks = [(t, d_theta_dot), ...]"""
    f0, d0 = rm.F_C, rm.DELAY
    rm.F_C, rm.DELAY = f0 * friction, delay          # 모델 오차 실험: 모듈 상수를 잠시 바꿈
    try:
        loop = rm.Loop(rm.Plant(theta0, b=b), RLController(lambda o: policy(o)[0]),
                       rm.Sensor(realistic), arm_limit_deg=140.0)
        kicks = sorted(kicks)
        log = {k: [] for k in ("t", "th", "w", "phi", "u")}
        while loop.plant.t < seconds:
            while kicks and loop.plant.t >= kicks[0][0]:
                loop.plant.kick(kicks.pop(0)[1])
            loop.advance(10)                               # 10 ms
            x = loop.plant.x
            for k, v in zip(log, (loop.plant.t, rm.wrap(x[0]), x[1], x[2], loop.u)):
                log[k].append(v)
    finally:
        rm.F_C, rm.DELAY = f0, d0
    log = {k: np.array(v) for k, v in log.items()}
    log["terminated"] = bool(loop.stop)
    log["stop"] = loop.stop
    return log


def rig_summary(lg):
    up = np.abs(lg["th"]) < math.radians(10)
    t_up = f"{lg['t'][np.argmax(up)]:.2f} s" if up.any() else "--"
    last = up[-200:].mean()                                 # 마지막 2 초
    return (f"{(lg['stop'] or ('balanced' if last > 0.95 else 'not balanced')):<26} first upright {t_up:>7}  "
            f"upright(last 2 s) {100 * last:3.0f}%  max|arm| {np.degrees(np.abs(lg['phi']).max()):4.0f} deg")


print("ideal sensor    :", rig_summary(run_on_rig_model(swingup_policy, realistic=False)))
print("realistic sensor:", rig_summary(run_on_rig_model(swingup_policy, realistic=True)))

# %% [markdown]
# ### 11.1 모델 오차에 대한 강건성
# 동정된 파라미터는 정확하지 않습니다. 실제 장비가 모델과 이만큼 다를 수 있다고 가정하고 같은 정책을 돌려 봅니다.
# 학습 중에는 **한 번도 본 적 없는** 조건들입니다.

# %%
cases = {"nominal": {}, "b -20%": dict(b=0.8 * rm.B_U), "b +20%": dict(b=1.2 * rm.B_U),
         "friction x0.5": dict(friction=0.5), "friction x2": dict(friction=2.0),
         "delay 12 ms": dict(delay=0.012), "kick 2 rad/s at t=7 s": dict(kicks=[(7.0, 2.0)], seconds=14.0)}
rig_logs = {}
for name, kw in cases.items():
    rig_logs[name] = run_on_rig_model(swingup_policy, **kw)
    print(f"{name:<22} {rig_summary(rig_logs[name])}")

# %% [markdown]
# **10 절의 DQN, REINFORCE 정책도 같은 장비 조건에서** — 이상적인 환경에서의 점수와 장비 조건에서의 결과가 같은지 확인합니다.

# %%
for algo in ("SAC", "DQN", "REINFORCE"):
    agent = algo_results[(algo, "swingup")]["agent"]
    pol = lambda o, ag=agent: np.atleast_1d(ag.act(o, True))
    for name, kw in [("nominal", {}), ("b -20%", dict(b=0.8 * rm.B_U)), ("delay 12 ms", dict(delay=0.012))]:
        print(f"{algo:<10} {name:<12} {rig_summary(run_on_rig_model(pol, **kw))}")

# %% [markdown]
# **외란(kick) 결과를 자세히 보세요.** 우리가 학습시킨 정책(시드 0)은 직립 상태에서 1~2 rad/s 로 치면 **넘어지지만,
# 스스로 다시 스윙업해서 2~3 초 뒤 균형을 되찾았습니다.** 장비의 4-상태 극점 배치 제어기는 약 2.6 rad/s 까지 넘어지지 않고 버팁니다.
# 즉 이 RL 정책은 *국소적인 외란 제거* 는 LQR 보다 약하지만, *넘어진 뒤 복구* 는 자동으로 합니다. 왜 그럴까요?
# 학습 중에 "직립에서 세게 맞는" 상황을 거의 겪지 않았기 때문입니다 (학습 분포!). 학습 중 무작위 kick 을 넣으면 달라집니다.
# (여러분이 학습한 정책은 시드·스텝 수에 따라 결과가 다를 수 있습니다.)
#
# 실패하는 조건이 있다면, 그것이 **sim-to-real gap** 입니다. 대표적인 해결책:
# - **Domain randomization**: 학습할 때 매 에피소드 $b$, 마찰, 지연을 무작위로 바꿔서 "모델이 조금 달라도 되는" 정책을 배우게 함 (실습 과제 5)
# - **관측에 지연/잡음 넣기**: 학습 환경에도 센서 양자화와 지연을 넣음
# - **보수적인 보상**: 입력 크기·변화율 벌점을 키워 부드러운 정책 → 지연에 덜 민감
#
# 실제 장비에 올리기 전에 반드시 이런 검증을 통과해야 합니다. 신경망 정책은 학습 분포 밖에서 *예측하기 어려운* 행동을 할 수 있습니다.

# %% [markdown]
# ---
# ## 12. 애니메이션 & 실시간 시뮬레이터
#
# ### 12.1 궤적 애니메이션
# 왼쪽은 옆에서 본 막대 (위 = 직립), 오른쪽 위는 위에서 본 팔입니다.

# %%
show_animation(animate_rotary(rig_logs["kick 2 rad/s at t=7 s"], "SAC swing-up on the rig model (kick at 7 s)"),
               "rotary_swingup_sac_rig_kick")

# %% [markdown]
# **알고리즘별 영상** — 같은 조건(매달린 상태, 이상적 환경 10 초 / balance 는 8° 에서 5 초)에서 세 알고리즘의 동작을 저장합니다.
# 점수표만 볼 때와 달리, DQN 의 떨리는 팔 움직임이나 REINFORCE 의 "계속 돌리기" 가 눈에 바로 보입니다.

# %%
for task, th0, secs in [("swingup", math.pi, 10.0), ("balance", math.radians(8), 5.0)]:
    for algo in ("SAC", "DQN", "REINFORCE"):
        agent = algo_results[(algo, task)]["agent"]
        lg = rollout(lambda o, ag=agent: ag.act(o, True), task, th0, seconds=secs)
        ani = animate_rotary(lg, f"{algo} - {task} (score {algo_results[(algo, task)]['score']:.0f})")
        show_animation(ani, f"rotary_{task}_{algo.lower()}")

# %% [markdown]
# ### 12.2 (로컬 PC 전용) 키보드로 조작하는 실시간 시뮬레이터
# Colab 에서는 창을 띄울 수 없으므로, 로컬 PC 에서 스크립트로 실행할 때만 쓰세요 (`RUN_REALTIME = True`).
# Jupyter 에서는 먼저 `%matplotlib qt` 또는 `%matplotlib tk` 를 실행해야 창이 뜹니다.
#
# | 키 | 동작 |
# |---|---|
# | ← / → | 막대 치기 ($\dot\theta$ ∓1 rad/s) |
# | ↑ | 세게 쳐서 넘어뜨리기 (+5 rad/s) → 다시 스윙업하는지 보기 |
# | `x` | 센서 realistic ↔ ideal |
# | `r` | 리셋 (매달린 상태) |

# %%
RUN_REALTIME = False


def run_realtime(policy):
    from matplotlib.animation import FuncAnimation

    def make(realistic=True):
        return rm.Loop(rm.Plant(math.pi + 0.02), RLController(lambda o: policy(o)[0]),
                       rm.Sensor(realistic), arm_limit_deg=140.0)

    S = {"lp": make(), "last": None, "T": [], "TH": []}
    fig, (ax_s, ax_p) = plt.subplots(1, 2, figsize=(12, 5))
    L = rm.L
    ax_s.set_aspect("equal"); ax_s.set_xlim(-1.3 * L, 1.3 * L); ax_s.set_ylim(-1.3 * L, 1.3 * L)
    ax_s.plot([0], [0], "ko")
    rod, = ax_s.plot([], [], lw=8, color="#E97820", solid_capstyle="round")
    info = ax_s.text(-1.25 * L, 1.15 * L, "", family="monospace", fontsize=9)
    line, = ax_p.plot([], [], color="#E97820")
    ax_p.set_ylim(-190, 190); ax_p.grid(alpha=.3); ax_p.set_xlabel("t [s]"); ax_p.set_ylabel("theta from up [deg]")

    def on_key(e):
        lp = S["lp"]
        if e.key == "left": lp.plant.kick(-1.0)
        elif e.key == "right": lp.plant.kick(1.0)
        elif e.key == "up": lp.plant.kick(5.0)
        elif e.key == "x": S.update(lp=make(not lp.sensor.realistic), T=[], TH=[], last=None)
        elif e.key == "r": S.update(lp=make(lp.sensor.realistic), T=[], TH=[], last=None)
    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_):
        lp, now = S["lp"], time.perf_counter()
        el = 0.0 if S["last"] is None else min(now - S["last"], 0.1)
        S["last"] = now
        lp.advance(int(el / rm.DT))
        th, phi = rm.wrap(lp.plant.x[0]), lp.plant.x[2]
        S["T"].append(lp.plant.t); S["TH"].append(math.degrees(th))
        rod.set_data([0, L * math.sin(th)], [0, L * math.cos(th)])
        info.set_text(f"t={lp.plant.t:6.2f}s theta={math.degrees(th):7.1f} arm={math.degrees(phi):7.1f}\n"
                      f"u={lp.u:+6.1f}  {'realistic' if lp.sensor.realistic else 'ideal'}  {lp.stop}")
        t0 = max(0.0, lp.plant.t - 10); ax_p.set_xlim(t0, t0 + 10)
        line.set_data(S["T"], S["TH"])
        return rod, info, line

    fig._ani = FuncAnimation(fig, update, interval=20, cache_frame_data=False)
    plt.tight_layout(); plt.show()


if RUN_REALTIME:
    run_realtime(swingup_policy)

# %% [markdown]
# ---
# # Part IV. 정책의 형태와 알고리즘 더 비교하기
#
# ## 13. 확률적(Gaussian) 정책 vs 결정적(deterministic) 정책
#
# 지금까지 연속 행동 알고리즘 (REINFORCE, SAC) 은 모두 정책이 **가우시안 분포** $\mathcal{N}(\mu(s), \sigma)$ 를 출력했고,
# 평가할 때만 평균 $\mu(s)$ 를 썼습니다. 그렇다면 **처음부터 결정적으로** $a = \mu(s)$ 만 출력하면 안 될까요?
#
# 가우시안을 쓰는 이유는 두 가지입니다.
# 1. **탐색**: 같은 상태에서 매번 조금씩 다른 행동을 해 봐야 더 나은 행동을 발견합니다.
# 2. **기울기 계산**: 정책 경사 $\nabla_\theta J = \mathbb{E}[\nabla_\theta\log\pi_\theta(a\mid s)\,G]$ 는 $\log\pi$ 가 있어야 계산됩니다.
#
# 결정적 정책은 $\pi(a\mid s) = \delta(a - \mu(s))$ (디랙 델타) 라서 $\log\pi$ 가 정의되지 않습니다. 그래서 출력을 결정적으로
# 바꾸려면 **위 두 역할을 다른 방법으로 해결** 해야 합니다.
#
# | 확률적 정책 알고리즘 | 결정적 버전 | 탐색은? | 기울기는? |
# |---|---|---|---|
# | SAC (actor-critic) | **TD3** (DDPG 개선판) | 행동에 잡음을 **밖에서** 더함: $a = \mu(s) + \epsilon$ | critic 을 통한 **결정적 정책 경사** $\nabla_\theta Q(s, \mu_\theta(s))$ |
# | REINFORCE (정책 경사) | **ARS** (random search) | **파라미터** 에 잡음: $\theta \pm \nu\delta$ | 두 결과의 차이로 **유한차분** 근사 |
#
# ### 13.1 SAC → TD3
# TD3 (Fujimoto et al., 2018) 는 SAC 에서 **확률적 actor 와 엔트로피 항을 빼고**, 대신 세 가지 안정화 장치를 둡니다.
#
# | | SAC | TD3 |
# |---|---|---|
# | actor 출력 | $[\mu, \log\sigma]$ → $a = \tanh(\mu + \sigma\epsilon)$ | $a = \tanh(\mu_\theta(s))$ |
# | 탐색 | 정책 자체의 무작위성 + 엔트로피 보너스 (자동 조절) | 고정 잡음 $\mathcal{N}(0, 0.1^2)$ 를 행동에 더함 |
# | actor 손실 | $\alpha\log\pi - \min_i Q_i$ | $-Q_1(s, \mu_\theta(s))$ |
# | critic 타깃 | $r + \gamma(\min Q' - \alpha\log\pi')$ | $r + \gamma\min Q'(s', \mu'(s') + \text{clip}(\epsilon))$ — **target smoothing** |
# | 기타 | — | actor 는 critic 2 번 업데이트에 1 번 (**delayed update**) |
#
# 학습 루프는 7.5 절의 `train_sac` 을 **그대로** 씁니다 (TD3 도 `act()` 와 `update()` 만 맞추면 됨).

# %%
class DetActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(128, 128)):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, hidden)

    def forward(self, s):
        return torch.tanh(self.net(s))


class TD3:
    """SAC 에서 '확률적 actor + 엔트로피' 를 빼고 '결정적 actor + 외부 탐색 잡음' 으로 바꾼 것 (Fujimoto et al., 2018)"""

    def __init__(self, obs_dim, act_dim, gamma=0.98, tau=0.005, lr=3e-4, hidden=(128, 128),
                 explore_std=0.1, target_std=0.2, target_clip=0.5, policy_delay=2):
        self.actor = DetActor(obs_dim, act_dim, hidden).to(DEVICE)
        self.actor_targ = copy.deepcopy(self.actor)
        self.critic = Critic(obs_dim, act_dim, hidden).to(DEVICE)
        self.critic_targ = copy.deepcopy(self.critic)
        for p in list(self.actor_targ.parameters()) + list(self.critic_targ.parameters()):
            p.requires_grad_(False)
        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma, self.tau = gamma, tau
        self.explore_std, self.target_std, self.target_clip, self.policy_delay = explore_std, target_std, target_clip, policy_delay
        self.n_updates = 0

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        s = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        a = self.actor(s).cpu().numpy()[0]
        if not deterministic:                                        # 탐색: 행동에 가우시안 잡음을 '밖에서' 더함
            a = np.clip(a + self.explore_std * np.random.randn(*a.shape), -1, 1)
        return a

    def update(self, buf, batch=256):
        s, a, r, s2, d = (x.to(DEVICE) for x in buf.sample(batch))
        with torch.no_grad():                                        # target policy smoothing
            noise = (torch.randn_like(a) * self.target_std).clamp(-self.target_clip, self.target_clip)
            a2 = (self.actor_targ(s2) + noise).clamp(-1, 1)
            y = r + self.gamma * (1 - d) * torch.min(*self.critic_targ(s2, a2))
        q1, q2 = self.critic(s, a)
        loss_q = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.q_opt.zero_grad(); loss_q.backward(); self.q_opt.step()
        self.n_updates += 1
        if self.n_updates % self.policy_delay == 0:                  # delayed policy update
            loss_pi = -self.critic(s, self.actor(s))[0].mean()       # deterministic policy gradient: max Q(s, mu(s))
            self.pi_opt.zero_grad(); loss_pi.backward(); self.pi_opt.step()
            with torch.no_grad():
                for net, targ in ((self.actor, self.actor_targ), (self.critic, self.critic_targ)):
                    for p, pt in zip(net.parameters(), targ.parameters()):
                        pt.mul_(1 - self.tau).add_(self.tau * p)
        return loss_q.item(), self.explore_std

td3_agents, td3_hists = {}, {}
for task, steps, evaluate in [("balance", BALANCE_STEPS, eval_balance), ("swingup", SWINGUP_STEPS, eval_swingup)]:
    agent = TD3(RotaryPendulumEnv.obs_dim, RotaryPendulumEnv.act_dim)

    def _train(agent=agent, task=task, steps=steps, evaluate=evaluate):
        torch.manual_seed(SEED); np.random.seed(SEED)
        print(f"--- TD3 {task} ---")
        return agent.actor, train_sac(RotaryPendulumEnv(task, seed=SEED), agent, steps, evaluate)

    hist = load_or_train(MODEL_DIR / f"rotary_{task}_td3.pt", agent.actor, _train, map_location=DEVICE)
    td3_agents[task], td3_hists[task] = agent, hist
    algo_results[("TD3", task)] = dict(agent=agent, steps=steps, hist=hist, time=(hist or {}).get("time"))
    algo_results[("TD3", task)]["score"], algo_results[("TD3", task)]["msg"] = evaluate(agent)
    print(f"TD3 {task}: eval {algo_results[('TD3', task)]['score']:.1f} | {algo_results[('TD3', task)]['msg']}")
    plot_training_curve(hist, f"TD3 - rotary {task}", f"rotary_{task}_td3_curve")

# %% [markdown]
# ### 13.2 REINFORCE → ARS (결정적 정책 + 파라미터 공간 탐색)
# REINFORCE 의 출력을 결정적으로 바꾸면 $\log\pi$ 가 없으니 정책 경사를 쓸 수 없습니다. 그러면 **행동이 아니라 정책 파라미터를
# 흔들어** 보면 됩니다. **ARS (Augmented Random Search, Mania et al., 2018)**:
# ```
# 정책: a = clip(M · (o - mean)/std)          <- 선형, 결정적 (관측은 실행 중 평균/표준편차로 정규화)
# 반복:
#     무작위 방향 δ_1..δ_8 (M 과 같은 크기의 가우시안 행렬)
#     각 방향마다 M + νδ_k, M - νδ_k 로 에피소드 1 개씩  -> 리턴 r+_k, r-_k
#     차이가 큰 상위 4 개 방향만 골라   M <- M + α/(4 σ_r) Σ (r+_k - r-_k) δ_k
# ```
# 리턴이 커지는 방향으로 파라미터를 옮기는 **유한차분 경사 추정** 입니다. 미분도, 신경망도, critic 도 없습니다.
# 정책이 **선형** ($u = -Kx$ 꼴, LQR 과 같은 구조) 이라는 점에 주목하세요 — balance 에는 충분하지만, 스윙업처럼
# "에너지 펌핑 ↔ 균형" 을 **전환** 해야 하는 비선형 전략은 표현할 수 없습니다.

# %%
class LinearPolicy(nn.Module):
    """a = clip(M (obs - mean) / std, -1, 1).  결정적 정책. 관측 정규화 통계도 버퍼로 저장"""

    def __init__(self, obs_dim=5, act_dim=1):
        super().__init__()
        self.register_buffer("M", torch.zeros(act_dim, obs_dim, dtype=torch.float64))
        self.register_buffer("mean", torch.zeros(obs_dim, dtype=torch.float64))
        self.register_buffer("var", torch.ones(obs_dim, dtype=torch.float64))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))

    def _norm(self, obs):
        return (np.asarray(obs, float) - self.mean.numpy()) / np.sqrt(self.var.numpy() + 1e-8)

    def act(self, obs, deterministic=True, M=None):
        M = self.M.numpy() if M is None else M
        return np.clip(M @ self._norm(obs), -1, 1)

    def update_stats(self, X):
        X = np.asarray(X, float)
        n, m, v, c = len(X), self.mean.numpy(), self.var.numpy(), float(self.count)
        bm, bv = X.mean(0), X.var(0)
        tot = c + n
        new_m = m + (bm - m) * n / tot
        new_v = (v * c + bv * n + (bm - m) ** 2 * c * n / tot) / tot
        self.mean[:] = torch.as_tensor(new_m); self.var[:] = torch.as_tensor(new_v); self.count.fill_(tot)


def train_ars(task, pi, total_steps, evaluate, n_dir=8, top=4, step_size=0.02, noise=0.03,
              eval_every=50_000, seed=SEED):
    """Augmented Random Search (Mania et al., 2018): 결정적 정책의 '파라미터' 에 잡음을 넣어 탐색"""
    env = RotaryPendulumEnv(task, seed=seed)
    rng = np.random.default_rng(seed)
    hist = {"step": [], "ret": [], "eval_step": [], "eval": []}
    step, next_eval, best, best_state, t0 = 0, eval_every, -np.inf, None, time.time()

    def episode(M):
        nonlocal step
        (obs, _), done, R, X = env.reset(), False, 0.0, []
        while not done:
            X.append(obs)
            obs, r, term, trunc, _ = env.step(pi.act(obs, M=M))
            R += r; done = term or trunc
        step += len(X)
        return R, X

    while step < total_steps:
        M = pi.M.numpy()
        deltas = rng.standard_normal((n_dir,) + M.shape)
        r_plus, r_minus, visited = np.zeros(n_dir), np.zeros(n_dir), []
        for k in range(n_dir):
            r_plus[k], X1 = episode(M + noise * deltas[k])
            r_minus[k], X2 = episode(M - noise * deltas[k])
            visited += X1 + X2
            hist["step"].append(step); hist["ret"].append((r_plus[k] + r_minus[k]) / 2)
        order = np.argsort(-np.maximum(r_plus, r_minus))[:top]           # 좋았던 방향 top 개만 사용
        sigma_r = np.concatenate([r_plus[order], r_minus[order]]).std() + 1e-8
        grad = sum((r_plus[k] - r_minus[k]) * deltas[k] for k in order)
        pi.M[:] = torch.as_tensor(M + step_size / (top * sigma_r) * grad)
        pi.update_stats(visited)
        if step >= next_eval:
            next_eval += eval_every
            score, msg = evaluate(pi)
            hist["eval_step"].append(step); hist["eval"].append(score)
            if score >= best:
                best, best_state = score, copy.deepcopy(pi.state_dict())
            print(f"step {step:8d} | eval {score:7.1f} | {msg} | {time.time() - t0:6.0f}s", flush=True)
    pi.load_state_dict(best_state)
    hist["time"] = time.time() - t0
    return pi, hist

for task, steps, evaluate in [("balance", 300_000, eval_balance), ("swingup", 1_000_000, eval_swingup)]:
    pi = LinearPolicy(RotaryPendulumEnv.obs_dim)

    def _train(pi=pi, task=task, steps=steps, evaluate=evaluate):
        print(f"--- ARS {task} ---")
        return train_ars(task, pi, steps, evaluate)

    hist = load_or_train(MODEL_DIR / f"rotary_{task}_ars.pt", pi, _train)
    algo_results[("ARS", task)] = dict(agent=pi, steps=steps, hist=hist, time=(hist or {}).get("time"))
    algo_results[("ARS", task)]["score"], algo_results[("ARS", task)]["msg"] = evaluate(pi)
    print(f"ARS {task}: eval {algo_results[('ARS', task)]['score']:.1f} | {algo_results[('ARS', task)]['msg']}")
    plot_training_curve(hist, f"ARS (linear, deterministic) - rotary {task}", f"rotary_{task}_ars_curve", ma=50)
print("ARS balance gain (normalised obs -> a):", np.round(algo_results[("ARS", "balance")]["agent"].M.numpy(), 3))

# %% [markdown]
# ### 13.3 σ (탐색 폭) 는 어떻게 둘까? — 고정 / 상태 무관 학습 / 상태 의존
# 10.2 절의 REINFORCE 는 $\sigma$ 를 **학습** 하지만 **모든 상태에서 같은 값** 입니다. "상황에 따라 분포가 넓거나 좁아야 할 텐데,
# 한 값으로 고정된 폭을 쓰는 것이 이상하지 않은가?" 는 좋은 질문입니다. 세 가지를 비교합니다.
#
# | 방식 | $\sigma$ | 쓰는 곳 | 장점 | 단점 |
# |---|---|---|---|---|
# | **고정** | 상수 (0.61) | 단순한 구현, DDPG/TD3 의 탐색 잡음 | 탐색이 줄어들지 않음 → 안정적 | 끝까지 잡음이 큼, 정밀 제어에 불리 |
# | **상태 무관, 학습** | 학습되는 파라미터 하나 | **PPO 표준 구현** (Stable-Baselines3 기본값), 10.2 절 | 학습이 진행되며 탐색을 스스로 줄임, 안정적 | 상태별 차이를 못 둠 |
# | **상태 의존** $\sigma(s)$ | 신경망 출력 | **SAC** | 필요한 곳만 넓게 탐색 (이론상 가장 유연) | on-policy 정책 경사에서는 불안정해지기 쉬움 |
#
# 왜 상태 무관 $\sigma$ 가 표준으로 쓰일까요?
# - 이 $\sigma$ 는 "이 상태에서 정답이 얼마나 불확실한가" 가 아니라 **탐색의 크기를 정하는 손잡이** 입니다. 최종 제어기는 평균 $\mu(s)$ 만 씁니다.
# - 정책 경사에서 $\sigma(s)$ 를 상태마다 학습하면, 우연히 좋았던 상태에서 $\sigma$ 가 빨리 0 으로 줄어 **그 상태의 탐색이 멈추거나**,
#   반대로 커져서 기울기 분산이 폭발하기 쉽습니다. 상태 무관 $\sigma$ 는 모든 데이터가 하나의 값을 함께 조정하므로 훨씬 안정적입니다.
# - SAC 가 $\sigma(s)$ 로도 안정적인 이유는 **엔트로피 보너스** 가 $\sigma$ 가 너무 빨리 줄어드는 것을 막고 (자동 온도 $\alpha$),
#   critic 이 기울기를 주기 때문입니다.
#
# REINFORCE 로 세 가지를 같은 조건 (스텝 수, 학습률, 시드) 에서 학습해 비교합니다. (상태 무관 학습 $\sigma$ 는 10.2 절 결과)

# %%
class GaussianPolicyS(nn.Module):
    """상태에 따라 달라지는 sigma(s): 신경망이 [mu, log_std] 를 함께 출력 (SAC 의 actor 와 같은 방식)"""

    def __init__(self, obs_dim, act_dim=1, hidden=(64, 64), log_std0=-0.5):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim, hidden)
        with torch.no_grad():                                        # 처음 sigma 를 다른 버전과 같게
            self.net[-1].bias[act_dim:] = log_std0
            self.net[-1].weight[act_dim:] *= 0.01

    def dist(self, obs):
        mu, log_std = self.net(obs).chunk(2, dim=-1)
        return torch.distributions.Normal(mu, log_std.clamp(-5, 1).exp())

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        s = torch.as_tensor(obs, dtype=torch.float32)
        d = self.dist(s)
        return (d.mean if deterministic else d.sample()).clamp(-1, 1).numpy()


def make_gaussian(kind, obs_dim=5):
    """kind: 'learned' (상태 무관, 학습) | 'fixed' (상태 무관, 고정) | 'state' (상태 의존 sigma(s))"""
    if kind == "state":
        return GaussianPolicyS(obs_dim)
    pi = GaussianPolicy(obs_dim)
    if kind == "fixed":
        pi.log_std.requires_grad_(False)                             # sigma = exp(-0.5) = 0.61 로 고정
    return pi

for kind in ("fixed", "state"):
    for task, steps, evaluate in [("balance", PG_BALANCE_STEPS, eval_balance), ("swingup", PG_SWINGUP_STEPS, eval_swingup)]:
        pi = make_gaussian(kind, RotaryPendulumEnv.obs_dim)

        def _train(pi=pi, task=task, steps=steps, evaluate=evaluate, kind=kind):
            torch.manual_seed(SEED); np.random.seed(SEED)
            print(f"--- REINFORCE ({kind} sigma) {task} ---")
            return pi, train_reinforce_env(RotaryPendulumEnv(task, seed=SEED), pi, steps, evaluate)

        hist = load_or_train(MODEL_DIR / f"rotary_{task}_reinforce_{kind}.pt", pi, _train)
        key = (f"REINFORCE-{kind}", task)
        algo_results[key] = dict(agent=pi, steps=steps, hist=hist, time=(hist or {}).get("time"))
        algo_results[key]["score"], algo_results[key]["msg"] = evaluate(pi)
        print(f"REINFORCE ({kind} sigma) {task}: eval {algo_results[key]['score']:.1f} | {algo_results[key]['msg']}")

# %% [markdown]
# **상태 의존 σ(s) 가 실제로 어떻게 생겼나** — 9 절 SAC (스윙업) 의 actor 가 출력하는 $\sigma(s)$ 를 $(\theta, \dot\theta)$ 평면에 그려 봅니다
# ($\phi = \dot\phi = 0$). SAC 는 엔트로피 보너스 덕분에 상태 의존 $\sigma$ 를 안정적으로 학습하는데, 어디서 넓게 / 좁게 탐색하도록 배웠을까요?
# (SAC 의 $\sigma$ 는 tanh 를 통과하기 전 값입니다.)

# %%
TH, W = np.meshgrid(np.linspace(-math.pi, math.pi, 181), np.linspace(-15, 15, 121))
grid = np.stack([np.cos(TH), np.sin(TH), W, np.zeros_like(TH), np.zeros_like(TH)], -1).reshape(-1, 5)
with torch.no_grad():
    _, log_std = swingup_agent.actor.net(torch.as_tensor(grid, dtype=torch.float32, device=DEVICE)).chunk(2, -1)
    sac_sigma = log_std.clamp(Actor.LOG_STD_MIN, Actor.LOG_STD_MAX).exp().cpu().numpy().reshape(TH.shape)
fig, ax = plt.subplots(figsize=(8, 4.2))
cs = ax.contourf(np.degrees(TH), W, np.log10(sac_sigma), 20, cmap="viridis")
fig.colorbar(cs, label="log10 sigma(s)")
E0 = 0.5 * W ** 2 + rm.A_G * (np.cos(TH) - 1)
ax.contour(np.degrees(TH), W, E0, levels=[0], colors="w", linewidths=1)     # 에너지 0 (직립에 도달하는 궤도)
ax.set_xlabel("theta from upright [deg]"); ax.set_ylabel("theta_dot [rad/s]")
ax.set_title("SAC swing-up actor: state-dependent exploration sigma(s)  (white: E = 0 orbit)")
plt.tight_layout(); save_figure(fig, "rotary_sac_sigma_map"); plt.show()

# %% [markdown]
# ---
# ## 14. PPO (Proximal Policy Optimization)
#
# REINFORCE 의 약점 (높은 분산, 한 번 쓰고 버리는 데이터, 불안정한 큰 업데이트) 을 고친 on-policy 알고리즘으로,
# 로봇 제어·게임·LLM 학습(RLHF) 까지 가장 널리 쓰이는 강화학습 알고리즘입니다 (Schulman et al., 2017).
#
# ### 14.1 REINFORCE 에서 PPO 로 — 세 가지 개선
# **(1) critic 으로 분산 줄이기 — 어드밴티지와 GAE.** 리턴 $G_t$ 대신 *평균보다 얼마나 나았나* 를 씁니다.
# 가치 신경망 $V_\psi(s)$ 를 함께 학습하고, TD 오차 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 를 지수 가중 평균한
# **GAE (Generalized Advantage Estimation)** 를 씁니다.
# $$
# \hat A_t = \sum_{k\ge0} (\gamma\lambda)^k\,\delta_{t+k}, \qquad \lambda = 0.95
# $$
# ($\lambda = 1$ 이면 REINFORCE 의 몬테카를로 리턴, $\lambda = 0$ 이면 한 스텝 TD — 분산과 편향 사이의 조절 손잡이)
#
# **(2) 같은 데이터로 여러 번 업데이트 — 중요도 비율.** 데이터를 모은 정책 $\pi_{old}$ 와 지금 정책 $\pi_\theta$ 의 확률비
# $r_t(\theta) = \pi_\theta(a_t\mid s_t) / \pi_{old}(a_t\mid s_t)$ 로 보정하면, 4096 스텝을 모아 10 번 (epoch) 재사용할 수 있습니다.
#
# **(3) 너무 많이 바뀌지 않게 — clipping.**
# $$
# L^{CLIP}(\theta) = \mathbb{E}_t\Big[\min\big(r_t(\theta)\hat A_t,\ \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\,\hat A_t\big)\Big], \qquad \epsilon = 0.2
# $$
# 확률비가 $[0.8, 1.2]$ 를 벗어나면 더 이상 이득이 없으므로 정책이 한 번에 크게 바뀌지 않습니다 —
# 3.4 절에서 본 REINFORCE 의 "갑자기 무너지는" 현상을 막는 장치입니다.
#
# ### 14.2 알고리즘
# ```
# 반복:
#     현재 정책으로 4096 스텝 수집 (에피소드 경계를 넘어 계속), 각 스텝의 log pi_old, V(s) 저장
#     GAE 로 어드밴티지 A_t, 가치 타깃 R_t = A_t + V(s_t) 계산 (시간 제한으로 끝나면 V(s_T) 로 bootstrap)
#     10 epoch × 미니배치 256:  loss = -L_CLIP + 0.5 (V(s) - R_t)^2,  기울기 크기 0.5 로 제한
# ```
# 정책의 $\sigma$ 는 13.3 절의 두 방식 — **상태 무관 학습 (표준)** 과 **상태 의존** — 을 모두 학습해 비교합니다.

# %%
class PPOAgent(nn.Module):
    """actor (가우시안) + critic V(s). sigma 는 'learned' (상태 무관) 또는 'state' (상태 의존)"""

    def __init__(self, obs_dim=5, act_dim=1, hidden=(64, 64), sigma="learned", log_std0=-0.5):
        super().__init__()
        self.sigma = sigma
        self.pi = mlp(obs_dim, act_dim * (2 if sigma == "state" else 1), hidden)
        self.v = mlp(obs_dim, 1, hidden)
        if sigma == "state":
            with torch.no_grad():
                self.pi[-1].bias[act_dim:] = log_std0
                self.pi[-1].weight[act_dim:] *= 0.01
        else:
            self.log_std = nn.Parameter(torch.full((act_dim,), log_std0))

    def dist(self, obs):
        out = self.pi(obs)
        if self.sigma == "state":
            mu, log_std = out.chunk(2, dim=-1)
            return torch.distributions.Normal(mu, log_std.clamp(-5, 1).exp())
        return torch.distributions.Normal(out, self.log_std.exp())

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        d = self.dist(torch.as_tensor(obs, dtype=torch.float32))
        return (d.mean if deterministic else d.sample()).clamp(-1, 1).numpy()


def train_ppo(env, ag, total_steps, evaluate, rollout_steps=4096, epochs=10, minibatch=256, gamma=0.98, lam=0.95,
              clip=0.2, lr=3e-4, vf_coef=0.5, max_grad_norm=0.5, eval_every=50_000):
    """PPO-clip (Schulman et al., 2017) + GAE.  REINFORCE 와 같은 on-policy 이지만
    (1) critic V(s) 로 분산을 줄이고 (2) 같은 데이터로 여러 번 업데이트하되 정책 변화량을 clip 으로 제한"""
    opt = torch.optim.Adam(ag.parameters(), lr=lr)
    hist = {"step": [], "ret": [], "eval_step": [], "eval": [], "sigma": []}
    (obs, _), ep_ret = env.reset(), 0.0
    step, next_eval, best, best_state, t0 = 0, eval_every, -np.inf, None, time.time()
    while step < total_steps:
        # ---- (1) 현재 정책으로 rollout_steps 만큼 수집
        S, A, LP, R, D, V = [], [], [], [], [], []
        for _ in range(rollout_steps):
            with torch.no_grad():
                s_t = torch.as_tensor(obs, dtype=torch.float32)
                d = ag.dist(s_t); a = d.sample()
                lp, v = d.log_prob(a).sum().item(), ag.v(s_t).item()
            obs2, r, term, trunc, _ = env.step(a.numpy())
            ep_ret += r
            if trunc and not term:                                   # 시간 제한: 남은 가치를 보상에 더해 bootstrap
                with torch.no_grad():
                    r += gamma * ag.v(torch.as_tensor(obs2, dtype=torch.float32)).item()
            S.append(obs); A.append(a.numpy()); LP.append(lp); R.append(r); D.append(term or trunc); V.append(v)
            obs = obs2
            step += 1
            if term or trunc:
                hist["step"].append(step); hist["ret"].append(ep_ret)
                (obs, _), ep_ret = env.reset(), 0.0
        with torch.no_grad():
            last_v = ag.v(torch.as_tensor(obs, dtype=torch.float32)).item()
        # ---- (2) GAE: A_t = sum (gamma lam)^k delta_{t+k},  delta_t = r_t + gamma V(s_{t+1}) - V(s_t)
        adv, gae = np.zeros(rollout_steps, np.float32), 0.0
        for t in reversed(range(rollout_steps)):
            nv = last_v if t == rollout_steps - 1 else V[t + 1]
            nonterm = 0.0 if D[t] else 1.0
            delta = R[t] + gamma * nv * nonterm - V[t]
            gae = delta + gamma * lam * nonterm * gae
            adv[t] = gae
        ret = adv + np.array(V, np.float32)
        S_t, A_t = torch.as_tensor(np.array(S)), torch.as_tensor(np.array(A))
        LP_t, ADV_t, RET_t = torch.as_tensor(LP), torch.as_tensor(adv), torch.as_tensor(ret)
        ADV_t = (ADV_t - ADV_t.mean()) / (ADV_t.std() + 1e-8)
        # ---- (3) 같은 데이터로 epochs 번, clipped objective
        for _ in range(epochs):
            for idx in torch.randperm(rollout_steps).split(minibatch):
                d = ag.dist(S_t[idx])
                ratio = (d.log_prob(A_t[idx]).sum(-1) - LP_t[idx]).exp()
                surr = torch.min(ratio * ADV_t[idx], ratio.clamp(1 - clip, 1 + clip) * ADV_t[idx])
                loss = -surr.mean() + vf_coef * F.mse_loss(ag.v(S_t[idx]).squeeze(-1), RET_t[idx])
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(ag.parameters(), max_grad_norm)
                opt.step()
        if step >= next_eval:
            next_eval += eval_every
            score, msg = evaluate(ag)
            with torch.no_grad():
                sig = ag.dist(S_t).stddev.mean().item()
            hist["eval_step"].append(step); hist["eval"].append(score); hist["sigma"].append(sig)
            if score >= best:
                best, best_state = score, copy.deepcopy(ag.state_dict())
            print(f"step {step:8d} | sigma {sig:.3f} | eval {score:7.1f} | {msg} | {time.time() - t0:6.0f}s", flush=True)
    ag.load_state_dict(best_state)
    hist["time"] = time.time() - t0
    return ag, hist

PPO_BALANCE_STEPS, PPO_SWINGUP_STEPS = 500_000, 2_000_000
for kind in ("learned", "state"):
    for task, steps, evaluate in [("balance", PPO_BALANCE_STEPS, eval_balance), ("swingup", PPO_SWINGUP_STEPS, eval_swingup)]:
        ag = PPOAgent(RotaryPendulumEnv.obs_dim, sigma=kind)
        suffix = "" if kind == "learned" else "_state"

        def _train(ag=ag, task=task, steps=steps, evaluate=evaluate, kind=kind):
            torch.manual_seed(SEED); np.random.seed(SEED)
            print(f"--- PPO ({kind} sigma) {task} ---")
            return train_ppo(RotaryPendulumEnv(task, seed=SEED), ag, steps, evaluate)

        hist = load_or_train(MODEL_DIR / f"rotary_{task}_ppo{suffix}.pt", ag, _train)
        key = ("PPO" if kind == "learned" else "PPO-state", task)
        algo_results[key] = dict(agent=ag, steps=steps, hist=hist, time=(hist or {}).get("time"))
        algo_results[key]["score"], algo_results[key]["msg"] = evaluate(ag)
        print(f"PPO ({kind} sigma) {task}: eval {algo_results[key]['score']:.1f} | {algo_results[key]['msg']}")
        plot_training_curve(hist, f"PPO ({kind} sigma) - rotary {task}", f"rotary_{task}_ppo{suffix}_curve",
                            extra=("sigma", "mean exploration sigma"))

# %% [markdown]
# ### 14.3 σ 비교 결과
# 아래 표는 같은 알고리즘 안에서 $\sigma$ 방식만 바꾼 결과입니다.

# %%
rows = [("REINFORCE", "fixed", "REINFORCE-fixed"), ("REINFORCE", "learned (state-indep.)", "REINFORCE"),
        ("REINFORCE", "state-dependent", "REINFORCE-state"), ("PPO", "learned (state-indep.)", "PPO"),
        ("PPO", "state-dependent", "PPO-state")]
print(f"{'algorithm':<10} {'sigma':<24} {'balance':>8}  {'':<12} {'swing-up':>8}  detail")
for algo, sig, key in rows:
    b, s = algo_results[(key, "balance")], algo_results[(key, "swingup")]
    print(f"{algo:<10} {sig:<24} {b['score']:8.1f}  {b['msg']:<12} {s['score']:8.1f}  {s['msg']}")

fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
for j, task in enumerate(("balance", "swingup")):
    for key, col, ls in [("PPO", "#2774B8", "-"), ("PPO-state", "#2774B8", "--")]:
        h = algo_results[(key, task)].get("hist")
        if h:
            ax[j].plot(h["eval_step"], h["sigma"], ls, color=col, marker="o", ms=3, label=key)
    ax[j].set_title(f"{task}: mean exploration sigma during PPO training"); ax[j].set_xlabel("environment step")
    ax[j].grid(alpha=.3); ax[j].legend(fontsize=8)
plt.tight_layout(); save_figure(fig, "rotary_ppo_sigma"); plt.show()

# %% [markdown]
# **저자가 시드 0 으로 실행한 결과** (같은 시드·스텝 수·학습률에서 σ 방식만 다름)
#
# | 알고리즘 | σ | balance | 스윙업 | 스윙업이 처음 100% 유지된 시점 |
# |---|---|---|---|---|
# | REINFORCE | 고정 (0.61) | 226 점, 1/4 넘어짐 | 3.5 s 에 위로 가지만 유지 7% | 실패 (100 만 스텝) |
# | REINFORCE | 상태 무관 학습 | 211 점, 2/4 넘어짐 | 2.5 s 에 위로 가지만 유지 9% | 실패 (100 만 스텝) |
# | REINFORCE | **상태 의존 σ(s)** | **91 점, 3/4 넘어짐** | 4.6 s 에 위로 가지만 유지 10% | 실패 (100 만 스텝) |
# | PPO | 상태 무관 학습 | 250 점, 0/4 | **1.0 s, 100% 유지 (467 점)** | **약 60 만 스텝** |
# | PPO | 상태 의존 σ(s) | 250 점, 0/4 | 1.4 s, 100% 유지 (453 점) | 약 120 만 스텝 |
#
# **1. σ 설계보다 알고리즘이 훨씬 중요합니다.** REINFORCE 는 σ 를 어떻게 두든 스윙업을 붙잡지 못했고, PPO 는 둘 다 성공했습니다.
# critic 으로 분산을 줄이고 clipping 으로 업데이트를 안정화한 효과가 σ 방식의 차이보다 큽니다.
#
# **2. 상태 의존 σ(s) 가 "이론상 더 유연" 하지만, on-policy 정책 경사에서는 오히려 불리했습니다.**
# - REINFORCE 에서는 balance 성능이 크게 나빠졌습니다 (3/4 넘어짐). 상태별 σ 가 기울기 잡음에 따라 제각각 움직이면서 학습이 흔들립니다.
# - PPO 에서도 상태 의존 σ 는 수렴이 두 배 느렸고, σ 가 0.05 수준까지 빠르게 줄어들었습니다 (위 그래프). 탐색이 너무 일찍 줄어든 것이고,
#   200 만 스텝 무렵에는 성능이 다시 떨어지기도 했습니다 (최고 체크포인트를 저장하는 이유).
#
# **3. 그럼 "상황에 상관없이 같은 폭" 이 이상하지 않은가?** 개념적으로는 맞는 지적입니다. 다만 이 σ 는 *최종 제어기의 일부가 아니라*
# **학습 중 탐색의 크기** 이고, 평가와 실제 사용에는 평균 $\mu(s)$ 만 씁니다. 상태 무관 σ 는 모든 데이터가 하나의 값을 함께 조정하므로
# 안정적이고, 학습이 진행되며 스스로 줄어듭니다 (PPO: 0.46 → 0.10). 그래서 PPO 의 표준 구현이 이 방식을 씁니다.
#
# **4. 상태 의존 σ 를 잘 쓰려면 σ 가 너무 빨리 줄지 않게 하는 장치가 필요합니다.** SAC 가 그 예입니다 — 엔트로피 보너스와
# 자동 온도 α 가 σ(s) 의 붕괴를 막습니다. 13.3 절의 SAC σ(s) 지도에서, SAC 가 어떤 상태에서 넓게/좁게 탐색하도록 배웠는지 확인해 보세요.

# %% [markdown]
# ---
# ## 15. 전체 비교 — 모방학습부터 PPO 까지
#
# 같은 장비 모델, 같은 평가 시나리오에서 이 튜토리얼의 모든 방법을 비교합니다. 마지막 열은 스윙업 정책을
# **실제 장비 조건** (11 절: 100 Hz, 4 ms 지연, 0.3° 엔코더) 에서 10 초 돌린 결과입니다.
#
# `est. time` 은 **같은 조건에서 잰 계산 비용** (1,000 환경 스텝당 시간, 저자 PC 의 CPU 1 스레드) × 사용한 스텝 수로 계산한 예상 학습 시간입니다.
# 실제 기록된 학습 시간은 다른 프로그램, 병렬 실행, 절전 등에 따라 크게 달라서 비교에 쓰지 않았습니다.

# %%
COST_PER_1K = {"SAC": 3.57, "TD3": 2.15, "DQN": 1.16, "PPO": 0.25, "PPO-state": 0.25, "REINFORCE": 0.14,
               "REINFORCE-fixed": 0.14, "REINFORCE-state": 0.14, "ARS": 0.08}      # [s / 1,000 steps], 저자 PC 측정
ALL = [("teacher", None), ("BC", None), ("DAgger", None), ("REINFORCE", "PG"), ("REINFORCE-fixed", "PG"),
       ("REINFORCE-state", "PG"), ("ARS", "random search"), ("PPO", "PG + critic"), ("PPO-state", "PG + critic"),
       ("DQN", "value"), ("SAC", "actor-critic"), ("TD3", "actor-critic")]
il_agents = {"teacher": teacher, "BC": bc_policy, "DAgger": dagger_policy}
summary = []
print(f"{'method':<16} {'env steps (b / s)':>20} {'est. time':>11}  {'balance':>8} {'swing-up':>9}  "
      f"{'swing-up detail':<32} rig (realistic)")
for name, _ in ALL:
    if name in il_agents:
        ag_s, ag_b = il_agents[name], il_agents[name]
        b, s = il_results[name]["balance"], il_results[name]["swingup"]
        steps = "--" if name == "teacher" else f"{(il_hist or {}).get('n_data', [0])[-1]:,} labels"
        tt = (il_hist or {}).get("time", 0) / 60 if name != "teacher" else None
    else:
        rb, rs = algo_results[(name, "balance")], algo_results[(name, "swingup")]
        ag_s, ag_b = rs["agent"], rb["agent"]
        b, s = (rb["score"], rb["msg"]), (rs["score"], rs["msg"])
        steps = f"{rb['steps'] / 1e3:.0f}k / {rs['steps'] / 1e3:.0f}k"
        tt = COST_PER_1K[name] * (rb["steps"] + rs["steps"]) / 1000 / 60
    rig = rig_summary(run_on_rig_model(lambda o, ag=ag_s: np.atleast_1d(ag.act(o, True))))
    summary.append((name, b, s, rig))
    print(f"{name:<16} {steps:>20} {'--' if tt is None else f'{tt:.1f}':>11}  {b[0]:8.1f} {s[0]:9.1f}  {s[1]:<32} {rig}")

# %% [markdown]
# **학습 곡선 한눈에 보기** — 평가 점수 vs 환경 스텝 (로그 축). 모방학습은 환경 스텝 대신 *선생님 레이블 수* 를 썼습니다.

# %%
COLORS = {"REINFORCE": "#7B3FA0", "REINFORCE-fixed": "#B07CC6", "REINFORCE-state": "#4B1F66", "ARS": "#8C6D31",
          "PPO": "#2774B8", "PPO-state": "#7FB2E5", "DQN": "#2E8B57", "SAC": "#E97820", "TD3": "#C0392B"}
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
for j, task in enumerate(("balance", "swingup")):
    for name, col in COLORS.items():
        h = algo_results[(name, task)].get("hist")
        if h:
            ax[j].semilogx(h["eval_step"], h["eval"], "-", color=col, lw=1.5, label=name)
    if il_hist is not None:
        ax[j].semilogx(il_hist["n_data"], il_hist[f"eval_{task}"], "k:", marker="s", ms=3, label="DAgger (labels)")
    ax[j].axhline(il_results["teacher"][task][0], color="k", ls="--", lw=1, label="teacher")
    ax[j].set_title(f"{task}: evaluation score"); ax[j].set_xlabel("environment steps (log)"); ax[j].grid(alpha=.3, which="both")
ax[1].legend(fontsize=7, ncol=2, loc="lower right")
plt.tight_layout(); save_figure(fig, "all_methods_curves"); plt.show()

# %% [markdown]
# **최종 모델 데모 영상** — 모든 방법의 스윙업 정책을 같은 조건 (매달린 상태, 10 초) 에서 녹화합니다.

# %%
for name, _ in ALL:
    if name in ("teacher", "BC", "DAgger", "SAC", "DQN", "REINFORCE"):
        continue                                         # B 절, 12 절에서 이미 저장함
    ag = algo_results[(name, "swingup")]["agent"]
    lg = rollout(lambda o, ag=ag: np.atleast_1d(ag.act(o, True)), "swingup", math.pi)
    show_animation(animate_rotary(lg, f"{name} - swing-up (score {algo_results[(name, 'swingup')]['score']:.0f})"),
                   f"rotary_swingup_{name.lower().replace('-', '_')}")

# %% [markdown]
# ### 15.1 해석
#
# **저자가 시드 0 으로 실행한 결과** (시드에 따라 달라질 수 있습니다). 예상 시간 = 측정한 계산 비용 × 스텝 수 (balance + swing-up).
#
# | 방법 | 학습 데이터 | 예상 시간 | balance | 스윙업 (이상적 환경) | 스윙업 (실제 장비 조건) |
# |---|---|---|---|---|---|
# | 선생님 (장비 제어기) | — | — | 0/4 넘어짐 | 2.6 s, 100% 유지 | 성공 — 지연 12 ms, b −20% 에서도 성공 |
# | BC | 레이블 2 만 | < 1 분 | **4/4 넘어짐** | 위로 가지만 유지 9% | 실패 (팔 한계) |
# | DAgger | 레이블 5.5 만 | 약 2 분 | 0/4 | 2.5 s, 100% | 성공 (b −20% 에서는 실패) |
# | REINFORCE (σ 3 종) | 50 만 / 100 만 스텝 | 약 4 분 | 1~3/4 넘어짐 | 위로 가지만 유지 7~10% | 실패 |
# | ARS (선형, 결정적) | 30 만 / 100 만 | 약 2 분 | **0/4, 최고점 (250)** | 위로 가지만 유지 13% | 실패 |
# | PPO | 50 만 / 200 만 | 약 11 분 | 0/4 | **1.0 s, 100% (최고점 467)** | 성공 — 지연 12 ms 도 성공 (b −20% 실패) |
# | PPO (σ(s)) | 50 만 / 200 만 | 약 11 분 | 0/4 | 1.4 s, 100% | **실패 (유지 30%)** |
# | DQN (5 단계) | 10 만 / 20 만 | 약 6 분 | 0/4 | 1.1 s, 100% | 성공 (팔을 135° 까지 씀, 지연 12 ms 에서 팔 한계) |
# | SAC | 3 만 / 20 만 | 약 14 분 | 0/4 | 1.0 s, 100% | **성공 — 지연 12 ms, b −20% 모두 성공** |
# | TD3 | 3 만 / 20 만 | 약 8 분 | 0/4 | 1.6 s, 유지 37% | 실패 |
#
# **1. balance 는 쉽고, swing-up 이 방법들을 갈라놓습니다.** 직립 근처의 균형은 사실상 선형 제어 문제 ($u = -Kx$) 라서
# 거의 모든 방법이 풀었고, **가장 단순한 선형 정책 + random search (ARS) 가 가장 빠르고 점수도 가장 높았습니다.**
# "가장 단순한 방법으로 풀리면 그것을 쓴다" 는 교훈입니다.
# 반면 스윙업은 "에너지 펌핑 → 붙잡기 → 균형" 을 **전환** 하는 비선형 전략이 필요해서,
# 선형 정책(ARS), 분산이 큰 REINFORCE, 탐색이 고정된 TD3 (20 만 스텝) 는 위로 올리기만 하고 붙잡지 못했습니다.
#
# **2. 모방학습은 빠르고 확실하지만 선생님을 넘지 못합니다.** DAgger 는 2 분, 레이블 5.5 만 개로 선생님 수준(2.5 s)에 도달했지만,
# 강화학습(SAC, PPO)은 보상만으로 **2.5 배 빠른 스윙업 (1.0 s)** 을 찾았습니다.
#
# **3. 이상적 환경의 점수 ≠ 실제 장비에서의 성공.** PPO(σ(s)) 는 이상적 환경에서 100% 유지했지만 지연·양자화가 있는 장비 조건에서는
# 30% 로 떨어졌고, DQN 은 팔을 한계(140°) 가까이 써서 지연이 늘면 실패했습니다. 반대로 **SAC 와 선생님은 모델 오차와 지연에도
# 모두 버텼습니다.** 실제 장비에 올릴 정책은 11 절 같은 검증을 반드시 거쳐야 하며, b −20% 에서 실패한 정책들은
# domain randomization (실습 과제 5) 으로 보완할 수 있습니다.
#
# **4. "스텝당 비용" 과 "필요한 스텝 수" 는 반대 방향입니다.** PPO 는 스텝당 계산이 SAC 의 1/14 이지만 스텝은 10 배 더 썼습니다.
# 시뮬레이터에서는 둘 다 10 분 남짓이지만, **실제 장비에서 학습한다면** 1 스텝 = 20 ms 실제 시간이므로
# SAC 스윙업 20 만 스텝 = 약 67 분, PPO 200 만 스텝 = 약 11 시간의 실험이 됩니다. 실제 로봇 학습에 off-policy 방법이 선호되는 이유입니다.
#
# **방법 고르기 — 이 실험에서 얻은 경험칙**
#
# | 상황 | 추천 |
# |---|---|
# | 잘 동작하는 제어기(선생님)가 있고, 임의의 상태에서 정답을 물어볼 수 있다 | **DAgger** (BC 만으로는 부족) |
# | 문제가 사실상 선형 (직립 균형 등) | 선형 제어 / **ARS** 같은 단순한 방법 |
# | 시뮬레이터가 빠르고 많이 돌릴 수 있다 | **PPO** (안정적, 구현·튜닝이 쉬움) |
# | 데이터가 비싸다 (실제 장비, 느린 시뮬레이터) | **SAC** (데이터 효율, 이 실험에서 가장 강건) |
# | 행동이 본래 이산적이다 | **DQN** |

# %% [markdown]
# ### 15.2 저장된 결과 파일 & 내려받기
# 지금까지 저장된 파일 목록입니다. Colab 에서는 `rl_results.zip` 을 만들어 바로 내려받습니다.
# (로컬 PC 에서는 `simulator/rl_results/` 폴더를 그대로 쓰면 됩니다.)

# %%
for sub in (MODEL_DIR, FIG_DIR, VIDEO_DIR):
    files = sorted(p for p in sub.iterdir() if p.is_file())
    print(f"{sub.relative_to(RESULTS_DIR.parent)}/  ({len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB)")
    for p in files:
        print(f"    {p.name:<44} {p.stat().st_size / 1e3:8.0f} kB")

if "google.colab" in sys.modules:
    zip_path = shutil.make_archive(str(HERE / "rl_results"), "zip", RESULTS_DIR)
    from google.colab import files as colab_files
    colab_files.download(zip_path)


# %% [markdown]
# ---
# ## 16. 정리 & 실습 과제
#
# ### 정리
# | | REINFORCE | DQN | SAC |
# |---|---|---|---|
# | 무엇을 배우나 | 정책 | Q 함수 | 정책 + Q 함수 |
# | 데이터 | on-policy (버림) | off-policy (replay) | off-policy (replay) |
# | 행동 | 이산/연속 | 이산 | **연속** |
# | 탐색 | 확률적 정책 | $\varepsilon$-greedy | 확률적 정책 + 엔트로피 보너스 |
# | 이 노트북의 문제 | CartPole | CartPole | 회전형 진자 균형·스윙업 |
#
# - 강화학습 제어기의 성능은 알고리즘만큼이나 **환경 설계** (관측, 행동 스케일, 제어 주기, 보상, 종료 조건, 초기 상태 분포) 에 좌우됩니다.
# - 강화학습은 모델 식 없이 시뮬레이터만으로 **비선형·다단계 전략** (스윙업 → 캐치 → 균형) 을 하나의 정책으로 찾아낼 수 있습니다.
# - 대신 학습 분포 밖에서는 보장이 없으므로, 실제 장비 조건(지연, 양자화, 모델 오차)에서의 **검증** 이 필수입니다.
#
# ### 실습 과제
# 1. **CartPole 보상 설계**: 수레가 가운데에 머물도록 보상을 바꿔 보세요 (예: $r = 1 - 0.1x^2$). 5 절의 $x$ 그래프가 어떻게 바뀌나요?
# 2. **하이퍼파라미터 민감도**: DQN 의 `tau`, `lr`, `eps_decay_steps` 를 하나씩 바꾸고, 시드 3 개로 학습 곡선을 비교하세요.
#    강화학습 결과는 시드에 따라 크게 다르므로 **한 번의 실행으로 결론 내리지 않는 습관** 이 중요합니다.
# 3. **RL vs 극점 배치**: 8 절의 `policy_gain()` 으로 얻은 $K$ 와 `rotary_model.place4()` 의 $K$ 를 비교하고,
#    선형화 모델 $(A, B)$ (`rotary_model.state_matrices()`) 로 각각의 폐루프 극점을 계산해 보세요.
# 4. **제어 주기**: `RotaryPendulumEnv.DT_CTRL` 을 0.01 (100 Hz) 로 바꾸면 같은 시야(1 s)를 유지하려면 `gamma` 를 얼마로 해야 할까요?
#    ($\gamma^{N} $ 이 같아지도록) 학습 속도와 균형 성능은 어떻게 변하나요?
# 5. **Domain randomization**: `RotaryPendulumEnv.reset()` 에서 매 에피소드 `self.b` 를 $0.8b \sim 1.2b$, 마찰을 0.5~2 배로
#    무작위화해 스윙업을 다시 학습하고, 11.1 의 강건성 표를 비교하세요.
# 6. **보상 함정 체험**: 스윙업 보상을 $r = -\theta^2$ (항상 음수) 로 바꾸고 학습해 보세요. 에이전트가 무엇을 하나요? 왜 그런가요?
# 7. **스윙업 학습 가속**: 9.1 의 세 가지 아이디어 중 하나를 골라 `RotaryPendulumEnv` 를 수정하고, 같은 시드·같은 스텝 수에서
#    `hold` 가 올라가는 시점을 원래 설정과 비교하세요. 시드 2 개 이상으로 비교해야 의미가 있습니다.
# 8. **DQN 행동 이산화**: 10.1 의 `DQN_LEVELS` 를 3 단계 $\{-1,0,1\}$, 11 단계로 바꿔 balance 를 다시 학습하고,
#    입력 $u$ 의 떨림과 점수를 비교하세요. 단계 수와 학습 난이도 사이의 trade-off 를 설명해 보세요.
# 9. **(모방학습) DAgger 반복 수**: B 절의 `train_bc_dagger(dagger_iters=...)` 를 2, 4, 8 로 바꿔 보고, 순수 BC 의 시연 수 `n_demo` 를
#    늘리는 것 (예: 40 → 200 에피소드) 과 DAgger 반복을 늘리는 것 중 어느 쪽이 데이터 대비 효과가 큰지 비교하세요.
# 10. **모방학습 + 강화학습**: DAgger 로 학습한 정책의 가중치로 SAC actor 의 평균 출력을 초기화(또는 BC 손실을 actor 손실에 더함)해
#    스윙업을 학습하면, 처음부터 학습할 때보다 빨라지나요? (시드 2 개 이상으로 비교)
# 11. **σ 설계**: 13.3 절의 고정 σ 값을 0.2, 0.6, 1.0 으로 바꿔 REINFORCE balance 를 학습해 보세요. 너무 작거나 크면 각각 어떤 문제가 생기나요?
# 12. **결정적 정책의 탐색**: TD3 의 `explore_std` 를 0.05, 0.1, 0.3 으로 바꿔 스윙업을 학습해 보세요. SAC 처럼 탐색을 자동 조절하지 못할 때의 trade-off 는?
# 13. **ARS 비선형 정책**: 13.2 절의 선형 정책을 작은 신경망 (예: 5 → 16 → 1) 으로 바꾸면 ARS 로 스윙업을 학습할 수 있을까요?
#    파라미터 수가 늘어나면 random search 의 효율은 어떻게 변하나요?
# 14. **(도전) PPO 직접 다시 짜 보기**: 3 절 REINFORCE 에 (a) 가치함수 baseline, (b) GAE, (c) clipped objective 를 차례로 추가해 14 절의 PPO 를 만들고
#    CartPole 과 회전형 진자 균형에 적용해 보세요.
#
# ### 참고 문헌
# - R. S. Sutton & A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018 — http://incompleteideas.net/book/
# - A. G. Barto, R. S. Sutton, C. W. Anderson, *Neuronlike adaptive elements that can solve difficult learning control problems*, IEEE SMC, 1983 — CartPole
# - R. J. Williams, *Simple statistical gradient-following algorithms for connectionist reinforcement learning*, Machine Learning, 1992 — REINFORCE
# - V. Mnih et al., *Human-level control through deep reinforcement learning*, Nature, 2015 — DQN
# - H. van Hasselt, A. Guez, D. Silver, *Deep Reinforcement Learning with Double Q-learning*, AAAI 2016
# - T. Haarnoja et al., *Soft Actor-Critic Algorithms and Applications*, arXiv:1812.05905, 2018 — SAC
# - K. J. Åström & K. Furuta, *Swinging up a pendulum by energy control*, Automatica, 2000
# - OpenAI Spinning Up — https://spinningup.openai.com
