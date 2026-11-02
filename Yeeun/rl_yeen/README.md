# rl_yeen — 보상 설계가 제어 성능을 어떻게 바꾸는가

회전형 역진자(STEVAL-EDUKIT01)의 동정 모델 위에서 **SAC**를 학습시키되,
**보상 함수의 항을 하나씩 바꿔가며** 무엇이 실제로 성능을 만드는지 측정한다.

앞선 연구 [`behavior_cloning_yeen/`](../../behavior_cloning_yeen) 의 결론에서 출발한다.

| 모방학습에서 남은 문제 | 원인 | 강화학습의 수단 |
|---|---|---|
| 세운 막대가 **8.9° 틀어진 채** 유지됨 | 교사의 숨은 상태 `b_hat` 을 학생이 못 봄 | 보상 `0.5(1+cos θ)` 가 **수직에 가까울수록** 직접 이득 |
| DAgger 31판이 **팔 한계로 폭주** | 팔 위치를 제약할 수단이 없음 | 보상 `−0.1(φ/140°)²` 가 팔을 가운데로 |

모방학습은 **"교사가 뭘 했는지"** 만 말할 수 있고, 강화학습은 **"결과가 어떠해야 하는지"** 를 말할 수 있다.
그렇다면 그 "말하는 방식"을 바꾸면 결과가 어떻게 달라지는가 — 이것이 이 폴더의 질문이다.

---

## 비교하는 보상 다섯 가지

```python
r = up(θ) − k_arm·(φ/140°)² − k_eff·a²        실패로 종료되면 + term
```

| 이름 | `up(θ)` | `k_arm` | `k_eff` | `term` | 무엇을 확인하나 |
|---|---|---|---|---|---|
| `full` | `0.5(1+cos θ)` | 0.1 | 0.01 | −50 | **기준선** (수업 노트북의 보상) |
| `no_arm` | `0.5(1+cos θ)` | **0** | 0.01 | −50 | 팔 항을 빼면 **DAgger 31처럼 팔이 폭주**하는가 |
| `no_effort` | `0.5(1+cos θ)` | 0.1 | **0** | −50 | 노력 항을 빼면 명령만 커지는가 |
| `sparse` | **±10° 안이면 1, 밖이면 0** | 0.1 | 0.01 | −50 | 기울기가 없는 보상으로 **학습이 시작되는가** |
| `trap` | **−θ²** | 0 | 0 | **0** | **종료가 이득이 되는 설계** — 일부러 실패시킨 것 |

`trap` 은 수업 노트북이 *글로만* 경고한 함정을 **실측**하려는 조건이다.
보상이 항상 음수고 실패에 벌점이 없으면, 매달린 채 손해를 누적하느니
**팔을 한계까지 돌려 에피소드를 끝내는 쪽이 리턴이 크다.**
무입력으로 이미 확인된다 — 같은 궤적의 누적 보상이 `full` 은 −33.34, `trap` 은 **−1.37**.

---

## 파일

| 파일 | 역할 |
|---|---|
| `rl_common.py` | 환경 `RotaryEnv`, 보상 정의 `REWARDS`, 평가 지표 `metrics`, numpy 추론 `NumpyActor` |
| `train_sac.py` | SAC 학습. `--reward` 로 보상 선택, `--sweep` 으로 5개 전부 |
| `eval_rl.py` | 체크포인트 일괄 평가 + 모델 오차 강건성 (numpy만 필요) |
| `reward_ablation.ipynb` | Colab용 — 학습부터 시뮬레이션·표까지 |
| `checkpoints/` | `sac_<task>_<reward>.npz` (actor 가중치 + 학습 보고서) |

### 환경 사양

```
관측   o = [cos θ, sin θ, θ', φ, φ']        θ: 직립 = 0, 매달림 = ±π
행동   a ∈ [-1, 1]  →  팔 가속도 u = a × 47.1 rad/s²
주기   물리 1 kHz RK4, 에이전트 50 Hz (20 ms)   ← 실기 루프는 100 Hz 라 같은 명령을 2번 보낸다
과제   balance  직립 근처 시작, 5 초, |θ|>30° 면 종료
       swingup  70% 매달림 / 30% 무작위, 10 초
종료   |φ| > 140°  (실기 호스트의 정지 조건과 동일)
```

모터 제약(최소 구동 속도 `SPEED_MIN`, 최대 속도 `SPEED_MAX`)을 **환경 안에 넣었다.**
빼고 학습하면 실기가 실행할 수 없는 명령을 배운다.

`simulator/` 는 이 폴더에서 위로 올라가며 찾는다(최대 5단계). 폴더를 옮겨도 된다.

---

## 평가 규칙 — 모방학습에서 배운 두 가지

**1. 모든 에이전트를 같은 보상으로 채점한다.**

```powershell
python Yeeun/rl_yeen/eval_rl.py --score full
```

자기가 학습한 보상으로 잰 리턴은 측정이 아니라 **자기 평가**다.
`trap` 은 짧은 에피소드에 `−θ²` 를 적용해 "우승"해버린다.

**2. 직립 판정을 ±10°, ±5°, ±2° 세 가지로 본다.**

모방학습 모델이 **8.9° 틀어진 채 ±10° 를 100% 통과**했다.
한 가지 기준만 보면 *버티는 정책* 과 *넘어지지만 않는 정책* 이 구분되지 않는다.

| 지표 | 뜻 |
|---|---|
| `bias` | 마지막 2초 평균 각도 — **중심이 0에서 얼마나 틀어졌나** |
| `wobble` | 그 구간 표준편차 — **중심 주위로 얼마나 떠는가** |
| `t_up` | 처음 ±10° 에 들어간 시각 |
| `arm` | 에피소드 중 최대 \|φ\| (한계 140°) |
| `sat%` | 명령이 포화한 비율 |

`bias` 와 `wobble` 은 **다른 문제**다. 안정적이면서 틀어져 있을 수 있다.

### 기준값 (teacher, 무오차 플랜트)

| 과제 | return | hold10 | hold2 | bias | arm |
|---|---|---|---|---|---|
| balance | 249.5 / 250 | 100% | 100% | +0.00° | 43.8° |
| swingup | 413.6 / 480 | 100% | 100% | −0.11° | 84.5° |

`teacher` 는 실기의 에너지 스윙업 + 극점 배치 제어기를 50 Hz 무상태 함수로 단순화한 것이다
(`eval_rl.teacher`). RL이 넘어야 할 선.

---

## 실행

저장소 루트에서 실행한다.

```powershell
conda activate control2026
$env:PYTHONPATH = "Yeeun/rl_yeen"
```

```powershell
# 보상 5개 전부 학습 (balance, 각 30,000 스텝 — CPU 로 전체 20~25분)
python Yeeun/rl_yeen/train_sac.py --sweep --task balance
```

```powershell
# 같은 잣대로 채점
python Yeeun/rl_yeen/eval_rl.py --task balance --score full
```

```powershell
# 모델 오차까지 (b ±20%, 마찰 2배)
python Yeeun/rl_yeen/eval_rl.py --task balance --suite robust
```

```powershell
# 가장 좋았던 보상으로 swing-up (200,000 스텝 — CPU 로 20~40분)
python Yeeun/rl_yeen/train_sac.py --task swingup --reward full --steps 200000
```

---

## 설계 메모

- **SAC를 쓴 이유** — 팔 가속도가 연속값이고, 환경 스텝이 (물리 20스텝 RK4라) 비싸다.
  off-policy 라 리플레이 버퍼에서 전이를 **여러 번 재사용**한다. REINFORCE 는 한 번 쓰고 버린다.
- **`alpha` 자동 조정** — 보상의 스케일을 의도적으로 바꾸는 연구라, 고정 `alpha` 는 조건마다
  다시 손봐야 한다. 목표 엔트로피를 정해 자동으로 맞춘다.
- **`truncated` 로는 부트스트랩을 끊지 않는다** — 시간 제한은 실패가 아니다. 끊으면
  "시간이 다 되는 것"을 실패로 배운다.
- **체크포인트는 `.npz`** — actor 평균 머리만 저장해서 추론에 numpy만 쓴다.
  실기 호스트에 torch 를 깔 필요가 없다. `check_numpy_matches_torch()` 가 내보내기를 검증한다.
- **`NumpyActor.as_rig_controller()`** — 같은 정책을 `rotary_model.Loop` 에 그대로 꽂아
  100 Hz · 엔코더 양자화 · 4 ms 지연 조건에서 돌릴 수 있다 (sim-to-real 점검).

## 이어지는 연구

- [`Yeeun/pipeline_yeen/`](../pipeline_yeen) — 모방학습과 강화학습의 **융합**.
  BC → DAgger → SAC fine-tune 을 하나의 정책으로 잇는다.
  여기서 다룬 보상(`full`)이 그 fine-tune 단계의 보상이다.

## 참고

- 환경 설계와 SAC 구성은 수업 노트북 `simulator/rl_tutorial_cartpole_rotary_v3.ipynb` 를 참고했다.
  플랜트 상수는 `simulator/rotary_model.py` 에서 가져온다.
- Haarnoja et al., *Soft Actor-Critic* (2018)
- Ross, Gordon & Bagnell, *A Reduction of Imitation Learning to No-Regret Online Learning* (2011) — 앞 연구
