# pipeline_yeen — 모방학습으로 시작해서 강화학습으로 넘어서기

**하나의 정책 네트워크**를 세 단계로 학습시킨다.

```
전문가 시연 ──BC──▶ π₀ ──DAgger──▶ π₁ ──SAC fine-tune──▶ π*
                                      └ 시연을 리플레이 버퍼에 미리 넣음
                                      └ BC 정규화 항을 감쇠시켜 떼어냄
```

## 왜 융합하는가

두 방법은 서로의 약점에 맞물린다.

| | 할 수 있는 것 | 못 하는 것 |
|---|---|---|
| **강화학습** | 보상을 직접 최대화 → **전문가보다 잘할 수 있다** | 어려운 과제를 **우연히** 성공해야 시작된다 |
| **모방학습** | 첫 에피소드부터 올바른 행동 | 베낀 것보다 **잘할 수 없다** |

`swingup` 이 그 "어려운 과제"다. 보상의 수직 항을 한 번이라도 받으려면 매달린 막대를
±180° 넘겨 **느리게** 도착시키고, 그걸 **팔 140° 예산 안에서** 해야 한다.
측정해 보면 (`fusion.ipynb` 2절):

```
무작위 정책 30판
  balance   27판이 수직을 스침 (이미 세워서 넘겨주므로)
  swingup    1판만 스침 -- 그 판의 시작 각도가 4.5도, 즉 처음부터 위에 있었다
             매달린 상태에서 올려 세운 판은 0판
```

critic 이 평가해야 할 행동을 **본 적이 없다.** 복제는 그 행동을 첫 판에 만들어 주고,
복제는 거기까지다. 그래서 복제한 정책을 **출발점**으로 쓴다.

## 성립 조건 — 같은 언어

하나의 네트워크를 물려주려면 모든 단계가 같은 관측·행동 공간을 써야 한다.
앞선 두 폴더는 달랐다.

| | `behavior_cloning_yeen` | `rl_yeen` |
|---|---|---|
| 관측 | `[sin θ, cos θ, θ', φ, φ']` × K프레임 | `[cos θ, sin θ, θ', φ, φ']` × 1 |
| 행동 | `u` [rad/s²] | `a ∈ [-1, 1]` (tanh) |
| 주기 | 100 Hz (`rm.Loop`) | 50 Hz (자체 RK4) |

그래서 **모방 단계를 RL 규약으로 다시 한다.** BC 목표는 행동 자체이고, squash를 통과시켜 잰다.

```
loss = mean( (tanh(mu(o)) − a*)² )
```

`atanh(a*)` 에 맞추지 않는 이유 — 전문가는 스윙업 구간에서 **포화 명령(|a*| > 0.99)을 약 2%**
의 시간 동안 낸다. 그 지점에서 `atanh` 는 발산한다.

## 망가지기 쉬운 세 지점

복제한 가중치는 SAC 기본 설정에서 **수백 스텝 안에 지워진다.** `fusion.finetune` 이 막는 것들.

| | 문제 | 대응 |
|---|---|---|
| 1 | critic 이 난수 → 그 기울기도 난수 | 시연만으로 critic 을 먼저 2,000스텝 (`critic_warm`) |
| 2 | 엔트로피 목표가 정책을 무작위로 끌어당김 | `log_std` ≈ 0.1 (`LOG_STD0`), `alpha` 를 0.02 (`ALPHA0`) 에서 시작 |
| 3 | 균일 난수 탐험 구간 | warm start 에서는 `start_steps = 0` — 그건 cold start 가 필요한 것 |

그리고 시연으로 끌어당기는 항을 감쇠시켜 결국 떼어낸다.

```
loss_actor = alpha·log π − min(Q₁,Q₂) + c(t)·‖tanh μ(o) − a*‖²      c: 1 → 0 (bc_decay)
```

`c(t) = 0` 이 된 뒤로 정책을 붙잡는 것은 보상뿐이다. **그때 전문가를 넘어서는지가 결과다.**

## 한 번의 롤아웃에서 두 가지가 나온다

DAgger 구현의 흔한 버그라 분리해 두었다 (`fusion.collect`).

| | 무엇 | 어디로 |
|---|---|---|
| `a_exec` | 실제로 적용된 행동 | **리플레이 버퍼** — 그것이 만든 보상·다음 상태와 함께 |
| `a_star` | 전문가가 여기서 할 행동 | **BC 데이터셋** — 라벨 |

## 파일

| 파일 | 역할 |
|---|---|
| `fusion.py` | 시연 수집 → BC → DAgger → SAC fine-tune. `--baseline` 로 cold 대조군까지 |
| `arena.py` | **공통 채점장** — `rm.Loop` 100 Hz, 양자화 엔코더, 4 ms 지연 (실기의 루프) |
| `animate.py` | 여러 제어기를 같은 에피소드에서 나란히 보여주는 애니메이션 / 궤적 |
| `fusion.ipynb` | Colab용 — 위 전부 |
| `checkpoints/` | `bc_<task>.npz`, `dagger_<task>.npz`, `sac_warm_<task>.npz`, `sac_cold_<task>.npz` |

### 왜 채점장을 따로 두는가

RL은 **50 Hz · 완벽한 센싱**에서 학습하고, 실기는 **100 Hz · 양자화 엔코더 · 4 ms 지연**이다.
그 불일치는 비교의 결함이 아니라 **sim-to-real 질문 그 자체**다. 그래서 모든 단계를 두 곳에서 잰다.

```
ret / hold2 / bias / arm   50 Hz 학습 환경 — SAC가 실제로 최적화한 숫자
arena_*                    100 Hz rm.Loop — 실기의 루프
```

왼쪽에서만 이기는 단계는 **전이에 실패한 것이다.**

`arena.py` 를 단독으로 돌리면 모방학습 쪽 체크포인트도 같은 잣대로 채점된다.

```powershell
python Yeeun/pipeline_yeen/arena.py --task swingup --suite robust
```

측정된 것 — **깨끗한 조건은 아무것도 구분하지 못한다.**

| hold2 [%] | clean | 2× 마찰 | **offset 2°** |
|---|---|---|---|
| expert | 100 | 100 | **100** |
| `bc_h1` | 66.7 | 57.7 | **0 — 3회 전부 팔 한계** |
| `bc_h5` | 100 | 59.3 | **0 — 3회 전부 팔 한계** |
| `bc_h30` | 100 | 100 | **100** |
| `dagger_h5_iter1~3` | 100 | 61~100 | **100** |

`offset 2°` 는 엔코더 영점이 2° 틀린 조건이다. 전문가는 숨은 적분기 `SwingUp.b_hat` 으로
그 2°를 스스로 찾아 흡수한다. 학생은 **그 상태를 관측하지 못한다** — 8.9° 편향 사건의 정체다.
RL 정책에도 적분기는 없다. **보상이 그 역할을 대신할 수 있는가**가 6절의 질문이다.

## 실행

저장소 루트에서 실행한다. `arena.py` 가 `simulator`, `rl_yeen`, `behavior_cloning_yeen` 을
**스스로 찾아** `sys.path` 에 넣으므로 `PYTHONPATH` 를 손으로 잡을 필요는 없다
(이 폴더에서 위로 올라가며 최대 5단계까지 찾는다 — 폴더를 또 옮겨도 된다).

```powershell
conda activate control2026
```

```powershell
# 배관 점검 — 네 단계를 아주 작은 숫자로 전부 통과 (약 1분). 긴 학습 전에 반드시.
python Yeeun/pipeline_yeen/fusion.py --task swingup --smoke
```

```powershell
# 본 실행. cold 대조군까지 (CPU 40~70분)
python Yeeun/pipeline_yeen/fusion.py --task swingup --steps 60000 --baseline --csv Yeeun/pipeline_yeen/fusion_swingup.csv
```

```powershell
# 공통 채점장에서 전부 다시 채점 (numpy만 필요)
python Yeeun/pipeline_yeen/arena.py --task swingup --suite robust
```

의미 있는 비교는 **같은 예산의 cold start** 하나뿐이다. "SAC가 풀었다" 는 출발점이 없던
실행과 나란히 놓이지 않으면 아무 말도 아니다.

## 앞뒤 연구

- [`behavior_cloning_yeen/`](../../behavior_cloning_yeen) — BC vs DAgger, 데이터 수 비교.
  여기서 8.9° 편향과 DAgger 하한선이 나왔다.
- [`rl_yeen/`](../../rl_yeen) — 보상 항 ablation. 위 표의 세 번째 줄("보상을 잘못 쓰면 그걸
  충실히 달성한다")을 따로 다룬다. `trap` 보상에서는 **일부러 실패하는 쪽이 이득**이다.

## 참고

- Ross, Gordon & Bagnell, *A Reduction of Imitation Learning to No-Regret Online Learning* (2011) — DAgger
- Haarnoja et al., *Soft Actor-Critic* (2018)
- Vecerik et al., *Leveraging Demonstrations for Deep RL* (2017) — 시연을 리플레이 버퍼에 넣는 방식
- Nair et al., *AWAC* (2020) — 사전학습 정책을 온라인 RL로 이어받을 때의 붕괴 문제
- 환경·SAC 구성은 수업 노트북 `simulator/rl_tutorial_cartpole_rotary_v3.ipynb` 를 참고했고,
  플랜트 상수는 `simulator/rotary_model.py` 에서 가져온다.
