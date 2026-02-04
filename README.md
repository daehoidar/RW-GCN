# RW-GCN: R-GCN with Learnable Graph Wavelet

> Relational Graph Convolutional Network에 Learnable Graph Wavelet 활용 가능성 탐구

서울시립대학교 통계학과 박민우

---

## 1. 서론

Knowledge Graph와 같이 relation type이 두 개 이상인 **Heterogeneous Graph**를 처리하기 위한 방법론으로 **R-GCN** (Relational Graph Convolutional Network) [[1]](#references)이 존재한다.

그러나 R-GCN은 relation별 메시지를 반복적으로 평균화하기 때문에, layer가 깊어질수록 노드 표현의 분산이 빠르게 감소하는 **over-smoothing** 문제가 발생한다. 이로 인해 R-GCN은 2~3 layer로 제한되며, 멀리 있는 노드의 정보를 전달받기 어렵다.

이를 개선하기 위해, **단일 layer 내에서 multi-hop 정보를 집계하는 Graph Wavelet 구조를 R-GCN에 추가**하는 방법을 실험하였다. 단일 layer에서 수행되므로 representation collapse를 유발하는 반복적 smoothing을 피할 수 있을 것으로 기대했다.

**핵심 아이디어:**
- **R-GCN**: relation-aware한 1-hop 정보 처리
- **Wavelet**: relation-agnostic한 multi-hop 구조 정보 처리
- 두 정보를 **additive**하게 결합

---

## 2. 핵심 이론

### 2.1 기존 Graph Wavelet [[2]](#references) [[3]](#references)

**Wavelet Transform:**

$$\Psi\_s = U \, \text{diag}(e^{-s\lambda\_1}, \ldots, e^{-s\lambda\_n}) \, U^T$$

**Heat Kernel** 기반이며, Chebyshev 다항식으로 근사 가능하다.

### 2.2 Learnable Graph Wavelet (제안)

기존 wavelet을 learnable하게 확장한 구조를 제안하였다.

$$\Psi\_s(X) = \sum\_{k=0}^{K} \alpha\_k \cdot e^{-s \cdot k} \cdot \hat{A}^k \cdot X$$

각 요소의 의미:

| 기호 | 설명 |
|:---:|:---|
| $\alpha\\_k$ | k-hop의 learnable weight (softmax 정규화) |
| $s$ | learnable scale parameter |
| $e^{-s \cdot k}$ | 거리에 따른 exponential decay |
| $\hat{A}$ | Symmetric normalized adjacency |

Hop별 의미:

| Hop | 수식 | 설명 |
|:---:|:---:|:---|
| k=0 | $I \cdot X$ | 자기 자신 (identity) |
| k=1 | $\hat{A} \cdot X$ | 1-hop 이웃 |
| k=2 | $\hat{A}^2 \cdot X$ | 2-hop 이웃 |

파라미터 수를 줄이기 위해 **hop 거리별로 동일한 파라미터를 공유**한다.

### 2.3 RW-GCN 최종 구조

R-GCN과 Wavelet을 additive하게 결합한다.

$$Y = \underbrace{f\_{\text{R-GCN}}(X)}\_{\text{1-hop, relation-aware}} + \lambda \cdot \underbrace{\Psi\_s(X) W}\_{\text{multi-hop, relation-agnostic}}$$

| 요소 | 설명 |
|:---:|:---|
| $f\\_{\text{R-GCN}}(X)$ | R-GCN (1-hop, relation-aware) |
| $\Psi\\_s(X) W$ | Wavelet (multi-hop) |
| $\lambda$ | Gate parameter (optional, sigmoid) |

---

## 3. 실험 결과

**Node Classification Benchmark** (10 runs 평균 ± 표준편차)

| Model | AIFB | MUTAG | BGS |
|:---|:---:|:---:|:---:|
| R-GCN (paper) | 95.83 | 73.23 | 83.10 |
| R-GCN (ours) | 95.83 ± 1.39 | 73.38 ± 2.23 | 86.90 ± 2.07 |
| w/ MLP | 95.56 ± - | 73.38 ± - | 89.31 ± - |
| **RW-GCN** | **96.67 ± 2.42** | **74.12 ± 4.17** | **88.97 ± 1.38** |
| RW-GCN + Gate | 96.94 ± 1.94 | 73.68 ± 4.23 | 89.66 ± 0.00 |

**Ablation Study:**

| Model | AIFB | MUTAG | BGS |
|:---|:---:|:---:|:---:|
| RW-GCN (full) | 96.67 | 74.12 | 88.97 |
| w/o Learnable α | 96.11 | 74.26 | 88.97 |
| w/o Decay | 96.67 | 78.62 | 88.62 |
| w/o Multi-hop (k=1) | 96.39 | 74.12 | 89.31 |

RW-GCN은 **0.22%의 파라미터 증가(350개)로 평균 +1.22%의 성능 향상**을 달성하였다.

AIFB와 MUTAG에서는 동일 파라미터 수의 MLP보다 높은 성능을 보여 graph 구조 활용의 효과를 확인하였으나, BGS에서는 MLP가 더 효과적이어서 데이터셋 특성에 따른 차이가 있음을 확인하였다.

---

## 4. 한계점

- Node classification task에서만 평가를 진행하였으며, link prediction task에 대한 검증이 필요하다.
- k=3으로 고정하여 실험하였으며, 데이터셋별 최적 k값 탐색이 이루어지지 않았다.
- Ablation study 결과, learnable hop weights와 exponential decay가 모든 데이터셋에서 효과적이지는 않았다.
- 제안한 구조가 over-smoothing을 실제로 완화하는지에 대한 이론적 분석이 부족하다.

---

## References

1. M. Schlichtkrull et al., ["Modeling Relational Data with Graph Convolutional Networks"](https://arxiv.org/abs/1703.06103), ESWC 2018.
2. D. K. Hammond et al., ["Wavelets on Graphs via Spectral Graph Theory"](https://arxiv.org/abs/0912.3848), Applied and Computational Harmonic Analysis, 2011.
3. B. Xu et al., ["Graph Wavelet Neural Network"](https://arxiv.org/abs/1904.07785), ICLR 2019.
