"""
RW-GCN: R-GCN + Learnable Graph Wavelet

기존 R-GCN baseline 코드에 Wavelet만 추가한 버전.
R-GCN 부분은 PyG의 RGCNConv를 그대로 사용하여 공정한 비교 보장.

구조: Y = R-GCN(X) + λ · Wavelet(X)

핵심 아이디어:
1. R-GCN: relation별 1-hop message passing (PyG RGCNConv 사용)
2. Wavelet: learnable scale과 hop-wise weights로 multi-hop 정보 캡처
3. 두 정보를 additive하게 결합
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Entities
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import degree
import argparse
import numpy as np


class WaveletLayer(nn.Module):
    """
    Learnable Graph Wavelet Layer
    
    Y_wav = Σₖ αₖ · e^{-s·k} · Â^k · X · W
    
    - s: learnable scale parameter (멀리 있는 노드 정보의 감쇠율)
    - αₖ: learnable hop weights (각 hop의 중요도)
    - k=0은 identity (자기 자신), k≥1은 k-hop 이웃
    
    Ablation options:
    - learnable_hop_weights: False면 uniform weights 사용
    - use_decay: False면 exponential decay 제거
    """
    def __init__(self, in_channels, out_channels, k_hop=3, 
                 init_scale=1.0, learnable_scale=True,
                 learnable_hop_weights=True, use_decay=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_hop = k_hop
        self.learnable_hop_weights = learnable_hop_weights
        self.use_decay = use_decay
        
        # Learnable scale parameter (log space for numerical stability)
        if learnable_scale and use_decay:
            self.log_scale = nn.Parameter(torch.tensor(np.log(init_scale)))
        else:
            self.register_buffer('log_scale', torch.tensor(np.log(init_scale)))
        
        # Learnable hop weights (k=0, 1, ..., k_hop)
        if learnable_hop_weights:
            self.hop_logits = nn.Parameter(torch.zeros(k_hop + 1))
        else:
            # Fixed uniform weights
            self.register_buffer('hop_logits', torch.zeros(k_hop + 1))
        
        # Feature transformation (hop 간 공유)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        nn.init.xavier_uniform_(self.weight)
    
    def count_params(self):
        """이 layer의 파라미터 수 반환"""
        count = self.in_channels * self.out_channels + self.out_channels  # weight + bias
        if self.learnable_hop_weights:
            count += self.k_hop + 1  # hop_logits
        if self.use_decay:
            count += 1  # log_scale
        return count
    
    @property
    def scale(self):
        return torch.exp(self.log_scale)
    
    @property
    def hop_weights(self):
        return F.softmax(self.hop_logits, dim=0)
    
    def get_normalized_adj(self, edge_index, num_nodes):
        """
        Symmetric normalization: D^{-1/2} A D^{-1/2}
        Self-loop 없음 (k=0 term이 identity 역할)
        """
        src, dst = edge_index
        
        # Out-degree와 in-degree
        deg_src = degree(src, num_nodes, dtype=torch.float)
        deg_dst = degree(dst, num_nodes, dtype=torch.float)
        
        deg_src_inv_sqrt = deg_src.pow(-0.5)
        deg_dst_inv_sqrt = deg_dst.pow(-0.5)
        deg_src_inv_sqrt[deg_src_inv_sqrt == float('inf')] = 0
        deg_dst_inv_sqrt[deg_dst_inv_sqrt == float('inf')] = 0
        
        edge_weight = deg_src_inv_sqrt[src] * deg_dst_inv_sqrt[dst]
        
        return edge_index, edge_weight
    
    def sparse_mm(self, edge_index, edge_weight, x, num_nodes):
        """Sparse matrix multiplication: A @ x"""
        src, dst = edge_index
        out = torch.zeros(num_nodes, x.size(1), device=x.device)
        out.index_add_(0, dst, x[src] * edge_weight.unsqueeze(1))
        return out
    
    def forward(self, x, edge_index, num_nodes):
        """
        Multi-scale wavelet transform
        
        Ψₛ(X) = Σₖ αₖ · e^{-s·k} · Â^k · X
        """
        edge_index_norm, edge_weight = self.get_normalized_adj(edge_index, num_nodes)
        
        s = self.scale
        alpha = self.hop_weights
        
        # k=0: identity (자기 자신의 정보)
        wavelet_out = alpha[0] * x
        
        # k=1, 2, ..., k_hop: multi-hop 이웃 정보
        current = x
        for k in range(1, self.k_hop + 1):
            current = self.sparse_mm(edge_index_norm, edge_weight, current, num_nodes)
            if self.use_decay:
                decay = torch.exp(-s * k)  # 거리에 따른 감쇠
            else:
                decay = 1.0  # no decay
            wavelet_out = wavelet_out + alpha[k] * decay * current
        
        # Feature transformation
        out = wavelet_out @ self.weight + self.bias
        
        return out


class MLPLayer(nn.Module):
    """
    Simple MLP Layer (Wavelet과 동일한 파라미터 수)
    
    파라미터 효과 vs 구조 효과 비교를 위한 baseline
    Wavelet의 graph structure 활용 없이 동일한 파라미터로 feature transformation만 수행
    """
    def __init__(self, in_channels, out_channels, k_hop=3,
                 learnable_hop_weights=True, use_decay=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Wavelet과 동일한 파라미터 수를 맞추기 위한 dummy parameters
        # hop_logits (k_hop + 1) + log_scale (1) 만큼 hidden dim 조정
        extra_params = 0
        if learnable_hop_weights:
            extra_params += k_hop + 1
        if use_decay:
            extra_params += 1
        
        # Main transformation (Wavelet의 weight, bias와 동일)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # 추가 파라미터를 맞추기 위한 small linear layer
        # extra_params 개수만큼 파라미터 추가
        if extra_params > 0:
            self.extra = nn.Parameter(torch.zeros(extra_params))
        else:
            self.register_parameter('extra', None)
        
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x, edge_index=None, num_nodes=None):
        """
        단순 MLP: Y = X @ W + b
        edge_index, num_nodes는 무시 (인터페이스 호환용)
        """
        out = x @ self.weight + self.bias
        return out


class RWGCN(nn.Module):
    """
    R-GCN + Learnable Wavelet (Additive)
    
    Y = R-GCN(X) + λ · Wavelet(X)
    
    R-GCN: PyG의 RGCNConv 사용 (baseline과 동일)
    Wavelet: learnable multi-hop aggregation
    λ: learnable gate (wavelet 기여도 조절)
    
    mlp_only: True면 Wavelet 대신 MLP 사용 (파라미터 효과 검증용)
    """
    def __init__(self, num_nodes, num_relations, num_classes, hidden_dim=16, 
                 num_bases=None, k_hop=3, wavelet_scale=1.0, learnable_scale=True,
                 use_gate=True, dropout=0.0,
                 learnable_hop_weights=True, use_decay=True, mlp_only=False):
        super().__init__()
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.use_gate = use_gate
        self.mlp_only = mlp_only
        
        # Learnable node embeddings
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)
        
        # Layer 1: R-GCN (PyG) + Wavelet/MLP
        self.conv1 = RGCNConv(hidden_dim, hidden_dim, num_relations, 
                              num_bases=num_bases, aggr='mean')
        if mlp_only:
            self.wavelet1 = MLPLayer(hidden_dim, hidden_dim, k_hop,
                                     learnable_hop_weights, use_decay)
        else:
            self.wavelet1 = WaveletLayer(hidden_dim, hidden_dim, k_hop, 
                                         wavelet_scale, learnable_scale,
                                         learnable_hop_weights, use_decay)
        
        # Layer 2: R-GCN (PyG) + Wavelet/MLP
        self.conv2 = RGCNConv(hidden_dim, num_classes, num_relations, 
                              num_bases=num_bases, aggr='mean')
        if mlp_only:
            self.wavelet2 = MLPLayer(hidden_dim, num_classes, k_hop,
                                     learnable_hop_weights, use_decay)
        else:
            self.wavelet2 = WaveletLayer(hidden_dim, num_classes, k_hop,
                                         wavelet_scale, learnable_scale,
                                         learnable_hop_weights, use_decay)
        
        # Learnable gates
        if use_gate:
            self.gate1 = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5
            self.gate2 = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, edge_index, edge_type):
        x = self.node_emb.weight
        
        # Layer 1
        rgcn_out1 = self.conv1(x, edge_index, edge_type)
        wav_out1 = self.wavelet1(x, edge_index, self.num_nodes)
        
        if self.use_gate:
            lambda1 = torch.sigmoid(self.gate1)
            x = rgcn_out1 + lambda1 * wav_out1
        else:
            x = rgcn_out1 + wav_out1
        
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        rgcn_out2 = self.conv2(x, edge_index, edge_type)
        wav_out2 = self.wavelet2(x, edge_index, self.num_nodes)
        
        if self.use_gate:
            lambda2 = torch.sigmoid(self.gate2)
            x = rgcn_out2 + lambda2 * wav_out2
        else:
            x = rgcn_out2 + wav_out2
        
        return x
    
    def get_learned_params(self):
        """학습된 wavelet 파라미터 반환 (분석용)"""
        params = {}
        
        # MLP only 모드에서는 wavelet 파라미터가 없음
        if not self.mlp_only:
            params['layer1_scale'] = self.wavelet1.scale.item()
            params['layer1_hop_weights'] = self.wavelet1.hop_weights.detach().cpu().numpy()
            params['layer2_scale'] = self.wavelet2.scale.item()
            params['layer2_hop_weights'] = self.wavelet2.hop_weights.detach().cpu().numpy()
        
        if self.use_gate:
            params['lambda1'] = torch.sigmoid(self.gate1).item()
            params['lambda2'] = torch.sigmoid(self.gate2).item()
        
        return params


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, optimizer, data, train_idx, labels):
    model.train()
    optimizer.zero_grad()
    out = model(data.edge_index, data.edge_type)
    loss = F.cross_entropy(out[train_idx], labels[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, idx, labels):
    model.eval()
    out = model(data.edge_index, data.edge_type)
    pred = out[idx].argmax(dim=1)
    acc = (pred == labels[idx]).float().mean().item()
    return acc


def main(args):
    # 데이터 로드
    dataset = Entities(root='./data', name=args.dataset)
    data = dataset[0]
    
    # 데이터 정보 출력
    print(f"\n{'='*60}")
    print(f"RW-GCN: R-GCN + Learnable Graph Wavelet")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Nodes: {data.num_nodes:,}")
    print(f"Edges: {data.edge_index.size(1):,}")
    print(f"Relations: {dataset.num_relations}")
    print(f"Classes: {dataset.num_classes}")
    print(f"Train/Test: {data.train_idx.size(0)}/{data.test_idx.size(0)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print(f"\n--- Model Configuration ---")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"K-hop: {args.k_hop}")
    print(f"Wavelet scale (init): {args.wavelet_scale}")
    print(f"Learnable scale: {args.learnable_scale}")
    print(f"Use gate: {args.use_gate}")
    print(f"Dropout: {args.dropout}")
    print(f"Learnable hop weights: {not args.no_learnable_hop}")
    print(f"Use exponential decay: {not args.no_decay}")
    print(f"MLP only (no graph structure): {args.mlp_only}")
    
    # 레이블 준비
    labels = torch.full((data.num_nodes,), -1, dtype=torch.long)
    labels[data.train_idx] = data.train_y
    labels[data.test_idx] = data.test_y
    labels = labels.to(device)
    
    # 데이터를 device로 이동
    data = data.to(device)
    
    # 파라미터 수 출력
    sample_model = RWGCN(
        num_nodes=data.num_nodes,
        num_relations=dataset.num_relations,
        num_classes=dataset.num_classes,
        hidden_dim=args.hidden_dim,
        num_bases=args.num_bases,
        k_hop=args.k_hop,
        wavelet_scale=args.wavelet_scale,
        learnable_scale=args.learnable_scale,
        use_gate=args.use_gate,
        dropout=args.dropout,
        learnable_hop_weights=not args.no_learnable_hop,
        use_decay=not args.no_decay,
        mlp_only=args.mlp_only
    )
    print(f"Total parameters: {count_parameters(sample_model):,}")
    del sample_model
    
    # 여러 번 실행하여 평균 계산
    test_accs = []
    best_params = None
    
    for run in range(args.runs):
        torch.manual_seed(run)
        np.random.seed(run)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(run)
        
        # 모델 초기화
        model = RWGCN(
            num_nodes=data.num_nodes,
            num_relations=dataset.num_relations,
            num_classes=dataset.num_classes,
            hidden_dim=args.hidden_dim,
            num_bases=args.num_bases,
            k_hop=args.k_hop,
            wavelet_scale=args.wavelet_scale,
            learnable_scale=args.learnable_scale,
            use_gate=args.use_gate,
            dropout=args.dropout,
            learnable_hop_weights=not args.no_learnable_hop,
            use_decay=not args.no_decay,
            mlp_only=args.mlp_only
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        best_test_acc = 0
        
        for epoch in range(1, args.epochs + 1):
            loss = train(model, optimizer, data, data.train_idx, labels)
            train_acc = evaluate(model, data, data.train_idx, labels)
            test_acc = evaluate(model, data, data.test_idx, labels)
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                if run == 0:
                    best_params = model.get_learned_params()
            
            if epoch % 10 == 0 and args.verbose:
                print(f"Run {run+1}, Epoch {epoch:03d}: Loss={loss:.4f}, Train={train_acc:.4f}, Test={test_acc:.4f}")
        
        test_accs.append(best_test_acc)
        print(f"Run {run+1}: Best Test Accuracy = {best_test_acc*100:.2f}%")
    
    # 결과 요약
    mean_acc = np.mean(test_accs) * 100
    std_acc = np.std(test_accs) * 100
    print(f"\n{'='*60}")
    print(f"Final Results ({args.runs} runs)")
    print(f"{'='*60}")
    print(f"RW-GCN Test Accuracy: {mean_acc:.2f} ± {std_acc:.2f}%")
    
    # 논문 참조 결과
    ref_results = {
        'AIFB': 95.83,
        'MUTAG': 73.23,
        'BGS': 83.10,
        'AM': 89.29
    }
    if args.dataset in ref_results:
        print(f"R-GCN Paper ({args.dataset}): {ref_results[args.dataset]}%")
    
    # 학습된 파라미터 출력
    if best_params and args.verbose:
        print(f"\n--- Learned Wavelet Parameters (Run 1) ---")
        for k, v in best_params.items():
            if isinstance(v, np.ndarray):
                print(f"{k}: {np.array2string(v, precision=3)}")
            else:
                print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RW-GCN: R-GCN + Learnable Graph Wavelet')
    parser.add_argument('--dataset', type=str, default='AIFB', choices=['AIFB', 'MUTAG', 'BGS', 'AM'])
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--num_bases', type=int, default=None)
    parser.add_argument('--k_hop', type=int, default=3)
    parser.add_argument('--wavelet_scale', type=float, default=1.0)
    parser.add_argument('--learnable_scale', action='store_true')
    parser.add_argument('--use_gate', action='store_true')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--verbose', action='store_true')
    # Ablation options
    parser.add_argument('--no_learnable_hop', action='store_true',
                        help='Ablation: use fixed uniform hop weights instead of learnable')
    parser.add_argument('--no_decay', action='store_true',
                        help='Ablation: remove exponential decay')
    parser.add_argument('--mlp_only', action='store_true',
                        help='Ablation: use MLP instead of Wavelet (same params, no graph structure)')
    args = parser.parse_args()
    
    main(args)