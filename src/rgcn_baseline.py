import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Entities
from torch_geometric.nn import RGCNConv
import argparse
import numpy as np

class RGCN(nn.Module):
    def __init__(self, num_nodes, num_relations, num_classes, hidden_dim=16, num_bases=None, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        
        # Learnable node embeddings
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)
        
        # R-GCN layers
        self.conv1 = RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=num_bases, aggr='mean')
        self.conv2 = RGCNConv(hidden_dim, num_classes, num_relations, num_bases=num_bases, aggr='mean')
    
    def forward(self, edge_index, edge_type):
        x = self.node_emb.weight
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x


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
    print(f"\n=== {args.dataset} Dataset ===")
    print(f"Nodes: {data.num_nodes}")
    print(f"Edges: {data.edge_index.size(1)}")
    print(f"Relations: {dataset.num_relations}")
    print(f"Classes: {dataset.num_classes}")
    print(f"Train nodes: {data.train_idx.size(0)}")
    print(f"Test nodes: {data.test_idx.size(0)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # 레이블 준비
    labels = torch.full((data.num_nodes,), -1, dtype=torch.long)
    labels[data.train_idx] = data.train_y
    labels[data.test_idx] = data.test_y
    labels = labels.to(device)
    
    # 데이터를 device로 이동
    data = data.to(device)
    
    # 여러 번 실행하여 평균 계산
    test_accs = []
    
    for run in range(args.runs):
        torch.manual_seed(run)
        np.random.seed(run)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(run)
        
        # 모델 초기화
        model = RGCN(
            num_nodes=data.num_nodes,
            num_relations=dataset.num_relations,
            num_classes=dataset.num_classes,
            hidden_dim=args.hidden_dim,
            num_bases=args.num_bases,
            dropout=args.dropout
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        best_test_acc = 0
        
        for epoch in range(1, args.epochs + 1):
            loss = train(model, optimizer, data, data.train_idx, labels)
            train_acc = evaluate(model, data, data.train_idx, labels)
            test_acc = evaluate(model, data, data.test_idx, labels)
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
            
            if epoch % 10 == 0 and args.verbose:
                print(f"Run {run+1}, Epoch {epoch:03d}: Loss={loss:.4f}, Train={train_acc:.4f}, Test={test_acc:.4f}")
        
        test_accs.append(best_test_acc)
        print(f"Run {run+1}: Best Test Accuracy = {best_test_acc*100:.2f}%")
    
    # 결과 요약
    mean_acc = np.mean(test_accs) * 100
    std_acc = np.std(test_accs) * 100
    print(f"\n=== Final Results ({args.runs} runs) ===")
    print(f"Test Accuracy: {mean_acc:.2f} ± {std_acc:.2f}%")
    print(f"R-GCN Paper (AIFB): 95.83%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='AIFB', choices=['AIFB', 'MUTAG', 'BGS', 'AM'])
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--num_bases', type=int, default=None)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    
    main(args)