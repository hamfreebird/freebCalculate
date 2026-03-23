"""
Corel 5k 图片关键词提取模型训练
使用多标签分类方法，从头训练神经网络
"""

import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import arff
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


# ==================== 1. 数据加载 ====================
def load_corel5k(arff_path):
    """
    加载Corel 5k ARFF文件
    返回: X特征矩阵, y标签矩阵, label_names
    """
    data, meta = arff.loadarff(arff_path)
    df = pd.DataFrame(data)

    # 特征列：前499列是数值特征
    # 标签列：最后374列是关键词标签（二进制）
    n_features = 499
    X = df.iloc[:, :n_features].values.astype(np.float32)
    y = df.iloc[:, n_features:].values

    # 将bytes类型标签转换为整数
    y = y.astype(np.int8)

    # 获取标签名称
    label_names = [name for name in df.columns[n_features:]]
    label_names = [
        name.decode("utf-8") if isinstance(name, bytes) else name
        for name in label_names
    ]

    print(f"数据加载完成:")
    print(f"  - 样本数: {X.shape[0]}")
    print(f"  - 特征维度: {X.shape[1]}")
    print(f"  - 关键词数量: {y.shape[1]}")
    print(f"  - 平均每图关键词数: {y.sum(axis=1).mean():.2f}")

    return X, y, label_names


# ==================== 2. PyTorch 数据集类 ====================
class Corel5kDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)  # 多标签使用float作为BCE目标

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==================== 3. 神经网络模型 ====================
class ImageKeywordModel(nn.Module):
    """
    从头构建的多标签分类模型
    输入: 499维特征向量
    输出: 374维（每个关键词的预测概率）
    """

    def __init__(
        self, input_dim=499, hidden_dims=[512, 256, 128], output_dim=374, dropout=0.3
    ):
        super(ImageKeywordModel, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ==================== 4. 训练函数 ====================
def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=50,
    device="cpu",
    patience=10,
):
    """
    训练多标签分类模型
    """
    model.to(device)
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_precision": []}

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)

        train_loss /= len(train_loader.dataset)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)

                # 使用sigmoid + 阈值0.5进行预测
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                all_preds.append(preds.cpu().numpy())
                all_labels.append(batch_y.cpu().numpy())

        val_loss /= len(val_loader.dataset)

        # 计算验证指标
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        # 计算每个标签的精确率、召回率、F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="micro", zero_division=0
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_precision"].append(precision)

        # 学习率调度
        if scheduler:
            scheduler.step(val_loss)

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Micro Precision: {precision:.4f}"
            )

    # 加载最佳模型
    model.load_state_dict(torch.load("best_model.pth"))

    return model, history


# ==================== 5. 评估函数 ====================
def evaluate_model(model, test_loader, label_names, device="cpu", top_k=5):
    """
    评估模型，输出详细的分类指标
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_y.numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)

    # 整体指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="micro", zero_division=0
    )

    # 每个标签的指标
    per_label_precision, per_label_recall, per_label_f1, _ = (
        precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
    )

    print("=" * 50)
    print("模型评估结果")
    print("=" * 50)
    print(f"Micro Precision: {precision:.4f}")
    print(f"Micro Recall: {recall:.4f}")
    print(f"Micro F1-score: {f1:.4f}")

    # 显示表现最好的10个关键词
    print("\n表现最好的10个关键词:")
    best_indices = np.argsort(per_label_f1)[-10:][::-1]
    for idx in best_indices:
        if per_label_f1[idx] > 0:
            print(
                f"  {label_names[idx]:15s} | "
                f"Precision: {per_label_precision[idx]:.4f} | "
                f"Recall: {per_label_recall[idx]:.4f} | "
                f"F1: {per_label_f1[idx]:.4f}"
            )

    # 示例预测
    print("\n示例预测（前5张测试图片）:")
    for i in range(min(5, len(test_loader.dataset))):
        true_labels = [label_names[j] for j in np.where(all_labels[i] == 1)[0]]

        # 获取Top-K预测
        probs_i = all_probs[i]
        top_indices = np.argsort(probs_i)[-top_k:][::-1]
        pred_keywords = [
            (label_names[idx], probs_i[idx])
            for idx in top_indices
            if probs_i[idx] > 0.3
        ]

        print(f"\n图片 {i + 1}:")
        print(f"  真实关键词: {true_labels[:5]}")
        print(f"  预测关键词: {[kw[0] for kw in pred_keywords[:5]]}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_label_f1": per_label_f1,
        "predictions": all_preds,
        "probabilities": all_probs,
    }


# ==================== 6. 主程序 ====================
def main():
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)

    # 设备选择
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 加载数据
    X, y, label_names = load_corel5k("core15k.arff")

    # 划分训练集、验证集、测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y.sum(axis=1) > 0
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.176,
        random_state=42,  # 0.15 * 0.176 ≈ 0.15
    )

    print(f"\n数据划分:")
    print(f"  训练集: {X_train.shape[0]} 张")
    print(f"  验证集: {X_val.shape[0]} 张")
    print(f"  测试集: {X_test.shape[0]} 张")

    # 特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 创建数据加载器
    batch_size = 64
    train_dataset = Corel5kDataset(X_train, y_train)
    val_dataset = Corel5kDataset(X_val, y_val)
    test_dataset = Corel5kDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    model = ImageKeywordModel(
        input_dim=499, hidden_dims=[512, 256, 128], output_dim=y.shape[1], dropout=0.3
    )
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()  # 多标签分类的标准损失
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # 训练模型
    print("\n开始训练...")
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=50,
        device=device,
        patience=10,
    )

    # 评估模型
    results = evaluate_model(model, test_loader, label_names, device=device)

    # 保存模型和预处理器
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler": scaler,
            "label_names": label_names,
            "input_dim": 499,
            "output_dim": y.shape[1],
        },
        "corel5k_model_complete.pth",
    )

    print("\n模型已保存至 corel5k_model_complete.pth")

    return model, scaler, label_names, results


if __name__ == "__main__":
    model, scaler, label_names, results = main()
