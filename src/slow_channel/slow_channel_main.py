"""
慢通道主训练脚本 (Slow Channel Main)

核心功能：
1. 完整训练流程（含早停）
2. CosineAnnealingLR调度器
3. 评估指标：Best R², IC, Sharpe Ratio
4. 输出：预测文件 + 模型权重

防泄漏保证：
- 所有数据shift处理在data_factory中完成
- 时序划分无shuffle
- 严格分离训练/验证/测试集

Author: Slow Channel Team
Date: 2025-12-22
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_factory import SlowChannelDataFactory
from model_zoo import SlowStreamTransformer, CautiousAdamW, CombinedLoss, pearson_correlation


# ==================== GPU优化配置 (参考fast_channel) ====================

def setup_device():
    """
    设备检测 + GPU优化设置
    与fast_channel保持一致的GPU优化策略
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ 使用设备: GPU ({torch.cuda.get_device_name(0)})")
        
        # GPU优化设置
        torch.backends.cudnn.benchmark = True  # 自动寻找最快卷积算法
        torch.backends.cudnn.deterministic = False  # 允许非确定性算法以提速
        
        # TF32加速（RTX30系及以上GPU支持）
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        
        print("  ✓ cudnn.benchmark = True")
        print("  ✓ TF32 enabled for faster computation")
        
        # 打印GPU内存信息
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  ✓ GPU显存: {total_mem:.1f} GB")
        
        return device
    else:
        print("⚠️ 未检测到GPU，使用CPU训练")
        print("  建议：在服务器上使用GPU可加速10-50倍")
        return torch.device("cpu")


class EarlyStopping:
    """早停机制：监控验证集损失，patience轮无改善则停止"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, val_loss: float, model: nn.Module):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
    
    def load_best_model(self, model: nn.Module):
        """加载最佳模型权重"""
        model.load_state_dict(self.best_model_state)


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    """
    计算年化夏普比率
    
    Args:
        returns: 日收益率序列
        risk_free_rate: 无风险利率（年化）
        periods_per_year: 每年交易日数
    
    Returns:
        年化夏普比率
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    # 日度无风险利率
    daily_rf = risk_free_rate / periods_per_year
    
    # 超额收益
    excess_returns = returns - daily_rf
    
    # 年化夏普比率
    sharpe = np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)
    
    return sharpe


def calculate_ic(pred: np.ndarray, actual: np.ndarray) -> float:
    """计算信息系数（Pearson相关）"""
    if len(pred) == 0:
        return 0.0
    
    pred = pred.flatten()
    actual = actual.flatten()
    
    corr = np.corrcoef(pred, actual)[0, 1]
    return corr if not np.isnan(corr) else 0.0


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer,
    criterion,
    device: torch.device
) -> float:
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion,
    device: torch.device
) -> float:
    """验证"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def predict(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> np.ndarray:
    """生成预测"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for X_batch, _ in data_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            predictions.append(pred.cpu().numpy())
    
    return np.concatenate(predictions, axis=0)


def main():
    """主训练流程"""
    
    # ==================== 超参数配置 ====================
    config = {
        # 数据配置
        'seq_length': 60,           # 输入序列长度（建议30-90天）
        # 修正：与Fusion目标一致，预测5日收益率而非波动率
        'predict_horizon': 5,        # 预测窗口: 5日收益率（与fusion一致）
        'task_type': 'regression',   # 任务类型: 改为回归以匹配fusion目标
        'ternary_threshold': 0.005,  # 三分类阈值（备用）
        
        # 模型配置
        'd_model': 64,              # 模型维度
        'nhead': 4,                 # 注意力头数
        'num_layers': 2,            # Transformer层数
        'dim_feedforward': 256,     # FFN维度
        'dropout': 0.2,             # Dropout率（增加正则化应对小样本）
        
        # 训练配置
        'batch_size': 32,           # 批大小
        'epochs': 100,              # 最大训练轮数
        'lr': 1e-4,                 # 学习率
        'weight_decay': 1e-5,       # 权重衰减
        'patience': 15,             # 早停耐心值（给模型更多学习机会）
        'warmup_epochs': 10,        # Warmup预热轮数（前N轮线性增加lr）
        'ic_loss_alpha': 0.1,       # IC损失权重（仅回归任务）
        
        # 数据划分
        'train_ratio': 0.7,         # 训练集比例
        'val_ratio': 0.15,          # 验证集比例
    }
    
    print("=" * 70)
    print("慢通道 Transformer 训练 (Slow Channel Training)")
    print("=" * 70)
    print("\n超参数配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # ==================== 设备配置 (GPU优化) ====================
    device = setup_device()
    use_gpu = device.type == 'cuda'
    
    # ==================== 数据准备 ====================
    print("\n" + "=" * 70)
    print("数据准备阶段")
    print("=" * 70)
    
    # 获取base_dir（智能检测数据文件位置）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    project_root = os.path.dirname(parent_dir)
    
    # 优先检查父目录，如果父目录没有sp500文件，则使用当前目录
    if os.path.exists(os.path.join(project_root, 'data', 'processed', 'sp500_with_indicators.csv')):
        base_dir = project_root
        print(f"  数据目录: {project_root} (项目根目录)")
    elif os.path.exists(os.path.join(parent_dir, 'sp500_with_indicators.csv')):
        base_dir = parent_dir
        print(f"  数据目录: {parent_dir} (父目录)")
    else:
        base_dir = script_dir
        print(f"  数据目录: {script_dir} (当前目录)")
    
    factory = SlowChannelDataFactory(
        base_dir=base_dir, 
        seq_length=config['seq_length'],
        predict_horizon=config['predict_horizon'],
        task_type=config['task_type'],
        ternary_threshold=config['ternary_threshold']
    )
    
    X_train, y_train, X_val, y_val, X_test, y_test = factory.prepare_sequences(
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio']
    )
    
    # 转换为PyTorch张量
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    
    # 创建DataLoader (GPU优化: pin_memory加速CPU->GPU传输)
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,  # 时序数据不shuffle
        pin_memory=use_gpu  # GPU优化
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        pin_memory=use_gpu
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        pin_memory=use_gpu
    )
    
    input_dim = X_train.shape[2]
    print(f"\n输入特征维度: {input_dim}")
    
    # ==================== 模型初始化 ====================
    print("\n" + "=" * 70)
    print("模型初始化")
    print("=" * 70)
    
    # 根据任务类型确定num_classes
    task_type = config['task_type']
    if task_type == 'binary':
        num_classes = 2
    elif task_type == 'ternary':
        num_classes = 3
    else:
        num_classes = 1  # 回归任务（包括regression和volatility）
    
    model = SlowStreamTransformer(
        input_dim=input_dim,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        seq_length=config['seq_length'],
        num_classes=num_classes
    ).to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})")
    print(f"任务类型: {task_type}, 输出维度: {num_classes}")
    
    # 优化器：Cautious AdamW
    optimizer = CautiousAdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # ==================== Warmup + CosineAnnealing 调度器 ====================
    # Warmup机制：前warmup_epochs个epoch线性增加学习率，之后使用CosineAnnealing
    warmup_epochs = config.get('warmup_epochs', 10)  # 默认10个epoch warmup
    
    # 使用PyTorch的LambdaLR实现Warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # 线性warmup: lr从0.1*base_lr增加到base_lr
            return 0.1 + 0.9 * (epoch / warmup_epochs)
        else:
            # Warmup后使用Cosine退火
            progress = (epoch - warmup_epochs) / (config['epochs'] - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    # 使用LambdaLR实现Warmup + Cosine组合调度
    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    print(f"调度器: Warmup({warmup_epochs} epochs) + CosineAnnealing")
    
    # 损失函数：根据任务类型选择
    if task_type in ['binary', 'ternary']:
        # 分类任务用CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()
        print(f"损失函数: CrossEntropyLoss")
    else:
        # 回归任务用MSE+IC组合损失
        criterion = CombinedLoss(alpha=config['ic_loss_alpha'])
        print(f"损失函数: MSE + {config['ic_loss_alpha']} * (1 - IC)")
    
    # 早停机制
    early_stopping = EarlyStopping(patience=config['patience'], verbose=True)
    
    print(f"优化器: CautiousAdamW (lr={config['lr']}, weight_decay={config['weight_decay']})")
    
    # ==================== 训练循环 ====================
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(1, config['epochs'] + 1):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        
        # 学习率更新
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 打印进度
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{config['epochs']} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {current_lr:.2e}")
        
        # 早停检查
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"\n⚠️ 早停触发于 Epoch {epoch}")
            break
    
    # 加载最佳模型
    early_stopping.load_best_model(model)
    print(f"\n✓ 已加载最佳模型 (Val Loss: {early_stopping.best_loss:.6f})")
    
    # ==================== 测试评估 ====================
    print("\n" + "=" * 70)
    print("测试集评估")
    print("=" * 70)
    
    # 生成预测
    y_pred_scaled = predict(model, test_loader, device)
    
    # 逆归一化
    y_pred = factory.inverse_transform_y(y_pred_scaled)
    y_true = factory.inverse_transform_y(y_test)
    
    # 计算指标
    # 1. R² (决定系数)
    r2 = r2_score(y_true, y_pred)
    
    # 2. IC (信息系数)
    ic = calculate_ic(y_pred, y_true)
    
    # 3. Sharpe Ratio (基于预测方向的简单策略)
    # 策略：预测为正则持有，预测为负则空仓
    strategy_returns = np.sign(y_pred) * y_true
    sharpe = calculate_sharpe_ratio(strategy_returns)
    
    # 4. 方向准确率 (Win Rate)
    direction_correct = (np.sign(y_pred) == np.sign(y_true)).sum()
    win_rate = direction_correct / len(y_true)
    
    # 5. MSE & MAE
    mse = np.mean((y_pred - y_true) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))
    
    print(f"\n{'=' * 40}")
    print("测试集指标:")
    print(f"{'=' * 40}")
    print(f"  Best R²:        {r2:.6f}")
    print(f"  IC:             {ic:.6f}")
    print(f"  Sharpe Ratio:   {sharpe:.4f}")
    print(f"  Win Rate:       {win_rate:.2%}")
    print(f"  MSE:            {mse:.8f}")
    print(f"  MAE:            {mae:.8f}")
    print(f"{'=' * 40}")
    
    # ==================== 保存结果 ====================
    print("\n" + "=" * 70)
    print("保存结果")
    print("=" * 70)
    
    output_dir = script_dir
    
    # 保存预测结果
    test_dates = factory.get_test_dates()
    predictions_df = pd.DataFrame({
        'date': test_dates,
        'pred_log_return': y_pred,
        'actual_log_return': y_true
    })
    predictions_path = os.path.join(output_dir, 'f_slow_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"✓ 预测结果保存到: {predictions_path}")
    
    # 保存模型权重
    model_path = os.path.join(output_dir, 'transformer_slow.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'metrics': {
            'r2': r2,
            'ic': ic,
            'sharpe': sharpe,
            'win_rate': win_rate
        }
    }, model_path)
    print(f"✓ 模型权重保存到: {model_path}")
    
    # 保存训练历史
    history_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    history_path = os.path.join(output_dir, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f"✓ 训练历史保存到: {history_path}")
    
    print("\n" + "=" * 70)
    print("✓ 慢通道训练完成!")
    print("=" * 70)
    
    # ==================== 导出Embedding (Phase 3: FiLM融合用) ====================
    print("\n" + "=" * 70)
    print("【额外步骤】导出宏观状态向量 (Macro-State Embeddings)")
    print("=" * 70)
    
    # 准备全部数据
    X_all = factory.prepare_all_data()
    all_dates = factory.get_all_dates()
    
    # 推理模式提取embedding
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        # 批量推理
        batch_size = 64
        for i in range(0, len(X_all), batch_size):
            batch_x = torch.FloatTensor(X_all[i:i+batch_size]).to(device)
            # 使用return_embedding=True获取CLS向量
            emb = model(batch_x, return_embedding=True)
            all_embeddings.append(emb.cpu().numpy())
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    
    # 保存embedding
    embeddings_path = os.path.join(output_dir, 'slow_channel_embeddings.npy')
    np.save(embeddings_path, embeddings)
    print(f"✓ Embeddings保存到: {embeddings_path}")
    print(f"  形状: {embeddings.shape}")  # (N, d_model=64)
    
    # 保存日期索引（确保对齐）
    dates_df = pd.DataFrame({'date': all_dates})
    dates_path = os.path.join(output_dir, 'slow_channel_dates.csv')
    dates_df.to_csv(dates_path, index=False)
    print(f"✓ 日期索引保存到: {dates_path}")
    print(f"  数量: {len(all_dates)} 行")
    
    # 验证对齐
    assert len(embeddings) == len(all_dates), f"对齐错误: {len(embeddings)} vs {len(all_dates)}"
    print("✓ Embedding与日期索引对齐验证通过!")
    
    print("\n📦 输出文件汇总:")
    print(f"  1. f_slow_predictions.csv - 测试集预测结果")
    print(f"  2. transformer_slow.pth - 模型权重")
    print(f"  3. slow_channel_embeddings.npy - 宏观状态向量 (用于FiLM融合)")
    print(f"  4. slow_channel_dates.csv - 日期索引")
    
    return {
        'r2': r2,
        'ic': ic,
        'sharpe': sharpe,
        'win_rate': win_rate
    }


if __name__ == "__main__":
    metrics = main()
