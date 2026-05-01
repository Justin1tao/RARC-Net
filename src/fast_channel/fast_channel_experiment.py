
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler  # 改用RobustScaler，对异常值更稳健
from typing import Dict, List, Tuple
import math

# ==================== C-AdamW 优化器 (Cautious AdamW) ====================

class CAdamW(torch.optim.Optimizer):
    """
    C-AdamW: Cautious AdamW 优化器
    论文: "Cautious Optimizers: Improving Training with One Line of Code"
    
    核心思想: 当梯度更新方向与动量方向不一致时，减少更新幅度
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(CAdamW, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('CAdamW does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Decoupled weight decay
                p.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = group['lr'] / bias_correction1
                
                # Compute denominator
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                # Compute update direction
                update = exp_avg / denom
                
                # ========== Cautious Mask (核心改进) ==========
                # 当梯度与动量方向一致时保留更新，否则减少更新
                mask = (update * grad > 0).to(grad.dtype)
                mask = mask / mask.mean().clamp_(min=1e-3)
                update = update * mask
                # ===============================================
                
                p.add_(update, alpha=-step_size)
        
        return loss


# ==================== 1. 数据预处理 ====================

# 训练配置
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

class DataPreprocessor:
    """数据预处理类 - 负责所有的数据清洗、特征工程和标准化"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.scalers = {}  # 存储归一化器
        
    def load_data(self) -> pd.DataFrame:
        """加载原始数据并合并"""
        print("加载数据...")
        
        # 读取主数据
        sp500 = pd.read_csv(os.path.join(self.data_dir, 'sp500_with_indicators.csv'))
        
        # 清理列名中的空格（避免 'change ' vs 'change' 的问题）
        sp500.columns = sp500.columns.str.strip()
        
        sp500['date'] = pd.to_datetime(sp500['date'])
        
        # 读取情绪指数 (如果存在)
        emotion_path = os.path.join(self.data_dir, 'esg_emotion_index.csv')
        if os.path.exists(emotion_path):
            emotion = pd.read_csv(emotion_path)
            # 清理并统一列名为小写
            emotion.columns = emotion.columns.str.strip().str.lower()
            emotion['date'] = pd.to_datetime(emotion['date'])
            # 重命名列以匹配
            if 'esg_sentiment_index' in emotion.columns:
                emotion = emotion.rename(columns={'esg_sentiment_index': 'esg_sentiment'})
            
            # 合并数据 - 同时保留 data_source 列用于后续加权
            merge_cols = ['date', 'esg_sentiment']
            if 'data_source' in emotion.columns:
                merge_cols.append('data_source')
            df = pd.merge(sp500, emotion[merge_cols], on='date', how='left')
            
            # 填充缺失的情绪数据
            df['esg_sentiment'] = df['esg_sentiment'].ffill().fillna(0)
            if 'data_source' in df.columns:
                df['data_source'] = df['data_source'].fillna('Forward_Fill')
            else:
                df['data_source'] = 'Unknown'
            print("✓ 已合并ESG情绪指数（含数据来源标记）")
        else:
            df = sp500
            df['esg_sentiment'] = 0
            df['data_source'] = 'Unknown'
            print("⚠️ 未找到情绪指数文件，仅使用技术指标")
            
        # 按日期排序
        df = df.sort_values('date').reset_index(drop=True)
        return df
    
    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建预测目标：标准化次日收益率
        用历史波动率标准化，让IC等指标更好
        """
        print("创建预测目标...")
        
        df = df.copy()
        
        # 1. 原始次日对数收益率
        df['next_return_raw'] = np.log(df['close'].shift(-1) / df['close'])
        
        # 2. 历史波动率（用于标准化）
        rolling_std = df['next_return_raw'].rolling(20, min_periods=5).std().shift(1)
        rolling_std = rolling_std.fillna(0.01)
        
        # 保存缩放因子（用于MSE还原）
        self.target_scale = rolling_std.mean()  # 保存平均缩放因子
        
        # 3. 标准化目标
        df['next_return'] = df['next_return_raw'] / (rolling_std + 1e-8)
        df['next_return'] = df['next_return'].clip(-3, 3)
        
        # 删除无效行
        df = df.dropna(subset=['next_return']).reset_index(drop=True)
        
        print(f"✓ 目标变量创建完成：标准化收益率，剩余 {len(df)} 条有效记录")
        print(f"  标准化范围: [{df['next_return'].min():.4f}, {df['next_return'].max():.4f}]")
        print(f"  缩放因子: {self.target_scale:.6f}")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, include_esg: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        准备特征列表 - 重点使用收益率相关特征
        核心改进：用历史收益率序列预测未来收益率
        """
        df = df.copy()
        
        feature_cols = []
        
        # ========== 1. 核心特征：历史对数收益率 ==========
        # 计算当天的对数收益率（这是预测的关键输入！）
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['log_return'] = df['log_return'].fillna(0)
        feature_cols.append('log_return')
        
        # 收益率的滞后特征（自回归结构）
        for lag in [1, 2, 3, 5]:
            df[f'return_lag{lag}'] = df['log_return'].shift(lag).fillna(0)
            feature_cols.append(f'return_lag{lag}')
        
        # ========== 2. 波动率特征 ==========
        # 历史波动率（关键！）
        df['volatility_5d'] = df['log_return'].rolling(window=5, min_periods=1).std().fillna(0)
        df['volatility_10d'] = df['log_return'].rolling(window=10, min_periods=1).std().fillna(0)
        df['volatility_20d'] = df['log_return'].rolling(window=20, min_periods=1).std().fillna(0)
        feature_cols.extend(['volatility_5d', 'volatility_10d', 'volatility_20d'])
        
        # 波动率变化率
        df['vol_change'] = (df['volatility_5d'] / df['volatility_20d'].replace(0, 1e-8)) - 1
        feature_cols.append('vol_change')
        
        # ========== 3. 动量特征 ==========
        # 累积收益率
        df['return_5d'] = df['log_return'].rolling(window=5, min_periods=1).sum()
        df['return_10d'] = df['log_return'].rolling(window=10, min_periods=1).sum()
        df['return_20d'] = df['log_return'].rolling(window=20, min_periods=1).sum()
        feature_cols.extend(['return_5d', 'return_10d', 'return_20d'])
        
        # 动量（当天收益相对于均值的偏离）
        df['return_ma5'] = df['log_return'].rolling(window=5, min_periods=1).mean()
        df['momentum'] = df['log_return'] - df['return_ma5']
        feature_cols.extend(['return_ma5', 'momentum'])
        
        # ========== 4. 技术指标（已经是平稳的）==========
        tech_features = ['ADX', 'RSI', 'MACD_hist']  # 只保留关键的
        for col in tech_features:
            if col in df.columns:
                feature_cols.append(col)
        
        # ========== 5. 日内特征（相对值）==========
        # 日内振幅（相对）
        df['intraday_range'] = (df['high'] - df['low']) / df['open']
        # 开盘缺口
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['gap'] = df['gap'].fillna(0)
        feature_cols.extend(['intraday_range', 'gap'])
        
        # ========== 6. ESG情绪特征（增强版）==========
        if include_esg:
            # 原始ESG列
            if 'esg_sentiment' in df.columns:
                esg_col = 'esg_sentiment'
            elif 'ESG_Sentiment_Index' in df.columns:
                esg_col = 'ESG_Sentiment_Index'
                df['esg_sentiment'] = df[esg_col]
            else:
                include_esg = False
                print("  ⚠️ 未找到ESG列，跳过ESG特征")
        
        if include_esg:
            # 6.1 数据源质量权重（ESG专题新闻更有价值）
            # 注意：Simulated是我们生成的2024模拟数据，质量经过验证，赋予1.2权重
            weight_map = {'ESG': 1.5, 'All_News': 1.0, 'Forward_Fill': 0.3, 'Simulated': 1.2}
            df['esg_weight'] = df['data_source'].map(weight_map).fillna(0.5)
            
            # 6.2 基础加权ESG (核心特征)
            df['esg_weighted'] = df['esg_sentiment'] * df['esg_weight']
            feature_cols.append('esg_weighted')
            
            # 6.3 ESG滞后特征 (只保留lag1，最相关)
            df['esg_lag1'] = df['esg_sentiment'].shift(1).fillna(0)
            feature_cols.append('esg_lag1')
            
            # 6.4 ESG滚动均值 (平滑趋势，降噪)
            df['esg_ma5'] = df['esg_sentiment'].rolling(5, min_periods=1).mean()
            feature_cols.append('esg_ma5')
            
            # 注意：移除了 esg_change 和 esg_zscore
            # 这些衍生特征在 Stacking/BiLSTM 上引入了噪声
            # 简化后的特征更稳定，对所有模型都有正向贡献
            
            source_counts = df['data_source'].value_counts()
            print(f"  ESG数据来源分布: {dict(source_counts)}")
            print(f"  ESG特征(精简版): esg_weighted, esg_lag1, esg_ma5")
        
        # ========== 7. 收益率Z-score（时序标准化）==========
        return_std_20 = df['log_return'].rolling(20, min_periods=1).std().fillna(0.01)
        return_mean_20 = df['log_return'].rolling(20, min_periods=1).mean()
        df['return_zscore'] = ((df['log_return'] - return_mean_20) / (return_std_20 + 1e-8)).clip(-3, 3)
        feature_cols.append('return_zscore')
        
        # 删除含有NaN的行
        df = df.dropna(subset=feature_cols).reset_index(drop=True)
        
        print(f"✓ 特征准备完成: {len(feature_cols)} 个特征 (ESG={'包含' if include_esg else '不包含'})")
        print(f"  核心特征: log_return + 滞后 + 波动率 + 动量")
        print(f"  特征列表: {feature_cols}")
        
        return df, feature_cols
    
    def split_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """按时间顺序划分训练集、验证集、测试集 (60/20/20)"""
        n = len(df)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
        
        splits = {
            'train': df.iloc[:train_end].reset_index(drop=True),
            'val': df.iloc[train_end:val_end].reset_index(drop=True),
            'test': df.iloc[val_end:].reset_index(drop=True)
        }
        
        print(f"数据集划分: 训练={len(splits['train'])}, 验证={len(splits['val'])}, 测试={len(splits['test'])}")
        
        return splits
    
    def normalize_features(self, splits: Dict[str, pd.DataFrame], feature_cols: List[str]) -> Dict[str, pd.DataFrame]:
        """归一化特征（仅在训练集上fit）- 使用RobustScaler对异常值更稳健"""
        print("归一化特征 (RobustScaler)...")
        
        # 使用RobustScaler，基于中位数和四分位距，对异常值更鲁棒
        self.scalers['features'] = RobustScaler()
        splits['train'][feature_cols] = self.scalers['features'].fit_transform(
            splits['train'][feature_cols]
        )
        
        # 在验证集和测试集上transform
        splits['val'][feature_cols] = self.scalers['features'].transform(
            splits['val'][feature_cols]
        )
        splits['test'][feature_cols] = self.scalers['features'].transform(
            splits['test'][feature_cols]
        )
        
        print("✓ 特征归一化完成 (RobustScaler)")
        
        return splits


class SequenceGenerator:
    """滑动窗口序列生成器"""
    
    @staticmethod
    def create_sequences(df: pd.DataFrame, feature_cols: List[str], 
                         target_cols: List[str], window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建滑动窗口序列
        X: (n_samples, window_size, n_features)
        y: (n_samples, n_targets)
        """
        X, y = [], []
        
        for i in range(window_size, len(df)):
            # 取过去 window_size 天的特征
            X.append(df[feature_cols].iloc[i-window_size:i].values)
            # 取当前时刻的目标
            y.append(df[target_cols].iloc[i].values)
        
        return np.array(X), np.array(y)


# ==================== 2. PyTorch 数据集 ====================

class TimeSeriesDataset(Dataset):
    """时间序列数据集"""
    
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==================== 3. 深度学习模型 ====================

class BiLSTM(nn.Module):
    """双向LSTM模型 - 增强LayerNorm正则化"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 output_dim: int, dropout: float = 0.2):
        super(BiLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # 双向LSTM输出维度是 hidden_dim * 2
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)  # 层归一化，稳定训练
        self.dropout1 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        # x: (batch, window, features)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        out = self.layer_norm(out)  # 层归一化
        out = self.dropout1(out)
        out = self.fc(out)
        return out


class BiGRU(nn.Module):
    """双向GRU模型 - 增强LayerNorm正则化"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 output_dim: int, dropout: float = 0.2):
        super(BiGRU, self).__init__()
        
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)  # 层归一化
        self.dropout1 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = gru_out[:, -1, :]
        out = self.layer_norm(out)  # 层归一化
        out = self.dropout1(out)
        out = self.fc(out)
        return out


# ==================== 4. 训练器 ====================

class ModelTrainer:
    """模型训练器 - GPU优化版：混合精度 + cudnn优化"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        # GPU优化设置
        if 'cuda' in str(device):
            torch.backends.cudnn.benchmark = True  # 自动寻找最快的卷积算法
            torch.backends.cudnn.deterministic = False  # 允许非确定性算法以提升速度
    
    def train_pytorch_model(self, model: nn.Module, train_loader: DataLoader, 
                           val_loader: DataLoader, epochs: int, lr: float) -> nn.Module:
        """训练PyTorch模型 - GPU优化版"""
        print(f"  > 开始训练 (Device: {self.device}, AMP: {'On' if 'cuda' in str(self.device) else 'Off'})")
        
        model = model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = CAdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 混合精度训练 (仅GPU)
        use_amp = 'cuda' in str(self.device)
        scaler = torch.amp.GradScaler('cuda') if use_amp else None
        
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        model.train()
        for epoch in range(epochs):
            train_loss = 0
            for X_batch, y_batch in train_loader:
                # 非阻塞传输加速
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
                
                # 混合精度前向传播
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                    
                    # 混合精度反向传播
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                train_loss += loss.item()
            
            # 验证
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device, non_blocking=True)
                    y_batch = y_batch.to(self.device, non_blocking=True)
                    
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            outputs = model(X_batch)
                            loss = criterion(outputs, y_batch)
                    else:
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                    
                    val_loss += loss.item()
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            
            model.train()
        
        if 'best_model_state' in locals():
            model.load_state_dict(best_model_state)
            
        return model

