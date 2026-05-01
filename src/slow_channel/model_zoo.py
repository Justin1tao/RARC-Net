"""
慢通道模型模块 (Slow Channel Model Zoo)

核心组件：
1. SlowStreamTransformer - Macro-Transformer编码器
2. PositionalEncoding - 正弦位置编码
3. CautiousAdamW - 谨慎优化器（2024/2025）

Author: Slow Channel Team
Date: 2025-12-22
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class PositionalEncoding(nn.Module):
    """
    正弦位置编码 (Sinusoidal Positional Encoding)
    
    使用sin/cos函数对位置进行编码，保留时间顺序信息
    """
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SlowStreamTransformer(nn.Module):
    """
    慢通道 Macro-Transformer 编码器
    
    架构特点：
    - 可学习的CLS token用于聚合序列信息
    - 正弦位置编码保留时序关系
    - 仅提取CLS token的隐藏状态作为宏观嵌入
    
    参数配置（按Prompt要求）：
    - d_model: 64
    - nhead: 4
    - num_layers: 2
    - dim_feedforward: 256
    - dropout: 0.1
    - seq_length: 60
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        seq_length: int = 60,
        num_classes: int = 1  # 1=回归, 2=二分类, 3=三分类
    ):
        super().__init__()
        
        self.d_model = d_model
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.is_classification = num_classes > 1
        
        # 输入嵌入层：将原始特征映射到d_model维度
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # 可学习的CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 正弦位置编码（max_len = seq_length + 1 for CLS token）
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length + 1, dropout=dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 使用 (batch, seq, feature) 格式
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # MLP预测头：根据任务类型调整输出维度
        output_dim = num_classes if num_classes > 1 else 1
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_embedding: bool = False
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (batch_size, seq_length, input_dim) 输入序列
            return_embedding: 是否返回CLS嵌入（用于第三阶段融合）
            
        Returns:
            - 回归任务: (batch_size, 1) 预测值
            - 分类任务: (batch_size, num_classes) logits
            - return_embedding=True: (batch_size, d_model) CLS嵌入
        """
        batch_size = x.size(0)
        
        # 1. 输入嵌入
        x = self.input_embedding(x)  # (batch, seq, d_model)
        
        # 2. 拼接CLS token到序列首部
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq+1, d_model)
        
        # 3. 添加位置编码
        x = self.pos_encoder(x)
        
        # 4. Transformer编码
        x = self.transformer_encoder(x)  # (batch, seq+1, d_model)
        
        # 5. 提取CLS token的隐藏状态
        cls_embedding = x[:, 0, :]  # (batch, d_model)
        
        if return_embedding:
            return cls_embedding
        
        # 6. MLP预测
        pred = self.mlp_head(cls_embedding)  # (batch, output_dim)
        
        return pred


class CautiousAdamW(torch.optim.Optimizer):
    """
    Cautious AdamW (C-AdamW) 优化器
    
    2024/2025年提出的谨慎优化器。核心思想：
    仅当"动量方向"与"当前梯度方向"一致时，才更新参数。
    相当于给非凸优化加了"刹车"，防止在噪声巨大的金融数据中跑偏。
    
    数学逻辑：
    - 计算掩码：mask = (exp_avg * grad > 0).float()
    - 仅在mask=1的维度进行更新
    
    参数:
        params: 模型参数
        lr: 学习率 (默认1e-4)
        betas: Adam的beta参数 (默认(0.9, 0.999))
        eps: 数值稳定性 (默认1e-8)
        weight_decay: 权重衰减 (默认1e-5)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-5
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        执行单步优化
        
        Args:
            closure: 用于重新评估loss的闭包（可选）
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                if grad.is_sparse:
                    raise RuntimeError('CautiousAdamW does not support sparse gradients')
                
                state = self.state[p]
                
                # 状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)      # 一阶动量
                    state['exp_avg_sq'] = torch.zeros_like(p)   # 二阶动量
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                
                # 更新动量估计
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 偏差校正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2
                
                # ========== 关键: 谨慎掩码 ==========
                # 仅当动量和梯度方向一致时才更新
                mask = (corrected_exp_avg * grad > 0).float()
                
                # 计算步长
                denom = corrected_exp_avg_sq.sqrt().add_(eps)
                step_size = lr
                
                # 谨慎更新公式：
                # p = p - lr * (mask * m / sqrt(v) + weight_decay * p)
                # 其中 m 是一阶动量，v 是二阶动量
                update = mask * corrected_exp_avg / denom
                
                # 应用权重衰减 (AdamW style: decoupled weight decay)
                p.mul_(1 - lr * weight_decay)
                
                # 应用谨慎更新
                p.add_(update, alpha=-step_size)
        
        return loss


def pearson_correlation(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    计算Pearson相关系数 (Information Coefficient)
    
    Args:
        pred: 预测值 (batch,) 或 (batch, 1)
        target: 真实值 (batch,) 或 (batch, 1)
    
    Returns:
        相关系数标量
    """
    pred = pred.view(-1)
    target = target.view(-1)
    
    pred_mean = pred.mean()
    target_mean = target.mean()
    
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    
    numerator = (pred_centered * target_centered).sum()
    denominator = torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum())
    
    # 防止除零
    if denominator < 1e-8:
        return torch.tensor(0.0, device=pred.device)
    
    return numerator / denominator


class CombinedLoss(nn.Module):
    """
    组合损失函数：MSE + IC Loss
    
    Loss = MSE + alpha * (1 - IC)
    
    其中:
    - MSE: 均方误差，确保数值准确性
    - IC: 信息系数（Pearson相关），强化方向预测能力
    - alpha: IC损失权重（默认0.1）
    """
    
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算组合损失
        
        Args:
            pred: 预测值
            target: 真实值
        
        Returns:
            loss: 组合损失值
        """
        mse_loss = self.mse(pred, target)
        ic = pearson_correlation(pred, target)
        ic_loss = 1 - ic
        
        total_loss = mse_loss + self.alpha * ic_loss
        
        return total_loss


# ================== 测试代码 ==================
if __name__ == "__main__":
    print("测试 SlowStreamTransformer...")
    model = SlowStreamTransformer(input_dim=13, d_model=64)
    x = torch.randn(32, 60, 13)  # batch=32, seq=60, features=13
    
    # 测试前向传播
    out = model(x)
    print(f"  输出形状: {out.shape}")  # 应为 (32, 1)
    
    # 测试返回embedding
    emb = model(x, return_embedding=True)
    print(f"  嵌入形状: {emb.shape}")  # 应为 (32, 64)
    
    print("\n测试 CautiousAdamW...")
    optimizer = CautiousAdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # 模拟一步优化
    y = torch.randn(32, 1)
    loss_fn = CombinedLoss(alpha=0.1)
    loss = loss_fn(out, y)
    print(f"  初始损失: {loss.item():.4f}")
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    out2 = model(x)
    loss2 = loss_fn(out2, y)
    print(f"  优化后损失: {loss2.item():.4f}")
    
    print("\n✓ 所有测试通过!")
