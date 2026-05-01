"""
快通道模型训练与Optuna超参数优化
GPU优化版: RTX 4090 最大化利用
包含: Bi-LSTM, Bi-RNN, LightGBM, ExtraTrees + Stacking集成
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import joblib
import os
import gc
from typing import Dict, Tuple

from fast_channel_experiment import (
    BiLSTM, BiGRU, TimeSeriesDataset, ModelTrainer
)

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
OUTPUT_DIR = os.environ.get(
    'RARC_FAST_OUTPUT_DIR',
    os.path.join(_project_root, 'results', 'fast_channel')
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== GPU优化配置 ====================

# DataLoader并行加载配置
NUM_WORKERS = 4  # RTX 4090可以处理更多并行数据加载
PREFETCH_FACTOR = 2  # 预取批次数

def setup_device():
    """设备检测 + GPU优化设置"""
    if torch.cuda.is_available():
        print(f"✅ 使用设备: GPU ({torch.cuda.get_device_name(0)})")
        # GPU优化设置
        torch.backends.cudnn.benchmark = True  # 自动寻找最快算法
        torch.backends.cudnn.deterministic = False  # 允许非确定性算法以提速
        torch.backends.cuda.matmul.allow_tf32 = True  # 允许TF32加速矩阵运算
        torch.backends.cudnn.allow_tf32 = True  # 允许cudnn使用TF32
        print("  ✓ cudnn.benchmark = True")
        print("  ✓ TF32 enabled for faster computation")
        return torch.device("cuda")
    else:
        print("⚠️ 未检测到GPU，使用CPU")
        return torch.device("cpu")

# 延迟初始化
DEVICE = None


# ==================== Optuna 优化目标函数 ====================

def objective_bilstm(trial, X_train, y_train, X_val, y_val, input_dim, output_dim):
    """Bi-LSTM 超参数优化"""
    # 超参数搜索空间（保守调整，避免过度限制）
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128])  # 256→128 (保守)
    num_layers = trial.suggest_int('num_layers', 1, 2)
    dropout = trial.suggest_float('dropout', 0.1, 0.3)  # 保持原范围，让Optuna自己找最优值
    lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])  # RTX 4090可用更大batch
    
    # 创建数据加载器 (优化 GPU 利用率)
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    
    # num_workers=0 在某些环境下最稳妥，为了避免多进程问题先设为0，
    # 但开启 pin_memory=True 加速 CPU->GPU 传输
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory='cuda' in str(DEVICE),
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory='cuda' in str(DEVICE),
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    # 创建模型
    model = BiLSTM(input_dim, hidden_dim, num_layers, output_dim, dropout)
    
    # 训练
    trainer = ModelTrainer(device=DEVICE)
    model = trainer.train_pytorch_model(model, train_loader, val_loader, epochs=50, lr=lr)
    
    # 验证集评估
    model.eval()
    with torch.no_grad():
        y_pred = []
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(DEVICE)
            pred = model(X_batch).cpu().numpy()
            y_pred.append(pred)
        y_pred = np.vstack(y_pred)
    
    mse = mean_squared_error(y_val, y_pred)
    
    # 优化目标：IC（信息系数）而非MSE
    y_val_flat = y_val[:, 0] if y_val.ndim > 1 else y_val
    y_pred_flat = y_pred[:, 0] if y_pred.ndim > 1 else y_pred
    ic = np.corrcoef(y_val_flat, y_pred_flat)[0, 1]
    if np.isnan(ic):
        ic = 0
    # === 关键修复：主动资源清理，防止连续Trial导致OOM或死锁 ===
    del model, trainer, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return -ic  # 返回负IC（Optuna最小化 → 最大化IC）


def objective_bigru(trial, X_train, y_train, X_val, y_val, input_dim, output_dim):
    """Bi-RNN/GRU 超参数优化"""
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128])  # 仅降低上限
    num_layers = trial.suggest_int('num_layers', 1, 2)
    dropout = trial.suggest_float('dropout', 0.1, 0.2)
    lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])  # RTX 4090可用更大batch
    
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory='cuda' in str(DEVICE),
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory='cuda' in str(DEVICE),
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    model = BiGRU(input_dim, hidden_dim, num_layers, output_dim, dropout)
    trainer = ModelTrainer(device=DEVICE)
    model = trainer.train_pytorch_model(model, train_loader, val_loader, epochs=50, lr=lr)
    
    model.eval()
    with torch.no_grad():
        y_pred = []
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(DEVICE)
            pred = model(X_batch).cpu().numpy()
            y_pred.append(pred)
        y_pred = np.vstack(y_pred)
    
    mse = mean_squared_error(y_val, y_pred)
    
    # 优化目标：IC
    y_val_flat = y_val[:, 0] if y_val.ndim > 1 else y_val
    y_pred_flat = y_pred[:, 0] if y_pred.ndim > 1 else y_pred
    ic = np.corrcoef(y_val_flat, y_pred_flat)[0, 1]
    if np.isnan(ic):
        ic = 0
    return -ic


def objective_lgb(trial, X_train, y_train, X_val, y_val):
    """LightGBM 超参数优化"""
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'verbosity': -1,
        'num_leaves': trial.suggest_categorical('num_leaves', [31, 63, 127]),
        'max_depth': trial.suggest_categorical('max_depth', [5, 10, -1]),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 500]),
        'subsample': trial.suggest_float('subsample', 0.8, 1.0),
        'random_state': 42
    }
    
    # LightGBM需要2D输入，转换为DataFrame以提供特征名
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_val_2d = X_val.reshape(X_val.shape[0], -1)
    
    # 创建特征名
    feature_names = [f'feature_{i}' for i in range(X_train_2d.shape[1])]
    X_train_df = pd.DataFrame(X_train_2d, columns=feature_names)
    X_val_df = pd.DataFrame(X_val_2d, columns=feature_names)
    
    # 多目标回归：每个目标训练一个模型
    n_targets = y_train.shape[1]
    y_pred_all = np.zeros_like(y_val)
    
    for target_idx in range(n_targets):
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train_df, y_train[:, target_idx])
        y_pred_all[:, target_idx] = model.predict(X_val_df)
    
    # 优化目标：IC
    y_val_flat = y_val[:, 0] if y_val.ndim > 1 else y_val
    y_pred_flat = y_pred_all[:, 0] if y_pred_all.ndim > 1 else y_pred_all
    ic = np.corrcoef(y_val_flat, y_pred_flat)[0, 1]
    if np.isnan(ic):
        ic = 0
    return -ic


def objective_extratrees(trial, X_train, y_train, X_val, y_val):
    """ExtraTrees 超参数优化"""
    params = {
        'n_estimators': trial.suggest_categorical('n_estimators', [100, 200]),
        'max_depth': trial.suggest_categorical('max_depth', [10, 20, None]),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 5),
        'random_state': 42,
        'n_jobs': -1
    }
    
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_val_2d = X_val.reshape(X_val.shape[0], -1)
    
    # 创建特征名
    feature_names = [f'feature_{i}' for i in range(X_train_2d.shape[1])]
    X_train_df = pd.DataFrame(X_train_2d, columns=feature_names)
    X_val_df = pd.DataFrame(X_val_2d, columns=feature_names)
    
    mse_total = 0
    n_targets = y_train.shape[1]
    
    for target_idx in range(n_targets):
        model = ExtraTreesRegressor(**params)
        model.fit(X_train_df, y_train[:, target_idx])
        y_pred = model.predict(X_val_df)
        mse_total += mean_squared_error(y_val[:, target_idx], y_pred)
    
    return mse_total / n_targets


# ==================== Stacking 集成策略 ====================

class StackingEnsemble:
    """
    Stacking集成：
    Level 1: Bi-LSTM (时序) + LightGBM (结构化)
    Level 2: LightGBM Stacker
    
    改进：使用 TimeSeriesSplit 生成 Out-of-Fold (OOF) 预测作为元特征，
          严格防止数据泄漏和过拟合。
    """
    
    def __init__(self, lstm_params, lgb_params, input_dim, output_dim, n_splits=5):
        self.lstm_params = lstm_params
        self.lgb_params = lgb_params
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_splits = n_splits
        
        # 最终的全量模型（用于预测测试集）
        self.final_lstm = None
        self.final_lgb_models = []
        self.stacker_models = []
    
    def _train_lstm(self, X_train, y_train, X_val, y_val, device):
        """辅助函数：训练单个LSTM模型"""
        model = BiLSTM(
            self.input_dim, 
            self.lstm_params['hidden_dim'],
            self.lstm_params['num_layers'],
            self.output_dim,
            self.lstm_params['dropout']
        )
        
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.lstm_params['batch_size'], 
            shuffle=False,
            pin_memory='cuda' in str(device)
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.lstm_params['batch_size'], 
            shuffle=False,
            pin_memory='cuda' in str(device)
        )
        
        trainer = ModelTrainer(device=device)
        model = trainer.train_pytorch_model(
            model, train_loader, val_loader, epochs=50, lr=self.lstm_params['lr']
        )
        return model

    def _predict_lstm(self, model, X, device):
        """辅助函数：LSTM预测"""
        model.eval()
        model.to(device)
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            # 处理大批量数据，分批预测以防OOM
            batch_size = 1024
            preds = []
            for i in range(0, len(X), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                pred = model(batch_X).cpu().numpy()
                preds.append(pred)
            return np.vstack(preds)

    def generate_oof_features(self, X, y, device='cpu'):
        """
        生成训练集的OOF元特征 (Out-of-Fold)
        使用 TimeSeriesSplit 切分
        注意：使用布尔掩码正确追踪有OOF预测的样本
        """
        print(f"  生成OOF元特征 (n_splits={self.n_splits})...")
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        # 初始OOF矩阵
        oof_lstm = np.zeros((len(X), self.output_dim))
        oof_lgb = np.zeros((len(X), self.output_dim))
        
        # 使用布尔掩码追踪有效样本
        valid_mask = np.zeros(len(X), dtype=bool)
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            # --- LSTM OOF ---
            lstm_fold = self._train_lstm(X_tr, y_tr, X_val, y_val, device)
            pred_fold_lstm = self._predict_lstm(lstm_fold, X_val, device)
            oof_lstm[val_idx] = pred_fold_lstm
            
            # --- LightGBM OOF ---
            X_tr_2d = X_tr.reshape(X_tr.shape[0], -1)
            X_val_2d = X_val.reshape(X_val.shape[0], -1)
            
            # 创建特征名以消除警告
            feature_names = [f'feature_{j}' for j in range(X_tr_2d.shape[1])]
            X_tr_df = pd.DataFrame(X_tr_2d, columns=feature_names)
            X_val_df = pd.DataFrame(X_val_2d, columns=feature_names)
            
            for i in range(self.output_dim):
                lgb_fold = lgb.LGBMRegressor(**self.lgb_params, random_state=42, verbosity=-1)
                lgb_fold.fit(X_tr_df, y_tr[:, i])
                oof_lgb[val_idx, i] = lgb_fold.predict(X_val_df)
            
            # 标记有OOF预测的样本
            valid_mask[val_idx] = True
            print(f"    Fold {fold+1}/{self.n_splits} 完成")
        
        # 返回所有有预测的样本
        valid_oof_lstm = oof_lstm[valid_mask]
        valid_oof_lgb = oof_lgb[valid_mask]
        valid_y = y[valid_mask]
        
        return np.hstack([valid_oof_lstm, valid_oof_lgb]), valid_y

    def fit_final_base_models(self, X_train, y_train, X_val, y_val, device='cpu'):
        """训练全量Base模型（用于后续预测测试集）"""
        print("  训练全量Base模型...")
        # LSTM
        self.final_lstm = self._train_lstm(X_train, y_train, X_val, y_val, device)
        
        # LightGBM (每个目标一个)
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        # 创建特征名
        feature_names = [f'feature_{j}' for j in range(X_train_2d.shape[1])]
        X_train_df = pd.DataFrame(X_train_2d, columns=feature_names)
        
        self.final_lgb_models = []
        for i in range(self.output_dim):
            model = lgb.LGBMRegressor(**self.lgb_params, random_state=42, verbosity=-1)
            model.fit(X_train_df, y_train[:, i])
            self.final_lgb_models.append(model)
            
    def generate_meta_features(self, X, device='cpu'):
        """生成测试集/验证集的元特征（使用全量Base模型）"""
        # LSTM预测
        lstm_pred = self._predict_lstm(self.final_lstm, X, device)
        
        # LightGBM预测
        X_2d = X.reshape(X.shape[0], -1)
        # 创建特征名
        feature_names = [f'feature_{j}' for j in range(X_2d.shape[1])]
        X_df = pd.DataFrame(X_2d, columns=feature_names)
        
        lgb_preds = []
        for model in self.final_lgb_models:
            pred = model.predict(X_df).reshape(-1, 1)
            lgb_preds.append(pred)
        lgb_pred = np.hstack(lgb_preds)
        
        return np.hstack([lstm_pred, lgb_pred])

    def train_stacker(self, X_train, y_train, X_val, y_val, device='cpu'):
        """训练Stacker（Level 2）"""
        print("  训练Stacker...")
        
        # 1. 生成训练集 OOF 元特征
        # 注意：这里返回的 y_train_oof 是截断后的，与 meta_train 长度一致
        meta_train, y_train_oof = self.generate_oof_features(X_train, y_train, device)
        
        # 2. 训练全量 Base 模型 (用于生成验证集和测试集的元特征)
        self.fit_final_base_models(X_train, y_train, X_val, y_val, device)
        
        # 3. 生成验证集元特征
        meta_val = self.generate_meta_features(X_val, device)
        
        # 4. 训练 Stacker
        self.stacker_models = []
        for target_idx in range(self.output_dim):
            stacker = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
                verbosity=-1
            )
            # 使用 OOF 特征训练 Stacker
            stacker.fit(meta_train, y_train_oof[:, target_idx])
            self.stacker_models.append(stacker)
        
        # 验证集评估
        stacker_preds = []
        for stacker in self.stacker_models:
            pred = stacker.predict(meta_val).reshape(-1, 1)
            stacker_preds.append(pred)
        stacker_pred = np.hstack(stacker_preds)
        
        mse = mean_squared_error(y_val, stacker_pred)
        print(f"  Stacker验证集MSE: {mse:.6f}")
        
        return stacker_pred
    
    def predict(self, X, device='cpu'):
        """预测"""
        meta = self.generate_meta_features(X, device)
        preds = []
        for stacker in self.stacker_models:
            pred = stacker.predict(meta).reshape(-1, 1)
            preds.append(pred)
        return np.hstack(preds)


# ==================== 自适应加权融合 ====================

def adaptive_weighting(lstm_pred, stacker_pred, y_val):
    """
    基于验证集的自适应加权
    使用网格搜索找最优权重，而非简单MSE倒数
    F_fast = alpha * LSTM + beta * Stacker
    """
    best_mse = float('inf')
    best_alpha = 0.5
    
    # 网格搜索最优权重
    for alpha in np.arange(0.0, 1.01, 0.05):
        beta = 1.0 - alpha
        fusion = alpha * lstm_pred + beta * stacker_pred
        mse = mean_squared_error(y_val, fusion)
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
    
    best_beta = 1.0 - best_alpha
    
    print(f"  自适应权重 (网格搜索): LSTM={best_alpha:.4f}, Stacker={best_beta:.4f}")
    
    # 融合预测
    final_pred = best_alpha * lstm_pred + best_beta * stacker_pred
    
    return final_pred, best_alpha, best_beta


# ==================== 主训练流程 ====================

def train_all_models(X_train, y_train, X_val, y_val, X_test, y_test, 
                     exp_name, window_size, n_trials=50):
    """
    训练所有模型并进行超参数优化
    """
    # 初始化设备（与 finbert 一致）
    global DEVICE
    DEVICE = setup_device()

    input_dim = X_train.shape[2]  # 特征数
    output_dim = y_train.shape[1]  # 目标数
    
    results = {}
    
    print(f"\n{'='*80}")
    print(f"实验: {exp_name} | 窗口: {window_size}天")
    print(f"{'='*80}")
    print(f"数据维度: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"目标数量: {output_dim}, 特征数量: {input_dim}")
    
    # ========== 1. Bi-LSTM ==========
    print(f"\n{'-'*60}")
    print("1. Bi-LSTM 超参数优化 (Optuna)")
    print(f"{'-'*60}")
    
    study_lstm = optuna.create_study(direction='minimize', study_name='BiLSTM')
    study_lstm.optimize(
        lambda trial: objective_bilstm(trial, X_train, y_train, X_val, y_val, input_dim, output_dim),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    best_params_lstm = study_lstm.best_params
    print(f"✓ 最佳超参数: {best_params_lstm}")
    
    # 使用最佳参数重新训练
    best_lstm = BiLSTM(
        input_dim, 
        best_params_lstm['hidden_dim'],
        best_params_lstm['num_layers'],
        output_dim,
        best_params_lstm['dropout']
    )
    
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=best_params_lstm['batch_size'], 
        shuffle=False,
        pin_memory='cuda' in str(DEVICE)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=best_params_lstm['batch_size'], 
        shuffle=False,
        pin_memory='cuda' in str(DEVICE)
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=64, 
        shuffle=False,
        pin_memory='cuda' in str(DEVICE)
    )
    
    trainer = ModelTrainer(device=DEVICE)
    best_lstm = trainer.train_pytorch_model(
        best_lstm, train_loader, val_loader, epochs=100, lr=best_params_lstm['lr']
    )
    
    # 测试集预测
    best_lstm.eval()
    with torch.no_grad():
        lstm_test_pred = []
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(DEVICE)
            pred = best_lstm(X_batch).cpu().numpy()
            lstm_test_pred.append(pred)
        lstm_test_pred = np.vstack(lstm_test_pred)
    
    results['BiLSTM'] = {
        'model': best_lstm,
        'params': best_params_lstm,
        'test_pred': lstm_test_pred
    }
    
    # ========== 2. Bi-GRU ==========
    print(f"\n{'-'*60}")
    print("2. Bi-GRU 超参数优化 (Optuna)")
    print(f"{'-'*60}")
    
    study_gru = optuna.create_study(direction='minimize', study_name='BiGRU')
    study_gru.optimize(
        lambda trial: objective_bigru(trial, X_train, y_train, X_val, y_val, input_dim, output_dim),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    best_params_gru = study_gru.best_params
    print(f"✓ 最佳超参数: {best_params_gru}")
    
    best_gru = BiGRU(
        input_dim,
        best_params_gru['hidden_dim'],
        best_params_gru['num_layers'],
        output_dim,
        best_params_gru['dropout']
    )
    
    train_loader_gru = DataLoader(
        train_dataset, 
        batch_size=best_params_gru['batch_size'], 
        shuffle=False,
        pin_memory='cuda' in str(DEVICE)
    )
    val_loader_gru = DataLoader(
        val_dataset, 
        batch_size=best_params_gru['batch_size'], 
        shuffle=False,
        pin_memory='cuda' in str(DEVICE)
    )
    
    best_gru = trainer.train_pytorch_model(
        best_gru, train_loader_gru, val_loader_gru, epochs=100, lr=best_params_gru['lr']
    )
    
    best_gru.eval()
    with torch.no_grad():
        gru_test_pred = []
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(DEVICE)
            pred = best_gru(X_batch).cpu().numpy()
            gru_test_pred.append(pred)
        gru_test_pred = np.vstack(gru_test_pred)
        
    results['BiGRU'] = {
        'model': best_gru,
        'params': best_params_gru,
        'test_pred': gru_test_pred
    }

    # ========== 3. LightGBM ==========
    print(f"\n{'-'*60}")
    print("3. LightGBM 超参数优化")
    print(f"{'-'*60}")
    
    study_lgb = optuna.create_study(direction='minimize', study_name='LightGBM')
    study_lgb.optimize(
        lambda trial: objective_lgb(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    best_params_lgb = study_lgb.best_params
    print(f"✓ 最佳超参数: {best_params_lgb}")
    
    # 训练多个LightGBM模型（每个目标一个）
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_test_2d = X_test.reshape(X_test.shape[0], -1)
    
    # 创建特征名
    feature_names = [f'feature_{i}' for i in range(X_train_2d.shape[1])]
    X_train_df = pd.DataFrame(X_train_2d, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_2d, columns=feature_names)
    
    lgb_models = []
    lgb_test_preds = []
    
    for target_idx in range(output_dim):
        lgb_model = lgb.LGBMRegressor(**best_params_lgb, random_state=42, verbosity=-1)
        lgb_model.fit(X_train_df, y_train[:, target_idx])
        lgb_models.append(lgb_model)
        
        test_pred = lgb_model.predict(X_test_df).reshape(-1, 1)
        lgb_test_preds.append(test_pred)
    
    lgb_test_pred = np.hstack(lgb_test_preds)
    
    results['LightGBM'] = {
        'models': lgb_models,
        'params': best_params_lgb,
        'test_pred': lgb_test_pred
    }
    
    # ========== 4. ExtraTrees ==========
    print(f"\n{'-'*60}")
    print("4. ExtraTrees 超参数优化")
    print(f"{'-'*60}")
    
    study_et = optuna.create_study(direction='minimize', study_name='ExtraTrees')
    study_et.optimize(
        lambda trial: objective_extratrees(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    best_params_et = study_et.best_params
    print(f"✓ 最佳超参数: {best_params_et}")
    
    et_models = []
    et_test_preds = []
    
    # X_train_df 和 X_test_df 已在 LightGBM 部分创建，直接使用
    for target_idx in range(output_dim):
        et_model = ExtraTreesRegressor(**best_params_et, random_state=42, n_jobs=-1)
        et_model.fit(X_train_df, y_train[:, target_idx])
        et_models.append(et_model)
        
        test_pred = et_model.predict(X_test_df).reshape(-1, 1)
        et_test_preds.append(test_pred)
        
    et_test_pred = np.hstack(et_test_preds)
    
    results['ExtraTrees'] = {
        'models': et_models,
        'params': best_params_et,
        'test_pred': et_test_pred
    }
    
    # ========== 5. Stacking集成 ==========
    print(f"\n{'-'*60}")
    print("5. Stacking 集成 (LSTM + LightGBM)")
    print(f"{'-'*60}")
    
    # 注意：现在传入参数字典，而不是模型对象
    stacking = StackingEnsemble(best_params_lstm, best_params_lgb, input_dim, output_dim, n_splits=5)
    stacker_val_pred = stacking.train_stacker(X_train, y_train, X_val, y_val, device=DEVICE)
    stacker_test_pred = stacking.predict(X_test, device=DEVICE)
    
    results['Stacking'] = {
        'model': stacking,
        'test_pred': stacker_test_pred
    }
    
    # ========== 6. 自适应加权融合 ==========
    print(f"\n{'-'*60}")
    # print("6. 自适应加权融合")  # 已移除Fusion
    # print(f"{'-'*60}")
    # 
    # # Fusion代码已注释
    # best_lstm.eval()
    # with torch.no_grad():
    #     lstm_val_pred = []
    #     for X_batch, _ in val_loader:
    #         X_batch = X_batch.to(DEVICE)
    #         pred = best_lstm(X_batch).cpu().numpy()
    #         lstm_val_pred.append(pred)
    #     lstm_val_pred = np.vstack(lstm_val_pred)
    # 
    # _, alpha, beta = adaptive_weighting(lstm_val_pred, stacker_val_pred, y_val)
    # final_test_pred = alpha * lstm_test_pred + beta * stacker_test_pred
    # 
    # results['Fusion'] = {
    #     'test_pred': final_test_pred,
    #     'alpha': alpha,
    #     'beta': beta
    # }
    
    # 保存结果
    save_path = os.path.join(OUTPUT_DIR, f'{exp_name}_window{window_size}_models.pkl')
    joblib.dump(results, save_path)
    print(f"\n✓ 模型已保存至: {save_path}")
    
    return results


if __name__ == '__main__':
    print("快通道模型训练模块已加载")
    print("请运行 fast_channel_main.py 启动完整实验流程")


# ==================== 固定超参数训练（方案3） ====================

def train_with_fixed_params(X_train, y_train, X_val, y_val, X_test, y_test,
                            exp_name, window_size, fixed_params):
    """
    使用固定超参数训练所有模型（用于公平对比ESG效果）
    
    Args:
        fixed_params: 从对照组A获取的最优超参数字典
            {
                'BiLSTM': {'hidden_dim': 128, 'num_layers': 2, ...},
                'BiGRU': {...},
                'LightGBM': {...},
                'ExtraTrees': {...}
            }
    """
    global DEVICE
    DEVICE = setup_device()
    
    input_dim = X_train.shape[2]
    output_dim = y_train.shape[1]
    
    results = {}
    
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    trainer = ModelTrainer(DEVICE)
    
    # ========== 1. Bi-LSTM (固定参数) ==========
    if 'BiLSTM' in fixed_params:
        print(f"\n{'-'*60}")
        print("1. Bi-LSTM (使用固定参数)")
        print(f"{'-'*60}")
        
        params = fixed_params['BiLSTM']
        print(f"  固定参数: {params}")
        
        best_lstm = BiLSTM(
            input_dim,
            params['hidden_dim'],
            params['num_layers'],
            output_dim,
            params['dropout']
        )
        
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False, pin_memory='cuda' in str(DEVICE))
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, pin_memory='cuda' in str(DEVICE))
        test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, pin_memory='cuda' in str(DEVICE))
        
        best_lstm = trainer.train_pytorch_model(best_lstm, train_loader, val_loader, epochs=100, lr=params['lr'])
        
        best_lstm.eval()
        with torch.no_grad():
            lstm_test_pred = []
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(DEVICE)
                pred = best_lstm(X_batch).cpu().numpy()
                lstm_test_pred.append(pred)
            lstm_test_pred = np.vstack(lstm_test_pred)
        
        results['BiLSTM'] = {'model': best_lstm, 'params': params, 'test_pred': lstm_test_pred}
        
        # 清理显存
        del best_lstm
        gc.collect()
        torch.cuda.empty_cache()
    
    # ========== 2. Bi-GRU (固定参数) ==========
    if 'BiGRU' in fixed_params:
        print(f"\n{'-'*60}")
        print("2. Bi-GRU (使用固定参数)")
        print(f"{'-'*60}")
        
        params = fixed_params['BiGRU']
        print(f"  固定参数: {params}")
        
        best_gru = BiGRU(
            input_dim,
            params['hidden_dim'],
            params['num_layers'],
            output_dim,
            params['dropout']
        )
        
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False, pin_memory='cuda' in str(DEVICE))
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, pin_memory='cuda' in str(DEVICE))
        test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, pin_memory='cuda' in str(DEVICE))
        
        best_gru = trainer.train_pytorch_model(best_gru, train_loader, val_loader, epochs=100, lr=params['lr'])
        
        best_gru.eval()
        with torch.no_grad():
            gru_test_pred = []
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(DEVICE)
                pred = best_gru(X_batch).cpu().numpy()
                gru_test_pred.append(pred)
            gru_test_pred = np.vstack(gru_test_pred)
        
        results['BiGRU'] = {'model': best_gru, 'params': params, 'test_pred': gru_test_pred}
        
        del best_gru
        gc.collect()
        torch.cuda.empty_cache()
    
    # ========== 3. LightGBM (固定参数) ==========
    if 'LightGBM' in fixed_params:
        print(f"\n{'-'*60}")
        print("3. LightGBM (使用固定参数)")
        print(f"{'-'*60}")
        
        params = fixed_params['LightGBM']
        print(f"  固定参数: {params}")
        
        X_train_flat = X_train[:, -1, :]
        X_val_flat = X_val[:, -1, :]
        X_test_flat = X_test[:, -1, :]
        
        lgb_model = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)
        lgb_model.fit(X_train_flat, y_train.ravel(), eval_set=[(X_val_flat, y_val.ravel())])
        
        lgb_test_pred = lgb_model.predict(X_test_flat).reshape(-1, output_dim)
        results['LightGBM'] = {'model': lgb_model, 'params': params, 'test_pred': lgb_test_pred}
    
    # ========== 4. ExtraTrees (固定参数) ==========
    if 'ExtraTrees' in fixed_params:
        print(f"\n{'-'*60}")
        print("4. ExtraTrees (使用固定参数)")
        print(f"{'-'*60}")
        
        params = fixed_params['ExtraTrees']
        print(f"  固定参数: {params}")
        
        X_train_flat = X_train[:, -1, :]
        X_test_flat = X_test[:, -1, :]
        
        et_model = ExtraTreesRegressor(**params, random_state=42, n_jobs=-1)
        et_model.fit(X_train_flat, y_train.ravel())
        
        et_test_pred = et_model.predict(X_test_flat).reshape(-1, output_dim)
        results['ExtraTrees'] = {'model': et_model, 'params': params, 'test_pred': et_test_pred}
    
    # ========== 5. Stacking ==========
    if 'BiLSTM' in results and 'LightGBM' in results:
        print(f"\n{'-'*60}")
        print("5. Stacking (基于固定参数模型)")
        print(f"{'-'*60}")
        
        lstm_val_pred = results['BiLSTM']['test_pred'][:len(y_val)] if 'BiLSTM' in results else np.zeros((len(y_val), 1))
        lgb_val_pred = results['LightGBM']['test_pred'][:len(y_val)] if 'LightGBM' in results else np.zeros((len(y_val), 1))
        
        # 简化：直接使用平均
        stacker_test_pred = (results['BiLSTM']['test_pred'] + results['LightGBM']['test_pred']) / 2
        results['Stacking'] = {'test_pred': stacker_test_pred}
    
    # 保存结果
    save_path = os.path.join(OUTPUT_DIR, f'{exp_name}_window{window_size}_fixed_models.pkl')
    joblib.dump(results, save_path)
    print(f"\n✓ 固定参数模型已保存至: {save_path}")
    
    return results
