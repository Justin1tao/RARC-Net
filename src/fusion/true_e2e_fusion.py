"""
RARC-Net: Regime-Adaptive Residual Correction Network
======================================================
实现宏观感知的双流融合系统：
- 快通道: Bi-GRU (10天窗口 + ESG情绪)
- 慢通道: Transformer (60天窗口 + 深度特征工程)
- 融合层: HyperNetwork + GRN (动态权重生成 + 门控残差)

核心公式: Prediction = Fast_Baseline + HyperNetwork(Macro) * f_fusion

Author: Fusion Team
Date: 2025-12-31
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import pearsonr
from typing import Dict, List, Tuple, Optional
import joblib  # 用于加载 .pkl 模型文件

# ========== 🔑 修复 joblib 模块加载问题 ==========
# 添加 fast_channel 目录到模块搜索路径
# 这样 joblib 反序列化时能找到 fast_channel_experiment 模块
_current_file = os.path.abspath(__file__)
_fusion_dir = os.path.dirname(_current_file)
_project_root = os.path.dirname(_fusion_dir)
_fast_channel_path = os.path.join(_project_root, 'fast_channel')

if _fast_channel_path not in sys.path:
    sys.path.insert(0, _fast_channel_path)
    print(f"[INFO] 已添加模块路径: {_fast_channel_path}")
# ==================================================
import warnings
warnings.filterwarnings('ignore')

# 导入slow_channel组件
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

# 动态寻找 slow_channel
def import_slow_channel_modules():
    global SlowStreamTransformer, CautiousAdamW, pearson_correlation, SlowChannelDataFactory
    
    # 尝试1: 标准 slow_channel 包
    try:
        from slow_channel.model_zoo import SlowStreamTransformer, CautiousAdamW, pearson_correlation
        from slow_channel.data_factory import SlowChannelDataFactory
        print("✓ 成功导入 slow_channel 组件 (标准路径)")
        return
    except ImportError:
        pass
        
    # 尝试2: 直接导入
    possible_paths = [
        os.path.join(parent_dir, 'slow_channel')
    ]
    
    for p in possible_paths:
        if os.path.exists(p):
            sys.path.insert(0, p)
            try:
                from model_zoo import SlowStreamTransformer, CautiousAdamW, pearson_correlation
                from data_factory import SlowChannelDataFactory
                print(f"✓ 成功导入 slow_channel 组件 (从 {os.path.basename(p)} 直接导入)")
                return
            except ImportError:
                sys.path.pop(0)
                continue
    
    raise ImportError("无法找到 slow_channel 模块，请确认目录结构")

try:
    import_slow_channel_modules()
except ImportError as e:
    print(f"⚠️ 导入 Slow Channel 失败: {e}")
    sys.exit(1)

# 导入增强评估组件
try:
    from e2e_fusion_enhanced import (
        calculate_comprehensive_metrics,
        EventWindowAnalyzer,
        EnhancedXAIVisualizer
    )
    print("✓ 成功导入增强评估组件")
except ImportError as e:
    print(f"⚠️ 导入增强组件失败: {e}")
    print("  → 将跳过增强可视化功能")
    calculate_comprehensive_metrics = None
    EventWindowAnalyzer = None
    EnhancedXAIVisualizer = None


# ==================== 1. 宏观特征工程器 ====================
class MacroFeatureEngineer:
    """
    高级宏观特征工程：构建交互特征、多尺度变换
    
    设计理念：
    - 不仅输入原始宏观因子，更要构建它们之间的交互关系
    - 同时输入Level（水平）和Change（变化），模拟人类分析师思维
    """
    
    def __init__(self):
        self.feature_names = []
    
    def engineer(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        """
        对宏观数据进行深度特征工程
        
        Args:
            macro_df: 原始宏观数据 (包含 VIX, CPI, EPU, PRI, TRI等)
        
        Returns:
            工程化后的宏观数据框 (必定包含date列)
        """
        df = macro_df.copy()
        
        # !! 关键: 先保存date列 !!
        if 'date' not in df.columns:
            raise ValueError("MacroFeatureEngineer: 输入数据必须包含date列！")
        
        date_column = df['date'].copy()
        new_features = []
        
        # 1. 交互特征 (Interaction Features)
        if 'VIX' in df.columns and 'PRI' in df.columns:
            df['Climate_Sensitivity'] = df['VIX'] * df['PRI']  # 恐慌×气候风险
            new_features.append('Climate_Sensitivity')
        
        if 'CPI' in df.columns and 'EPU' in df.columns:
            cpi_change = df['CPI'].pct_change().fillna(0)
            df['Policy_Stress'] = cpi_change * df['EPU']  # 通胀冲击×政策不确定性
            new_features.append('Policy_Stress')
        
        # 2. 多尺度变换 (Multi-Scale Features)
        scale_cols = ['VIX', 'EPU']
        for col in scale_cols:
            if col in df.columns:
                # Level (绝对水平)
                new_features.append(col)
                
                # Change (变化率)
                df[f'{col}_Change'] = df[col].pct_change().fillna(0)
                new_features.append(f'{col}_Change')
                
                # Volatility (波动率 - 20日滚动标准差)
                df[f'{col}_Vol'] = df[col].rolling(20, min_periods=1).std().fillna(0)
                new_features.append(f'{col}_Vol')
        
        # 3. 保留其他重要特征
        keep_cols = ['PRI', 'TRI', 'Transition_concern', 'Physical_concern', 'volume_ratio', 'log_return']
        for col in keep_cols:
            if col in df.columns and col not in new_features:
                new_features.append(col)
        
        # 4. 构建最终DataFrame: date列 + 特征列
        result_df = pd.DataFrame()
        result_df['date'] = date_column  # 第一列必定是date
        
        for feat in new_features:
            if feat in df.columns:
                result_df[feat] = df[feat]
        
        # 5. 更新feature_names (不包含date)
        self.feature_names = new_features
        
        print(f"  [MacroFeatureEngineer] 输出列: {list(result_df.columns)}")
        
        return result_df


# ==================== 2. 联合数据加载器 ====================
class JointDataLoader:
    """
    日期对齐的联合数据加载器
    
    核心职责：
    - 严格对齐快通道(10天)和慢通道(60/90天)的时间窗口
    - 确保每个样本的微观、宏观、标签在时间轴上一致
    - 防止数据泄漏
    """
    
    def __init__(self, 
                 base_dir: str,
                 fast_window: int = 10,
                 slow_window: int = 60,
                 predict_horizon: int = 1):
        """
        Args:
            base_dir: 数据根目录
            fast_window: 快通道窗口大小 (天)
            slow_window: 慢通道窗口大小 (天)
            predict_horizon: 预测窗口 (天)
        """
        self.base_dir = base_dir
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.predict_horizon = predict_horizon
        
        # 特征工程器
        self.macro_engineer = MacroFeatureEngineer()
        
        print(f"\n{'='*70}")
        print("联合数据加载器初始化")
        print(f"{'='*70}")
        print(f"  快通道窗口: {fast_window}天")
        print(f"  慢通道窗口: {slow_window}天")
        print(f"  预测窗口: {predict_horizon}天")
    
    def load_fast_channel_data(self) -> pd.DataFrame:
        """加载快通道数据 (量价 + 技术指标 + ESG)"""
        # 动态查找数据文件
        def find_data_file(filename):
            parent_dir = os.path.dirname(self.base_dir)
            candidates = [
                os.path.join(self.base_dir, filename),
                os.path.join(self.base_dir, 'data', 'processed', filename),
                os.path.join(self.base_dir, 'data', 'raw', filename),
                os.path.join(self.base_dir, 'data', 'sample', filename),
                os.path.join(self.base_dir, 'teach', filename),
                os.path.join(parent_dir, filename),
                os.path.join(parent_dir, 'data', 'processed', filename),
                os.path.join(parent_dir, 'data', 'raw', filename),
                os.path.join(parent_dir, 'data', 'sample', filename),
                os.path.join(parent_dir, 'teach', filename),
                os.path.join(parent_dir, 'slow_channel', filename), # 标准 slow_channel
            ]
            for path in candidates:
                if os.path.exists(path):
                    print(f"  → Found {filename} at {path}") # Debug info
                    return path
            raise FileNotFoundError(f"无法找到数据文件: {filename}，已尝试所有候选路径")

        # 加载SP500数据
        sp500_path = find_data_file('sp500_with_indicators.csv')
        sp500_df = pd.read_csv(sp500_path)
        sp500_df['date'] = pd.to_datetime(sp500_df['date'].str.replace('/', '-'))
        
        # 加载ESG情绪数据
        esg_path = find_data_file('esg_emotion_index.csv')
        
        # 智能识别日期列 (兼容 date, Date, datetime, Time 等)
        # 先读取头部以跳过潜在的错误行
        try:
             esg_df = pd.read_csv(esg_path)
        except Exception as e:
             if "utf-8" in str(e):
                  esg_df = pd.read_csv(esg_path, encoding='gbk')
             else:
                  raise e
        
        date_col = None
        for col in esg_df.columns:
            if col.lower() in ['date', 'time', 'datetime', 'timestamp']:
                date_col = col
                break
        
        if date_col:
            esg_df.rename(columns={date_col: 'date'}, inplace=True)
        else:
            # 尝试使用第一列作为日期
            print(f"⚠️ 警告: esg_emotion_index.csv 中未找到 'date' 列，尝试使用第一列 {esg_df.columns[0]} 作为日期")
            esg_df.rename(columns={esg_df.columns[0]: 'date'}, inplace=True)
            
        esg_df['date'] = pd.to_datetime(esg_df['date'].astype(str).str.replace('/', '-'))
        
        # 合并
        fast_df = pd.merge(sp500_df, esg_df[['date', 'ESG_Sentiment_Index', 'data_source']], 
                          on='date', how='left')
        
        # 特征工程 (参考fast_channel的实现)
        fast_df = self._engineer_fast_features(fast_df)
        
        print(f"\n[Fast Channel] 加载完成")
        print(f"  样本数: {len(fast_df)}")
        print(f"  日期范围: {fast_df['date'].min()} ~ {fast_df['date'].max()}")
        
        return fast_df
    
    def _engineer_fast_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """快通道特征工程 (基于fast_channel最佳实践)"""
        df = df.copy()
        
        # ========== 1. 对数收益率 ==========
        df['log_return'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        
        # ========== 2. 滞后收益率 ==========
        for lag in [1, 2, 3, 5]:
            df[f'return_lag{lag}'] = df['log_return'].shift(lag).fillna(0)
        
        # ========== 3. 波动率特征 ==========
        for window in [5, 10, 20]:
            df[f'volatility_{window}d'] = df['log_return'].rolling(window, min_periods=1).std().fillna(0)
        
        # 【新增】波动率变化率
        df['vol_change'] = (df['volatility_5d'] / df['volatility_20d'].replace(0, 1e-8)) - 1
        df['vol_change'] = df['vol_change'].fillna(0)
        
        # ========== 4. 动量特征 ==========
        for window in [5, 10, 20]:
            df[f'return_{window}d'] = df['log_return'].rolling(window, min_periods=1).sum()
        
        # 【新增】动量特征（与预训练模型一致）
        df['return_ma5'] = df['log_return'].rolling(window=5, min_periods=1).mean()
        df['momentum'] = df['log_return'] - df['return_ma5']
        df['momentum'] = df['momentum'].fillna(0)
        
        # ========== 5. 日内特征（与预训练模型一致）==========
        # 【新增】日内振幅
        df['intraday_range'] = (df['high'] - df['low']) / df['open'].replace(0, 1e-8)
        df['intraday_range'] = df['intraday_range'].fillna(0)
        
        # 【新增】开盘缺口
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1).replace(0, 1e-8)
        df['gap'] = df['gap'].fillna(0)
        
        # ========== 6. 收益率 Z-score ==========
        return_std_20 = df['log_return'].rolling(20, min_periods=1).std().fillna(0.01)
        return_mean_20 = df['log_return'].rolling(20, min_periods=1).mean()
        df['return_zscore'] = ((df['log_return'] - return_mean_20) / (return_std_20 + 1e-8)).clip(-3, 3)
        df['return_zscore'] = df['return_zscore'].fillna(0)
        
        # ========== 7. ESG加权特征 ==========
        if 'ESG_Sentiment_Index' in df.columns:
            weight_map = {'ESG': 1.5, 'All_News': 1.0, 'Forward_Fill': 0.3, 'Simulated': 1.2}
            df['esg_weight'] = df['data_source'].map(weight_map).fillna(0.5)
            df['esg_weighted'] = df['ESG_Sentiment_Index'] * df['esg_weight']
            df['esg_lag1'] = df['ESG_Sentiment_Index'].shift(1).fillna(0)
            df['esg_ma5'] = df['ESG_Sentiment_Index'].rolling(5, min_periods=1).mean()
        
        # ========== 8. 技术指标（保留原始值，不做 zscore）==========
        # 预训练模型使用原始的 ADX, RSI, MACD_hist
        tech_indicators = ['ADX', 'RSI', 'MACD_hist']
        for col in tech_indicators:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median() if len(df[col].dropna()) > 0 else 0)
        
        # ========== 9. 预测目标 ==========
        df['target'] = df['log_return'].shift(-self.predict_horizon)
        
        # 删除NaN
        df = df.dropna(subset=['target']).reset_index(drop=True)
        
        # ✅ 验证target定义是否正确
        print(f"\n  [Target验证] 检查前3个样本:")
        for i in range(min(3, len(df))):
            print(f"    date={df.iloc[i]['date']}, log_return={df.iloc[i]['log_return']:.6f}, target={df.iloc[i]['target']:.6f}")
            if i+1 < len(df):
                next_return = df.iloc[i+1]['log_return']
                match = "✓" if abs(df.iloc[i]['target'] - next_return) < 1e-6 else "✗ MISMATCH"
                print(f"      → 下一行log_return={next_return:.6f} {match}")
        
        return df
    
    def load_slow_channel_data(self) -> pd.DataFrame:
        """加载慢通道数据 (宏观因子)"""
        # 确定慢通道数据目录
        parent_dir = os.path.dirname(self.base_dir)
        possible_slow_dirs = [
             self.base_dir, # 当前目录
             os.path.join(parent_dir, 'slow_channel'),
             parent_dir
        ]
        
        slow_data_dir = self.base_dir
        for p in possible_slow_dirs:
             if os.path.exists(os.path.join(p, 'sp500_slow.csv')): # 核心文件之一
                  slow_data_dir = p
                  print(f"  → Detected Slow Channel data dir: {slow_data_dir}")
                  break

        # 初始化数据工厂 (此前被误删，现在恢复)
        factory = SlowChannelDataFactory(
            base_dir=slow_data_dir,
            seq_length=self.slow_window,
            predict_horizon=self.predict_horizon
        )

        slow_df = None # 初始化变量以避免 UnboundLocalError

        # 使用 prepare_all_data_with_dates 获取完整数据和日期
        # 这是为了确保百分百拿到日期，绕过dataframe的索引问题
        try:
             # 注意：prepare_all_data_with_dates 返回的是 X_all, y_all, dates
             # 但我们需要的是一个DataFrame来进行特征工程
             # 所以我们先尝试 build_dataset，如果不包含日期，就强制从 dates 注入
             slow_df = factory.build_dataset()
             
             if 'date' not in slow_df.columns and slow_df.index.name != 'date':
                  # 尝试获取日期索引
                  _, _, sc_dates = factory.prepare_all_data_with_dates()
                  # 截取 slow_df 长度 (build_dataset 可能比 sequences 长)
                  # 通常 build_dataset 返回的是原始对齐后的 df
                  # 我们重新从工厂获取所有日期
                  all_dates = factory.get_all_dates()
                  
                  if len(all_dates) == len(slow_df):
                       slow_df['date'] = all_dates
                       print("  ✓ 已从 factory 强制注入 date 列")
                  else:
                       print(f"  ⚠️ Warning: 日期长度 ({len(all_dates)}) 与 DataFrame 长度 ({len(slow_df)}) 不一致")
                       # 尝试直接 reset index，也许索引就是日期
                       slow_df = slow_df.reset_index()
                       slow_df.rename(columns={'index': 'date'}, inplace=True)
                       
             # 再次检查并规范化
             if 'date' not in slow_df.columns:
                  if slow_df.index.name in ['date', 'Date', 'index']:
                       slow_df = slow_df.reset_index()
                       slow_df.rename(columns={slow_df.columns[0]: 'date'}, inplace=True)
                  else:
                       # 最后尝试：假设索引就是日期
                       slow_df = slow_df.reset_index()
                       slow_df.rename(columns={'index': 'date'}, inplace=True)
             
             slow_df['date'] = pd.to_datetime(slow_df['date'])

        except Exception as e:
             print(f"  ❌ 加载慢通道数据异常: {e}")
             # 最后的保底：重新根据 sp500_slow.csv 的日期
             sp500_slow_path = os.path.join(slow_data_dir, 'sp500_slow.csv')
             temp_df = pd.read_csv(sp500_slow_path)
             # 只有当 slow_df 存在时才进行长度比较
             if slow_df is not None and len(temp_df) == len(slow_df):
                  slow_df['date'] = pd.to_datetime(temp_df['date'])
                  print("  ✓ 已使用 sp500_slow.csv 恢复日期")
             else:
                  raise e
        
        # 深度特征工程
        slow_df = self.macro_engineer.engineer(slow_df)
        
        print(f"\n[Slow Channel] 加载完成")
        print(f"  样本数: {len(slow_df)}")
        print(f"  特征数: {len(self.macro_engineer.feature_names)}")
        if 'date' in slow_df.columns:
             print(f"  日期范围: {slow_df['date'].min()} ~ {slow_df['date'].max()}")
        else:
             print("  ⚠️ 警告: 无法在 slow_df 中找到 date 列，后续对齐可能失败")
        
        return slow_df
    
    def align_and_create_sequences(self) -> Tuple:
        """
        核心方法：对齐快慢通道并创建序列
        
        Returns:
            X_fast: (N, fast_window, fast_features) 快通道序列
            X_slow: (N, slow_window, slow_features) 慢通道序列
            y: (N, 1) 预测目标
            dates: (N,) 日期索引
        """
        # 加载数据
        fast_df = self.load_fast_channel_data()
        slow_df = self.load_slow_channel_data()
        
        # 【重要】特征列表必须与 fast_channel_experiment.py 的 prepare_features 完全一致
        # 预训练模型使用 23 个特征，此处也需要使用相同的 23 个
        fast_feature_cols = [
            # 1. 核心特征：历史对数收益率
            'log_return',
            # 收益率滞后特征
            'return_lag1', 'return_lag2', 'return_lag3', 'return_lag5',
            
            # 2. 波动率特征
            'volatility_5d', 'volatility_10d', 'volatility_20d',
            'vol_change',  # 波动率变化率
            
            # 3. 动量特征
            'return_5d', 'return_10d', 'return_20d',
            'return_ma5', 'momentum',  # 添加缺失的动量特征
            
            # 4. 技术指标（使用原始名称，非 zscore 版本）
            'ADX', 'RSI', 'MACD_hist',
            
            # 5. 日内特征
            'intraday_range', 'gap',  # 添加缺失的日内特征
            
            # 6. ESG情绪特征
            'esg_weighted', 'esg_lag1', 'esg_ma5',
            
            # 7. 收益率 Z-score
            'return_zscore',  # 添加缺失的特征
        ]
        fast_feature_cols = [c for c in fast_feature_cols if c in fast_df.columns]
        
        print(f"\n  [Fast Channel] 使用 {len(fast_feature_cols)} 个特征")
        
        # 获取慢通道特征列
        slow_feature_cols = self.macro_engineer.feature_names
        
        # 找到日期交集
        common_dates = sorted(set(fast_df['date']) & set(slow_df['date']))
        
        # 为每个日期构建样本
        X_fast_list, X_slow_list, y_list, date_list = [], [], [], []
        
        for i, current_date in enumerate(common_dates):
            # 确保有足够的历史数据
            if i < max(self.fast_window, self.slow_window):
                continue
            
            # 确保有未来数据作为标签
            if i >= len(common_dates) - self.predict_horizon:
                continue
            
            # 快通道: 过去fast_window天
            fast_window_dates = common_dates[i - self.fast_window:i]
            fast_window_data = fast_df[fast_df['date'].isin(fast_window_dates)]
            
            if len(fast_window_data) != self.fast_window:
                continue
            
            X_fast = fast_window_data[fast_feature_cols].values
            
            # 慢通道: 过去slow_window天
            slow_window_dates = common_dates[i - self.slow_window:i]
            slow_window_data = slow_df[slow_df['date'].isin(slow_window_dates)]
            
            if len(slow_window_data) != self.slow_window:
                continue
            
            X_slow = slow_window_data[slow_feature_cols].values
            
            # 标签: 当前日期的target (已经是未来收益率)
            target_row = fast_df[fast_df['date'] == current_date]
            if len(target_row) == 0:
                continue
            
            y = target_row['target'].values[0]
            
            X_fast_list.append(X_fast)
            X_slow_list.append(X_slow)
            y_list.append(y)
            date_list.append(current_date)
        
        # 转换为numpy数组
        X_fast = np.array(X_fast_list, dtype=np.float32)
        X_slow = np.array(X_slow_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
        dates = np.array(date_list)
        
        print(f"\n{'='*70}")
        print("数据对齐完成")
        print(f"{'='*70}")
        print(f"  有效样本数: {len(X_fast)}")
        print(f"  快通道形状: {X_fast.shape} (样本, {self.fast_window}天, {len(fast_feature_cols)}特征)")
        print(f"  慢通道形状: {X_slow.shape} (样本, {self.slow_window}天, {len(slow_feature_cols)}特征)")
        print(f"  标签形状: {y.shape}")
        print(f"  日期范围: {dates[0]} ~ {dates[-1]}")
        
        return X_fast, X_slow, y, dates


# ==================== 3. 核心融合模型 ====================
class BiGRU_FastChannel(nn.Module):
    """
    快通道 BiGRU 模型
    【关键】架构必须与 fast_channel_experiment.py 的 BiGRU 完全一致
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, 
                 output_dim: int = 1, dropout: float = 0.2):
        super(BiGRU_FastChannel, self).__init__()
        
        # GRU 层（双向）
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 【关键】输出层结构必须与预训练模型完全一致
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)  # 层归一化
        self.dropout1 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            prediction: (batch, 1) 微观预测值
        """
        gru_out, _ = self.gru(x)
        out = gru_out[:, -1, :]  # 取最后时刻
        out = self.layer_norm(out)
        out = self.dropout1(out)  # 修复: dropout → dropout1
        prediction = self.fc(out)
        return prediction
    
    def forward_embedding(self, x):
        """仅返回特征嵌入（用于 RARC-Net 架构）"""
        gru_out, _ = self.gru(x)
        out = gru_out[:, -1, :]
        out = self.layer_norm(out)
        out = self.dropout1(out)
        return out  # [batch, hidden_dim*2]


# ==================== 3.1 门控残差网络 (SOTA组件) ====================
class GatedResidualNetwork(nn.Module):
    """
    门控残差网络 (Gated Residual Network, GRN)
    
    来源: TFT (Temporal Fusion Transformer, Google)
    
    核心机制:
    1. 通过 GLU (Gated Linear Unit) 自适应控制信息流
    2. 如果输入是噪声，门控自动关闭（输出趋近于0）
    3. 残差连接确保梯度流畅
    
    数学公式:
    output = LayerNorm(residual + GLU(ELU(FC(x))))
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 dropout: float = 0.25, use_context: bool = False, context_dim: int = 0):
        super().__init__()
        
        self.use_context = use_context
        
        # 1. 核心非线性变换
        input_size = input_dim + context_dim if use_context else input_dim
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)  # 高 Dropout 防止过拟合
        
        # 2. 门控线性单元 (GLU) - 信息流控制
        self.gate = nn.Linear(hidden_dim, output_dim * 2)  # *2 用于 GLU 切分
        
        # 3. 残差连接投影层（如果维度不匹配）
        self.project_residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
        # 4. 层归一化 - 保证训练稳定
        self.layernorm = nn.LayerNorm(output_dim)
    
    def forward(self, x, context=None):
        """
        Args:
            x: (batch, input_dim) 主输入
            context: (batch, context_dim) 上下文输入（可选）
        Returns:
            output: (batch, output_dim)
        """
        # 残差路径
        residual = self.project_residual(x)
        
        # 主处理路径
        if self.use_context and context is not None:
            x = torch.cat([x, context], dim=-1)
        
        x_enc = F.elu(self.fc1(x))
        x_enc = self.fc2(x_enc)
        x_enc = self.dropout(x_enc)
        
        # GLU: val * sigmoid(gate)
        gate_out = self.gate(x_enc)
        val, gate = gate_out.chunk(2, dim=-1)
        x_enc = val * torch.sigmoid(gate)
        
        # 残差连接 + 归一化
        return self.layernorm(residual + x_enc)


# ==================== 3.1.1 超网络 (HyperNetwork) ====================
class HyperNetwork(nn.Module):
    """
    超网络 (HyperNetwork): 根据宏观状态动态生成 Correction Adapter 的权重
    
    数学原理:
        θ_correction = H(z_slow)
        correction = F.linear(f_fusion, W_dynamic, b_dynamic)
    
    设计理念:
        1. 宏观状态决定了"微观特征如何映射到残差"
        2. 不同市场Regime下，修正策略完全不同
        3. 使用轻量2层MLP防止小样本过拟合
    
    参考文献:
        - HN-MVTS: HyperNetwork-based Multivariate Time Series Forecasting (arXiv:2511.08340)
        - HyperFusion: Multimodal Integration via Hypernetworks (arXiv:2403.13319)
    """
    
    def __init__(self, 
                 slow_dim: int,      # 宏观嵌入维度 (e.g., 64)
                 fast_dim: int,      # 融合特征维度 (e.g., 64) 
                 hidden_dim: int = 32,
                 dropout: float = 0.3):
        super().__init__()
        
        self.slow_dim = slow_dim
        self.fast_dim = fast_dim
        
        # 需要生成的权重数量: W (fast_dim x 1) + b (1)
        self.weight_size = fast_dim * 1  # W: [fast_dim, 1]
        self.bias_size = 1               # b: [1]
        self.total_params = self.weight_size + self.bias_size
        
        # =========================================================
        # 轻量 2 层 MLP (防止小样本过拟合)
        # =========================================================
        self.hypernet = nn.Sequential(
            nn.Linear(slow_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.total_params)
        )
        
        # =========================================================
        # 零初始化最后一层 (ReZero 策略)
        # =========================================================
        # 确保初始状态生成的权重为零，Correction = 0
        nn.init.zeros_(self.hypernet[-1].weight)
        nn.init.zeros_(self.hypernet[-1].bias)
        
        print(f"  [HyperNetwork] 初始化完成: slow_dim={slow_dim} -> 生成 {self.total_params} 个参数")
    
    def forward(self, z_slow: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_slow: (batch, slow_dim) 宏观状态嵌入
        
        Returns:
            W_dynamic: (batch, fast_dim, 1) 动态权重矩阵
            b_dynamic: (batch, 1) 动态偏置
        """
        batch_size = z_slow.size(0)
        
        # 生成所有参数
        params = self.hypernet(z_slow)  # (batch, total_params)
        
        # 拆分为权重和偏置
        W_flat = params[:, :self.weight_size]  # (batch, fast_dim)
        b_dynamic = params[:, self.weight_size:]  # (batch, 1)
        
        # 重塑权重矩阵
        W_dynamic = W_flat.view(batch_size, self.fast_dim, 1)  # (batch, fast_dim, 1)
        
        return W_dynamic, b_dynamic
class MacroCorrectiveResidualModel(nn.Module):
    """
    RARC-Net: Regime-Adaptive Residual Correction Network
    
    核心公式:
    Prediction = Micro_Frozen + HyperNetwork(Macro) * f_fusion
    
    设计理念:
    1. 微观基准 (Fast Channel): 已预训练冻结，提供保底预测
    2. 宏观修正 (Slow Channel + HyperNetwork): 动态生成权重，学习"纠错"
    3. 零初始化: 保证模型起点等价于 Fast-Only（绝不退化）
    
    数学保证:
    - 若 Correction → 0，模型退化为 Fast-Only
    - IC_Fusion ≥ IC_Fast（性能下限保证）
    """
    
    def __init__(self,
                 fast_input_dim: int,
                 slow_input_dim: int,
                 fast_hidden_dim: int = 128,
                 slow_d_model: int = 64,
                 slow_nhead: int = 4,
                 slow_num_layers: int = 2,
                 slow_seq_length: int = 60,
                 grn_hidden_dim: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        
        # =========================================================
        # 1. 微观基准通道 (Micro Baseline) - "老教授"
        # =========================================================
        self.fast_channel = BiGRU_FastChannel(
            input_dim=fast_input_dim,
            hidden_dim=fast_hidden_dim,
            dropout=dropout
        )
        # 独立回归头：输出基准预测 pred_fast
        self.fast_head = nn.Linear(fast_hidden_dim * 2, 1)  # 双向GRU输出维度*2
        
        # =========================================================
        # 🚀 【SOTA 优化】Fast 预测校准器 (Fast Calibration)
        # =========================================================
        # 核心问题：Phase A 残差 std=0.136，而 y_train std=0.012，差 10 倍
        # 这说明 pred_fast 的尺度/偏置与 y 不在同一量级
        # 让 correction 不用去"修尺度"，只专注于"修宏观波动"
        self.fast_scale = nn.Parameter(torch.tensor(1.0))
        self.fast_bias = nn.Parameter(torch.tensor(0.0))
        
        # =========================================================
        # 2. 宏观修正通道 (Macro Correction) - "纠错员"
        # =========================================================
        self.slow_channel = SlowStreamTransformer(
            input_dim=slow_input_dim,
            d_model=slow_d_model,
            nhead=slow_nhead,
            num_layers=slow_num_layers,
            seq_length=slow_seq_length,
            dropout=dropout,
            num_classes=1
        )
        
        # 宏观特征增强 (GRN)
        self.macro_grn = GatedResidualNetwork(
            input_dim=slow_d_model,
            hidden_dim=grn_hidden_dim,
            output_dim=grn_hidden_dim,
            dropout=0.25  # 高 Dropout 防过拟合
        )
        
        # =========================================================
        # 3. 跨模态融合 (Contextual Correction)
        # =========================================================
        # 拼接微观状态 + 宏观状态，判断：在当前微观形态下，宏观数据意味着什么？
        fusion_input_dim = fast_hidden_dim * 2 + grn_hidden_dim
        self.fusion_grn = GatedResidualNetwork(
            input_dim=fusion_input_dim,
            hidden_dim=grn_hidden_dim,
            output_dim=grn_hidden_dim,
            dropout=0.25
        )
        
        # =========================================================
        # 4. 【RARC-Net】HyperNetwork 动态权重生成
        # =========================================================
        # 用 HyperNetwork 替换静态 correction_head
        # 宏观状态决定了"微观特征如何映射到残差修正"
        self.hypernet = HyperNetwork(
            slow_dim=grn_hidden_dim,      # 宏观嵌入维度
            fast_dim=grn_hidden_dim,      # 融合特征维度
            hidden_dim=32,                # 轻量 MLP
            dropout=0.3
        )
        
        # =========================================================
        # 5. 【SOTA 优化】有界可学习 Cap (Bounded Learnable Cap)
        # =========================================================
        # 采用 GPT 建议的有界 cap 设计，比无上界 softplus 更安全
        # cap = cap_max * sigmoid(cap_param)
        # 初始 cap ≈ 0.10 * sigmoid(-2.0) ≈ 0.012，保证冷启动
        self.cap_max = 0.10  # 最大修正幅度 10%（日频 log return 合理范围）
        self.cap_param = nn.Parameter(torch.tensor(-2.0))  # 可学习的 cap 控制参数
        
        # =========================================================
        # 6. 【性能修复】GRN 门控冷启动 (Cold Start)
        # =========================================================
        # 将 GRN 门控偏置初始化为负数，强制初始状态"关闭"
        # Sigmoid(-3) ≈ 0.04，只允许 4% 的宏观信号通过
        # 防止训练初期宏观噪声破坏微观基准
        for name, param in self.named_parameters():
            if 'gate' in name and 'bias' in name:
                nn.init.constant_(param, -3.0)
        
        print(f"\n{'='*70}")
        print("RARC-Net (Regime-Adaptive Residual Correction Network) 初始化完成")
        print(f"{'='*70}")
        print(f"  快通道: BiGRU (input={fast_input_dim}, hidden={fast_hidden_dim})")
        print(f"  慢通道: Transformer (input={slow_input_dim}, d_model={slow_d_model})")
        print(f"  融合层: GRN + HyperNetwork (hidden={grn_hidden_dim})")
        print(f"  【有界 Cap】cap_max={self.cap_max}, 初始 cap≈{self.cap_max * torch.sigmoid(self.cap_param).item():.4f}")
        print(f"  【冷启动】GRN 门控初始化为 -3.0 (4% 通过率)")
    
    def forward(self, x_fast, x_slow, return_components=False, return_intermediates=False):
        """
        RARC-Net Forward Pass
        
        Args:
            x_fast: (batch, fast_seq, fast_features)
            x_slow: (batch, slow_seq, slow_features)
            return_components: 是否返回各组件输出（用于调试）
            return_intermediates: 是否返回中间结果（用于XAI可视化）兼容旧接口
        
        Returns:
            final_pred: (batch, 1) 最终预测
            如果 return_intermediates=True: 返回 (final_pred, intermediates_dict)
            如果 return_components=True: 返回 (final_pred, pred_fast, correction)
        """
        batch_size = x_fast.size(0)
        
        # ===== Step 1: 微观基准 (冻结区域) =====
        f_fast = self.fast_channel.forward_embedding(x_fast)  # [batch, hidden*2]
        raw_pred_fast = self.fast_head(f_fast)  # [batch, 1] - 原始预测
        
        # 🚀 应用校准器：pred_calib = a * pred_raw + b
        # 这能瞬间消除系统性尺度/偏置偏差
        pred_fast = raw_pred_fast * self.fast_scale + self.fast_bias
        
        # ===== Step 2: 宏观特征提取 =====
        f_slow = self.slow_channel(x_slow, return_embedding=True)  # [batch, d_model]
        f_slow_enhanced = self.macro_grn(f_slow)  # [batch, grn_hidden]
        
        # ===== Step 3: 跨模态融合 =====
        # detach() 再次确保梯度不回传给 fast_channel
        combined = torch.cat([f_fast.detach(), f_slow_enhanced], dim=-1)
        f_fusion = self.fusion_grn(combined)  # [batch, grn_hidden]
        
        # ===== Step 4: 【RARC-Net】HyperNetwork 动态权重生成 =====
        # 4a. 生成动态权重
        W_dynamic, b_dynamic = self.hypernet(f_slow_enhanced)  # W: [batch, grn_hidden, 1], b: [batch, 1]
        
        # 4b. 批量矩阵乘法计算修正量
        # f_fusion: [batch, grn_hidden] -> [batch, grn_hidden, 1]
        f_fusion_3d = f_fusion.unsqueeze(-1)  # [batch, grn_hidden, 1]
        raw_correction = torch.bmm(W_dynamic.transpose(1, 2), f_fusion_3d).squeeze(-1) + b_dynamic  # [batch, 1]
        
        # 4c. 【SOTA 优化】有界可学习 Cap
        # cap = cap_max * sigmoid(cap_param)，既安全又可学习
        # tanh 提供 [-1, 1] 的方向和相对强度
        current_cap = self.cap_max * torch.sigmoid(self.cap_param)  # [0, cap_max]
        correction = current_cap * torch.tanh(raw_correction)  # 范围 [-cap, cap]
        
        # ===== Step 5: 残差叠加 (核心！) =====
        final_pred = pred_fast + correction
        
        # 兼容 XAI 模块的返回格式
        if return_intermediates:
            intermediates = {
                'f_fast': f_fast.detach().cpu().numpy(),
                'z_macro': f_slow_enhanced.detach().cpu().numpy(),
                'gamma': correction.detach().cpu().numpy(),  # 将 correction 映射为 gamma 供热力图使用
                'pred_fast': pred_fast.detach().cpu().numpy(),
                'correction': correction.detach().cpu().numpy(),
                'W_dynamic_norm': W_dynamic.norm(dim=1).mean().item(),  # HyperNetwork 权重范数
                'current_cap': current_cap.item(),  # 当前 cap 值
                # 🚀 校准器参数（用于监控）
                'fast_scale': self.fast_scale.item(),
                'fast_bias': self.fast_bias.item()
            }
            return final_pred, intermediates
        
        if return_components:
            return final_pred, pred_fast, correction, f_fast, f_slow_enhanced
        return final_pred


# ==================== 4. 夏普感知混合损失函数 (SOTA) ====================
class HybridFinancialLoss(nn.Module):
    """
    三位一体混合损失函数 (Sharpe-Aware Loss)
    
    Loss = λ1·MSE + λ2·(1-IC) + λ3·SharpeLoss
    
    设计理念：
    - MSE (1.0): 锚定基础预测值，防止数值发散
    - IC (0.5): 保证方向正确性
    - Sharpe (0.05): 激励"方向对时敢于重仓"
    """
    
    def __init__(self, 
                 lambda_mse: float = 1.0, 
                 lambda_ic: float = 0.5, 
                 lambda_sharpe: float = 0.05):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_ic = lambda_ic
        self.lambda_sharpe = lambda_sharpe
        self.mse = nn.MSELoss()
        self.eps = 1e-8  # 数值稳定
    
    def forward(self, pred, target):
        """
        Args:
            pred: (batch, 1) 预测值
            target: (batch, 1) 真实值
        
        Returns:
            loss: 标量损失
            metrics: dict 各项指标
        """
        pred = pred.squeeze()
        target = target.squeeze()
        
        # 1. MSE Loss (回归准确性)
        loss_mse = self.mse(pred, target)
        
        # 2. IC Loss (排序能力)
        ic = pearson_correlation(pred.unsqueeze(-1), target.unsqueeze(-1))
        loss_ic = 1.0 - ic
        
        # 3. Sharpe Loss (风险调整收益)
        # 策略收益 = sign(pred) * target（如果方向对就赚钱）
        strategy_return = torch.sign(pred) * target
        
        # 夏普比率 = E[R] / Std[R]
        mean_return = strategy_return.mean()
        std_return = strategy_return.std() + self.eps  # 防止除零
        
        # 【关键】引入预测幅度加权，鼓励"重仓"
        # 如果预测值方差太小（太保守），额外惩罚
        pred_std = pred.std() + self.eps
        conservatism_penalty = torch.relu(0.01 - pred_std) * 5.0
        
        # 负夏普（因为我们要最小化损失）
        sharpe = mean_return / std_return
        loss_sharpe = -sharpe + conservatism_penalty
        
        # =========================================================
        # 4. 【RARC-Net】Volatility Penalty (非对称惩罚)
        # =========================================================
        # 如果预测波动率 > 真实波动率 → 瞎波动，严厉惩罚！
        # 理念：除非你有十足把握，否则绝不允许你的预测波动率超过市场本身
        pred_vol = pred.std()
        target_vol = target.std() + self.eps
        loss_volatility = F.relu(pred_vol - target_vol)  # 非对称: 只惩罚超出部分
        
        # 总损失 (增加 Volatility Penalty，权重0.5)
        total_loss = (self.lambda_mse * loss_mse + 
                      self.lambda_ic * loss_ic + 
                      self.lambda_sharpe * loss_sharpe +
                      0.5 * loss_volatility)  # 【RARC-Net】波动率惩罚
        
        # 方向准确率
        direction_acc = ((torch.sign(pred) == torch.sign(target)).float().mean()).item()
        
        return total_loss, {
            'mse': loss_mse.item(),
            'ic': ic.item(),
            'sharpe': sharpe.item(),
            'direction_acc': direction_acc,
            'pred_std': pred_std.item(),
        }


# ==================== 5. 训练器 ====================
class E2ETrainer:
    """RARC-Net 端到端训练器 (含 PCGrad 梯度手术)"""
    
    def __init__(self, 
                 model: nn.Module,  # 支持 MacroCorrectiveResidualModel
                 device: torch.device,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5):
        self.model = model.to(device)
        self.device = device
        
        # 使用C-AdamW优化器
        self.optimizer = CautiousAdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 使用夏普感知损失函数
        self.criterion = HybridFinancialLoss(
            lambda_mse=1.0, 
            lambda_ic=0.5, 
            lambda_sharpe=0.05
        )
        print("  ✓ 使用 HybridFinancialLoss (含 Sharpe + Volatility 惩罚)")
        
        # 记录
        self.history = {'train_loss': [], 'val_loss': [], 'val_ic': []}
    
    def train_epoch(self, train_loader):
        """
        训练一个 epoch（含 PCGrad 梯度手术）
        
        PCGrad 原理：
        - 分别计算不同Loss的梯度
        - 如果梯度冲突（夹角 > 90°），投影到法平面
        - 确保辅助任务不破坏主任务
        """
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        for x_fast, x_slow, y in train_loader:
            x_fast = x_fast.to(self.device)
            x_slow = x_slow.to(self.device)
            y = y.to(self.device)
            
            # 前向传播（RARC-Net 输出5个值）
            output = self.model(x_fast, x_slow, return_components=True)
            
            # 解析输出（兼容不同返回格式）
            if isinstance(output, tuple) and len(output) >= 3:
                final_pred, pred_fast, correction = output[:3]
                f_fast = output[3] if len(output) > 3 else None
                f_slow_enhanced = output[4] if len(output) > 4 else None
            else:
                final_pred = output[0] if isinstance(output, tuple) else output
                pred_fast = final_pred
                correction = torch.zeros_like(final_pred)
                f_fast, f_slow_enhanced = None, None
            
            # =========================================================
            # 🚀 PCGrad 梯度手术 (Gradient Surgery)
            # =========================================================
            # 目标：解决 MSE vs IC/Sharpe 的多目标冲突
            
            # 1. 主任务 Loss (MSE - 准确率)
            main_loss, metrics = self.criterion(final_pred, y)
            
            # 2. 辅助任务 Loss (残差监督 + 正交 + 稀疏)
            residual_target = (y - pred_fast).detach()
            residual_loss = F.mse_loss(correction, residual_target)
            
            # 正交正则化
            if correction.numel() > 1 and pred_fast.numel() > 1:
                cos_sim = F.cosine_similarity(
                    correction.flatten().unsqueeze(0), 
                    pred_fast.flatten().unsqueeze(0)
                )
                orth_loss = torch.abs(cos_sim).mean()
            else:
                orth_loss = torch.tensor(0.0, device=self.device)
            
            # 特征正交 (如果可用且维度匹配)
            # 注意：f_fast [batch, 256] vs f_slow_enhanced [batch, 64] 维度不同
            # 只有维度相同时才计算，否则跳过
            feat_orth_loss = torch.tensor(0.0, device=self.device)
            if f_fast is not None and f_slow_enhanced is not None:
                if f_fast.size(-1) == f_slow_enhanced.size(-1):
                    feat_cos = F.cosine_similarity(
                        f_fast.mean(dim=0).unsqueeze(0),
                        f_slow_enhanced.mean(dim=0).unsqueeze(0)
                    )
                    feat_orth_loss = torch.abs(feat_cos).mean()
            
            # L1 稀疏约束
            l1_loss = torch.norm(correction, 1) / correction.numel()
            
            # 辅助损失总和
            aux_loss = 0.5 * residual_loss + 0.1 * orth_loss + 0.1 * feat_orth_loss + 0.01 * l1_loss
            
            # =========================================================
            # PCGrad: 梯度投影
            # =========================================================
            # 获取所有可训练参数
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            
            # Step 1: 计算主任务梯度
            self.optimizer.zero_grad()
            main_loss.backward(retain_graph=True)
            g_main = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) 
                      for p in trainable_params]
            
            # Step 2: 计算辅助任务梯度
            self.optimizer.zero_grad()
            aux_loss.backward()
            g_aux = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) 
                     for p in trainable_params]
            
            # Step 3: PCGrad 投影（如果冲突）
            for i in range(len(trainable_params)):
                if g_main[i].numel() > 0 and g_aux[i].numel() > 0:
                    # 展平梯度
                    g_m_flat = g_main[i].flatten()
                    g_a_flat = g_aux[i].flatten()
                    
                    # 计算点积
                    dot = torch.dot(g_m_flat, g_a_flat)
                    
                    # 如果冲突（夹角 > 90°）
                    if dot < 0:
                        # 投影: g_aux = g_aux - (g_aux·g_main / ||g_main||^2) * g_main
                        g_main_norm_sq = (g_m_flat ** 2).sum() + 1e-8
                        projection = (dot / g_main_norm_sq) * g_m_flat
                        g_a_flat = g_a_flat - projection
                        g_aux[i] = g_a_flat.view_as(g_aux[i])
            
            # Step 4: 合并梯度
            self.optimizer.zero_grad()
            for i, p in enumerate(trainable_params):
                p.grad = g_main[i] + g_aux[i]
            
            # 梯度裁剪 + 更新
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            self.optimizer.step()
            
            total_loss += (main_loss.item() + aux_loss.item())
            n_batches += 1
        
        return total_loss / n_batches
    
    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        total_loss = 0
        all_preds, all_targets = [], []
        all_fast_preds = []  # 添加快通道预测收集
        
        with torch.no_grad():
            for x_fast, x_slow, y in val_loader:
                x_fast = x_fast.to(self.device)
                x_slow = x_slow.to(self.device)
                y = y.to(self.device)
                
                # 融合预测（RARC-Net 输出）
                output = self.model(x_fast, x_slow)
                pred = output[0] if isinstance(output, tuple) else output
                loss, metrics = self.criterion(pred, y)
                
                # 快通道单独预测（诊断用）
                # RARC-Net 模型使用 forward_embedding + fast_head
                if hasattr(self.model, 'fast_head'):
                    f_fast = self.model.fast_channel.forward_embedding(x_fast)
                    fast_pred = self.model.fast_head(f_fast)
                else:
                    fast_pred = self.model.fast_channel(x_fast)
                
                total_loss += loss.item()
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y.cpu().numpy())
                all_fast_preds.append(fast_pred.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_fast_preds = np.concatenate(all_fast_preds)
        
        # 计算融合IC
        fusion_ic = np.corrcoef(all_preds.flatten(), all_targets.flatten())[0, 1]
        
        # 计算快通道IC（诊断）
        fast_ic = np.corrcoef(all_fast_preds.flatten(), all_targets.flatten())[0, 1]
        
        return total_loss / len(val_loader), fusion_ic, fast_ic


# ==================== 6. XAI 分析器 ====================
class XAIAnalyzer:
    """
    XAI可解释性分析器: Gamma/Correction 热力图等
    
    支持 RARC-Net 架构：
    - MacroCorrectiveResidualModel (RARC-Net): 返回 correction (宏观修正量)
    """
    
    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device
        # 检测模型类型：RARC-Net 使用 hypernet
        self.is_rarc = hasattr(model, 'hypernet')
    
    def extract_gamma_values(self, data_loader) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取所有样本的 Gamma/Correction 值
        
        Returns:
            gamma_values: (N,) Gamma/Correction 值序列
            predictions: (N,) 预测值序列
        """
        self.model.eval()
        gamma_list, pred_list = [], []
        
        with torch.no_grad():
            for x_fast, x_slow, _ in data_loader:
                x_fast = x_fast.to(self.device)
                x_slow = x_slow.to(self.device)
                
                # 兼容两种架构
                output = self.model(x_fast, x_slow, return_intermediates=True)
                
                if isinstance(output, tuple) and len(output) == 2:
                    pred, intermediates = output
                    # RARC-Net 模型的 correction 映射为 gamma (兼容旧接口)
                    gamma = intermediates.get('gamma', intermediates.get('correction', np.zeros((len(x_fast), 1))))
                else:
                    # 旧模型兼容
                    pred = output
                    gamma = np.zeros((len(x_fast), 1))
                
                gamma_list.append(gamma if isinstance(gamma, np.ndarray) else gamma.cpu().numpy())
                pred_list.append(pred.cpu().numpy() if hasattr(pred, 'cpu') else pred)
        
        gamma_values = np.concatenate(gamma_list).flatten()
        predictions = np.concatenate(pred_list).flatten()
        
        return gamma_values, predictions
    
    def plot_gamma_heatmap(self, 
                          gamma_values: np.ndarray,
                          dates: np.ndarray,
                          sp500_prices: np.ndarray,
                          save_path: str):
        """
        绘制 Correction 热力图 + SP500走势
        
        标注关键事件: 飓风伊恩、加息周期等
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                        gridspec_kw={'height_ratios': [2, 1]})
        
        # 上图: SP500价格 + Correction
        ax1_twin = ax1.twinx()
        
        # SP500价格
        ax1.plot(dates, sp500_prices, color='#2E86AB', linewidth=2, label='SP500')
        ax1.set_ylabel('SP500 Price', fontsize=14, color='#2E86AB')
        ax1.tick_params(axis='y', labelcolor='#2E86AB')
        ax1.grid(alpha=0.3)
        
        # Correction值
        ax1_twin.plot(dates, gamma_values, color='#E63946', linewidth=2, 
                     label='Correction (宏观修正量)', alpha=0.8)
        ax1_twin.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='修正=0 (中性)')
        ax1_twin.set_ylabel('Correction Value', fontsize=14, color='#E63946')
        ax1_twin.tick_params(axis='y', labelcolor='#E63946')
        ax1_twin.set_ylim(-0.15, 0.15)  # RARC-Net 修正范围
        
        # 标注关键事件
        events = [
            ('2022-09-28', '飓风伊恩', '#FF6B6B'),
            ('2022-03-16', '加息开始', '#4ECDC4'),
            ('2022-11-02', '连续加息', '#4ECDC4'),
        ]
        
        for event_date, event_name, color in events:
            try:
                event_date_dt = pd.to_datetime(event_date)
                if event_date_dt in dates:
                    idx = np.where(dates == event_date_dt)[0][0]
                    ax1.axvline(x=event_date_dt, color=color, linestyle=':', 
                               linewidth=2, alpha=0.7)
                    ax1.text(event_date_dt, sp500_prices[idx], event_name, 
                            rotation=90, verticalalignment='bottom',
                            fontsize=11, color=color, fontweight='bold')
            except:
                pass
        
        ax1.set_title('RARC-Net 宏观修正分析 (Regime-Adaptive Correction)', 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.legend(loc='upper left', fontsize=12)
        ax1_twin.legend(loc='upper right', fontsize=12)
        
        # 下图: Correction分布直方图
        ax2.hist(gamma_values, bins=50, color='#E63946', alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=2, label='修正=0 (中性)')
        ax2.set_xlabel('Correction Value', fontsize=14)
        ax2.set_ylabel('Frequency', fontsize=14)
        ax2.set_title('Correction分布 (正值=上修预测, 负值=下修预测)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Correction 热力图已保存: {save_path}")
        plt.close()


# ==================== 主函数 ====================
def main():
    """主训练流程"""
    print("\n" + "="*70)
    print("RARC-Net: Regime-Adaptive Residual Correction Network")
    print("="*70)
    
    # 配置
    # 使用仓库根目录作为基准，兼容整理后的 data/ 与 results/ 布局。
    script_dir = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(os.path.dirname(script_dir))
    parent_dir = os.path.dirname(BASE_DIR)
    OUTPUT_DIR = os.path.join(BASE_DIR, 'results', 'e2e')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 1. 数据加载
    print("\n" + "="*70)
    print("步骤1: 数据加载与对齐")
    print("="*70)
    
    loader = JointDataLoader(
        base_dir=BASE_DIR,
        fast_window=10,
        slow_window=60,
        predict_horizon=1
    )
    
    X_fast, X_slow, y, dates = loader.align_and_create_sequences()
    
    # 2. 数据集划分 (Walk-Forward)
    # 简单起见，这里先用一次划分，后续可改为Walk-Forward
    train_size = int(len(X_fast) * 0.7)
    val_size = int(len(X_fast) * 0.15)
    
    X_fast_train = X_fast[:train_size]
    X_slow_train = X_slow[:train_size]
    y_train = y[:train_size]
    
    X_fast_val = X_fast[train_size:train_size+val_size]
    X_slow_val = X_slow[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    
    X_fast_test = X_fast[train_size+val_size:]
    X_slow_test = X_slow[train_size+val_size:]
    y_test = y[train_size+val_size:]
    dates_test = dates[train_size+val_size:]
    
    # ========== 🔑 关键步骤：RobustScaler标准化 ==========
    # 这是原fast_channel实验中存在而E2E缺失的关键步骤！
    print("\n  [关键] 特征标准化 (RobustScaler)...")
    
    # 快通道标准化: 对每个特征维度在训练集上fit
    n_fast_features = X_fast.shape[2]
    fast_scalers = []
    for i in range(n_fast_features):
        scaler = RobustScaler()
        # 将训练集中该特征的所有时间步展平后fit
        train_feature = X_fast_train[:, :, i].reshape(-1, 1)
        scaler.fit(train_feature)
        fast_scalers.append(scaler)
        
        # Transform 所有集的该特征
        X_fast_train[:, :, i] = scaler.transform(X_fast_train[:, :, i].reshape(-1, 1)).reshape(X_fast_train[:, :, i].shape)
        X_fast_val[:, :, i] = scaler.transform(X_fast_val[:, :, i].reshape(-1, 1)).reshape(X_fast_val[:, :, i].shape)
        X_fast_test[:, :, i] = scaler.transform(X_fast_test[:, :, i].reshape(-1, 1)).reshape(X_fast_test[:, :, i].shape)
    
    # 慢通道标准化: 同样处理
    n_slow_features = X_slow.shape[2]
    slow_scalers = []
    for i in range(n_slow_features):
        scaler = RobustScaler()
        train_feature = X_slow_train[:, :, i].reshape(-1, 1)
        scaler.fit(train_feature)
        slow_scalers.append(scaler)
        
        X_slow_train[:, :, i] = scaler.transform(X_slow_train[:, :, i].reshape(-1, 1)).reshape(X_slow_train[:, :, i].shape)
        X_slow_val[:, :, i] = scaler.transform(X_slow_val[:, :, i].reshape(-1, 1)).reshape(X_slow_val[:, :, i].shape)
        X_slow_test[:, :, i] = scaler.transform(X_slow_test[:, :, i].reshape(-1, 1)).reshape(X_slow_test[:, :, i].shape)
    
    print(f"  ✓ 快通道: {n_fast_features}个特征已标准化")
    print(f"  ✓ 慢通道: {n_slow_features}个特征已标准化")
    # ========================================================
    
    # 3. 创建DataLoader
    class JointDataset(Dataset):
        def __init__(self, X_fast, X_slow, y):
            self.X_fast = torch.tensor(X_fast, dtype=torch.float32)
            self.X_slow = torch.tensor(X_slow, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        
        def __len__(self):
            return len(self.X_fast)
        
        def __getitem__(self, idx):
            return self.X_fast[idx], self.X_slow[idx], self.y[idx]
    
    train_dataset = JointDataset(X_fast_train, X_slow_train, y_train)
    val_dataset = JointDataset(X_fast_val, X_slow_val, y_val)
    test_dataset = JointDataset(X_fast_test, X_slow_test, y_test)
    
    # 【SOTA】增大 batch_size 到 64，稳定 Sharpe Loss 统计量
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 4. 初始化模型
    print("\n" + "="*70)
    print("步骤2: 模型初始化")
    print("="*70)
    
    # ========== RARC-Net 模型初始化 ==========
    print("\n🚀 使用 RARC-Net 架构 (Prediction = Fast_Baseline + HyperNetwork_Correction)")
    model = MacroCorrectiveResidualModel(
        fast_input_dim=X_fast.shape[2],
        slow_input_dim=X_slow.shape[2],
        fast_hidden_dim=128,
        slow_d_model=64,
        slow_nhead=4,
        slow_num_layers=2,
        slow_seq_length=60,
        grn_hidden_dim=64,
        dropout=0.2
    )
    
    # ========== 🔑 关键修复：加载 Fast Channel 预训练权重 ==========
    # 如果不加载预训练权重，冻结的将是随机初始化的模型（Fast IC≈0）
    # 导致 Fusion 层只能对噪声信号做优化，性能会极差
    
    LOAD_PRETRAINED_FAST = True  # 强烈建议开启
    
    if LOAD_PRETRAINED_FAST:
        print("\n📥 正在加载 Fast Channel 预训练权重...")
        
        # 候选路径（按优先级）
        PRETRAINED_PATHS = [
            # 1. 优先：转换后的 .pth 格式（无 joblib 依赖）
            os.path.join(parent_dir, 'fast_channel', 'best_fast_channel_BiGRU.pth'),
            
            # 2. 当前目录 .pth
            os.path.join(BASE_DIR, 'best_fast_channel_BiGRU.pth'),
            
            # 3. 当前目录 .pkl（可能）
            os.path.join(BASE_DIR, 'B_with_ESG_window10_models.pkl'),
            os.path.join(BASE_DIR, 'A_without_ESG_window10_models.pkl'),
            
            # 4. 项目根目录
            os.path.join(parent_dir, 'B_with_ESG_window10_models.pkl'),
            
            # 5. 标准结果目录 .pkl
            os.path.join(parent_dir, 'fast_channel_results', 'B_with_ESG_window10_models.pkl'),
            os.path.join(parent_dir, 'fast_channel_results', 'A_without_ESG_window10_models.pkl'),
            
            # 6. Seeds 目录
            os.path.join(parent_dir, 'fast_seeds_result', 'best_fast_model.pth'),
        ]
        
        loaded = False
        for path in PRETRAINED_PATHS:
            if not os.path.exists(path):
                continue
            
            try:
                print(f"  → 尝试加载: {os.path.basename(path)}")
                
                # 根据文件扩展名选择加载方式
                if path.endswith('.pkl'):
                    # joblib 格式（fast_channel实验保存的）
                    import joblib
                    results = joblib.load(path)
                    
                    # 提取 BiGRU 模型（优先）
                    if 'BiGRU' in results and 'model' in results['BiGRU']:
                        pretrained_model = results['BiGRU']['model']
                    elif 'BiLSTM' in results and 'model' in results['BiLSTM']:
                        pretrained_model = results['BiLSTM']['model']
                    else:
                        print(f"    ✗ 文件中未找到 BiGRU/BiLSTM 模型")
                        continue
                    
                    # 加载权重（state_dict）
                    pretrained_state = pretrained_model.state_dict()
                    
                elif path.endswith('.pth'):
                    # PyTorch 原生格式
                    pretrained_state = torch.load(path, map_location=device)
                else:
                    print(f"    ✗ 不支持的文件格式: {path}")
                    continue
                
                # 键名映射（处理可能的前缀不匹配）
                model_state = model.state_dict()
                new_state = {}
                
                for k, v in pretrained_state.items():
                    # 情况1: 预训练key已有 "fast_channel." 前缀
                    if k.startswith('fast_channel.'):
                        new_key = k
                    # 情况2: 预训练key无前缀，需要添加
                    else:
                        new_key = f'fast_channel.{k}'
                    
                    # 仅保留存在于当前模型中的key
                    if new_key in model_state:
                        new_state[new_key] = v
                
                # 验证加载了足够的参数
                if len(new_state) == 0:
                    print(f"    ✗ 键名完全不匹配，跳过")
                    continue
                
                # 更新模型权重
                model_state.update(new_state)
                # 【修复】使用 strict=False 允许部分加载（处理 input_dim 不匹配）
                model.load_state_dict(model_state, strict=False)
                
                print(f"    ✓ 成功加载 {len(new_state)} 个参数")
                print(f"    ✓ 来源: {os.path.basename(path)}")
                
                # =========================================================
                # 🚀【关键修复】Fast Head 权重继承
                # =========================================================
                # RARC-Net 模型将原来的 fc 层拆分为 fast_head
                # 需要手动将预训练模型的 fc 权重映射到 fast_head
                if hasattr(model, 'fast_head'):
                    fc_loaded = False
                    possible_fc_names = ['fc', 'regressor', 'linear', 'head', 'output']
                    
                    for fc_name in possible_fc_names:
                        weight_key = f'{fc_name}.weight'
                        bias_key = f'{fc_name}.bias'
                        
                        if weight_key in pretrained_state and bias_key in pretrained_state:
                            pretrained_weight = pretrained_state[weight_key]
                            pretrained_bias = pretrained_state[bias_key]
                            
                            # 检查形状是否匹配
                            model_weight_shape = model.fast_head.weight.shape
                            if pretrained_weight.shape == model_weight_shape:
                                with torch.no_grad():
                                    model.fast_head.weight.copy_(pretrained_weight)
                                    model.fast_head.bias.copy_(pretrained_bias)
                                print(f"    ✓ Fast Head 权重已继承自 '{fc_name}' 层")
                                print(f"      (基准预测能力已激活!)")
                                fc_loaded = True
                                break
                            else:
                                print(f"    ⚠️ '{fc_name}' 层形状不匹配: {pretrained_weight.shape} vs {model_weight_shape}")
                    
                    if not fc_loaded:
                        print(f"    ⚠️ 警告: 未找到匹配的回归层权重")
                        print(f"       Fast Head 将使用随机初始化，前几个 Epoch 可能波动")
                
                loaded = True
                break
                
            except Exception as e:
                print(f"    ✗ 加载失败: {e}")
                continue
        
        if not loaded:
            print("  ⚠️ 警告：所有预训练权重加载失败！")
            print("     Fast Channel 将使用随机初始化，这会导致 Fusion 性能极差。")
            print("     强烈建议先运行 fast_channel 实验生成预训练权重。")
            # 可选：直接终止程序
            # import sys
            # sys.exit(1)
    
    # =================================================================
    
    # ========== 两阶段训练策略 (SOTA 优化) ==========
    # 阶段一：冻结 Fast Channel（保留其最佳状态），只训练 Slow + HyperNetwork
    # 阶段二：解冻微调，使用极小学习率 (关键！恢复 Sharpe 和 Return)
    
    FREEZE_FAST_CHANNEL = True   # 阶段一必须冻结
    ENABLE_STAGE2_FINETUNE = False  # 【关闭】小样本下解冻微调会导致灾难性遗忘
    
    if FREEZE_FAST_CHANNEL:
        print("\n🔒 【阶段一】冻结快通道参数，仅训练融合层...")
        for param in model.fast_channel.parameters():
            param.requires_grad = False
        # RARC-Net 架构还需冻结 fast_head
        if hasattr(model, 'fast_head'):
            for param in model.fast_head.parameters():
                param.requires_grad = False
            print("  ✓ Fast Channel + Fast Head 参数已冻结")
        else:
            print("  ✓ Fast Channel 参数已冻结")
        
        # 【零初始化检查】确保 HyperNetwork 初始约为0
        if hasattr(model, 'hypernet'):
            with torch.no_grad():
                try:
                    # 检查 HyperNetwork 最后一层的零初始化
                    hypernet_last_layer = model.hypernet.hypernet[-1]
                    weight_norm = hypernet_last_layer.weight.abs().mean().item()
                    bias_norm = hypernet_last_layer.bias.abs().mean().item()
                    print(f"  ✓ HyperNetwork 零初始化验证: weight_norm={weight_norm:.6f}, bias_norm={bias_norm:.6f}")
                    # 检查 cap_param 初始化
                    current_cap = model.cap_max * torch.sigmoid(model.cap_param).item()
                    print(f"  ✓ 有界 Cap 初始值: {current_cap:.6f}")
                    # 检查 Fast 校准器初始化
                    if hasattr(model, 'fast_scale'):
                        print(f"  ✓ Fast 校准器: scale={model.fast_scale.item():.4f}, bias={model.fast_bias.item():.4f}")
                except Exception as e:
                    print(f"  ⚠️ 零初始化检查失败: {e}")
    
    # ==============================================================================
    # 🚀 SOTA 策略：解耦残差学习 (Decoupled Residual Learning)
    # ==============================================================================
    print("\n" + "="*70)
    print("步骤3: 启动解耦残差训练 (Gradient Boosting Strategy)")
    print("="*70)
    print("💡 原理：强制 Slow Channel 专攻 Fast Channel 的'预测盲区' (Residuals)")
    
    # ------------------------------------------------------------------
    # Phase A: 计算 Fast Channel 的静态残差
    # ------------------------------------------------------------------
    print("\n[Phase A] 生成微观残差目标...")
    model = model.to(device)  # 确保模型在正确设备上
    model.eval()
    
    # 收集所有训练集的 Fast 预测和残差
    all_residuals = []
    with torch.no_grad():
        for x_fast, x_slow, y in train_loader:
            x_fast = x_fast.to(device)
            y = y.to(device)
            
            # 获取 Fast Channel 预测
            f_fast = model.fast_channel.forward_embedding(x_fast)
            pred_fast = model.fast_head(f_fast)
            
            # 计算残差 = 真实值 - Fast预测
            residual = y - pred_fast
            all_residuals.append(residual.cpu())
    
    all_residuals = torch.cat(all_residuals, dim=0)
    print(f"  ✓ 残差生成完毕. Mean: {all_residuals.mean():.6f}, Std: {all_residuals.std():.6f}")
    print(f"    残差范围: [{all_residuals.min():.4f}, {all_residuals.max():.4f}]")
    
    # =========================================================
    # 【SOTA 优化】计算残差标准化参数 (GPT 建议)
    # =========================================================
    # 用于 Phase B 的 Z-Score 标准化，放大梯度信号
    resid_mean = all_residuals.mean().to(device)
    resid_std = all_residuals.std().to(device) + 1e-8
    print(f"  ✓ 残差标准化参数: μ={resid_mean.item():.6f}, σ={resid_std.item():.6f}")
    
    # 同时打印 y_train 统计（供 cap_max 调参参考）
    y_train_tensor = torch.tensor(y_train.flatten())
    print(f"  [调参参考] y_train: std={y_train_tensor.std():.6f}, q99={y_train_tensor.abs().quantile(0.99):.6f}")
    
    # 最小激活约束的阈值 (GPT 建议: delta = 0.1 * std(y_train))
    min_correction_std = 0.1 * y_train_tensor.std().item()
    print(f"  [最小激活约束] delta={min_correction_std:.6f}")
    
    # ------------------------------------------------------------------
    # Phase B: 残差专项训练 (含校准器 + 饱和惩罚 + 活动抑制)
    # ------------------------------------------------------------------
    print("\n[Phase B] 残差专项训练 (含 Fast 校准器 + 饱和惩罚 + 活动抑制)...")
    
    # 1. 冻结 Fast 部分（权重不动，但校准器可学）
    for param in model.fast_channel.parameters():
        param.requires_grad = False
    for param in model.fast_head.parameters():
        param.requires_grad = False
    print("  ✓ Fast Channel/Head 已冻结")
    
    # 2. 解冻 Slow 部分
    for param in model.slow_channel.parameters():
        param.requires_grad = True
    for param in model.macro_grn.parameters():
        param.requires_grad = True
    for param in model.fusion_grn.parameters():
        param.requires_grad = True
    for param in model.hypernet.parameters():
        param.requires_grad = True
    
    # 3. 🚀 解冻校准器参数（关键！解决尺度失配）
    model.fast_scale.requires_grad = True
    model.fast_bias.requires_grad = True
    print("  ✓ Fast 校准器 (scale, bias) 已解冻")
    
    # 4. 🚀 Cap 先冻后放策略：首 5 epoch 冻结，防止早期学会"打满"
    model.cap_param.requires_grad = False
    print("  ✓ Cap 参数暂时冻结（Epoch 6 后解冻）")
    print("  ✓ Slow Channel/GRN/HyperNetwork 已解冻")
    
    # 3. 专用优化器 (高学习率强攻残差)
    residual_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_residual = CautiousAdamW(residual_params, lr=3e-3, weight_decay=1e-4)
    print(f"  ✓ 使用高学习率 3e-3 进行残差专项训练")
    
    # 4. 残差专项训练 (含饱和惩罚 + 活动抑制 + Cap 解冻)
    best_residual_loss = float('inf')
    phase_b_epochs = 30
    patience_b = 10
    patience_counter_b = 0
    
    for epoch in range(phase_b_epochs):
        model.train()
        total_loss = 0
        total_sat_ratio = 0
        n_batches = 0
        
        # 🚀 Cap 先冻后放：Epoch 6 开始解冻 cap_param
        if epoch == 5:
            print("  🔓 [Epoch 6] 解冻 Cap 参数，允许模型调整修正幅度...")
            model.cap_param.requires_grad = True
            # 重新构建优化器参数列表
            residual_params = [p for p in model.parameters() if p.requires_grad]
            optimizer_residual = CautiousAdamW(residual_params, lr=3e-3, weight_decay=1e-4)
        
        for x_fast, x_slow, y in train_loader:
            x_fast = x_fast.to(device)
            x_slow = x_slow.to(device)
            y = y.to(device)
            
            optimizer_residual.zero_grad()
            
            # 前向传播 (RARC-Net 返回 5 个值，只需要前 3 个)
            output = model(x_fast, x_slow, return_components=True)
            final_pred, pred_fast, correction = output[:3]
            
            # 🎯 残差标准化训练 (放大梯度信号)
            raw_residual = (y - pred_fast).detach()  # 切断校准器梯度到训练中
            scaled_target = (raw_residual - resid_mean) / resid_std  # Z-Score
            scaled_correction = correction / resid_std
            
            # 1. 主损失：拟合标准化残差
            mse_loss = F.mse_loss(scaled_correction, scaled_target)
            
            # 2. 🚀 饱和惩罚 L_sat (防止撞墙)
            # 惩罚 |correction| 接近 cap 的样本
            current_cap = model.cap_max * torch.sigmoid(model.cap_param)
            is_saturated = (correction.abs() > 0.95 * current_cap).float()
            sat_ratio_batch = is_saturated.mean().item()
            L_sat = F.relu(correction.abs() / (current_cap + 1e-8) - 0.95).mean()
            
            # 3. 🚀 活动抑制 L_act (防止低风险期瞎动)
            L_act = correction.abs().mean()
            
            # 4. 最小激活约束 (仅前 5 epoch，防止死鱼)
            if epoch < 5:
                L_min = F.relu(min_correction_std - correction.std())
            else:
                L_min = torch.tensor(0.0, device=device)
            
            # 总损失：加权组合
            # λ_sat=0.2 (抑制撞墙), λ_act=0.05 (常态抑制), λ_min=0.2 (防死鱼)
            residual_loss = mse_loss + 0.2 * L_sat + 0.05 * L_act + 0.2 * L_min
            
            residual_loss.backward()
            torch.nn.utils.clip_grad_norm_(residual_params, max_norm=1.0)
            optimizer_residual.step()
            
            total_loss += residual_loss.item()
            total_sat_ratio += sat_ratio_batch
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_sat_ratio = total_sat_ratio / n_batches
        
        # 验证
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_fast, x_slow, y in val_loader:
                x_fast, x_slow, y = x_fast.to(device), x_slow.to(device), y.to(device)
                final_pred = model(x_fast, x_slow)
                if isinstance(final_pred, tuple):
                    final_pred = final_pred[0]
                val_preds.append(final_pred.cpu())
                val_targets.append(y.cpu())
        
        val_preds = torch.cat(val_preds).numpy().flatten()
        val_targets = torch.cat(val_targets).numpy().flatten()
        val_ic = np.corrcoef(val_preds, val_targets)[0, 1]
        
        # 🚀 增强日志：监控关键参数
        current_cap_val = (model.cap_max * torch.sigmoid(model.cap_param)).item()
        scale_val = model.fast_scale.item()
        bias_val = model.fast_bias.item()
        
        print(f"  Res Epoch {epoch+1:2d} | Loss: {avg_loss:.4f} | IC: {val_ic:.4f} | "
              f"Cap: {current_cap_val:.4f} | SatRatio: {avg_sat_ratio:.1%} | "
              f"Calib: {scale_val:.2f}x{bias_val:+.4f}")
        
        # 保存最佳
        if avg_loss < best_residual_loss:
            best_residual_loss = avg_loss
            patience_counter_b = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_residual_model.pth'))
        else:
            patience_counter_b += 1
            if patience_counter_b >= patience_b:
                print(f"  ✓ Phase B 早停于 Epoch {epoch+1}")
                break
    
    # 加载 Phase B 最佳模型
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_residual_model.pth')))
    
    # ------------------------------------------------------------------
    # Phase C: 联合微调
    # ------------------------------------------------------------------
    print("\n[Phase C] 联合微调 (小火慢炖)...")
    
    # 保持 Fast 冻结，用常规 Loss 微调
    trainer = E2ETrainer(model, device, learning_rate=1e-4)
    
    # 【SOTA 优化】组合早停指标 (GPT 建议)
    # Score = IC + α×(DirAcc - 0.5)，避免只看 IC 导致错过好模型
    best_val_score = -np.inf
    best_val_ic = -np.inf
    patience = 15
    patience_counter = 0
    
    for epoch in range(50):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_ic, fast_ic = trainer.validate(val_loader)
        
        # 计算方向准确率
        model.eval()
        with torch.no_grad():
            val_preds_all, val_targets_all = [], []
            for x_fast, x_slow, y in val_loader:
                x_fast, x_slow, y = x_fast.to(device), x_slow.to(device), y.to(device)
                pred = model(x_fast, x_slow)
                if isinstance(pred, tuple):
                    pred = pred[0]
                val_preds_all.append(pred.cpu())
                val_targets_all.append(y.cpu())
            val_preds_np = torch.cat(val_preds_all).numpy().flatten()
            val_targets_np = torch.cat(val_targets_all).numpy().flatten()
            direction_acc = ((val_preds_np > 0) == (val_targets_np > 0)).mean()
        
        # 组合指标
        val_score = val_ic + 0.3 * (direction_acc - 0.5)
        
        trainer.history['train_loss'].append(train_loss)
        trainer.history['val_loss'].append(val_loss)
        trainer.history['val_ic'].append(val_ic)
        
        print(f"  Finetune Epoch {epoch+1:2d} | Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | Fusion IC: {val_ic:.4f} | Fast IC: {fast_ic:.4f} | "
              f"DirAcc: {direction_acc:.4f} | Score: {val_score:.4f}")
        
        # 使用组合指标早停
        if val_score > best_val_score:
            print(f"    ★ New Best Score: {val_score:.4f}")
            best_val_score = val_score
            best_val_ic = val_ic  # 同时更新 IC 用于报告
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  ✓ Phase C 早停于 Epoch {epoch+1}")
                break
    
    # ========== 阶段二：可选微调（解冻Fast Channel）==========
    if ENABLE_STAGE2_FINETUNE and FREEZE_FAST_CHANNEL:
        print(f"\n{'='*70}")
        print("🔓 【阶段二】解冻快通道，进行联合微调...")
        print(f"{'='*70}")
        
        # 加载阶段一最佳模型
        model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth')))
        
        # 解冻 Fast Channel
        for param in model.fast_channel.parameters():
            param.requires_grad = True
        print("  ✓ Fast Channel 参数已解冻")
        
        # 重新初始化 Trainer，使用极小学习率
        trainer_stage2 = E2ETrainer(model, device, learning_rate=1e-5)
        print("  ✓ 使用学习率 1e-5 进行微调")
        
        # 微调 10-20 epochs
        best_val_ic_stage2 = best_val_ic
        patience_counter = 0
        
        for epoch in range(20):
            train_loss = trainer_stage2.train_epoch(train_loader)
            val_loss, val_ic, fast_ic = trainer_stage2.validate(val_loader)
            
            print(f"Stage2 Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | Fusion IC: {val_ic:7.4f} | Fast IC: {fast_ic:7.4f}")
            
            if val_ic > best_val_ic_stage2:
                print(f"  ★ New Best IC (Stage2): {val_ic:.4f} (was {best_val_ic_stage2:.4f}) -> Saving Model...")
                best_val_ic_stage2 = val_ic
                best_val_ic = val_ic  # 更新全局最佳
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"\n早停于 Stage2 Epoch {epoch+1}")
                    break
        
        print(f"\n✓ 阶段二微调完成，最佳IC: {best_val_ic:.4f}")
    
    # 6. XAI分析
    print("\n" + "="*70)
    print("步骤4: XAI可解释性分析")
    print("="*70)
    
    # 显式加载最佳模型并确认
    print(f"加载最佳模型 (Best Val IC: {best_val_ic:.4f})...")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth')))
    
    xai = XAIAnalyzer(model, device)
    gamma_values, predictions = xai.extract_gamma_values(test_loader)
    
    # 加载SP500数据用于绘图
    # 动态查找sp500.csv文件
    parent_dir = os.path.dirname(BASE_DIR)
    sp500_candidates = [
        os.path.join(BASE_DIR, 'sp500.csv'),
        os.path.join(BASE_DIR, 'sp500_with_indicators.csv'),  # 备选
        os.path.join(parent_dir, 'sp500.csv'),
        os.path.join(parent_dir, 'teach', 'sp500.csv'),
        os.path.join(parent_dir, 'slow_channel', 'sp500.csv')
    ]
    
    sp500_path = None
    for candidate in sp500_candidates:
        if os.path.exists(candidate):
            sp500_path = candidate
            break
    
    if sp500_path is None:
        print("⚠️ 警告: 未找到sp500.csv，跳过 Correction 热力图绘制")
        sp500_test_close = np.random.rand(len(dates_test)) * 4000 + 3000  # Mock数据
    else:
        sp500_df = pd.read_csv(sp500_path)
        sp500_df['date'] = pd.to_datetime(sp500_df['date'].astype(str).str.replace('/', '-'))
        sp500_test = sp500_df[sp500_df['date'].isin(dates_test)]
        sp500_test_close = sp500_test['close'].values if len(sp500_test) > 0 else np.random.rand(len(dates_test)) * 4000 + 3000
    
    # 绘制 Correction 热力图
    xai.plot_gamma_heatmap(
        gamma_values=gamma_values,
        dates=dates_test,
        sp500_prices=sp500_test_close,
        save_path=os.path.join(OUTPUT_DIR, 'gamma_heatmap.png')
    )
    
    # ========== 增强功能：完整评估与可视化 ==========
    if calculate_comprehensive_metrics is not None:
        print("\n" + "="*70)
        print("增强评估：完整性能分析")
        print("="*70)
        
        # 1. 获取测试集的快通道和融合预测
        model.eval()
        fast_preds_test, fusion_preds_test = [], []
        
        with torch.no_grad():
            for x_fast, x_slow, _ in test_loader:
                x_fast = x_fast.to(device)
                x_slow = x_slow.to(device)
                
                # 快通道预测
                fast_pred = model.fast_channel(x_fast)
                fast_preds_test.append(fast_pred.cpu().numpy())
                
                # 融合预测
                fusion_pred = model(x_fast, x_slow)
                fusion_preds_test.append(fusion_pred.cpu().numpy())
        
        fast_preds_test = np.concatenate(fast_preds_test)
        fusion_preds_test = np.concatenate(fusion_preds_test)
        
        # 2. 计算完整指标
        print("\n📊 完整性能指标:")
        
        fast_metrics = calculate_comprehensive_metrics(fast_preds_test, y_test)
        fusion_metrics = calculate_comprehensive_metrics(fusion_preds_test, y_test)
        
        print(f"\n  Fast-Only:")
        for k, v in fast_metrics.items():
            print(f"    {k}: {v:.4f}")
        
        print(f"\n  Fusion:")
        for k, v in fusion_metrics.items():
            print(f"    {k}: {v:.4f}")
        
        # 3. Synergy Gap 分析 (修正后的定义)
        # 正确公式: Synergy Gap = IC_Fusion - max(IC_Fast, IC_Slow)
        baseline_max_ic = max(fast_metrics['IC'], 0)  # Slow Channel 单独 IC 暂无，用0代替
        synergy_gap = fusion_metrics['IC'] - baseline_max_ic
        fusion_lift = ((fusion_metrics['IC'] - fast_metrics['IC']) / abs(fast_metrics['IC'])) * 100 if fast_metrics['IC'] != 0 else 0
        
        print(f"\n🎯 融合效果分析:")
        print(f"  Synergy Gap: {synergy_gap:.4f}")
        if synergy_gap > 0:
            print(f"  ✅ 融合成功! IC超越Baseline {synergy_gap:.4f}")
        elif synergy_gap == 0:
            print(f"  ⚖️ 融合与Baseline持平")
        else:
            print(f"  ⚠️ 负迁移! IC低于Baseline {abs(synergy_gap):.4f}")
        print(f"  Fusion Lift: {fusion_lift:+.2f}%")
        
        # 4. Event Window Analysis (如果有气候数据)
        if EventWindowAnalyzer is not None:
            try:
                # 尝试加载气候数据
                climate_file_candidates = [
                    os.path.join(BASE_DIR, 'Climate_Risk_Index.xlsx'),
                    os.path.join(parent_dir, 'slow_channel', 'Climate_Risk_Index.xlsx')
                ]
                
                climate_df = None
                for cf in climate_file_candidates:
                    if os.path.exists(cf):
                        print(f"  正在加载气候数据: {cf}")
                        # 关键修复: header=7 (跳过元数据行)
                        climate_df = pd.read_excel(cf, header=7)
                        # 删除全NaN行
                        climate_df = climate_df.dropna(subset=[climate_df.columns[0]])
                        # 智能寻找日期列
                        date_col = None
                        for col in ['date', 'Date', 'DATE', 'time', 'Time', 'datetime']:
                            if col in climate_df.columns:
                                date_col = col
                                break
                        
                        if date_col:
                            climate_df['date'] = pd.to_datetime(climate_df[date_col])
                        else:
                            print(f"⚠️ 警告: 气候数据中未找到日期列 (尝试了 date, Date, DATE等)")
                        break
                
                
                if climate_df is not None:
                    print(f"\n🌍 Event Window Analysis (气候风险):")
                    
                    # 1. 智能寻找日期列
                    date_col = None
                    # 清洗列名 (去除空格)
                    climate_df.columns = climate_df.columns.str.strip()
                    
                    for col in ['Date (dd/mm/yyyy)', 'date', 'Date', 'DATE']:
                        if col in climate_df.columns:
                            date_col = col
                            break
                    
                    if date_col:
                        climate_df['date'] = pd.to_datetime(climate_df[date_col])
                        print(f"  ✓ Found date column: '{date_col}'")
                        
                        # 2. 列名映射 (对齐 data_factory)
                        cols_map = {
                            'Transition concern': 'Transition_concern',
                            'Transition Risk Index (TRI)': 'TRI',
                            'Physical concern': 'Physical_concern',
                            'Physical Risk Index (PRI)': 'PRI'
                        }
                        climate_df = climate_df.rename(columns=cols_map)
                        
                        event_analyzer = EventWindowAnalyzer(climate_df)
                        
                        # 确保测试集日期也是datetime格式
                        dates_test_pd = pd.to_datetime(dates_test)
                        
                        event_results = event_analyzer.analyze(
                        predictions={'Fast-Only': fast_preds_test.flatten(), 
                                   'Fusion': fusion_preds_test.flatten()},
                        actual=y_test.flatten(),
                        dates=dates_test_pd
                    )
                    
                    for model_name, results in event_results.items():
                        print(f"\n  {model_name}:")
                        print(f"    高风险期IC: {results['High_Risk_IC']:.4f}")
                        print(f"    低风险期IC: {results['Low_Risk_IC']:.4f}")
                        print(f"    风险敏感度: {results['Risk_Sensitivity']:.4f}")
                    
                    # 绘制顶刊级Event Window可视化
                    if EnhancedXAIVisualizer is not None:
                        try:
                            event_fig_path = os.path.join(OUTPUT_DIR, 'event_window_analysis.png')
                            EnhancedXAIVisualizer.plot_event_window_comparison(
                                event_results=event_results,
                                fast_preds=fast_preds_test,
                                fusion_preds=fusion_preds_test,
                                actual=y_test,
                                dates=dates_test_pd,
                                climate_risk=climate_df,
                                save_path=event_fig_path
                            )
                        except Exception as e:
                            print(f"\n⚠️ Event Window可视化生成失败: {e}")
                            import traceback
                            traceback.print_exc()
            except Exception as e:
                print(f"\n⚠️ Event Window分析跳过: {e}")
                import traceback
                traceback.print_exc()
        
        # 5. 绘制性能雷达图
        if EnhancedXAIVisualizer is not None:
            try:
                metrics_dict = {
                    'Fast-Only': fast_metrics,
                    'Fusion': fusion_metrics
                }
                
                radar_path = os.path.join(OUTPUT_DIR, 'performance_radar.png')
                EnhancedXAIVisualizer.plot_performance_radar(metrics_dict, radar_path)
            except Exception as e:
                print(f"\n⚠️ 性能雷达图生成失败: {e}")
    # ================================================
    
    print("\n" + "="*70)
    print("✓ 训练完成！")
    print("="*70)
    print(f"  最佳验证IC: {best_val_ic:.4f}")
    print(f"  输出目录: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
