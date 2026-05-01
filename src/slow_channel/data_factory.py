"""
慢通道数据工程模块 (Slow Channel Data Factory)

核心职责：
1. 加载、对齐、合并所有宏观数据源
2. 严格执行防泄漏协议（shift宏观数据）
3. NaN诊断与分级处理
4. 生成时序训练数据集

数据源（慢通道独立训练，不含快通道预测）：
- SP500基础行情 (2002-2024)
- 成交量因子 (volume_ratio, volume_zscore)
- 利率因子 (Nominal_Rate, Term_Spread, Credit_Spread)
- EPU经济政策不确定性 (日度)
- VIX恐慌指数 (日度)
- CPI通胀指标 (月度 -> 日度)
- 气候风险指数 (TRI, PRI, Concerns)

注意：f_fast快通道预测仅在融合层使用，不作为慢通道输入特征

Author: Slow Channel Team
Date: 2025-12-22
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class SlowChannelDataFactory:
    """
    慢通道数据工厂：负责数据加载、防泄漏处理和序列生成
    
    支持任务类型:
    - 'volatility': 波动率预测任务（慢通道推荐），预测未来20日波动率
    - 'regression': 回归任务，预测log_return数值
    - 'binary': 二分类任务，预测涨/跌 (target = 1 if return > 0 else 0)
    - 'ternary': 三分类任务，预测大涨/震荡/大跌 (threshold=0.005)
    """
    
    def __init__(
        self, 
        base_dir: str = None, 
        seq_length: int = 60,
        predict_horizon: int = 1,
        task_type: str = 'regression',
        ternary_threshold: float = 0.005
    ):
        """
        初始化数据工厂
        
        Args:
            base_dir: 数据目录根路径
            seq_length: 输入时序窗口长度（默认60个交易日≈一季度，建议30-90天）
            predict_horizon: 预测窗口（默认1天，可设5-20天捕捉中期趋势）
            task_type: 任务类型 ('regression', 'binary', 'ternary')
            ternary_threshold: 三分类阈值（默认0.5%）
        """
        if base_dir is None:
            # 获取当前文件所在目录的父目录
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.base_dir = base_dir
        self.seq_length = seq_length
        self.predict_horizon = predict_horizon
        self.task_type = task_type
        self.ternary_threshold = ternary_threshold
        
        self.scaler_y = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.feature_cols = None
        
        print(f"[DataFactory] 任务类型: {task_type}, 输入窗口: {seq_length}天, 预测窗口: {predict_horizon}天")

    def _find_file(self, filename: str) -> str:
        """在整理版仓库的数据目录中查找文件，兼容旧的扁平目录结构。"""
        candidates = [
            os.path.join(self.base_dir, filename),
            os.path.join(self.base_dir, 'data', 'processed', filename),
            os.path.join(self.base_dir, 'data', 'raw', filename),
            os.path.join(os.path.dirname(self.base_dir), 'data', 'processed', filename),
            os.path.join(os.path.dirname(self.base_dir), 'data', 'raw', filename),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]
        
    def load_sp500(self) -> pd.DataFrame:
        """加载SP500基础行情数据"""
        path = self._find_file('sp500_with_indicators.csv')
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'].str.replace('/', '-'))
        df = df.set_index('date').sort_index()
        
        # 防泄漏：change列必须shift(1)
        # 将百分比字符串转为浮点数
        df['change'] = df['change'].str.replace('%', '').astype(float) / 100
        df['change'] = df['change'].shift(1)
        
        # 选择需要的列
        cols = ['open', 'close', 'high', 'low', 'change']
        print(f"[SP500] 加载完成: {df.shape[0]}行, 日期范围: {df.index.min()} ~ {df.index.max()}")
        return df[cols]
    
    def load_volume(self) -> pd.DataFrame:
        """
        加载SP500成交量数据（量在价先核心特征）
        
        数据处理：
        1. 对数变换处理偏度（原始成交量偏度~2.4）
        2. 使用相对指标避免训练/测试集数据漂移
        """
        path = self._find_file('sp500_volume.csv')
        
        if not os.path.exists(path):
            print(f"[Volume] ⚠️ 未找到成交量文件: {path}")
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # 对数变换处理偏度
        df['log_volume'] = np.log1p(df['volume'])
        
        # 量比（今日对数成交量 / 5日均值）- 捕捉放量信号
        log_vol_ma5 = df['log_volume'].rolling(window=5, min_periods=1).mean()
        df['volume_ratio'] = (df['log_volume'] / log_vol_ma5.replace(0, 1e-8)) - 1
        df['volume_ratio'] = df['volume_ratio'].clip(-1, 1)
        
        # 成交量Z-score（20日滚动标准化）- 捕捉异常放量
        log_vol_mean_20 = df['log_volume'].rolling(20, min_periods=5).mean()
        log_vol_std_20 = df['log_volume'].rolling(20, min_periods=5).std()
        df['volume_zscore'] = ((df['log_volume'] - log_vol_mean_20) / (log_vol_std_20 + 1e-8)).clip(-3, 3)
        
        # 只返回处理后的特征列
        result = df[['volume_ratio', 'volume_zscore']]
        print(f"[Volume] 加载完成: {result.shape[0]}行, 特征: volume_ratio, volume_zscore")
        return result
    
    def load_rate_data(self) -> pd.DataFrame:
        """
        加载宏观利率数据（货币政策核心特征）
        
        数据源: sp500_slow.csv
        特征:
        - Nominal_Rate: 名义利率（资金成本）
        - Term_Spread: 期限利差（收益率曲线斜率）
        - Credit_Spread: 信用利差（系统性信用风险）
        
        处理: 一阶差分 + 20日滚动Z-score标准化
        """
        path = self._find_file('sp500_slow.csv')
        
        if not os.path.exists(path):
            print(f"[Rate] ⚠️ 未找到利率文件: {path}")
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['DATE'])
        df = df.set_index('date').sort_index()
        
        result_cols = []
        for col in ['Nominal_Rate', 'Term_Spread', 'Credit_Spread']:
            if col not in df.columns:
                print(f"[Rate] ⚠️ 缺少列: {col}")
                continue
            
            # 一阶差分：捕捉"变化"而非"水平"
            diff_col = f'{col}_diff'
            df[diff_col] = df[col].diff()
            
            # 20日滚动Z-score标准化
            mean = df[diff_col].rolling(20, min_periods=5).mean()
            std = df[diff_col].rolling(20, min_periods=5).std()
            zscore_col = f'{col}_zscore'
            df[zscore_col] = ((df[diff_col] - mean) / (std + 1e-8)).clip(-3, 3)
            result_cols.append(zscore_col)
        
        if not result_cols:
            return pd.DataFrame()
        
        result = df[result_cols]
        print(f"[Rate] 加载完成: {result.shape[0]}行, 特征: {', '.join(result_cols)}")
        return result
    
    def load_epu(self) -> pd.DataFrame:
        """加载EPU经济政策不确定性指数"""
        path = self._find_file('EPU.csv')
        df = pd.read_csv(path)
        
        # 构造日期
        df['date'] = pd.to_datetime(
            df['year'].astype(str) + '-' + 
            df['month'].astype(str).str.zfill(2) + '-' + 
            df['day'].astype(str).str.zfill(2)
        )
        df = df.set_index('date').sort_index()
        
        # 防泄漏：shift(1)，假设当日EPU在收盘后才公布
        df['EPU'] = df['daily_policy_index'].shift(1)
        
        print(f"[EPU] 加载完成: {df.shape[0]}行, 日期范围: {df.index.min()} ~ {df.index.max()}")
        return df[['EPU']]
    
    def load_vix(self) -> pd.DataFrame:
        """加载VIX恐慌指数"""
        path = self._find_file('VIX.csv')
        if not os.path.exists(path):
            print(f"[VIX] ⚠️ 未找到VIX文件: {path}，将跳过VIX特征")
            return pd.DataFrame()
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['DATE'])
        df = df.set_index('date').sort_index()
        df = df.rename(columns={'VIX': 'VIX'})
        
        # VIX是实时数据，无需额外滞后，但为保守起见也shift(1)
        df['VIX'] = df['VIX'].shift(1)
        
        print(f"[VIX] 加载完成: {df.shape[0]}行, 日期范围: {df.index.min()} ~ {df.index.max()}")
        return df[['VIX']]
    
    def load_cpi(self) -> pd.DataFrame:
        """加载CPI通胀指标（月度->日度）"""
        path = self._find_file('medium_CPI.csv')
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['observation_date'].str.replace('/', '-'))
        df = df.set_index('date').sort_index()
        
        # 防泄漏：月度数据滞后1个月（T月数据T+1月中旬才发布）
        df['CPI'] = df['CPI'].shift(1)
        
        # 月度->日度：前向填充
        df = df[['CPI']].resample('D').ffill()
        
        print(f"[CPI] 加载完成(月->日): {df.shape[0]}行, 日期范围: {df.index.min()} ~ {df.index.max()}")
        return df
    
    def load_climate_risk(self) -> pd.DataFrame:
        """加载气候风险指数"""
        path = self._find_file('Climate_Risk_Index.xlsx')
        df = pd.read_excel(path, header=7)
        
        # 清洗：删除日期为NaN的行（末尾脏数据）
        df = df.dropna(subset=['Date (dd/mm/yyyy)']).copy()
        
        df['date'] = pd.to_datetime(df['Date (dd/mm/yyyy)'])
        df = df.set_index('date').sort_index()
        
        # 选择关键列并重命名
        cols_map = {
            'Transition concern': 'Transition_concern',
            'Transition Risk Index (TRI)': 'TRI',
            'Physical concern': 'Physical_concern',
            'Physical Risk Index (PRI)': 'PRI'
        }
        df = df.rename(columns=cols_map)
        climate_cols = ['TRI', 'PRI', 'Transition_concern', 'Physical_concern']
        
        # 防泄漏：shift(1)
        for col in climate_cols:
            df[col] = df[col].shift(1)
        
        print(f"[Climate Risk] 加载完成: {df.shape[0]}行, 日期范围: {df.index.min()} ~ {df.index.max()}")
        return df[climate_cols]
    
    # 注意：已删除load_fast_predictions函数
    # f_fast快通道预测不再作为慢通道输入特征
    # 慢通道独立学习宏观规律，f_fast仅在融合层使用
    
    def handle_nan(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """
        分级NaN处理：先诊断，后治理
        
        策略：
        - NaN比例>30%：警告，需人工审核
        - NaN比例>0且<=30%：前向填充+后向填充
        """
        nan_ratio = df.isna().sum() / len(df)
        has_nan = nan_ratio[nan_ratio > 0]
        
        if len(has_nan) > 0:
            print(f"  [{source_name}] NaN诊断:")
            for col, ratio in has_nan.items():
                if ratio > 0.3:
                    print(f"    ⚠️ {col}: {ratio:.2%} (>30%, 需人工审核)")
                else:
                    print(f"    • {col}: {ratio:.2%} (ffill+bfill处理)")
        
        # 分级处理
        for col in df.columns:
            ratio = nan_ratio[col]
            if ratio > 0 and ratio <= 0.3:
                df[col] = df[col].ffill().bfill()
            elif ratio > 0.3:
                # 高NaN列：用0填充（或可选丢弃）
                df[col] = df[col].fillna(0)
        
        return df
    
    def validate_no_nan(self, df: pd.DataFrame, stage_name: str):
        """验证无NaN"""
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            nan_cols = df.columns[df.isna().any()].tolist()
            raise ValueError(f"[{stage_name}] 仍存在{nan_count}个NaN! 问题列: {nan_cols}")
        print(f"  ✓ [{stage_name}] NaN检查通过")
    
    def build_dataset(self) -> pd.DataFrame:
        """
        构建完整数据集
        
        Returns:
            合并后的DataFrame，包含所有特征和目标变量
        """
        print("=" * 60)
        print("开始构建慢通道数据集...")
        print("=" * 60)
        
        # 1. 加载所有数据源
        sp500 = self.load_sp500()
        volume = self.load_volume()  # 成交量特征（量在价先）
        rate_data = self.load_rate_data()  # 利率特征（货币政策）
        epu = self.load_epu()
        vix = self.load_vix()
        cpi = self.load_cpi()
        climate = self.load_climate_risk()
        # 注意：f_fast不作为慢通道训练特征，应在融合层使用
        # 慢通道独立学习宏观规律，不依赖快通道预测
        
        # 2. 合并数据（以SP500日期为基准）
        print("\n合并数据...")
        df = sp500.copy()
        if not volume.empty:
            df = df.join(volume, how='left')  # 合并成交量特征
        if not rate_data.empty:
            df = df.join(rate_data, how='left')  # 合并利率特征
        df = df.join(epu, how='left')
        df = df.join(vix, how='left')
        df = df.join(cpi, how='left')
        df = df.join(climate, how='left')
        
        print(f"  合并后形状: {df.shape}")
        
        # 3. 分级NaN处理
        print("\nNaN处理...")
        df = self.handle_nan(df, "合并数据")
        
        # 4. 裁剪到有效区间（气候数据开始日期之后，2005年）
        # 慢通道独立训练，使用完整宏观数据
        climate_start = climate.index.min() if not climate.empty else pd.Timestamp('2005-01-03')
        df = df[df.index >= climate_start].copy()
        print(f"\n裁剪到气候数据起始日期后: {df.shape[0]}行, {climate_start} ~ {df.index.max()}")
        
        # 再次处理裁剪后可能的边界NaN
        df = self.handle_nan(df, "裁剪后数据")
        
        # 5. 构建目标变量：支持多预测窗口和波动率预测
        # 先计算log_return用于各种target
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # 根据任务类型构建标签
        if self.task_type == 'volatility':
            # 波动率预测任务（慢通道推荐使用）
            # 宏观因子对波动率有更强预测力
            df['target'] = df['log_return'].rolling(20, min_periods=5).std()
            df['target'] = df['target'].shift(-1)  # 预测未来20日波动率
            print(f"  波动率预测任务: 目标为未来20日波动率, 范围 {df['target'].min():.4f} ~ {df['target'].max():.4f}")
        elif self.task_type == 'binary':
            # 二分类：涨=1, 跌=0
            df['target'] = (df['log_return'].shift(-1) > 0).astype(int)
            print(f"  二分类标签分布: 涨={df['target'].sum()}, 跌={(~df['target'].astype(bool)).sum()}")
        elif self.task_type == 'ternary':
            # 三分类：大涨=2, 震荡=1, 大跌=0
            next_return = df['log_return'].shift(-1)
            df['target'] = 1  # 默认震荡
            df.loc[next_return > self.ternary_threshold, 'target'] = 2   # 大涨
            df.loc[next_return < -self.ternary_threshold, 'target'] = 0  # 大跌
            print(f"  三分类标签分布: 大涨={len(df[df['target']==2])}, 震荡={len(df[df['target']==1])}, 大跌={len(df[df['target']==0])}")
        else:
            # 回归任务（预测次日收益率）
            if self.predict_horizon == 1:
                df['target'] = df['log_return'].shift(-1)
            else:
                # 多日预测窗口：计算未来N天的累积收益率
                df['target'] = np.log(df['close'].shift(-self.predict_horizon) / df['close'])
        
        # 删除NaN行（首行或末尾N行）
        df = df.dropna(subset=['log_return', 'target'])
        
        # 6. 最终验证
        self.validate_no_nan(df, "最终数据")
        
        # 7. 记录特征列
        preferred_feature_cols = [
            'open', 'close', 'high', 'low', 'change',  # 大盘因子 (5)
            'volume_ratio', 'volume_zscore',            # 成交量因子 (2)
            'Nominal_Rate_zscore', 'Term_Spread_zscore', 'Credit_Spread_zscore',  # 利率因子 (3)
            'CPI', 'VIX', 'EPU',                        # 宏观因子 (3)
            'TRI', 'PRI', 'Transition_concern', 'Physical_concern',  # 气候因子 (4)
        ]  # 共17个特征（不含f_fast，f_fast仅在融合层使用）
        self.feature_cols = [col for col in preferred_feature_cols if col in df.columns]
        missing_cols = sorted(set(preferred_feature_cols) - set(self.feature_cols))
        if missing_cols:
            print(f"  ⚠️ 跳过缺失特征: {missing_cols}")
        
        print(f"\n✓ 数据集构建完成!")
        print(f"  任务类型: {self.task_type}")
        print(f"  预测窗口: {self.predict_horizon}天")
        print(f"  形状: {df.shape}")
        print(f"  日期范围: {df.index.min()} ~ {df.index.max()}")
        print(f"  特征列: {self.feature_cols}")
        print("=" * 60)
        
        self.data = df
        return df
    
    def prepare_sequences(
        self, 
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        准备时序训练数据
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
        
        Returns:
            X_train, y_train, X_val, y_val, X_test, y_test
        """
        if self.data is None:
            self.build_dataset()
        
        df = self.data.copy()
        
        # 提取特征
        X_raw = df[self.feature_cols].values
        
        # 根据任务类型处理目标变量
        if self.task_type in ['binary', 'ternary']:
            # 分类任务：直接使用整数标签，不归一化
            y_raw = df['target'].values.astype(np.int64)
            self._is_classification = True
        else:
            # 回归任务：即使是回归，也可以选择不归一化（直接预测收益率）
            # 或者使用Scaler。这里为了端到端训练的物理意义明确，我们直接使用原始值
            # 因为y如果是log_return，本身数值就很小，不需要scaler
            # 如果是volatility，可能需要。
            # 为了兼容性，如果是regression且没有明确要求scaler，我们不scaler
            y_raw = df['target'].values.reshape(-1, 1).astype(np.float32)
            self._is_classification = False
        
        # 归一化特征
        X_scaled = self.scaler_X.fit_transform(X_raw)
        
        # 生成时序序列
        X_seq, y_seq = [], []
        dates_seq = []
        
        indices = df.index # 获取日期索引
        
        for i in range(self.seq_length, len(X_scaled)):
            X_seq.append(X_scaled[i - self.seq_length:i])
            y_seq.append(y_raw[i])
            dates_seq.append(indices[i]) # 记录对应样本的日期
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        dates_seq = np.array(dates_seq)
        
        # 时序划分（不shuffle）
        n_samples = len(X_seq)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        X_train, y_train = X_seq[:train_end], y_seq[:train_end]
        d_train = dates_seq[:train_end]
        
        X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
        d_val = dates_seq[train_end:val_end]
        
        X_test, y_test = X_seq[val_end:], y_seq[val_end:]
        d_test = dates_seq[val_end:]
        
        # 获取测试集日期索引 (兼容旧接口)
        self.test_dates = df.index[self.seq_length + val_end:]
        
        print(f"\n数据集划分 (seq_length={self.seq_length}):")
        print(f"  训练集: {X_train.shape[0]} 样本")
        print(f"  验证集: {X_val.shape[0]} 样本")
        print(f"  测试集: {X_test.shape[0]} 样本")
        
        # 返回 9 个值
        return X_train, y_train, d_train, X_val, y_val, d_val, X_test, y_test, d_test
    
    def inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """将归一化的y逆变换回原始尺度"""
        return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    
    def get_test_dates(self) -> pd.DatetimeIndex:
        """获取测试集日期索引"""
        return self.test_dates
    
    def get_all_dates(self) -> pd.DatetimeIndex:
        """获取全部数据集日期索引（用于导出embedding）"""
        if self.data is None:
            raise ValueError("请先调用 build_dataset()")
        # 从seq_length开始，因为前面的数据用于构建序列
        return self.data.index[self.seq_length:]
    
    def prepare_all_data(self) -> np.ndarray:
        """
        准备全部数据的时序序列（用于导出embedding）
        
        Returns:
            X_all: (N, seq_length, n_features) 全部数据的时序特征
        """
        if self.data is None:
            raise ValueError("请先调用 build_dataset()")
        
        df = self.data
        X_raw = df[self.feature_cols].values
        
        # 使用已拟合的scaler归一化
        X_scaled = self.scaler_X.transform(X_raw)
        
        # 生成时序序列
        X_seq = []
        for i in range(self.seq_length, len(X_scaled)):
            X_seq.append(X_scaled[i - self.seq_length:i])
        
        return np.array(X_seq)
    
    def prepare_all_data_with_dates(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        准备全部数据的时序序列，包含目标变量和日期索引
        用于端到端训练时做快慢通道时间交集过滤
        
        Returns:
            X_all: (N, seq_length, n_features) 全部数据的时序特征
            y_all: (N,) 或 (N,1) 目标变量
            dates: (N,) 日期索引
        """
        if self.data is None:
            self.build_dataset()
        
        df = self.data
        X_raw = df[self.feature_cols].values
        y_raw = df['target'].values
        
        # 使用scaler归一化特征
        X_scaled = self.scaler_X.fit_transform(X_raw)
        
        # 生成时序序列
        X_seq, y_seq, dates_seq = [], [], []
        indices = df.index
        
        for i in range(self.seq_length, len(X_scaled)):
            X_seq.append(X_scaled[i - self.seq_length:i])
            y_seq.append(y_raw[i])
            dates_seq.append(indices[i])
        
        return np.array(X_seq), np.array(y_seq), np.array(dates_seq)


# ================== 测试代码 ==================
if __name__ == "__main__":
    # 测试数据工厂
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    factory = SlowChannelDataFactory(
        base_dir=project_root,
        seq_length=60
    )
    
    # 构建数据集
    df = factory.build_dataset()
    
    # 准备时序数据
    X_train, y_train, X_val, y_val, X_test, y_test = factory.prepare_sequences()
    
    print(f"\n数据形状验证:")
    print(f"  X_train: {X_train.shape}")  # (N, 60, 13)
    print(f"  y_train: {y_train.shape}")  # (N, 1)
