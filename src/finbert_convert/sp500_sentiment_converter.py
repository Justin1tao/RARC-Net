"""
SP500 新闻标题情绪指数转换器
===============================
基于 FinBERT 模型，将新闻标题转换为日度情绪指数

输入文件: sp500_headlines_2008_2024.csv (Title, Date, CP)
输出文件: sp500_sentiment_index.csv (Date, ESG_Sentiment, data_source)
时间范围: 2020-06-12 至 2024-01-01
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from typing import Optional

# ==========================================
# 1. 全局配置 (Global Configuration)
# ==========================================
CONFIG = {
    "INPUT_FILE": "sp500_headlines_2008_2024.csv",
    "OUTPUT_FILE": "sp500_sentiment_index.csv",
    "MODEL_NAME": "ProsusAI/finbert",
    "BATCH_SIZE": 32,
    "MAX_LEN": 64,
    "EPSILON": 1e-8,
    "COL_TITLE": "Title",       # 新闻标题列名
    "COL_DATE": "Date",         # 日期列名
    "START_DATE": "2020-06-12", # 开始日期
    "END_DATE": "2024-01-01",   # 结束日期
}


def setup_device():
    """设置计算设备 (GPU/CPU)"""
    if torch.cuda.is_available():
        print(f"✅ 使用设备: GPU ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    else:
        print("⚠️ 未检测到GPU，使用CPU (处理速度较慢)")
        return torch.device("cpu")


def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    加载并预处理数据
    
    处理步骤:
    1. 尝试多种编码读取CSV文件
    2. 转换日期格式
    3. 按时间范围筛选数据
    4. 清洗标题文本
    """
    print(f"\n[1/5] 加载数据: {file_path}")
    
    # 尝试多种编码
    encodings = ['utf-8', 'ISO-8859-1', 'latin1', 'cp1252', 'gbk']
    df = None
    
    for encoding in encodings:
        try:
            print(f"   尝试编码: {encoding}")
            df = pd.read_csv(
                file_path, 
                encoding=encoding,
                on_bad_lines='skip',
                engine='python'
            )
            print(f"   ✅ 成功使用 {encoding} 编码加载")
            break
        except Exception as e:
            print(f"   ❌ {encoding} 失败: {str(e)[:80]}")
            continue
    
    if df is None:
        raise Exception("无法加载CSV文件，尝试了所有编码方式")
    
    print(f"   原始数据行数: {len(df)}")

    # 日期转换
    df[CONFIG["COL_DATE"]] = pd.to_datetime(df[CONFIG["COL_DATE"]], errors='coerce')

    # 基础清洗
    df = df.dropna(subset=[CONFIG["COL_DATE"], CONFIG["COL_TITLE"]])
    df[CONFIG["COL_TITLE"]] = df[CONFIG["COL_TITLE"]].astype(str).str.strip()
    df = df[df[CONFIG["COL_TITLE"]].str.len() > 3]  # 去除过短标题
    
    # 按时间范围筛选
    start_date = pd.to_datetime(CONFIG["START_DATE"])
    end_date = pd.to_datetime(CONFIG["END_DATE"])
    
    df = df[(df[CONFIG["COL_DATE"]] >= start_date) & (df[CONFIG["COL_DATE"]] <= end_date)]
    
    print(f"   清洗后数据行数: {len(df)}")
    print(f"   时间范围: {CONFIG['START_DATE']} 至 {CONFIG['END_DATE']}")

    return df


def get_sentiment_probabilities(texts, tokenizer, model, device):
    """
    使用 FinBERT 批量获取情感概率
    
    FinBERT 原理:
    - FinBERT 是基于 BERT 的金融领域预训练模型
    - 输出三类情感概率: positive, negative, neutral
    - 使用 softmax 将模型输出转换为概率分布
    
    返回: numpy array, shape (N, 3), 每行为 [positive, negative, neutral] 概率
    """
    probs_list = []
    model.eval()
    print(f"\n[3/5] 开始 FinBERT 推理 (共 {len(texts)} 条新闻)...")

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), CONFIG["BATCH_SIZE"]), unit="batch"):
            batch_texts = texts[i: i + CONFIG["BATCH_SIZE"]]
            
            # Tokenize: 将文本转换为模型可处理的张量
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=CONFIG["MAX_LEN"]
            ).to(device)
            
            # 模型推理
            outputs = model(**inputs)
            
            # Softmax: 将 logits 转换为概率
            probs = F.softmax(outputs.logits, dim=-1)
            probs_list.append(probs.cpu().numpy())

    return np.concatenate(probs_list, axis=0)


def calculate_daily_sentiment(df: pd.DataFrame) -> pd.Series:
    """
    计算日度情绪指数
    
    情绪得分公式:
        sentiment_score = (P_positive - P_negative) / (P_positive + P_negative + ε)
    
    其中:
    - P_positive: 正面情绪概率
    - P_negative: 负面情绪概率
    - ε: 防止除零的小常数
    
    日度指数: 当日所有新闻情绪得分的算术平均
    """
    # 文章级别情绪得分
    df['sentiment_score'] = (df['pos_prob'] - df['neg_prob']) / \
                            (df['pos_prob'] + df['neg_prob'] + CONFIG["EPSILON"])

    # 按日期聚合: 等权平均
    daily_series = df.groupby(CONFIG["COL_DATE"])['sentiment_score'].mean()

    return daily_series


def fill_missing_dates(daily_series: pd.Series, start_date: str, end_date: str) -> pd.DataFrame:
    """
    填充缺失日期
    
    策略:
    1. 有新闻的日期: 使用计算的情绪指数，标记为 "All_News"
    2. 无新闻的日期: 使用前向填充 (forward fill)，标记为 "Forward_Fill"
    """
    print("\n[4/5] 填充缺失日期...")
    
    # 创建完整日期范围
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 创建结果 DataFrame
    result_df = pd.DataFrame(index=date_range)
    result_df.index.name = 'Date'
    
    # 对齐原始数据
    result_df['ESG_Sentiment'] = daily_series
    
    # 标记数据来源
    result_df['data_source'] = 'Forward_Fill'
    result_df.loc[result_df['ESG_Sentiment'].notna(), 'data_source'] = 'All_News'
    
    # 前向填充
    result_df['ESG_Sentiment'] = result_df['ESG_Sentiment'].ffill()
    
    # 统计信息
    news_days = (result_df['data_source'] == 'All_News').sum()
    fill_days = (result_df['data_source'] == 'Forward_Fill').sum()
    
    print(f"   -> 有新闻日期: {news_days} 天")
    print(f"   -> 前向填充日期: {fill_days} 天")
    print(f"   -> 总计: {len(result_df)} 天")
    
    return result_df


def main():
    """主程序入口"""
    print("=" * 60)
    print("SP500 新闻情绪指数转换器 (基于 FinBERT)")
    print("=" * 60)
    
    # 1. 设置计算设备
    device = setup_device()

    # 2. 加载数据
    if not os.path.exists(CONFIG["INPUT_FILE"]):
        print(f"❌ 找不到输入文件: {CONFIG['INPUT_FILE']}")
        return
    
    df = load_and_preprocess_data(CONFIG["INPUT_FILE"])
    
    if len(df) == 0:
        print("❌ 筛选后数据为空，程序终止。")
        return
    
    print(f"\n[2/5] 数据统计:")
    print(f"   -> 新闻条数: {len(df)}")
    print(f"   -> 日期范围: {df[CONFIG['COL_DATE']].min().date()} 至 {df[CONFIG['COL_DATE']].max().date()}")

    # 3. 加载 FinBERT 模型
    print("\n正在加载 FinBERT 模型...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["MODEL_NAME"])
        model = AutoModelForSequenceClassification.from_pretrained(CONFIG["MODEL_NAME"]).to(device)
        print("   ✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载错误: {e}")
        return

    # 获取标签索引
    label2id = model.config.label2id
    pos_idx = label2id.get('positive', 0)
    neg_idx = label2id.get('negative', 1)
    
    print(f"   -> 标签映射: positive={pos_idx}, negative={neg_idx}")

    # 4. FinBERT 推理
    texts = df[CONFIG["COL_TITLE"]].tolist()
    probs = get_sentiment_probabilities(texts, tokenizer, model, device)
    
    # 提取正负情绪概率
    df['pos_prob'] = probs[:, pos_idx]
    df['neg_prob'] = probs[:, neg_idx]

    # 5. 计算日度情绪指数
    daily_series = calculate_daily_sentiment(df)

    # 6. 填充缺失日期
    result_df = fill_missing_dates(
        daily_series, 
        CONFIG["START_DATE"], 
        CONFIG["END_DATE"]
    )

    # 7. 保存结果
    print(f"\n[5/5] 保存文件至: {CONFIG['OUTPUT_FILE']}")
    result_df.to_csv(CONFIG["OUTPUT_FILE"])

    # 输出摘要
    print("\n" + "=" * 60)
    print("✅ 处理完成!")
    print("=" * 60)
    print(f"   输入文件: {CONFIG['INPUT_FILE']}")
    print(f"   输出文件: {CONFIG['OUTPUT_FILE']}")
    print(f"   时间范围: {CONFIG['START_DATE']} 至 {CONFIG['END_DATE']}")
    print(f"   处理新闻: {len(df)} 条")
    print(f"   生成天数: {len(result_df)} 天")
    print(f"   情绪均值: {result_df['ESG_Sentiment'].mean():.4f}")
    print(f"   情绪范围: [{result_df['ESG_Sentiment'].min():.4f}, {result_df['ESG_Sentiment'].max():.4f}]")


if __name__ == "__main__":
    main()
