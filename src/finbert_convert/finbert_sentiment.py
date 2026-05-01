import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os
import re
from typing import List, Optional


# ==========================================
# 1. 全局配置 (Global Configuration)
# ==========================================
CONFIG={
    "INPUT_FILE": "esgnews2012_2020.csv",
    "OUTPUT_FILE": "esg_sentiment_index.csv",
    "MODEL_NAME": "ProsusAI/finbert",
    "BATCH_SIZE": 32,
    "MAX_LEN": 64,
    "EPSILON": 1e-8,
    "COL_TITLE": "Article_title",
    "COL_DATE": "Date",
    "MIN_REQUIRED_ROWS": 50000  # 您的硬性数据量要求
}
# ==========================================
# 2. 广义 ESG 关键词库 (High Recall Strategy)
# ==========================================
# 只要包含以下任意词根（不区分大小写），即视为 ESG 相关。
# 为了保证数据量 > 5w，我们纳入了包含隐含 ESG 信息的行业词和风险词。
ESG_KEYWORDS={
    # Environmental (环境)
    "E": [
        "climate", "carbon", "emission", "green", "sustainab", "environment", "energy",
        "pollut", "waste", "clean", "solar", "wind", "renewable", "oil", "gas", "fuel",
        "water", "bio", "nature", "forest", "mining", "resource", "spill", "hazard",
        "toxic", "eco", "electric", "battery", "conservation", "warm", "planet"
    ],
    # Social (社会)
    "S": [
        "labor", "worker", "employee", "human right", "strike", "union", "gender",
        "divers", "inclus", "community", "health", "safety", "consumer", "customer",
        "product", "privacy", "data", "charity", "donat", "social", "welfare",
        "wage", "salary", "pay", "job", "recruit", "talent", "train", "equali",
        "supply chain", "vendor", "supplier", "rights", "demo", "protest"
    ],
    # Governance (治理)
    "G": [
        "board", "executive", "ceo", "cfo", "management", "shareholder", "stakeholder",
        "audit", "complian", "regul", "law", "legal", "suit", "case", "court",
        "fine", "bribe", "corrupt", "fraud", "scandal", "ethics", "transparen",
        "disclos", "report", "govern", "policy", "vote", "elect", "risk",
        "investor", "acquisition", "merger", "insider", "antitrust", "monopoly"
    ],
    # Broad/Financial Impact (泛ESG影响词 - 增加召回率)
    "Broad": [
        "esg", "csr", "sdg", "impact", "responsib", "crisis", "damage", "fine",
        "penalty", "sanction", "ban", "approv", "reject", "standard", "rule",
        "violat", "fail", "success", "perform", "future", "outlook"
    ]
}


def setup_device():
    if torch.cuda.is_available():
        print(f"✅ 使用设备: GPU ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    else:
        print("⚠️ 未检测到GPU，使用CPU")
        return torch.device("cpu")


def build_keyword_regex():
    """将关键词列表编译为高效的正则表达式"""
    all_keywords=[]
    for category in ESG_KEYWORDS.values():
        all_keywords.extend(category)
    # 使用 | 连接所有词，构建 (word1|word2|...)
    pattern='|'.join(all_keywords)
    return pattern


def filter_by_esg(df: pd.DataFrame) -> pd.DataFrame:
    """使用关键词正则筛选 ESG 相关新闻"""
    print("\n[2/5] 正在进行 ESG 关键词筛选...")
    initial_count=len(df)

    # 构建正则模式
    pattern=build_keyword_regex()

    # 向量化筛选 (Case Insensitive)
    # 只要标题中包含关键词列表中的任意一个，就保留
    mask=df[CONFIG["COL_TITLE"]].str.contains(pattern, case=False, regex=True, na=False)
    esg_df=df[mask].copy()

    filtered_count=len(esg_df)
    drop_count=initial_count - filtered_count

    print(f"   -> 关键词库规模: {len(pattern.split('|'))} 个词根")
    print(f"   -> 筛选结果: {filtered_count} 条 (剔除: {drop_count})")

    # 数据量检查
    if filtered_count < CONFIG["MIN_REQUIRED_ROWS"]:
        print(f"⚠️ 警告: 筛选后数据量 ({filtered_count}) 低于目标值 ({CONFIG['MIN_REQUIRED_ROWS']})!")
        print("   -> 建议: 检查 'ESG_KEYWORDS' 列表，或在 filter_by_esg 函数中放宽条件。")
    else:
        print(f"✅ 数据量满足要求 (>= {CONFIG['MIN_REQUIRED_ROWS']})")

    return esg_df


def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    print(f"\n[1/5] 加载数据: {file_path}")
    
    # 尝试多种编码和参数组合
    encodings = ['utf-8', 'ISO-8859-1', 'latin1', 'cp1252', 'gbk']
    df = None
    
    for encoding in encodings:
        try:
            print(f"   尝试编码: {encoding}")
            df = pd.read_csv(
                file_path, 
                encoding=encoding,
                on_bad_lines='skip',  # 跳过损坏的行
                engine='python'       # 使用更宽容的解析引擎
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
    df[CONFIG["COL_DATE"]]=pd.to_datetime(df[CONFIG["COL_DATE"]], errors='coerce')

    # 基础清洗
    df=df.dropna(subset=[CONFIG["COL_DATE"], CONFIG["COL_TITLE"]])
    df[CONFIG["COL_TITLE"]]=df[CONFIG["COL_TITLE"]].astype(str).str.strip()
    df=df[df[CONFIG["COL_TITLE"]].str.len() > 3]  # 去除过短标题
    
    print(f"   清洗后数据行数: {len(df)}")

    return df


def get_sentiment_probabilities(texts, tokenizer, model, device):
    probs_list=[]
    model.eval()
    print(f"\n[3/5] 开始 FinBERT 推理 (N={len(texts)})...")

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), CONFIG["BATCH_SIZE"]), unit="batch"):
            batch_texts=texts[i: i + CONFIG["BATCH_SIZE"]]
            inputs=tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True,
                             max_length=CONFIG["MAX_LEN"]).to(device)
            outputs=model(**inputs)
            probs=F.softmax(outputs.logits, dim=-1)
            probs_list.append(probs.cpu().numpy())

    return np.concatenate(probs_list, axis=0)


def calculate_daily_index(df: pd.DataFrame) -> pd.DataFrame:
    """计算日度情绪指数（单个数据集）"""
    # 1. 文章级得分: (Pos - Neg) / (Pos + Neg + eps)
    df['sentiment_score']=(df['pos_prob'] - df['neg_prob']) / \
                          (df['pos_prob'] + df['neg_prob'] + CONFIG["EPSILON"])

    # 2. 聚合: 默认等权平均 (如有影响力数据，可在此修改)
    daily_series=df.groupby(CONFIG["COL_DATE"])['sentiment_score'].mean()

    return daily_series


def merge_with_fallback(esg_series: pd.Series, all_series: pd.Series, 
                        date_range: pd.DatetimeIndex) -> pd.DataFrame:
    """
    三层回退策略：
    1. 优先使用ESG新闻情绪指数
    2. 如无ESG新闻，使用全量新闻情绪指数
    3. 如完全无新闻，使用前向填充
    """
    print("\n[4/5] 应用三层回退策略...")
    
    # 创建完整日期范围的DataFrame
    result_df = pd.DataFrame(index=date_range)
    
    # 将两个序列对齐到完整日期
    result_df['esg_index'] = esg_series
    result_df['all_index'] = all_series
    
    # 应用回退逻辑
    result_df['ESG_Sentiment_Index'] = result_df['esg_index'].fillna(result_df['all_index'])
    
    # 前向填充（处理完全无新闻的日期）
    result_df['ESG_Sentiment_Index'] = result_df['ESG_Sentiment_Index'].fillna(method='ffill')
    
    # 标注数据来源
    result_df['data_source'] = 'empty'
    result_df.loc[result_df['esg_index'].notna(), 'data_source'] = 'ESG'
    result_df.loc[(result_df['esg_index'].isna()) & (result_df['all_index'].notna()), 'data_source'] = 'All_News'
    result_df.loc[(result_df['esg_index'].isna()) & (result_df['all_index'].isna()) & 
                  (result_df['ESG_Sentiment_Index'].notna()), 'data_source'] = 'Forward_Fill'
    
    # 统计信息
    esg_days = (result_df['data_source'] == 'ESG').sum()
    all_days = (result_df['data_source'] == 'All_News').sum()
    ffill_days = (result_df['data_source'] == 'Forward_Fill').sum()
    
    print(f"   -> ESG新闻日期: {esg_days} 天")
    print(f"   -> 全量新闻日期: {all_days} 天")
    print(f"   -> 前向填充日期: {ffill_days} 天")
    print(f"   -> 总计: {len(result_df)} 天")
    
    # 只保留最终列
    return result_df[['ESG_Sentiment_Index', 'data_source']]


def main():
    device=setup_device()

    # 1. 加载原始数据
    if not os.path.exists(CONFIG["INPUT_FILE"]):
        print("❌ 找不到输入文件")
        return
    df_all=load_and_preprocess_data(CONFIG["INPUT_FILE"])
    
    if len(df_all) == 0:
        print("❌ 原始数据为空，程序终止。")
        return

    # 2. ESG 筛选（保留两份数据）
    print("\n[2/5] 准备ESG和全量两组数据...")
    df_esg = filter_by_esg(df_all)
    
    print(f"   -> ESG相关新闻: {len(df_esg)} 条")
    print(f"   -> 全量新闻: {len(df_all)} 条")

    # 3. 模型准备
    print("\n正在加载 FinBERT...")
    try:
        tokenizer=AutoTokenizer.from_pretrained(CONFIG["MODEL_NAME"])
        model=AutoModelForSequenceClassification.from_pretrained(CONFIG["MODEL_NAME"]).to(device)
    except Exception as e:
        print(f"❌ 模型加载错误: {e}")
        return

    # 识别标签ID
    label2id=model.config.label2id
    pos_idx=label2id.get('positive', 0)
    neg_idx=label2id.get('negative', 1)

    # 4. 推理 - ESG数据
    esg_series = pd.Series(dtype='float64')
    if len(df_esg) > 0:
        print("\n[3a/5] 处理ESG新闻...")
        texts_esg = df_esg[CONFIG["COL_TITLE"]].tolist()
        probs_esg = get_sentiment_probabilities(texts_esg, tokenizer, model, device)
        df_esg['pos_prob'] = probs_esg[:, pos_idx]
        df_esg['neg_prob'] = probs_esg[:, neg_idx]
        esg_series = calculate_daily_index(df_esg)
    else:
        print("⚠️ 无ESG新闻，跳过ESG处理")

    # 5. 推理 - 全量数据
    print("\n[3b/5] 处理全量新闻...")
    texts_all = df_all[CONFIG["COL_TITLE"]].tolist()
    probs_all = get_sentiment_probabilities(texts_all, tokenizer, model, device)
    df_all['pos_prob'] = probs_all[:, pos_idx]
    df_all['neg_prob'] = probs_all[:, neg_idx]
    all_series = calculate_daily_index(df_all)

    # 6. 获取完整日期范围
    start_date = df_all[CONFIG["COL_DATE"]].min()
    end_date = df_all[CONFIG["COL_DATE"]].max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # 7. 应用三层回退策略
    result_df = merge_with_fallback(esg_series, all_series, date_range)

    # 8. 保存结果
    print(f"\n[5/5] 保存文件至: {CONFIG['OUTPUT_FILE']}")
    result_df.to_csv(CONFIG["OUTPUT_FILE"])

    print(f"\n✅ 完成!")
    print(f"   时间范围: {start_date.date()} 至 {end_date.date()}")
    print(f"   总天数: {len(result_df)} 天")
    print(f"   ESG文章数: {len(df_esg)}")
    print(f"   全量文章数: {len(df_all)}")


if __name__ == "__main__":
    main()