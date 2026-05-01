import pandas as pd
import pandas_ta as ta

df = pd.read_csv("sp500.csv")

# ---------- 重命名 ----------
df = df.rename(columns={
    "Price": "close",
    "Open": "open",
    "High": "high",
    "Low": "low"
})

# ---------- 数字清洗 ----------
for col in ["open", "high", "low", "close"]:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", "", regex=False)  # 去掉千分位
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["open", "high", "low", "close"])

# ---------- 技术指标（无 AD） ----------
df["ADX"] = ta.adx(df["high"], df["low"], df["close"])["ADX_14"]
df["EMA_12"] = ta.ema(df["close"], length=12)
df["KAMA"] = ta.kama(df["close"])
df["MA_10"] = ta.sma(df["close"], length=10)

macd = ta.macd(df["close"])
df["MACD"] = macd["MACD_12_26_9"]
df["MACD_signal"] = macd["MACDs_12_26_9"]
df["MACD_hist"] = macd["MACDh_12_26_9"]

df["RSI"] = ta.rsi(df["close"], length=14)

psar = ta.psar(df["high"], df["low"], df["close"])
df["PSAR"] = psar["PSARl_0.02_0.2"].fillna(psar["PSARs_0.02_0.2"])

df["SMA_20"] = ta.sma(df["close"], length=20)

# ---------- 保留两位小数 ----------
df = df.round(2)

df.to_csv("sp500_with_indicators.csv", index=False)
print("成功完成！")
