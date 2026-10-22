"""
v3：把 4 个 CSV 整理为一个 Excel。
- 不再拆 core/extra，每个表对应一个 sheet。
- sheet 名用中文。
- 删全空列、Unnamed:* 占位列、清洗空白与 nan 字符串。
- 新增 Overview（行/列数）和 Column_NaN_Rate（每列 NaN 占比）sheet。
"""
import os
import re
import pandas as pd

BASE = r"D:\trae\solo\ampmodel\data_processing\all_antifungal"
OUT  = os.path.join(BASE, "merged_antifungal_summary_v3.xlsx")

SHEETS = [
    ("原始合并",
     os.path.join(BASE, "merged_antifungal_all_improved.csv"),
     "原始合并（未标准化）"),
    ("全量标准化",
     os.path.join(BASE, "merged_antifungal_all_improved_standardized.csv"),
     "全量 + 列名/字段标准化"),
    ("多库交叉",
     os.path.join(BASE, "merged_antifungal_2plus_improved_standardized.csv"),
     "至少在 2 个数据库出现 + 标准化"),
    ("≤60aa 终版",
     os.path.join(BASE, "merged_antifungal_le60_final.csv"),
     "长度 <= 60 aa 终版"),
]


def read_safe(p):
    for enc in ("utf-8-sig", "gb18030", "latin1"):
        try:
            return pd.read_csv(p, low_memory=False, encoding=enc,
                               encoding_errors="replace")
        except Exception:
            continue
    raise RuntimeError(p)


def clean_df(df):
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed:")]
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = (df[c].astype(str).str.strip()
                          .replace({"nan": "", "NaN": "", "None": ""}))
    return df


ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def write_sheet(writer, df, sheet_name):
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = (df[c].astype(str)
                       .str.replace(ILLEGAL, "", regex=True)
                       .str.slice(0, 32000))
    df.to_excel(writer, sheet_name=sheet_name, index=False)


# 主流程
overview_rows = []
nan_rows = []
loaded = []

for name, path, desc in SHEETS:
    print(f"\n--- {os.path.basename(path)} ---")
    df = clean_df(read_safe(path))
    print(f"cleaned: {df.shape}")
    overview_rows.append({
        "Sheet": name,
        "Source File": os.path.basename(path),
        "Description": desc,
        "Rows": len(df),
        "Columns": len(df.columns),
        "File Size (KB)": round(os.path.getsize(path) / 1024, 1),
    })
    for col, r in df.isna().mean().round(4).items():
        nan_rows.append({
            "Sheet": name,
            "Column": col,
            "NaN_Rate": r,
            "NonNull_Count": int(df[col].notna().sum()),
        })
    loaded.append((name, df))


print(f"\nwriting {OUT} ...")
with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
    pd.DataFrame(overview_rows).to_excel(writer, sheet_name="概览", index=False)
    pd.DataFrame(nan_rows).to_excel(writer, sheet_name="列缺失率", index=False)
    for name, df in loaded:
        write_sheet(writer, df, name)
        print(f"  wrote sheet: {name}")

print(f"\nDONE -> {OUT}")
print(f"size: {os.path.getsize(OUT)/1024:.1f} KB")
