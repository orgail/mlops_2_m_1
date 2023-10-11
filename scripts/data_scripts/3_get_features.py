import sys
import os
import io
import pandas as pd

# Работаем с оставшимися нужными признаками, сохраняем файл в stage2

# python3 ./get_features.py data-file
# & C:/Python311/python.exe f:/Local/URFU_L/Python/MLOP_II_M_1_scripts/data_scripts/3_get_features.py data/stage1/train.csv


if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 get_features.py data-file\n")
    sys.exit(1)

f_input = sys.argv[1]
os.makedirs(os.path.join("data", "stage2"), exist_ok=True)

# забираем датасет для обработки
df = pd.read_csv(f_input)


to_cat = []
for column in df.select_dtypes(include='number').columns:
    if len(df[column].value_counts().index) < 25:
        to_cat.append(column)


for quality_col in to_cat:
    df[quality_col] = df[quality_col].astype(object)


for cat_col in df.select_dtypes(exclude='number').columns:
    _overall = df[cat_col].count()
    _most_samples = df[cat_col].value_counts().iloc[0]
    _coef = _most_samples / _overall
    if _coef > 0.9:
        df.drop(cat_col, axis=1, inplace=True)


df.to_csv("data/stage2/train.csv", index=False)
