import sys
import os
import pandas as pd

# Преобразовываем признаки текстовые в числовые, меняем типы,
# сохраняем файл в stage3

# & C:/Python311/python.exe f:/Local/URFU_L/Python/MLOP_II_M_1_scripts/data_scripts/4_change_text_to_numeric.py data/stage2/train.csv

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 change_text_to_numeric.py data-file\n")
    sys.exit(1)

f_input = sys.argv[1]
os.makedirs(os.path.join("data", "stage3"), exist_ok=True)

# забираем датасет для обработки
df = pd.read_csv(f_input)

# Удалим признаки, которые не оказывают значитального влияния на модель обучения
df.drop([
"YrSold", "MSSubClass", "LotConfig", "RoofStyle", "HouseStyle", "LotShape",
"Exterior1st", "Exterior2nd", "BsmtFullBath", "BedroomAbvGr", "HalfBath",
"BsmtFinType2", "MoSold", "YrSold"
],
axis=1, inplace=True)

# заменим текстовые признаки на числовые
df["SaleCondition"] = pd.factorize(df["SaleCondition"])[0]
df["SaleType"] = pd.factorize(df["SaleType"])[0]
df["Condition1"] = pd.factorize(df["Condition1"])[0]


df.to_csv("data/stage3/train.csv", index=False)
