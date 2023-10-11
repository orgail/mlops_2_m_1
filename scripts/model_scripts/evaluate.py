import os
import sys
import pickle
import json
import pandas as pd

# этим файлом получаем метрики работы модели на тестовых данных
# & C:/Python311/python.exe f:/Local/URFU_L/Python/MLOP_II_M_1_scripts/model_scripts/evaluate.py data/stage4/test.csv models/model.pkl

# забираем тестовый файл и обученную модель
if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython evaluate.py data-file model\n")
    sys.exit(1)

# загружаем тренировочный датасет и разделяем признаки и метки
# df = pd.read_csv(sys.argv[1], header=None)
df = pd.read_csv(sys.argv[1])

# Отделяем признаки и метки
X_test = df.iloc[:,0:-1]
y_test = df.iloc[:,-1].astype('int')

# Загружаем модель
with open(sys.argv[2], "rb") as fd:
    clf = pickle.load(fd)

# Получаем метрики работы модели на тестовых данных
score = clf.score(X_test, y_test)

prc_file = os.path.join("evaluate", "score.json")
os.makedirs(os.path.join("evaluate"), exist_ok=True)

with open(prc_file, "w") as fd:
    json.dump({"score": score}, fd)
