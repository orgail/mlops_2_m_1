import sys
import os
import yaml
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis


# Здесь обучаем и сохраняем модель в *.pkl
# & C:/Python311/python.exe f:/Local/URFU_L/Python/MLOP_II_M_1_scripts/model_scripts/dt.py data/stage4/train.csv model.pkl


if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython dt.py data-file model \n")
    sys.exit(1)

f_input = sys.argv[1]
f_output = os.path.join("models", sys.argv[2])
os.makedirs(os.path.join("models"), exist_ok=True)

CLASSIFICATION_TARGET = "OverallQual"
params = yaml.safe_load(open("params.yaml"))["train"]
p_n_neighbors = params["n_neighbors"]
p_weights = params["weights"]

# загружаем тренировочный датасет и разделяем признаки и метки
df = pd.read_csv(f_input)

# Отделяем признаки и метки
X_train = df.iloc[:,0:-1]
y_train = df.iloc[:,-1].astype('int')

# Пишем пайп финальной подготовки и обучения датасета
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()), # StandardScaler or MinMaxScaler
    ('NCA', NeighborhoodComponentsAnalysis(n_components=8))
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("one_hot", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False))
])

preprocessors = ColumnTransformer(transformers=[
    ("num", num_pipe, df.select_dtypes(include="number").columns.drop(CLASSIFICATION_TARGET)),
    ("cat", cat_pipe, df.select_dtypes(exclude="number").columns)
])
                                  
model_pipe = Pipeline([
    ("preprocessing", preprocessors),
    ("model", KNeighborsClassifier(n_neighbors=p_n_neighbors, weights=p_weights))
])

# Обучаем модель
model_pipe.fit(X_train, y_train.astype('int'))

# Сохраняем обученную модель
with open(f_output, "wb") as fd:
    pickle.dump(model_pipe, fd)
