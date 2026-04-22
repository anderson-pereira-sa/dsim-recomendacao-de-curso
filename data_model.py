import streamlit as st
import pickle

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier


dataset_0 = pickle.load(open('check_dataset_0.pkl', 'rb'))

categoricas_ohencoder = ['ANO','UNIDADE', 'CURSO']
numericas_ohencoder = [
    'QTD_EMPRESAS',
    'QTD_VINCULOS',
    'QTD_CONC_INEP',
    'QTD_MAT_CONC',
    'SALARIO_MEDIO',
    'SALDO_EMPREGO'
]

X_ohencoder = dataset_0[categoricas_ohencoder + numericas_ohencoder]

le = LabelEncoder()
dataset_0['FAIXA_MAT_ENC'] = le.fit_transform(dataset_0['FAIXA_MAT'])
y_ohencoder = dataset_0['FAIXA_MAT_ENC']
# y_ohencoder = dataset_0['FAIXA_MAT']
encoder = OneHotEncoder(drop='first',
                        sparse_output=False,
                        handle_unknown='ignore')

X_cat_ohencoder = encoder.fit_transform(X_ohencoder[categoricas_ohencoder])
X_num_ohencoder = X_ohencoder[numericas_ohencoder].values
X_final_ohencoder = np.hstack([X_num_ohencoder, X_cat_ohencoder])

# >>>>>>> - Preparação dos dados - <<<<<<< #
X_train, X_test, y_train, y_test = train_test_split(X_final_ohencoder, y_ohencoder, 
                                                    test_size=0.2, 
                                                    random_state=123, 
                                                    stratify= y_ohencoder)

# >>>>>>> - Parâmetros do modelo - <<<<<<< #
xgbc = XGBClassifier(
    objective='multi:softprob',
    num_class=5,
    eval_metric='mlogloss',
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=2,
    subsample=0.95,
    colsample_bytree=0.95,
    n_estimators=1000,
    random_state=123
)

classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced',
                                     classes = classes,
                                     y = y_train)

class_weight_dict = dict(zip(classes, class_weights))

# >>>>>>> - Treinamento - <<<<<<< #
xgbc.fit(X_train, y_train, sample_weight=y_train.map(class_weight_dict))
xgbc.save_model("modelo_xgbc.json")

with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

with open("categoricas.pkl", "wb") as f:
    pickle.dump(categoricas_ohencoder, f)

with open("numericas.pkl", "wb") as f:
    pickle.dump(numericas_ohencoder, f)

with open("feature_order.pkl", "wb") as f:
    pickle.dump(numericas_ohencoder + list(encoder.get_feature_names_out()),f)


print('Modelo salvo com sucesso!')