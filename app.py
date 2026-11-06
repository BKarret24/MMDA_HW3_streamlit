import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

st.set_page_config(page_title='Подбор хороших вин', layout='wide')
app_dir = Path(__file__).parent


@st.cache_resource
def load_model():
    return joblib.load(app_dir / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(app_dir / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


model = load_model()
test = load_test()
features = test.drop(columns='target')
y_test = test['target']
proba = test_proba(model, features)
default_threshold = json.loads((app_dir / 'config.json').read_text(encoding='utf-8'))['threshold']

st.title('Подбор хороших вин')
threshold = st.sidebar.slider('Порог вероятности', 0.0, 1.0,
                              value=float(default_threshold), step=0.001)
st.sidebar.write(f'Текущий порог: {threshold:.3f}')

tab_wine, tab_metrics, tab_batch, tab_coef = st.tabs([
    'Одно вино', 'Модельные метрики', 'Пакетное предсказание', 'Коэффициенты'
])

with tab_wine:
    st.subheader('Характеристики вина')
    values = {}
    for col in features.columns:
        if col == 'is_red':
            values[col] = int(st.toggle('Красное вино (иначе белое)',
                                        value=bool(features[col].median())))
        else:
            values[col] = st.number_input(
                col,
                min_value=float(features[col].min()),
                max_value=float(features[col].max()),
                value=float(features[col].median())
            )

    wine = pd.DataFrame([values])[features.columns]
    probability = model.predict_proba(wine)[0, 1]
    st.write(f'Вероятность хорошего вина: **{probability:.1%}**')
    st.write('Решение:', '**Рекомендовать**' if probability >= threshold else '**Не рекомендовать**')

with tab_metrics:
    prediction = (proba >= threshold).astype(int)
    precision = metrics.precision_score(y_test, prediction, zero_division=0)
    recall = metrics.recall_score(y_test, prediction)
    f1 = metrics.f1_score(y_test, prediction, zero_division=0)

    c1, c2, c3 = st.columns(3)
    c1.metric('Precision', f'{precision:.3f}')
    c2.metric('Recall', f'{recall:.3f}')
    c3.metric('F1', f'{f1:.3f}')

    fig, ax = plt.subplots(figsize=(5, 4))
    metrics.ConfusionMatrixDisplay.from_predictions(
        y_test, prediction, ax=ax, cmap='Blues'
    )
    ax.set_title('Матрица ошибок')
    st.pyplot(fig)
    plt.close(fig)

    fpr, tpr, _ = metrics.roc_curve(y_test, proba)
    fp_rate = np.sum((y_test == 0) & (prediction == 1)) / np.sum(y_test == 0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label='ROC')
    ax.scatter(fp_rate, recall, label='Текущий порог')
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC-кривая')
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    prec_curve, rec_curve, _ = metrics.precision_recall_curve(y_test, proba)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rec_curve, prec_curve, label='PR')
    ax.scatter(recall, precision, label='Текущий порог')
    ax.axhline(y_test.mean(), linestyle='--', color='gray', label='Случайная модель')
    ax.set(xlabel='Recall', ylabel='Precision', title='PR-кривая')
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.info('Если увеличить порог, вин в подборке станет меньше, но обычно возрастёт точность рекомендаций. '
            'Если уменьшить порог, хороших вин найдём больше, но ошибочных рекомендаций тоже станет больше.')

with tab_batch:
    uploaded = st.file_uploader('Загрузите CSV с признаками вин (без target)', type='csv')
    if uploaded is not None:
        new_wines = pd.read_csv(uploaded)
        missing = [col for col in features.columns if col not in new_wines.columns]
        if missing:
            st.error('Не хватает столбцов: ' + ', '.join(missing))
        elif new_wines.empty:
            st.error('CSV-файл не содержит строк с винами.')
        else:
            try:
                probability_batch = model.predict_proba(new_wines[features.columns])[:, 1]
                result = new_wines.copy()
                result['probability'] = probability_batch
                result['recommend'] = (probability_batch >= threshold).astype(int)
                st.dataframe(result)
                st.download_button('Скачать результаты',
                                   data=result.to_csv(index=False).encode('utf-8'),
                                   file_name='wine_predictions.csv', mime='text/csv')
            except ValueError:
                st.error('Проверьте значения признаков: они должны быть числовыми и без пропусков.')

with tab_coef:
    coefficients = model.named_steps['logisticregression'].coef_[0]
    coef_table = pd.Series(coefficients, index=features.columns).sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))
    coef_table.plot(kind='barh', ax=ax)
    ax.set(xlabel='Коэффициент после стандартизации', ylabel='Признак',
           title='Коэффициенты логистической регрессии')
    st.pyplot(fig)
    plt.close(fig)

    st.write('Сильнее всего за хорошее вино:', ', '.join(coef_table.tail(3).index))
    st.write('Сильнее всего против:', ', '.join(coef_table.head(3).index))
    st.write('Положительный коэффициент увеличивает вероятность хорошего вина, '
             'а отрицательный уменьшает её при прочих равных.')

# cache_resource хранит один объект модели, а cache_data — копии таблиц и рассчитанных вероятностей.
# Без кэша joblib.load и predict_proba повторялись бы при каждом движении ползунка.
