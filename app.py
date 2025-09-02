# app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
from typing import Dict, Any

# Конфигурация страницы
st.set_page_config(
    page_title="Предсказание недвижимости",
    page_icon="🏠",
    layout="wide"
)

# Заголовок приложения
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <h1 style="color: #1976d2;">🔮 Предсказание параметров недвижимости</h1>
    <p><b>Машинное обучение для анализа рынка недвижимости</b></p>
</div>
""", unsafe_allow_html=True)

# Создание боковой панели для ввода параметров
st.sidebar.header("Ввод параметров недвижимости")

# Основные параметры
st.sidebar.subheader("Основные характеристики")
size = st.sidebar.number_input("Площадь (м²)", min_value=10, max_value=1000, value=80, step=1)
room_count = st.sidebar.selectbox("Количество комнат", [1, 2, 3, 4, 5, 6], index=1)
building_age_numeric = st.sidebar.slider("Возраст здания (лет)", 0, 50, 5)

# Расположение
st.sidebar.subheader("Расположение")
floor_no = st.sidebar.number_input("Этаж", min_value=-2, max_value=50, value=3, step=1)
total_floor_count = st.sidebar.number_input("Этажность здания", min_value=1, max_value=50, value=5, step=1)

# Дополнительные параметры
st.sidebar.subheader("Дополнительные параметры")
days_on_market = st.sidebar.slider("Дней на рынке", 0, 365, 30)
heating_type = st.sidebar.selectbox(
    "Тип отопления",
    options=[
        ("Нет", 0), ("Соба (Уголь)", 1), ("Соба (Газ)", 2),
        ("Калорифер (Уголь)", 3), ("Калорифер (Газ)", 4),
        ("Калорифер (Топливо)", 5), ("Комби (Газ)", 6),
        ("Комби (Электричество)", 7), ("Кат Калорифер", 8),
        ("Центральная система", 9), ("Центральная система (Измеритель тепла)", 10),
        ("Теплый пол", 11), ("Кондиционер", 12), ("Фанкойл", 13),
        ("Солнечная энергия", 14), ("Геотермальная", 15)
    ],
    format_func=lambda x: x[0]
)[1]

# URL API (можно изменить при необходимости)
API_URL = "http://localhost:8000"

# Функция для получения предсказаний от API
def get_predictions(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        response = requests.post(f"{API_URL}/predict", json=data, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Ошибка API: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Не удалось подключиться к API. Убедитесь, что сервер запущен.")
        return None
    except requests.exceptions.Timeout:
        st.error("⏰ Превышено время ожидания ответа от API.")
        return None
    except Exception as e:
        st.error(f"❌ Произошла ошибка: {str(e)}")
        return None


# Кнопка для получения предсказаний
if st.sidebar.button("🔮 Получить предсказания", type="primary"):
    # Подготовка данных для отправки
    input_data = {
        "total_floor_count": total_floor_count,
        "floor_no": floor_no,
        "room_count": room_count,
        "size": size,
        "building_age_numeric": building_age_numeric,
        "days_on_market": days_on_market,
        "heating_type": heating_type
    }

    # Отправка запроса к API
    with st.spinner("Анализируем данные..."):
        predictions = get_predictions(input_data)

    if predictions:
        # Отображение результатов
        st.success("✅ Предсказания получены!")

        # Создание двух колонок для отображения результатов
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("💰 Предсказанная цена")
            price = predictions.get("predicted_price", 0)
            st.metric(
                label="Цена недвижимости",
                value=f"{price:,.0f} ₺",
                delta=None
            )

            # Дополнительная информация о цене
            price_per_sqm = price / size if size > 0 else 0
            st.info(f"Цена за м²: {price_per_sqm:,.0f} ₺")

        with col2:
            st.subheader("🏷 Тип недвижимости")
            subtype = predictions.get("predicted_subtype", "Неизвестно")
            subtype_mapping = {
                0: "Квартира (Flat)",
                1: "Жилой комплекс (Rezidans)",
                2: "Вилла (Villa)",
                3: "Индивидуальный дом (Müstakil Ev)",
                4: "Кооператив (Kooperatif)",
                5: "Комплекс зданий (Komple Bina)",
                6: "Дача (Yazlık)",
                7: "Сборный дом (Prefabrik Ev)",
                8: "Особняк/Дворец/Водный дом (Köşk / Konak / Yalı)",
                9: "Фермерский дом (Çiftlik Evi)",
                10: "Водная квартира (Yalı Dairesi)",
                11: "Лофт (Loft)"
            }
            subtype_name = subtype_mapping.get(subtype, f"Тип {subtype}")
            st.metric(
                label="Классификация",
                value=subtype_name,
                delta=None
            )

        # Дополнительная информация
        st.subheader("📊 Детали анализа")
        col3, col4 = st.columns(2)

        with col3:
            st.write("**Введенные параметры:**")
            st.json(input_data)

        with col4:
            st.write("**Предсказания модели:**")
            st.json(predictions)

        # Визуализация
        st.subheader("📈 Визуализация")
        chart_data = pd.DataFrame({
            'Параметр': ['Площадь', 'Комнаты', 'Этаж', 'Возраст'],
            'Значение': [size, room_count, floor_no, building_age_numeric]
        })
        st.bar_chart(chart_data.set_index('Параметр'))

    else:
        st.error("Не удалось получить предсказания. Проверьте подключение к API.")

else:
    # Информационная панель
    st.info("ℹ️ Введите параметры недвижимости в боковой панели и нажмите кнопку 'Получить предсказания'")

    # Пример данных
    st.subheader("📝 Пример заполнения")
    st.markdown("""
    - **Площадь**: 85 м²
    - **Комнаты**: 3 (2+1)
    - **Этаж**: 3 из 5
    - **Возраст здания**: 10 лет
    - **Дней на рынке**: 45
    - **Отопление**: Центральное (газ)
    """)

# Информация о моделях
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ О моделях")
st.sidebar.markdown("""
**Модели машинного обучения:**
- **Регрессия**: StackingRegressor
- **Классификация**: StackingClassifier
- **Точность**: Высокая
""")

# Футер
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p><b>Создано для курса "Машинное обучение и большие данные" 📊</b></p>
    <p>Мухитова Азалия, Каспранов Камиль — 22П-2</p>
</div>
""", unsafe_allow_html=True)
