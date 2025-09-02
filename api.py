# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import uvicorn
from typing import Dict, Any

# Создание приложения FastAPI
app = FastAPI(
    title="API для предсказания недвижимости",
    description="API для предсказания цены и типа недвижимости",
    version="1.0.0"
)

# Загрузка моделей
try:
    with open('stacking_regressor_model.pkl', 'rb') as f:
        regressor_model = pickle.load(f)

    with open('stacking_classifier_model.pkl', 'rb') as f:
        classifier_model = pickle.load(f)

    print("✅ Модели успешно загружены")
except Exception as e:
    print(f"❌ Ошибка загрузки моделей: {e}")
    regressor_model = None
    classifier_model = None


# Модель данных для запроса
class PropertyFeatures(BaseModel):
    total_floor_count: int
    floor_no: int
    room_count: int
    size: float
    building_age_numeric: int
    days_on_market: int
    heating_type: int


# Корневой эндпоинт
@app.get("/")
async def root():
    return {
        "message": "API для предсказания недвижимости",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Предсказание цены и типа недвижимости"
        }
    }


# Эндпоинт для предсказаний
@app.post("/predict")
async def predict_price_and_type(features: PropertyFeatures) -> Dict[str, Any]:
    if not regressor_model or not classifier_model:
        return {
            "error": "Модели не загружены",
            "predicted_price": 0,
            "predicted_subtype": -1
        }

    try:
        # Создаем DataFrame с признаками для регрессора
        # Признаки для регрессии (без price и price_per_sqm)
        regressor_features = pd.DataFrame([[
            features.total_floor_count,
            features.floor_no,
            features.room_count,
            features.size,
            features.building_age_numeric,
            features.days_on_market,
            features.heating_type
        ]], columns=[
            'total_floor_count', 'floor_no', 'room_count', 'size',
            'building_age_numeric', 'days_on_market', 'heating_type'
        ])

        # Предсказание цены
        predicted_price = float(regressor_model.predict(regressor_features)[0])

        # Рассчитываем цену за квадратный метр
        price_per_sqm = predicted_price / features.size if features.size > 0 else 0

        # Создаем DataFrame с признаками для классификатора
        classifier_features = pd.DataFrame([[
            features.total_floor_count,
            features.floor_no,
            features.room_count,
            features.size,
            features.building_age_numeric,
            features.days_on_market,
            price_per_sqm,
            predicted_price,
            features.heating_type
        ]], columns=[
            'total_floor_count', 'floor_no', 'room_count', 'size',
            'building_age_numeric', 'days_on_market', 'price_per_sqm',
            'price', 'heating_type'
        ])

        # Предсказание типа недвижимости
        predicted_subtype = int(classifier_model.predict(classifier_features)[0])

        return {
            "predicted_price": round(predicted_price, 2),
            "predicted_subtype": predicted_subtype,
            "price_per_sqm": round(price_per_sqm, 2)
        }

    except Exception as e:
        return {
            "error": f"Ошибка при предсказании: {str(e)}",
            "predicted_price": 0,
            "predicted_subtype": -1
        }


# Эндпоинт для проверки состояния
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": regressor_model is not None and classifier_model is not None
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
