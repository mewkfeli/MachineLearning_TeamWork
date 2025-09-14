from fastapi import FastAPI
import pandas as pd
import numpy as np
import pickle
from pydantic import BaseModel, Field
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

    with open('Bagging_classifier_model.pkl', 'rb') as f:
        classifier_model = pickle.load(f)

    print("✅ Модели успешно загружены")

    # Получаем имена признаков из моделей
    try:
        regressor_features = regressor_model.feature_names_in_
        print(f"Признаки регрессора ({len(regressor_features)}): {list(regressor_features)}")
    except:
        regressor_features = None
        print("Не удалось получить признаки регрессора")

    try:
        classifier_features = classifier_model.feature_names_in_
        print(f"Признаки классификатора ({len(classifier_features)}): {list(classifier_features)}")
    except:
        classifier_features = None
        print("Не удалось получить признаки классификатора")

except Exception as e:
    print(f"❌ Ошибка загрузки моделей: {e}")
    regressor_model = None
    classifier_model = None
    regressor_features = None
    classifier_features = None


# Модель данных для запроса
# Модель данных для запроса
class PropertyFeatures(BaseModel):
    total_floor_count: int
    floor_no: int
    room_count: int
    size: float
    building_age_numeric: int
    days_on_market: int = Field(ge=0, le=3000)  # Изменено с 365 на 3000
    heating_type: int
    listing_type: int = 1
    sub_type: int = 0


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
        # Создаем словарь со всеми признаками
        feature_dict = {
            'total_floor_count': features.total_floor_count,
            'floor_no': features.floor_no,
            'room_count': features.room_count,
            'size': features.size,
            'building_age_numeric': features.building_age_numeric,
            'days_on_market': features.days_on_market,
            'heating_type': features.heating_type,
            'listing_type': features.listing_type,
            'sub_type': features.sub_type
        }

        # Создаем DataFrame для регрессора
        if regressor_features is not None:
            regressor_data = {col: [feature_dict[col]] for col in regressor_features}
            regressor_df = pd.DataFrame(regressor_data)
        else:
            # Предполагаем, что регрессор использует все 9 признаков
            regressor_df = pd.DataFrame([[
                features.total_floor_count,
                features.floor_no,
                features.room_count,
                features.size,
                features.building_age_numeric,
                features.days_on_market,
                features.heating_type,
                features.listing_type,
                features.sub_type
            ]], columns=[
                'total_floor_count', 'floor_no', 'room_count', 'size',
                'building_age_numeric', 'days_on_market', 'heating_type',
                'listing_type', 'sub_type'
            ])

        # Предсказание цены
        predicted_price = float(regressor_model.predict(regressor_df)[0])

        # Создаем DataFrame для классификатора (только 7 признаков)
        if classifier_features is not None:
            classifier_data = {col: [feature_dict[col]] for col in classifier_features}
            classifier_df = pd.DataFrame(classifier_data)
        else:
            # Классификатор использует только основные 7 признаков (без listing_type, sub_type, price_per_sqm, price)
            classifier_df = pd.DataFrame([[
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

        # Предсказание типа недвижимости
        predicted_subtype = int(classifier_model.predict(classifier_df)[0])

        return {
            "predicted_price": round(predicted_price, 2),
            "predicted_subtype": predicted_subtype,
            "regressor_features_count": len(regressor_df.columns),
            "classifier_features_count": len(classifier_df.columns)
        }

    except Exception as e:
        return {
            "error": f"Ошибка при предсказании: {str(e)}",
            "predicted_price": 0,
            "predicted_subtype": -1
        }


# Эндпоинт для получения информации о признаках моделей
@app.get("/model_features")
async def get_model_features():
    return {
        "regressor_features": list(regressor_features) if regressor_features is not None else "Не доступно",
        "classifier_features": list(classifier_features) if classifier_features is not None else "Не доступно",
        "regressor_features_count": len(regressor_features) if regressor_features is not None else 0,
        "classifier_features_count": len(classifier_features) if classifier_features is not None else 0
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