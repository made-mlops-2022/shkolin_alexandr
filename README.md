# Задача классификации

## Данные

Датасеты взяты [отсюда](https://www.kaggle.com/competitions/hotel-booking-demand-3/data)

## Использование

### Установка
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Обучение по двум конфигурациям

Стандартная логистическая регрессия
```
python src/train_pipeline.py configs/train_lr_default.yaml
```

Логистическая регрессия с параметрами
```
python src/train_pipeline.py configs/train_lr_custom.yaml
```

### Предикт
```
python src/predict_pipeline.py
```

## Структура
```commandline
├── configs                       <- Сonfiguration files
├── data
│   ├── predictions         <- predicted .scv files
│   └── raw                 <- original .scv files
│
├── models                        <- Trained model, transformers and metrics
│
├── notebooks                     <- Jupyter notebooks - EDA
│
├── requirements.txt              <- Requirements
│
├── src                           <- Source code for use in this project.
│   ├── __init__.py         <- Makes src a Python module
│   │
│   ├── data                <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── entity              <- Scripts to read configuration file
│   │
│   ├── models              <- Scripts to train models and then use trained models to make
│   │                    predictions
│   ├── predict_pipeline.py <- Pipeline for prediction
│   ├── train_pipeline.py   <- Pipeline for training model
├── tests                         <- Tests folder
```