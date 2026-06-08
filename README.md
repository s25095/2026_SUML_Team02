# 2026_SUML_Team02

Webowa aplikacja FastAPI do szacowania ceny auta uzywanego w PLN na podstawie cech z ogloszenia.

## Cel projektu

Projekt wykorzystuje dataset `bartoszpieniak/poland-cars-for-sale-dataset` z Kaggle. Targetem jest `Price`, czyli cena ofertowa auta. Aplikacja przyjmuje cechy auta, takie jak marka, rok produkcji, przebieg, moc, pojemnosc, paliwo, naped, skrzynia biegow, typ nadwozia i liczba drzwi, a nastepnie zwraca szacowana cene w PLN.

Praktyczne zastosowanie aplikacji to szybka orientacyjna wycena samochodu uzywanego przed zakupem albo przed wystawieniem ogloszenia sprzedazy. Kupujacy moze porownac cene z ogloszenia z predykcja modelu i latwiej wykryc oferty znacznie odbiegajace od typowych cen dla podobnych aut. Sprzedajacy moze oszacowac realistyczny poziom ceny wyjsciowej, aby nie zanizyc wartosci auta ani nie ustawic ceny zbyt wysokiej wzgledem rynku. Aplikacja nie zastepuje rzeczoznawcy, bo nie widzi stanu technicznego konkretnego egzemplarza, historii serwisowej ani uszkodzen, ale daje powtarzalny punkt odniesienia oparty na danych z wielu ogloszen.

## Setup

Projekt uzywa `uv` jako package managera i runnera. Docelowa wersja Pythona to 3.12.

```bash
uv sync --python 3.12
```

Skonfiguruj zmienne srodowiskowe:

```bash
cp .env.example .env
```

Uzupelnij w `.env` tokenem Kaggle zaczynajacym sie od `KGAT_`:

```bash
KAGGLE_API_TOKEN=twoj_kaggle_access_token
```

Konfiguracja aplikacji jest ladowana przez Pydantic Settings z `.env` oraz zmiennych systemowych.

## Workflow

1. Pobierz dataset z Kaggle:

```bash
uv run download-data
```

Kaggle API uzyje `KAGGLE_API_TOKEN` z `.env`.

2. Przygotuj dane:

```bash
uv run preprocess-data
```

3. Przeprowadz EDA i porownanie modeli w notebooku:

```bash
uv run jupyter lab notebooks/car_price_eda_model_selection.ipynb
```

Notebook dokumentuje rozklad targetu, braki danych, outliery, wybrane cechy i porownanie modeli. Produkcyjny trening nie odbywa sie w notebooku, tylko w powtarzalnym skrypcie.

4. Wytrenuj model produkcyjny:

```bash
uv run train-model
```

Skrypt porownuje baseline, modele klasyczne i LightGBM. LightGBM jest preferowany, jesli bije baseline i jest w granicy 5% najlepszego RMSE. Artefakty trafiaja do `models/`.

5. Uruchom aplikacje:

```bash
uv run serve
```

Strona HTML jest dostepna pod `http://127.0.0.1:8000/`.

## API

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Predykcja JSON:

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Condition": "Used",
    "Vehicle_brand": "Toyota",
    "Production_year": 2018,
    "Mileage_km": 120000,
    "Power_HP": 150,
    "Displacement_cm3": 1998,
    "Fuel_type": "Gasoline",
    "Drive": "Front wheels",
    "Transmission": "Manual",
    "Type": "SUV",
    "Doors_number": 5
  }'
```

## Testy

```bash
uv run pytest
uv run python -m json.tool notebooks/car_price_eda_model_selection.ipynb
```

## Struktura

- `src/car_price_prediction/config.py` - centralne sciezki, nazwy kolumn, ustawienia datasetu i modelu.
- `src/car_price_prediction/data/` - pobieranie i preprocessing danych.
- `src/car_price_prediction/model/` - trening, wybor modelu i predykcja.
- `src/car_price_prediction/app/` - FastAPI, formularz HTML i endpoint JSON.
- `notebooks/` - EDA i uzasadnienie wyboru modelu.
- `tests/` - testy jednostkowe i testy endpointow.
