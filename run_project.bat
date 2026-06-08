@echo off
cd /d "%~dp0"

echo ======================================
echo Car Price Prediction Project
echo ======================================
echo.

echo Step 0: Installing dependencies
echo --------------------------------------
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo Dependency installation failed.
    goto end
)

echo Dependencies installed successfully.
echo.

set RAW_DATA=data\raw\Car_sale_ads.csv
set PROCESSED_DATA=data\processed\car_prices_clean.csv
set KAGGLE_TOKEN=%USERPROFILE%\.kaggle\access_token

echo Step 1: Data preprocessing
echo --------------------------------------

if not exist "%RAW_DATA%" (
    if not exist "%KAGGLE_TOKEN%" (
        echo Kaggle API token not found:
        echo %KAGGLE_TOKEN%
        echo.
        echo Create Kaggle API token and save it as access_token in:
        echo %USERPROFILE%\.kaggle
        goto end
    )

    echo Raw dataset not found. Downloading from Kaggle...
    python -m kaggle datasets download -d bartoszpieniak/poland-cars-for-sale-dataset -p data\raw --unzip

    if errorlevel 1 (
        echo.
        echo Kaggle dataset download failed.
        echo Make sure your Kaggle API token is configured correctly.
        echo Expected token path on Windows: %KAGGLE_TOKEN%
        goto end
    )

    if not exist "%RAW_DATA%" (
        echo.
        echo Kaggle download finished, but expected CSV was not found:
        echo %RAW_DATA%
        goto end
    )

    echo Kaggle dataset downloaded successfully.
)

if exist "%PROCESSED_DATA%" (
    echo Processed dataset already exists:
    echo %PROCESSED_DATA%
    echo.
    goto ask_overwrite
)

:run_preprocessing
echo.
echo Running data preprocessing...
python src\data\preprocessing.py

if errorlevel 1 (
    echo.
    echo Data preprocessing failed.
    goto end
)

echo Data preprocessing finished successfully.
goto preprocessing_done

:ask_overwrite
set /p overwrite=Overwrite it? [Y/N]: 

if /i "%overwrite%"=="Y" goto run_preprocessing
if /i "%overwrite%"=="YES" goto run_preprocessing

echo Skipping data preprocessing.
goto preprocessing_done

:preprocessing_done
echo.
echo Project setup finished.
echo.
echo Kolejne kroki tbd:
echo - trenowanie modelu AutoGluonem
echo - FastAPI
echo - Streamlit

:end
echo.
pause
