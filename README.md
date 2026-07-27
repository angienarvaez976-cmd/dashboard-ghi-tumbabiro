# Dashboard interactivo — Pronóstico de GHI, Planta Fotovoltaica Tumbabiro

Dashboard en Streamlit que responde a la pregunta central de la tesis: **¿qué modelo
predice mejor la irradiancia global horizontal (GHI), el enfoque clásico
(ARIMA/SARIMAX) o los de Machine Learning (Random Forest, XGBoost, LSTM)?**

Construido como insumo del objetivo específico 3 (modificado según indicación del
tutor), separando el modelado (ya cerrado en el notebook) de la capa de visualización.

## Estructura

```
dashboard_tumbabiro/
├── app.py                          # Aplicación Streamlit
├── requirements.txt
├── README.md
└── data/
    ├── predicciones_todas_series.csv   # fecha, serie, real y 4 pronósticos (1022 filas)
    ├── resumen_comparacion.csv         # MAPE por serie y modelo, ganador absoluto (14 filas)
    └── importancia_variables.csv       # importancia RF/XGBoost por variable y serie (140 filas)
```

## Cómo ejecutarlo localmente

1. Crear un entorno virtual (opcional pero recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Ejecutar el dashboard:
   ```bash
   streamlit run app.py
   ```

4. Se abrirá automáticamente en el navegador (por defecto `http://localhost:8501`).

## Secciones del dashboard

- **Resumen general:** KPIs globales, conteo de victorias por modelo y mapa de calor
  de MAPE por serie × modelo.
- **Comparación por serie:** selector de serie (14 disponibles), gráfico interactivo
  Real vs. los 4 pronósticos, con rango de fechas ajustable y selección de modelos.
- **Importancia de variables:** importancia relativa de cada variable exógena
  (Random Forest y XGBoost) por serie, más el promedio general.
- **Tabla de resultados:** tabla consolidada de las 14 series con el ganador
  absoluto resaltado, descargable en CSV.

## Actualizar los datos

Si vuelves a correr el notebook (Paso 40) y generas nuevas versiones de los 3 CSV,
solo reemplázalos dentro de `data/` manteniendo el mismo nombre y estructura de
columnas — el dashboard no requiere ningún cambio de código.
