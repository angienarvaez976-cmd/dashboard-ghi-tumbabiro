# =============================================================
# Dashboard interactivo — Pronóstico de GHI, Planta Fotovoltaica Tumbabiro
# Proyecto de titulación — Comparación de modelos clásicos vs. Machine Learning
# =============================================================
# Insumos (generados en el notebook, Paso 40):
#   - data/predicciones_todas_series.csv
#   - data/resumen_comparacion.csv
#   - data/importancia_variables.csv
#
# Ejecutar con:  streamlit run app.py

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# -----------------------------------------------------------------
# Configuración general de la página
# -----------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard GHI — Tumbabiro",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------
# Estilos: tipografía de portada + efecto "pasar página" al cambiar
# de sección (el contenedor de cada sección tiene una key distinta,
# así que al cambiar de sección React lo vuelve a montar y la
# animación se reproduce; al interactuar dentro de la misma sección
# la key no cambia y no se repite la animación).
# -----------------------------------------------------------------
st.markdown(
    """
    <style>
    @keyframes pageTurn {
        0%   { opacity: 0; transform: translateX(35px) rotateY(-6deg); }
        100% { opacity: 1; transform: translateX(0)     rotateY(0deg); }
    }
    div[class*="st-key-page_"] {
        animation: pageTurn 0.55s ease;
        transform-origin: left center;
    }
    div[class*="st-key-portada_"] {
        animation: pageTurn 0.7s ease;
        transform-origin: left center;
    }
    .portada-uce {
        font-family: Georgia, 'Times New Roman', serif;
        text-align: center;
        padding: 0.5rem 1rem 1rem 1rem;
    }
    .portada-uce .institucion {
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 0.15rem;
    }
    .portada-uce .facultad {
        font-size: 1.05rem;
        color: #444;
        margin-bottom: 1.2rem;
    }
    .portada-uce hr {
        width: 60%;
        margin: 1.5rem auto;
        border: none;
        border-top: 2px solid #DD8452;
    }
    .portada-uce .titulo-tesis {
        font-size: 1.65rem;
        font-weight: 700;
        line-height: 1.4;
        max-width: 850px;
        margin: 0 auto 1.8rem auto;
        color: #1a1a1a;
    }
    .portada-uce .subtitulo {
        font-size: 1.02rem;
        font-style: italic;
        color: #555;
        max-width: 700px;
        margin: 0 auto;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODELOS_COLOR = {
    "Real": "#1a1a1a",
    "SARIMAX": "#4C72B0",
    "Random Forest": "#DD8452",
    "XGBoost": "#55A868",
    "LSTM": "#8172B2",
}

NOMBRE_SERIE_LEGIBLE = {
    "GHI_Diaria": "Diaria (7h–17h)",
    "GHI_Manana": "Mañana (7h–11h)",
    "GHI_Tarde": "Tarde (12h–17h)",
}
for h in range(7, 18):
    NOMBRE_SERIE_LEGIBLE[f"GHI_hora_{h:02d}h"] = f"Hora {h:02d}h"


# -----------------------------------------------------------------
# Carga de datos (cacheada)
# -----------------------------------------------------------------
@st.cache_data
def cargar_datos():
    pred = pd.read_csv("data/predicciones_todas_series.csv", parse_dates=["fecha"])
    resumen = pd.read_csv("data/resumen_comparacion.csv")
    importancia = pd.read_csv("data/importancia_variables.csv")
    return pred, resumen, importancia


pred_df, resumen_df, imp_df = cargar_datos()
SERIES = list(resumen_df["Serie"])

# -----------------------------------------------------------------
# Navegación: barra superior con íconos + botones anterior/siguiente
# -----------------------------------------------------------------
SECCIONES = [
    {"nombre": "Portada", "icono": "🎓"},
    {"nombre": "Resumen general", "icono": "📊"},
    {"nombre": "Comparación por serie", "icono": "📈"},
    {"nombre": "Importancia de variables", "icono": "🧩"},
    {"nombre": "Tabla de resultados", "icono": "📋"},
]

if "page_idx" not in st.session_state:
    st.session_state.page_idx = 0

st.markdown(
    """
    <style>
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button {
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h3 style='margin-bottom:0.2rem;'>☀️ Dashboard Tumbabiro</h3>"
    "<p style='color:#888; margin-top:0; margin-bottom:0.8rem;'>"
    "Pronóstico de Irradiancia Global Horizontal (GHI) — Modelos clásicos vs. Machine Learning"
    "</p>",
    unsafe_allow_html=True,
)

nav_cols = st.columns(len(SECCIONES))
for i, sec in enumerate(SECCIONES):
    with nav_cols[i]:
        es_activa = st.session_state.page_idx == i
        if st.button(
            f"{sec['icono']}  {sec['nombre']}",
            key=f"nav_{i}",
            type="primary" if es_activa else "secondary",
            use_container_width=True,
        ):
            st.session_state.page_idx = i
            st.rerun()

st.divider()

seccion = SECCIONES[st.session_state.page_idx]["nombre"]


def boton_navegacion(idx):
    """Botones Anterior/Siguiente a mitad del dashboard, como pasar página."""
    st.markdown("<div style='margin-top:2.5rem;'></div>", unsafe_allow_html=True)
    st.divider()
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if idx > 0:
            if st.button("⬅  Sección anterior", key=f"prev_{idx}", use_container_width=True):
                st.session_state.page_idx = idx - 1
                st.rerun()
    with c2:
        st.markdown(
            f"<p style='text-align:center; color:#999;'>Sección {idx + 1} de {len(SECCIONES)}</p>",
            unsafe_allow_html=True,
        )
    with c3:
        if idx < len(SECCIONES) - 1:
            if st.button("Siguiente sección  ➡", key=f"next_{idx}", use_container_width=True):
                st.session_state.page_idx = idx + 1
                st.rerun()


# ===================================================================
# SECCIÓN 0 — PORTADA
# ===================================================================
if seccion == "Portada":
    with st.container(key="portada_page"):
        st.markdown(
            '<div class="portada-uce">'
            '<div class="institucion">UNIVERSIDAD CENTRAL DEL ECUADOR</div>'
            '<div class="facultad">Facultad de Ciencias Económicas</div>'
            '<hr>'
            '<div class="titulo-tesis">'
            'Análisis predictivo de la Irradiancia Global Horizontal (GHI) '
            'mediante modelos clásicos y modelos de aprendizaje supervisado: '
            'Caso Tumbabiro'
            '</div>'
            '<div class="subtitulo">'
            'Trabajo de titulación – Opción: Proyecto de investigación '
            'presentado para obtener el grado académico de Ingeniera '
            'Estadística.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown(
            "<p style='text-align:center; color:#777;'>"
            "Usa el menú superior o el botón de abajo para explorar los resultados: "
            "resumen general, comparación por serie, importancia de variables y la "
            "tabla consolidada de resultados."
            "</p>",
            unsafe_allow_html=True,
        )
        boton_navegacion(st.session_state.page_idx)

# ===================================================================
# SECCIÓN 1 — RESUMEN GENERAL
# ===================================================================
elif seccion == "Resumen general":
    with st.container(key="page_resumen"):
        st.title("Resumen general de resultados")
        st.markdown(
            "Comparación del modelo ganador por cada una de las **14 series de GHI** "
            "(diaria, mañana, tarde y 11 series horarias), evaluadas con MAPE sobre el "
            "conjunto de prueba (80/20, partición cronológica)."
        )

        # --- KPIs ---
        conteo = resumen_df["Ganador_absoluto"].value_counts()
        mape_prom = resumen_df["MAPE_ganador"].mean()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAPE promedio (mejor modelo por serie)", f"{mape_prom:.2f}%")
        col2.metric("Series totales analizadas", f"{len(resumen_df)}")
        col3.metric(
            "Modelo más frecuentemente ganador",
            conteo.idxmax(),
            f"{conteo.max()} de {len(resumen_df)} series",
        )
        col4.metric(
            "Mejor serie individual (menor MAPE)",
            resumen_df.loc[resumen_df["MAPE_ganador"].idxmin(), "Serie"],
            f"{resumen_df['MAPE_ganador'].min():.2f}%",
        )

        st.divider()

        c1, c2 = st.columns([1, 1.4])

        with c1:
            st.subheader("Conteo de victorias por modelo")
            conteo_df = conteo.reset_index()
            conteo_df.columns = ["Modelo", "Series ganadas"]
            fig_conteo = px.bar(
                conteo_df,
                x="Modelo",
                y="Series ganadas",
                color="Modelo",
                color_discrete_map={
                    "Random Forest": MODELOS_COLOR["Random Forest"],
                    "XGBoost": MODELOS_COLOR["XGBoost"],
                    "SARIMAX exógenas": MODELOS_COLOR["SARIMAX"],
                    "LSTM": MODELOS_COLOR["LSTM"],
                },
                text="Series ganadas",
            )
            fig_conteo.update_layout(showlegend=False, yaxis_title="N° de series")
            st.plotly_chart(fig_conteo, use_container_width=True)

        with c2:
            st.subheader("Mapa de calor — MAPE (%) por serie y modelo")
            heat_df = resumen_df.set_index("Serie")[
                ["MAPE_mejor", "MAPE_RF", "MAPE_XGB", "MAPE_LSTM"]
            ]
            heat_df.columns = ["Mejor clásico", "Random Forest", "XGBoost", "LSTM"]
            heat_df.index = [NOMBRE_SERIE_LEGIBLE.get(s, s) for s in heat_df.index]

            fig_heat = px.imshow(
                heat_df,
                text_auto=".1f",
                color_continuous_scale="RdYlGn_r",
                aspect="auto",
                labels=dict(color="MAPE (%)"),
            )
            fig_heat.update_layout(height=500)
            st.plotly_chart(fig_heat, use_container_width=True)

        st.info(
            "💡 **Lectura del mapa de calor:** verde = menor error (mejor), rojo = mayor error. "
            "Random Forest y XGBoost dominan en las horas centrales del día (07h–14h), mientras "
            "que SARIMAX con exógenas es superior en las horas de la tarde (15h–17h)."
        )
        boton_navegacion(st.session_state.page_idx)

# ===================================================================
# SECCIÓN 2 — COMPARACIÓN POR SERIE
# ===================================================================
elif seccion == "Comparación por serie":
    with st.container(key="page_comparacion"):
        st.title("Comparación de pronósticos por serie")

        col_sel, col_info = st.columns([1, 2])
        with col_sel:
            serie_sel = st.selectbox(
                "Selecciona una serie de GHI",
                SERIES,
                format_func=lambda s: NOMBRE_SERIE_LEGIBLE.get(s, s),
            )

        fila = resumen_df[resumen_df["Serie"] == serie_sel].iloc[0]
        df_serie = pred_df[pred_df["serie"] == serie_sel].sort_values("fecha")

        with col_info:
            st.markdown(
                f"**Ganador absoluto:** {fila['Ganador_absoluto']} "
                f"(MAPE = {fila['MAPE_ganador']:.2f}%)  \n"
                f"**Mejor enfoque clásico:** {fila['Mejor_enfoque']} "
                f"(MAPE = {fila['MAPE_mejor']:.2f}%)"
            )

        # --- Métricas por modelo ---
        st.subheader("MAPE por modelo — " + NOMBRE_SERIE_LEGIBLE.get(serie_sel, serie_sel))
        metricas = {
            "SARIMAX exógenas / mejor clásico": fila["MAPE_mejor"],
            "Random Forest": fila["MAPE_RF"],
            "XGBoost": fila["MAPE_XGB"],
            "LSTM": fila["MAPE_LSTM"],
        }
        ganador_valor = min(metricas.values())
        cols = st.columns(4)
        for c, (nombre, valor) in zip(cols, metricas.items()):
            delta = "🏆 Ganador" if valor == ganador_valor else None
            c.metric(nombre, f"{valor:.2f}%", delta)

        st.divider()

        # --- Rango de fechas ---
        fecha_min, fecha_max = df_serie["fecha"].min(), df_serie["fecha"].max()
        rango = st.slider(
            "Rango de fechas (conjunto de prueba)",
            min_value=fecha_min.to_pydatetime(),
            max_value=fecha_max.to_pydatetime(),
            value=(fecha_min.to_pydatetime(), fecha_max.to_pydatetime()),
            format="DD-MMM",
        )
        df_plot = df_serie[(df_serie["fecha"] >= rango[0]) & (df_serie["fecha"] <= rango[1])]

        modelos_mostrar = st.multiselect(
            "Modelos a mostrar",
            ["SARIMAX", "Random Forest", "XGBoost", "LSTM"],
            default=["SARIMAX", "Random Forest", "XGBoost", "LSTM"],
        )

        # --- Gráfico comparativo ---
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_plot["fecha"], y=df_plot["GHI_real"],
                name="Real", mode="lines",
                line=dict(color=MODELOS_COLOR["Real"], width=3),
            )
        )
        columnas_modelo = {
            "SARIMAX": "pred_SARIMAX",
            "Random Forest": "pred_RF",
            "XGBoost": "pred_XGBoost",
            "LSTM": "pred_LSTM",
        }
        for modelo in modelos_mostrar:
            col = columnas_modelo[modelo]
            fig.add_trace(
                go.Scatter(
                    x=df_plot["fecha"], y=df_plot[col],
                    name=modelo, mode="lines",
                    line=dict(color=MODELOS_COLOR[modelo], width=1.6, dash="dash"),
                )
            )
        fig.update_layout(
            title=f"Real vs. pronósticos — {NOMBRE_SERIE_LEGIBLE.get(serie_sel, serie_sel)}",
            xaxis_title="Fecha",
            yaxis_title="GHI (Wh/m²)",
            hovermode="x unified",
            height=520,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Tabla de datos + descarga ---
        with st.expander("Ver tabla de datos de esta serie"):
            st.dataframe(df_plot, use_container_width=True, hide_index=True)
            st.download_button(
                "Descargar CSV de esta serie",
                df_plot.to_csv(index=False).encode("utf-8"),
                file_name=f"predicciones_{serie_sel}.csv",
                mime="text/csv",
            )
        boton_navegacion(st.session_state.page_idx)

# ===================================================================
# SECCIÓN 3 — IMPORTANCIA DE VARIABLES
# ===================================================================
elif seccion == "Importancia de variables":
    with st.container(key="page_importancia"):
        st.title("Importancia de variables (Random Forest / XGBoost)")
        st.markdown(
            "Importancia relativa de cada variable exógena en la predicción de GHI, "
            "según el modelo de ensamble seleccionado."
        )

        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            serie_sel = st.selectbox(
                "Serie",
                SERIES,
                format_func=lambda s: NOMBRE_SERIE_LEGIBLE.get(s, s),
                key="imp_serie",
            )
        with col_sel2:
            modelo_sel = st.radio("Modelo", ["Random Forest", "XGBoost", "Ambos"], horizontal=True)

        df_imp = imp_df[imp_df["Serie"] == serie_sel].copy()
        if modelo_sel != "Ambos":
            df_imp = df_imp[df_imp["Modelo"] == modelo_sel]

        df_imp = df_imp.sort_values("Importancia", ascending=True)

        fig_imp = px.bar(
            df_imp,
            x="Importancia",
            y="Variable",
            color="Modelo",
            orientation="h",
            barmode="group",
            color_discrete_map={
                "Random Forest": MODELOS_COLOR["Random Forest"],
                "XGBoost": MODELOS_COLOR["XGBoost"],
            },
            text_auto=".3f",
        )
        fig_imp.update_layout(
            title=f"Importancia de variables — {NOMBRE_SERIE_LEGIBLE.get(serie_sel, serie_sel)}",
            xaxis_title="Importancia relativa",
            yaxis_title="",
            height=450,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        st.subheader("Importancia promedio por variable (las 14 series)")
        prom_var = (
            imp_df.groupby(["Variable", "Modelo"])["Importancia"]
            .mean()
            .reset_index()
            .sort_values("Importancia", ascending=False)
        )
        fig_prom = px.bar(
            prom_var,
            x="Variable",
            y="Importancia",
            color="Modelo",
            barmode="group",
            color_discrete_map={
                "Random Forest": MODELOS_COLOR["Random Forest"],
                "XGBoost": MODELOS_COLOR["XGBoost"],
            },
        )
        fig_prom.update_layout(yaxis_title="Importancia promedio", xaxis_title="")
        st.plotly_chart(fig_prom, use_container_width=True)
        boton_navegacion(st.session_state.page_idx)

# ===================================================================
# SECCIÓN 4 — TABLA DE RESULTADOS
# ===================================================================
elif seccion == "Tabla de resultados":
    with st.container(key="page_tabla"):
        st.title("Tabla comparativa completa (14 series)")
        st.markdown(
            "Tabla consolidada con el MAPE de cada enfoque por serie y el ganador absoluto. "
            "Los valores mínimos de cada fila están resaltados en verde."
        )

        tabla = resumen_df.copy()
        tabla["Serie"] = tabla["Serie"].map(lambda s: NOMBRE_SERIE_LEGIBLE.get(s, s))
        tabla_mostrar = tabla[
            ["Serie", "Mejor_enfoque", "MAPE_mejor", "MAPE_RF", "MAPE_XGB", "MAPE_LSTM",
             "Ganador_absoluto", "MAPE_ganador"]
        ].rename(columns={
            "Mejor_enfoque": "Mejor enfoque clásico",
            "MAPE_mejor": "MAPE clásico (%)",
            "MAPE_RF": "MAPE RF (%)",
            "MAPE_XGB": "MAPE XGB (%)",
            "MAPE_LSTM": "MAPE LSTM (%)",
            "Ganador_absoluto": "Ganador absoluto",
            "MAPE_ganador": "MAPE ganador (%)",
        })

        def resaltar_min(row):
            cols_mape = ["MAPE clásico (%)", "MAPE RF (%)", "MAPE XGB (%)", "MAPE LSTM (%)"]
            minimo = row[cols_mape].min()
            return [
                "background-color: #c6efce" if (c in cols_mape and row[c] == minimo) else ""
                for c in row.index
            ]

        st.dataframe(
            tabla_mostrar.style.apply(resaltar_min, axis=1).format(
                {c: "{:.2f}" for c in
                 ["MAPE clásico (%)", "MAPE RF (%)", "MAPE XGB (%)", "MAPE LSTM (%)", "MAPE ganador (%)"]}
            ),
            use_container_width=True,
            hide_index=True,
            height=560,
        )

        st.download_button(
            "Descargar tabla completa (CSV)",
            tabla_mostrar.to_csv(index=False).encode("utf-8"),
            file_name="resumen_comparacion_dashboard.csv",
            mime="text/csv",
        )

        st.caption(
            "Fuente: Paso 40 del pipeline de modelado (notebook), consolidando SARIMAX/ARIMA, "
            "Random Forest, XGBoost y LSTM (mejor configuración por grid search) sobre las "
            "14 series de GHI, partición cronológica 80/20."
        )
        boton_navegacion(st.session_state.page_idx)
