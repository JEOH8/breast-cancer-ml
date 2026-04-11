# -*- coding: utf-8 -*-
"""
app.py
======
Demo interactiva de clasificación de cáncer de mama.
Desplegada en Streamlit Community Cloud.

Dos modos:
  1. Predicción manual (30 características morfológicas)
  2. Predicción por lote (CSV con múltiples pacientes)
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Configuración de página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Clasificador de Cáncer de Mama",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cargar pipeline ────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    pipeline     = joblib.load("models/full_pipeline.joblib")
    feature_names = joblib.load("models/feature_names.joblib")
    feature_stats = joblib.load("models/feature_stats.joblib")
    return pipeline, feature_names, feature_stats

pipeline, feature_names, feature_stats = load_pipeline()

# ── Descripciones de features ──────────────────────────────────────────────
FEATURE_DESCRIPTIONS = {
    "radius_mean":             "Radio medio (media de distancias centro-perímetro)",
    "texture_mean":            "Textura media (desv. estándar de escala de grises)",
    "perimeter_mean":          "Perímetro medio del núcleo",
    "area_mean":               "Área media del núcleo",
    "smoothness_mean":         "Suavidad media (variación local en longitudes de radio)",
    "compactness_mean":        "Compacidad media (perímetro² / área - 1.0)",
    "concavity_mean":          "Concavidad media (severidad de porciones cóncavas)",
    "concave points_mean":     "Puntos cóncavos medios (número de porciones cóncavas)",
    "symmetry_mean":           "Simetría media",
    "fractal_dimension_mean":  "Dimensión fractal media ('aproximación costera' - 1)",
    "radius_se":               "Error estándar del radio",
    "texture_se":              "Error estándar de la textura",
    "perimeter_se":            "Error estándar del perímetro",
    "area_se":                 "Error estándar del área",
    "smoothness_se":           "Error estándar de la suavidad",
    "compactness_se":          "Error estándar de la compacidad",
    "concavity_se":            "Error estándar de la concavidad",
    "concave points_se":       "Error estándar de los puntos cóncavos",
    "symmetry_se":             "Error estándar de la simetría",
    "fractal_dimension_se":    "Error estándar de la dimensión fractal",
    "radius_worst":            "Radio peor (media de los 3 valores más grandes)",
    "texture_worst":           "Textura peor",
    "perimeter_worst":         "Perímetro peor",
    "area_worst":              "Área peor",
    "smoothness_worst":        "Suavidad peor",
    "compactness_worst":       "Compacidad peor",
    "concavity_worst":         "Concavidad peor",
    "concave points_worst":    "Puntos cóncavos peor",
    "symmetry_worst":          "Simetría peor",
    "fractal_dimension_worst": "Dimensión fractal peor",
}

# ── Funciones auxiliares ───────────────────────────────────────────────────

def predict_single(values: dict) -> dict:
    """Predice para una muestra de 30 features."""
    df    = pd.DataFrame([values])[feature_names]
    proba = pipeline.predict_proba(df)[0]
    pred  = pipeline.predict(df)[0]
    return {
        "prediction":        "Maligno" if pred == 1 else "Benigno",
        "prob_malignant":    float(proba[1]),
        "prob_benign":       float(proba[0]),
        "confidence":        float(max(proba)),
    }

def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Predice para un DataFrame con múltiples pacientes."""
    df_features = df[feature_names]
    probas      = pipeline.predict_proba(df_features)
    preds       = pipeline.predict(df_features)
    df_result   = df.copy()
    df_result["Diagnóstico"]          = ["Maligno" if p == 1 else "Benigno" for p in preds]
    df_result["Prob. Maligno (%)"]    = (probas[:, 1] * 100).round(2)
    df_result["Prob. Benigno (%)"]    = (probas[:, 0] * 100).round(2)
    df_result["Confianza (%)"]        = (np.max(probas, axis=1) * 100).round(2)
    return df_result

def gauge_chart(prob_malignant: float) -> go.Figure:
    """Medidor visual de probabilidad."""
    color = "#e74c3c" if prob_malignant >= 0.5 else "#2ecc71"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_malignant * 100,
        number={"suffix": "%", "font": {"size": 36}},
        title={"text": "Probabilidad de Malignidad", "font": {"size": 16}},
        gauge={
            "axis":  {"range": [0, 100], "tickwidth": 1},
            "bar":   {"color": color},
            "steps": [
                {"range": [0,  40], "color": "#d5f5e3"},
                {"range": [40, 60], "color": "#fdebd0"},
                {"range": [60, 100], "color": "#fadbd8"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=0, l=20, r=20))
    return fig


# ── Header ─────────────────────────────────────────────────────────────────
st.title("🔬 Clasificador de Cáncer de Mama")
st.markdown(
    "Sistema de apoyo diagnóstico basado en características morfológicas celulares. "
    "Desarrollado sobre el **Wisconsin Breast Cancer Dataset** con Regresión Logística "
    "optimizada mediante Optimización Bayesiana."
)

# Métricas del modelo
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy",  "99.12%")
col2.metric("Precision", "100.00%")
col3.metric("Recall",    "97.62%")
col4.metric("F1-Score",  "98.80%")
col5.metric("AUC",       "0.998")

st.divider()

# Advertencia médica
st.warning(
    "⚠️ **Aviso:** Este sistema es una herramienta de apoyo académica/investigativa. "
    "No reemplaza el diagnóstico clínico profesional. "
    "Ante cualquier resultado, consulte siempre con un especialista médico."
)

# ── Tabs principales ───────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🧪 Predicción Manual",
    "📂 Predicción por Lote (CSV)",
    "📊 Acerca del Modelo",
])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICCIÓN MANUAL
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Ingrese las características morfológicas del tumor")
    st.markdown(
        "Ajuste los valores de las 30 características. "
        "Los valores predeterminados corresponden a la **media del dataset**."
    )

    st.info(
        "🔬 **Contexto clínico de los datos**\n\n"
        "Las 30 características morfológicas de este modelo **no provienen de mamografías**. "
        "Se calculan computacionalmente a partir de imágenes digitalizadas de "
        "**biopsias de aspiración con aguja fina (FNA)**, analizadas bajo microscopio. "
        "Variables como radio, textura, concavidad y dimensión fractal son extraídas "
        "automáticamente por software especializado de análisis de imagen (originalmente WBC).\n\n"
        "En un entorno clínico real, estos valores serían calculados de forma automática "
        "a partir de la muestra celular digitalizada — no ingresados manualmente. "
        "Esta demo ilustra el funcionamiento del modelo predictivo con fines académicos e investigativos."
    )

    # Organizar features en 3 grupos
    groups = {
        "📐 Características Medias (mean)": [f for f in feature_names if "_mean" in f],
        "📏 Errores Estándar (se)":         [f for f in feature_names if "_se" in f],
        "⚠️  Valores Peores (worst)":       [f for f in feature_names if "_worst" in f],
    }

    input_values = {}

    for group_name, group_features in groups.items():
        with st.expander(group_name, expanded=(group_name == "📐 Características Medias (mean)")):
            cols = st.columns(2)
            for i, feat in enumerate(group_features):
                with cols[i % 2]:
                    val = st.slider(
                        label=feat,
                        min_value=float(feature_stats["min"][feat]),
                        max_value=float(feature_stats["max"][feat]),
                        value=float(feature_stats["mean"][feat]),
                        format="%.4f",
                        help=FEATURE_DESCRIPTIONS.get(feat, feat),
                    )
                    input_values[feat] = val

    st.divider()

    if st.button("🔍 Predecir", type="primary", use_container_width=True):
        result = predict_single(input_values)

        col_gauge, col_result = st.columns([1, 1])

        with col_gauge:
            st.plotly_chart(gauge_chart(result["prob_malignant"]),
                            use_container_width=True)

        with col_result:
            st.markdown("### Resultado")
            if result["prediction"] == "Maligno":
                st.error(f"🔴 **{result['prediction']}**")
                st.markdown("**Recomendación:** CONSULTAR CON ONCÓLOGO")
            else:
                st.success(f"🟢 **{result['prediction']}**")
                st.markdown("**Recomendación:** Seguimiento rutinario")

            st.markdown("---")
            st.metric("Prob. Maligno",  f"{result['prob_malignant']*100:.2f}%")
            st.metric("Prob. Benigno",  f"{result['prob_benign']*100:.2f}%")
            st.metric("Confianza",      f"{result['confidence']*100:.2f}%")

            st.markdown("---")
            st.caption(
                "El modelo aplica internamente StandardScaler + PCA (17 componentes, "
                "99.1% varianza) + Regresión Logística optimizada con Optimización Bayesiana."
            )


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDICCIÓN POR LOTE (CSV)
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Subir archivo CSV con múltiples pacientes")

    st.markdown(
        "El CSV debe contener las **30 columnas de características** del dataset Wisconsin. "
        "Puede incluir columnas adicionales (ID, nombre) — solo se usarán las 30 features."
    )

    # Plantilla de descarga
    template_df = pd.DataFrame(
        [list(feature_stats["mean"].values())],
        columns=feature_names
    )
    st.download_button(
        label="⬇️ Descargar plantilla CSV",
        data=template_df.to_csv(index=False),
        file_name="plantilla_pacientes.csv",
        mime="text/csv",
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Seleccionar archivo CSV",
        type=["csv"],
        help="El archivo debe contener las 30 columnas de características."
    )

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            st.success(f"✓ Archivo cargado: {len(df_input)} pacientes, {len(df_input.columns)} columnas")

            # Verificar columnas
            missing_cols = [c for c in feature_names if c not in df_input.columns]
            if missing_cols:
                st.error(f"Columnas faltantes: {missing_cols}")
            else:
                st.dataframe(df_input.head(), use_container_width=True)

                if st.button("🔍 Predecir todos", type="primary"):
                    with st.spinner("Procesando..."):
                        df_result = predict_batch(df_input)

                    st.subheader("Resultados")

                    # Resumen
                    n_malignant = (df_result["Diagnóstico"] == "Maligno").sum()
                    n_benign    = (df_result["Diagnóstico"] == "Benigno").sum()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total pacientes", len(df_result))
                    c2.metric("Malignos detectados", n_malignant,
                              delta=f"{n_malignant/len(df_result)*100:.1f}%")
                    c3.metric("Benignos",            n_benign,
                              delta=f"{n_benign/len(df_result)*100:.1f}%")

                    # Gráfico de distribución
                    fig_dist = px.histogram(
                        df_result, x="Prob. Maligno (%)",
                        color="Diagnóstico",
                        color_discrete_map={"Maligno": "#e74c3c", "Benigno": "#2ecc71"},
                        nbins=20,
                        title="Distribución de Probabilidades de Malignidad",
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

                    # Tabla de resultados
                    result_cols = ["Diagnóstico", "Prob. Maligno (%)",
                                   "Prob. Benigno (%)", "Confianza (%)"]
                    extra_cols  = [c for c in df_input.columns if c not in feature_names]
                    display_cols = extra_cols + result_cols
                    st.dataframe(
                        df_result[display_cols].style.applymap(
                            lambda v: "background-color: #fadbd8" if v == "Maligno"
                            else "background-color: #d5f5e3" if v == "Benigno" else "",
                            subset=["Diagnóstico"]
                        ),
                        use_container_width=True,
                    )

                    # Descargar resultados
                    st.download_button(
                        label="⬇️ Descargar resultados CSV",
                        data=df_result.to_csv(index=False),
                        file_name="resultados_prediccion.csv",
                        mime="text/csv",
                    )

        except Exception as e:
            st.error(f"Error procesando el archivo: {e}")


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — ACERCA DEL MODELO
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Pipeline y métricas del modelo")

    col_pipe, col_metrics = st.columns([1, 1])

    with col_pipe:
        st.markdown("### Pipeline sin data leakage")
        st.code("""
1. Carga del Wisconsin Breast Cancer Dataset
2. Preprocesamiento estructural
3. Split estratificado (80% train / 20% test)
   ↑ aquí se separan los datos
4. StandardScaler  ← fit SOLO en train
5. PCA (17 componentes, 99.1% varianza) ← fit SOLO en train
6. Regresión Logística optimizada con
   BayesSearchCV (50 iteraciones, CV 5-fold)
7. Validación cruzada 10-fold
        """, language="text")

        st.markdown("### Verificación de data leakage")
        leakage_data = {
            "Métrica":      ["Accuracy", "Precision", "Recall", "F1", "AUC"],
            "CV-10 (train)": ["97.58%", "97.18%", "96.47%", "96.73%", "99.42%"],
            "Holdout (test)": ["99.12%", "100.00%", "97.62%", "98.80%", "99.80%"],
            "Δ":             ["+1.54% ✅", "+2.82% ⚠️", "+1.15% ✅", "+2.06% ⚠️", "+0.38% ✅"],
        }
        st.dataframe(pd.DataFrame(leakage_data), use_container_width=True)
        st.caption(
            "⚠️ Diferencia positiva (holdout > CV) NO indica leakage — "
            "el leakage se manifiesta como caída en test. "
            "Varianza esperable con 114 muestras de prueba."
        )

    with col_metrics:
        st.markdown("### Comparativa de modelos")
        metrics_data = {
            "Modelo": [
                "Reg. Logística (base)", "KNN (k=5)", "SVM (RBF)",
                "Árbol de Decisión", "Random Forest",
                "Reg. Logística (optimizada) ⭐", "Red Neuronal Profunda"
            ],
            "Accuracy":  [0.9825, 0.9561, 0.9649, 0.9474, 0.9474, 0.9912, 0.9912],
            "Precision": [1.0000, 0.9744, 0.9750, 0.9500, 0.9500, 1.0000, 1.0000],
            "Recall":    [0.9524, 0.9048, 0.9286, 0.9048, 0.9048, 0.9762, 0.9762],
            "F1":        [0.9756, 0.9383, 0.9512, 0.9268, 0.9268, 0.9880, 0.9880],
            "AUC":       [0.9967, 0.9823, 0.9954, 0.9382, 0.9919, 0.9980, 0.9970],
        }
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
        st.caption(
            "* AUC de la Red Neuronal varía entre 0.995–0.999 entre ejecuciones "
            "por aleatoriedad de TensorFlow en CPU — comportamiento documentado y esperado."
        )

        st.markdown("### Conclusiones clave")
        st.info(
            "**PCA fue determinante:** eliminó 21 pares de variables con correlación >0.9, "
            "reduciendo dimensionalidad 43% sin perder información relevante.\n\n"
            "**Precision = 1.0:** cuando el modelo dice 'Maligno', siempre tiene razón. "
            "Cero falsos positivos en el conjunto de prueba.\n\n"
            "**Regresión Logística ≈ Red Neuronal:** en datos tabulares de escala moderada, "
            "un modelo lineal bien optimizado iguala al deep learning con mayor estabilidad "
            "e interpretabilidad — ventaja crítica en contextos clínicos."
        )

    st.divider()
    st.markdown(
        "**Repositorio completo:** [github.com/JEOH8/breast-cancer-ml](https://github.com/JEOH8/breast-cancer-ml)  \n"
        "**Desarrollado por:** Juan Esteban Ospina — Ingeniero Biomédico  \n"
        "**Stack:** Python · scikit-learn · TensorFlow · Streamlit · Plotly"
    )
