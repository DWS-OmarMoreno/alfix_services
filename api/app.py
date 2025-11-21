# -*- coding: utf-8 -*-
# /api/api_analysis.py
# -----------------------------------------------------------------------------
# Este script es una función serverless para Vercel que recibe 7 variables
# y devuelve un análisis completo: score, categoría, análisis de variables,
# recomendaciones y un cupo recomendado.
#
# CÓMO USAR:
# 1. Guarda tu modelo entrenado como 'alfix_model.pkl' en el mismo directorio /api.
# 2. Envía una solicitud POST al endpoint '/api/api_analysis' con un JSON
#    que contenga las 7 variables requeridas.
#
# Ejemplo de prueba con cURL:
# curl -X POST https://<tu-dominio-vercel>.app/api/api_analysis \
# -H "Content-Type: application/json" \
# -d '{
#   "profit_cont_ops": 1305954.0,
#   "total_equity": 4008240.0,
#   "total_liab_cur_excl_disposal": 119646380.0,
#   "total_liab_cur_ex_hfs": 2604677.0,
#   "nonfin_liab_other_cur": 3787620.0,
#   "fin_liab_other_cur": 10368470.0,
#   "prov_cur_total": 2564660.0
# }'
# -----------------------------------------------------------------------------

from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import bisect
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# --- 1. Cargar Artefactos y Constantes ---

#try:
    # Cargar modelo
    
    #model_path = os.path.join(os.path.dirname(__file__), 'alfix_model.pkl')
    #model = joblib.load(model_path)
#except FileNotFoundError:

model = None

def get_model():
    global model
    if model is None:
        model_path = os.path.join(os.path.dirname(__file__), 'alfix_model.pkl')
        model = joblib.load('alfix_model.pkl')
    return model

# Parámetros de escalamiento del score
SCORING_OFFSET = 437.9502843417596
SCORING_FACTOR = 95.25948800838954

# Lista de las 7 variables finales
FINAL_COLUMNS = [
    'profit_cont_ops', 'total_equity', 'total_liab_cur_excl_disposal',
    'total_liab_cur_ex_hfs', 'nonfin_liab_other_cur',
    'fin_liab_other_cur', 'prov_cur_total'
]

# --- Constantes del Notebook (Params, Meta, Consejos, Cupos) ---

params = {
    "profit_cont_ops": {"mean": 4.016873e+06, "p25": 3.296000e+03, "p50": 7.443860e+05, "p75": 3.911169e+06},
    "total_equity": {"mean": 2.599826e+08, "p25": 7.245390e+06, "p50": 3.483531e+07, "p75": 2.130313e+08},
    "total_liab_cur_excl_disposal": {"mean": 3.244306e+08, "p25": 9.307000e+07, "p50": 1.196464e+08, "p75": 1.555798e+08},
    "total_liab_cur_ex_hfs": {"mean": 5.578154e+06, "p25": 2.377950e+06, "p50": 2.377950e+06, "p75": 2.377950e+06},
    "nonfin_liab_other_cur": {"mean": 2.198911e+07, "p25": 2.253170e+06, "p50": 3.787620e+06, "p75": 5.434850e+06},
    "fin_liab_other_cur": {"mean": 6.297570e+07, "p25": 4.722060e+06, "p50": 1.036847e+07, "p75": 2.205482e+07},
    "prov_cur_total": {"mean": 1.316643e+07, "p25": 1.122660e+06, "p50": 2.564660e+06, "p75": 5.515620e+06}
}

meta = {
    "score": {"label": "Puntaje Riesgo Crediticio"},
    "profit_cont_ops": {"label": "Utilidad operativa"},
    "total_equity": {"label": "Patrimonio total"},
    "total_liab_cur_excl_disposal": {"label": "Pasivos corrientes sin disposiciones"},
    "total_liab_cur_ex_hfs": {"label": "Pasivos corrientes ajustados"},
    "nonfin_liab_other_cur": {"label": "Otros pasivos corrientes no financieros"},
    "fin_liab_other_cur": {"label": "Otros pasivos corrientes financieros"},
    "prov_cur_total": {"label": "Provisiones corrientes totales"},
}

definiciones = {
    "profit_cont_ops": "¿Cuál fue la utilidad operacional de tu empresa en el último año?",
    "total_equity": "¿Cuál es el valor total del patrimonio de tu empresa?",
    "total_liab_cur_excl_disposal": "¿A cuánto ascienden las deudas y obligaciones de corto plazo de tu empresa (≤ 1 año)?",
    "total_liab_cur_ex_hfs": "Indica el total de pasivos corrientes (obligaciones que vencen en el corto plazo).",
    "nonfin_liab_other_cur": "¿Obligaciones de corto plazo no financieras? Monto total.",
    "fin_liab_other_cur": "¿Valor de las deudas financieras de corto plazo (créditos, leasing, pagarés)?",
    "prov_cur_total": "¿Provisiones de corto plazo (litigios, indemnizaciones, obligaciones fiscales)? ¿Por cuánto?",
}

consejos = {
    "profit_cont_ops": {
        "bajo": "Tu margen operativo es bajo. Revisa tus precios y el tipo de productos que ofreces. Reduce gastos innecesarios, renegocia con proveedores y busca formas de hacer más eficiente la operación.",
        "medio-bajo": "Tu rentabilidad puede mejorar. Controla los gastos operativos, estandariza procesos, y busca agilizar la rotación de inventarios y cuentas por cobrar.",
        "medio-alto": "Tu gestión es buena. Mantén el control de costos, revisa variaciones entre presupuestos y resultados, y considera automatizar procesos clave para mejorar la eficiencia.",
        "alto": "Tu rentabilidad operativa es sólida. Documenta las buenas prácticas, protege tus márgenes ante posibles aumentos de costos y monitorea la calidad de tus ingresos."
    },
    "total_equity": {
        "bajo": "Tu patrimonio es débil. Considera retener más utilidades, reducir retiros, y revisar posibles pérdidas acumuladas para fortalecer la estructura financiera.",
        "medio-bajo": "Tu solidez patrimonial puede mejorar. Aumenta la rentabilidad, ajusta la política de dividendos y mejora el control del gasto para fortalecer el capital propio.",
        "medio-alto": "Tu nivel de patrimonio es saludable. Mantén controlado el nivel de endeudamiento y utiliza parte de las utilidades para reforzar las reservas de capital.",
        "alto": "Tienes una estructura patrimonial sólida. Mantén políticas claras de reinversión y asegúrate de conservar un margen de seguridad ante posibles cambios del entorno."
    },
    "total_liab_cur_excl_disposal": {
        "bajo": "Tus deudas de corto plazo están en niveles sanos. Mantén una buena relación con proveedores y evita tomar deuda que no necesites.",
        "medio-bajo": "Tus obligaciones de corto plazo son manejables. Asegúrate de coordinar los plazos de cobro y pago para mantener un flujo de caja equilibrado.",
        "medio-alto": "Tienes cierta presión de caja. Acelera la cobranza, mejora la rotación de inventarios y busca descuentos por pronto pago con tus proveedores.",
        "alto": "Tu nivel de deuda de corto plazo es alto. Renegocia plazos con tus acreedores, refinancia parte a largo plazo y refuerza tus políticas de crédito y cobro."
    },
    "total_liab_cur_ex_hfs": {
        "bajo": "Tu manejo de obligaciones de corto plazo es adecuado. Sigue cumpliendo puntualmente con los pagos.",
        "medio-bajo": "Refina tu calendario de pagos para priorizar las obligaciones más importantes y evitar retrasos innecesarios.",
        "medio-alto": "Tienes cierta concentración de deuda. Revisa los principales acreedores y evita depender demasiado de uno solo.",
        "alto": "Tienes alta exposición en pasivos de corto plazo. Considera refinanciar parte a plazos más largos y organiza mejor tus flujos de pago para reducir el riesgo."
    },
    "nonfin_liab_other_cur": {
        "bajo": "Estás cumpliendo bien tus obligaciones no financieras. Mantén tus pagos y compromisos al día.",
        "medio-bajo": "Ajusta los acuerdos con proveedores y evita atrasos que puedan generar multas o intereses.",
        "medio-alto": "Mejora la gestión de pagos y aprobaciones para evitar retrasos. Negocia plazos más convenientes si es posible.",
        "alto": "Tienes atrasos en tus compromisos no financieros. Negocia planes de pago, prioriza las obligaciones fiscales y evita sanciones o sobrecostos."
    },
    "fin_liab_other_cur": {
        "bajo": "Tus deudas financieras de corto plazo están bajo control. Aun así, revisa si puedes reducir costos financieros.",
        "medio-bajo": "Monitorea tus créditos. Compara tasas y usa las líneas solo cuando realmente se necesiten para cubrir estacionalidades.",
        "medio-alto": "Tu deuda de corto plazo es significativa. Evalúa mover parte a largo plazo y define límites internos para no sobreendeudarte.",
        "alto": "Tienes un nivel alto de deuda de corto plazo. Renegocia tasas y plazos, refinancia parte a largo plazo y evita depender de créditos rotativos."
    },
    "prov_cur_total": {
        "bajo": "Tus provisiones son razonables. Asegúrate de documentar bien los criterios que usas para calcularlas.",
        "medio-bajo": "Revisa tus contingencias y actualiza las provisiones con información reciente para evitar subestimaciones.",
        "medio-alto": "Tus provisiones son elevadas. Intenta cerrar litigios o acuerdos pendientes y respalda las estimaciones con evidencia sólida.",
        "alto": "Tus provisiones son muy altas. Identifica las causas principales y busca soluciones, como seguros o acuerdos, para reducir los riesgos futuros."
    }
}

# Bandas por categoría de score -> % del patrimonio
CUPOS_POR_SCORE = {
    "Muy Bueno": 0.30,  # hasta 30% del patrimonio
    "Bueno":     0.20,  # hasta 20%
    "Medio":     0.10,  # hasta 10%
    "Malo":      0.03,  # tope bajo
    "Muy Malo":  0.00   # sin cupo
}

# Ajuste por presión de corto plazo: razón Pasivo Corriente / Patrimonio
AJUSTE_LIQ = [
    (0.50, 1.00), (1.00, 0.85), (2.00, 0.70), (float("inf"), 0.50)
]

# Ajuste por deuda financiera de corto plazo dentro del pasivo corriente
AJUSTE_FIN_CP = [
    (0.10, 1.00), (0.30, 0.90), (0.50, 0.80), (float("inf"), 0.65)
]


# --- 2. Funciones de Análisis del Notebook ---

def pd_to_score(pd_value, offset=SCORING_OFFSET, factor=SCORING_FACTOR):
    """Convierte Probabilidad de Default (PD) a Score."""
    # Asegura que pd_value esté dentro de un rango seguro para evitar log(0) o división por cero
    pd_safe = max(1e-9, min(pd_value, 1 - 1e-9))
    return offset + factor * np.log((1 - pd_safe) / pd_safe)

def score_classify(s):
    """Clasifica un score (0-1000) en categorías de riesgo."""
    if s < 0 or s > 1000: return "fuera de rango"
    bins   = [0, 200, 400, 600, 800, 1000]
    labels = ["Muy Malo", "Malo", "Medio", "Bueno", "Muy Bueno"]
    if s == 1000: return "Muy Bueno"
    i = bisect.bisect_left(bins, s) - 1
    return labels[i]

def _nivel_por_percentil(valor: float, p: dict) -> str:
    """Clasifica un valor en 'bajo', 'medio-bajo', 'medio-alto', 'alto' según sus percentiles."""
    if valor < p["p25"]:
        return "bajo"
    elif valor < p["p50"]:
        return "medio-bajo"
    elif valor < p["p75"]:
        return "medio-alto"
    else:
        return "alto"

def analyze_variables(sample: dict, params: dict, meta: dict) -> list:
    """Compara cada variable del sample con su media y percentiles."""
    resultados = []
    for var, val in sample.items():
        if var not in params or var not in meta:
            continue
        
        p = params[var]
        m = p["mean"]
        
        # Comparativa vs media
        desvio = (val - m) / m if m != 0 else (val - m)
        lado = "por encima" if desvio > 0 else "por debajo" if desvio < 0 else "igual a"
        
        # Percentil más cercano
        difs = {k: abs(val - p[k]) for k in ("p25", "p50", "p75")}
        cercano = min(difs, key=difs.get)
        pct = {"p25": 25, "p50": 50, "p75": 75}[cercano]
        
        resultados.append({
            "variable": var,
            "etiqueta": meta[var]['label'],
            "valor": f"{val:,.0f}",
            "comparativa_media": f"{abs(desvio)*100:.1f}% {lado} de la media.",
            "distribucion": f"Valor cercano al percentil {pct}."
        })
    return resultados

def recomendar_por_variable(sample: dict, params: dict, meta: dict, consejos: dict, definiciones: dict) -> list:
    """Genera recomendaciones para cada variable basado en su nivel de percentil."""
    resultados = []
    for var, val in sample.items():
        if var not in params or var not in meta or var not in consejos or var not in definiciones:
            continue
            
        p = params[var]
        nivel = _nivel_por_percentil(val, p)
        sug = consejos[var][nivel]
        etiqueta = meta[var]["label"]
        pregunta = definiciones[var]

        resultados.append({
            "variable": var,
            "etiqueta": etiqueta,
            "valor": float(val),
            "nivel": nivel,
            "consejo": sug,
            "captura_dato": pregunta
        })
    return resultados

def _tramo(valor, tramos):
    """Devuelve el multiplicador según el primer umbral que el valor no excede."""
    for umbral, mult in tramos:
        if valor <= umbral:
            return mult
    return tramos[-1][1] # Devuelve el último multiplicador si excede todos los umbrales

def cupo_recomendado(sample: dict, pd_hat: float, score_raw: float, categoria: str) -> dict:
    """Calcula el cupo recomendado basado en el score y ratios financieros."""
    
    # 1) Cupo base por patrimonio (usa la categoría ya calculada)
    pct_equity = CUPOS_POR_SCORE.get(categoria, 0.0)
    equity = float(sample["total_equity"])
    cupo_base = max(0.0, pct_equity * max(equity, 0.0))

    # 2) Ajuste por presión de corto plazo
    liab_cp = float(sample["total_liab_cur_excl_disposal"])
    razon_lc_equity = liab_cp / max(equity, 1.0) # Evita división por cero si equity es 0
    mult_liq = _tramo(razon_lc_equity, AJUSTE_LIQ)

    # 3) Ajuste por concentración de deuda financiera de corto plazo
    fin_cp = float(sample["fin_liab_other_cur"])
    razon_fin_sobre_pc = fin_cp / max(liab_cp, 1.0) # Evita división por cero
    mult_fin = _tramo(razon_fin_sobre_pc, AJUSTE_FIN_CP)

    # 4) Cupo final
    mult_total = mult_liq * mult_fin
    cupo_final = cupo_base * mult_total

    return {
        "pd": pd_hat,
        "score_raw": score_raw,
        "categoria": categoria,
        "porcentaje_equity_base": pct_equity,
        "equity": equity,
        "cupo_base_por_equity": cupo_base,
        "razon_pc_equity": razon_lc_equity,
        "ajuste_liquidez": mult_liq,
        "razon_fin_cp_sobre_pc": razon_fin_sobre_pc,
        "ajuste_financiero_cp": mult_fin,
        "multiplicador_total": mult_total,
        "cupo_recomendado": cupo_final
    }

# --- 3. Definir el Endpoint de la API ---

@app.route('/api_analysis', methods=['POST'])
def handler():
    """
    Recibe los datos financieros, calcula el score y devuelve un análisis completo.
    """
    if model is None:
        return jsonify({"error": "Modelo no encontrado. Asegúrate de que 'alfix_model.pkl' esté en el directorio /api."}), 500

    try:
        # --- 3.1. Validar los Datos de Entrada ---
        user_data = request.get_json()
        if not user_data:
            return jsonify({"error": "No se recibieron datos en formato JSON."}), 400

        missing_keys = [key for key in FINAL_COLUMNS if key not in user_data]
        if missing_keys:
            return jsonify({"error": f"Faltan las siguientes variables: {', '.join(missing_keys)}"}), 400

        # --- 3.2. Preparar Datos y Calcular Score (UNA SOLA VEZ) ---
        input_df = pd.DataFrame([user_data], columns=FINAL_COLUMNS)
        
        # Predecir Probabilidad de Default (PD)
        #pd_probability = model.predict_proba(input_df)[0][1]
        
        mdl = get_model()
        pd_probability = mdl.predict_proba(input_df)[0][1]


        # Calcular Score (crudo y final)
        score_raw = pd_to_score(pd_probability) # Usa la función con protección
        score_final = max(0, min(1000, round(score_raw)))
        
        # Clasificar
        categoria = score_classify(score_final)
        
        # --- 3.3. Ejecutar Análisis ---
        
        # Análisis de variables vs. percentiles/media
        analisis_vars = analyze_variables(user_data, params, meta)
        
        # Recomendaciones por variable
        recomendaciones_list = recomendar_por_variable(user_data, params, meta, consejos, definiciones)
        
        # Cálculo de cupo (reutilizando valores)
        cupo_info = cupo_recomendado(user_data, pd_probability, score_raw, categoria)

        # --- 3.4. Devolver el Resultado Completo ---
        output = {
            "score_calculado": {
                "score": score_final,
                "categoria": categoria,
                "pd_estimada": pd_probability,
                "score_raw": score_raw
            },
            "cupo_recomendado": cupo_info,
            "analisis_variables": analisis_vars,
            "recomendaciones": recomendaciones_list
        }
        
        return jsonify(output)

    except Exception as e:
        # Capturar cualquier otro error durante la ejecución
        return jsonify({"error": f"Error interno del servidor: {str(e)}"}), 500
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)