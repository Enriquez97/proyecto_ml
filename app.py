import streamlit as st
import pandas as pd
import joblib
from train_model import train_churn_model  # Importar la función de entrenamiento

# Configuración de página
st.set_page_config(
    page_title="Proyecto Grupo 3",
    #page_icon="",
    page_icon="assets/logo-upn-nuevo.svg",
    layout="wide"
)

# CSS personalizado
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        height: 3em;
    }
    .stMetric-value {
        font-size: 3em !important;
    }
    </style>
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
        }
    </style>
    """, unsafe_allow_html=True)

# Función para cargar modelo
@st.cache_resource
def load_model():
    try:
        model = joblib.load('churn_model_pipeline.pkl')
        return model
    except Exception as e:
        return None

# --- NAVEGACIÓN LATERAL ---
st.sidebar.image("assets/logo-upn-nuevo.svg", use_container_width=True)
st.sidebar.title("Proyecto Grupo 3")
page = st.sidebar.radio("Ir a:", ["Predicción (Consumir Modelo)", "Entrenamiento del Modelo"])

# ==========================================
# PÁGINA 1: PREDICCIÓN
# ==========================================
if page == "Predicción (Consumir Modelo)":
    st.title("Predicción de Abandono (Churn)")
    st.markdown("Ingrese los detalles del cliente para predecir la probabilidad de abandono.")

    model = load_model()
    
    if model is None:
        st.error("⚠️ No se encontró ningún modelo entrenado. Por favor, ve a la página de 'Entrenamiento' primero.")
    else:
        
        with st.form("churn_prediction_form"):
            st.header("📝 Detalles del Cliente")
            
            col_1, col_2, col_3, col_4 = st.columns(4)
            
            
            with col_1:
                st.subheader("📊 Numéricos")
                tenure = st.number_input("Antigüedad (Meses)", min_value=0, max_value=120, value=12)
                monthly_charges = st.number_input("Cargos Mensuales ($)", min_value=0.0, value=50.0)
                total_charges = st.number_input("Cargos Totales ($)", min_value=0.0, value=500.0)

            
            with col_2:
                st.subheader("👤 Demografía")
                gender = st.selectbox("Género", ['Male', 'Female'])
                senior_citizen = st.selectbox("Cliente Anciano", ['No', 'Yes'])
                partner = st.selectbox("Pareja", ['No', 'Yes'])
                dependents = st.selectbox("Dependientes", ['No', 'Yes'])

           
            with col_3:
                st.subheader("🛠️ Servicios")
                phone_service = st.selectbox("Telf. Servicio", ['Yes', 'No'])
                multiple_lines = st.selectbox("Líneas Múltiples", ['No', 'Yes', 'No phone service'])
                internet_service = st.selectbox("Internet", ['DSL', 'Fiber optic', 'No'])
                online_security = st.selectbox("Seguridad", ['Yes', 'No', 'No internet service'])
                online_backup = st.selectbox("Backup", ['Yes', 'No', 'No internet service'])

            
            with col_4:
                st.subheader("🛠️ Servicios")
                device_protection = st.selectbox("Protección Disp.", ['No', 'Yes', 'No internet service'])
                streaming_movies = st.selectbox("Streaming Pelis", ['No', 'Yes', 'No internet service'])
                # Cuenta
                st.subheader("💳 Cuenta")
                contract = st.selectbox("Contrato", ['Month-to-month', 'Two year', 'One year'])
                payment_method = st.selectbox("Pago", ['Mailed check', 'Electronic check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
                

            
            with st.expander("Otras opciones (Tech Support, TV, Facturación...)"):
                ex_col1, ex_col2, ex_col3 = st.columns(3)
                with ex_col1:
                    tech_support = st.selectbox("Soporte Técnico", ['No', 'Yes', 'No internet service'])
                with ex_col2:
                    streaming_tv = st.selectbox("Streaming TV", ['No', 'Yes', 'No internet service'])
                with ex_col3:
                    paperless_billing = st.selectbox("Facturación sin Papel", ['Yes', 'No'])

            submitted = st.form_submit_button("Predecir Churn")

        if submitted:
            # Preparar datos de entrada
            input_data = pd.DataFrame({
                'Tenure Months': [tenure],
                'Monthly Charges': [monthly_charges],
                'Total Charges': [total_charges],
                'Gender': [gender],
                'Senior Citizen': [senior_citizen],
                'Partner': [partner],
                'Dependents': [dependents],
                'Phone Service': [phone_service],
                'Multiple Lines': [multiple_lines],
                'Internet Service': [internet_service],
                'Online Security': [online_security],
                'Online Backup': [online_backup],
                'Device Protection': [device_protection],
                'Tech Support': [tech_support],
                'Streaming TV': [streaming_tv],
                'Streaming Movies': [streaming_movies],
                'Contract': [contract],
                'Paperless Billing': [paperless_billing],
                'Payment Method': [payment_method]
            })
            
            with st.spinner("Procesando..."):
                prediction = model.predict(input_data)[0]
                prediction_proba = model.predict_proba(input_data)[0][1] # Probabilidad de Churn (1)

            st.divider()
            r_col1, r_col2 = st.columns([2, 1])

            with r_col1:
                st.subheader("Resultado de la Predicción")
                if prediction_proba < 0.4:
                    st.success("✅ **Bajo Riesgo de Abandono**")
                    st.markdown(f"Es probable que este cliente se quede.")
                elif prediction_proba < 0.7:
                    st.warning("⚠️ **Riesgo Medio de Abandono**")
                    st.markdown(f"El riesgo de abandono es moderado. Se recomienda seguimiento.")
                else:
                    st.error("🚨 **Alto Riesgo de Abandono**")
                    st.markdown(f"Es muy probable que este cliente abandone el servicio.")
                    
            with r_col2:
                st.subheader("Probabilidad")
                st.metric(label="Riesgo", value=f"{prediction_proba:.2%}")
                st.progress(prediction_proba)

# ==========================================
# PÁGINA 2: ENTRENAMIENTO
# ==========================================
elif page == "Entrenamiento del Modelo":
    st.title("Entrenamiento del Modelo")
    st.markdown("Aquí podemos re-entrenar el modelo ajustando sus hiperparámetros.")

    with st.container():
        col_params, col_action = st.columns([2, 1])
        
        with col_params:
            st.subheader("Hiperparámetros")
            max_iter = st.slider("Iteraciones Máximas (max_iter)", 100, 5000, 1000, step=100)
            c_param = st.number_input("Fuerza de Regularización (C) - Menor valor = Mayor regularización", 0.01, 10.0, 1.0, step=0.01)

        with col_action:
            st.write("")
            st.write("") 
            train_btn = st.button("🚀 Iniciar Entrenamiento")

    if train_btn:
        with st.status("Entrenando modelo...", expanded=True) as status:
            st.write("Cargando datos y preprocesando...")
           
            model, metrics = train_churn_model('./data/Telco_customer_churn.xlsx', max_iter=max_iter, c_parameter=c_param)
            
            if model:
                st.write("Entrenamiento completado.")
                st.write("Guardando modelo en disco...")
                status.update(label="¡Entrenamiento Exitoso!", state="complete", expanded=False)
                
                # Mostrar métricas
                st.divider()
                st.subheader("📊 Métricas del Nuevo Modelo")
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Exactitud (Accuracy)", f"{metrics['accuracy']:.2%}")
                m_col2.metric("Precisión (Clase 1)", f"{metrics['report']['1']['precision']:.2%}")
                m_col3.metric("Recall (Clase 1)", f"{metrics['report']['1']['recall']:.2%}")
                
                st.success("El modelo ha sido actualizado y guardado como `churn_model_pipeline.pkl`. Ya puedes usarlo en la pestaña de Predicción.")
                
                
                st.cache_resource.clear()
            else:
                status.update(label="Error en el entrenamiento", state="error")
                st.error(metrics) 
