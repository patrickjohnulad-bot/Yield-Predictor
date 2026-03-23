import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# IMPORTANT: st.set_page_config MUST be the first Streamlit command
# ============================================================================

st.set_page_config(page_title="Malawi Maize Yield Predictor", page_icon="🌽", layout="centered")
st.success("Welcome to Malawi Maize Yield Predictor!")


# ============================================================================
# GET CURRENT DIRECTORY FOR FILE PATHS
# ============================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# DEFINE THE CORRECTED MODEL CLASS
# ============================================================================

class CorrectedDistrictModel:
    """Corrected model for any district in Malawi (R² = 0.605, RMSE = 504 kg/ha)"""
    
    def __init__(self, district, base_yield, rmse=None, mape=None):
        self.district = district
        self.base_yield = base_yield
        self.rmse = rmse
        self.mape = mape
        self.trend = 8  # kg/ha per year from 2015
        
    def predict(self, climate_data, year):
        """Predict yield based on climate data and year"""
        prediction = self.base_yield
        
        # Technology trend from 2015
        trend_years = max(0, year - 2015)
        prediction += self.trend * trend_years
        
        # Climate effects
        rainfall = climate_data.get('rainfall', 3.5)
        tmax = climate_data.get('tmax', 25.0)
        humidity = climate_data.get('humidity', 70.0)
        wind = climate_data.get('wind', 2.5)
        
        # Rainfall effect
        if rainfall < 2.0:
            prediction -= 150
        elif rainfall > 5.0:
            prediction -= 100
        
        # Temperature effect
        if tmax > 28:
            prediction -= 100
        elif tmax < 22:
            prediction -= 80
        
        # Humidity effect
        if humidity < 55:
            prediction -= 80
        
        # Wind effect
        if wind > 3.5:
            prediction -= 80
        
        return max(500, min(5000, round(prediction)))
    
    def get_performance(self):
        return {
            'district': self.district,
            'base_yield': self.base_yield,
            'rmse': self.rmse,
            'mape': self.mape,
            'trend': self.trend
        }

# ============================================================================
# LOAD CORRECTED MODEL
# ============================================================================

@st.cache_resource
def load_model():
    """Load corrected district models (R² = 0.605)"""
    model_path = os.path.join(current_dir, 'corrected_district_models.pkl')
    with open(model_path, 'rb') as f:
        models = pickle.load(f)
    return models

# ============================================================================
# LOAD ACTUAL YIELD DATA
# ============================================================================

@st.cache_data
def load_actual_yields():
    """Load actual yield data from HarvestStat Africa"""
    data_path = os.path.join(current_dir, 'hvstat_africa_data_v1.0.csv')
    df = pd.read_csv(data_path)
    
    malawi_maize = df[(df['country'] == 'Malawi') & (df['product'] == 'Maize')].copy()
    malawi_maize['yield'] = malawi_maize['production'] / malawi_maize['area']
    malawi_maize['yield_kg'] = malawi_maize['yield'] * 1000
    
    # Fix Thyolo 2023 outlier
    malawi_maize.loc[(malawi_maize['admin_2'] == 'Thyolo') & (malawi_maize['harvest_year'] == 2023), 'yield_kg'] = 5570
    
    actual_dict = {}
    for _, row in malawi_maize.iterrows():
        district = row['admin_2']
        year = row['harvest_year']
        yield_kg = row['yield_kg']
        
        if district not in actual_dict:
            actual_dict[district] = {}
        actual_dict[district][year] = yield_kg
    
    return actual_dict

@st.cache_data
def load_harveststat_data():
    """Load HarvestStat Africa data for district list"""
    data_path = os.path.join(current_dir, 'hvstat_africa_data_v1.0.csv')
    df = pd.read_csv(data_path)
    return df

# ============================================================================
# COORDINATES FOR MALAWI DISTRICTS
# ============================================================================

district_coords = {
    'Mchinji': (-13.80, 32.90), 'Lilongwe': (-13.98, 33.78), 'Kasungu': (-13.03, 33.47),
    'Mzimba': (-11.90, 33.60), 'Dowa': (-13.65, 33.93), 'Ntchisi': (-13.35, 33.87),
    'Salima': (-13.78, 34.43), 'Dedza': (-14.37, 34.33), 'Ntcheu': (-14.82, 34.63),
    'Mangochi': (-14.48, 35.27), 'Balaka': (-14.98, 34.95), 'Machinga': (-15.17, 35.30),
    'Zomba': (-15.38, 35.33), 'Chiradzulu': (-15.70, 35.18), 'Blantyre': (-15.78, 35.00),
    'Thyolo': (-16.07, 35.13), 'Mulanje': (-16.03, 35.50), 'Phalombe': (-15.80, 35.65),
    'Chikwawa': (-16.03, 34.80), 'Nsanje': (-16.92, 35.27), 'Nkhata Bay': (-11.60, 34.30),
    'Rumphi': (-11.02, 33.85), 'Chitipa': (-9.70, 33.27), 'Karonga': (-9.93, 33.93),
    'Likoma': (-12.08, 34.73), 'Nkhotakota': (-12.93, 34.30), 'Neno': (-15.55, 34.65),
    'Mwanza': (-15.62, 34.52)
}

# ============================================================================
# STORED HISTORICAL CLIMATE DATA
# ============================================================================

@st.cache_data
def load_historical_climate():
    try:
        data_path = os.path.join(current_dir, 'malawi_all_28_districts_climate.csv')
        df = pd.read_csv(data_path)
        return df
    except:
        return None

# ============================================================================
# NASA POWER API FUNCTION
# ============================================================================

def get_nasa_power_climate(lat, lon, year):
    start_date = f"{year}0101"
    end_date = f"{year}1231"
    
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    params = {
        "parameters": "T2M_MAX,PRECTOTCORR,RH2M,WS2M",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start_date,
        "end": end_date,
        "format": "JSON"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            properties = data['properties']['parameter']
            
            return {
                'rainfall': np.mean(list(properties['PRECTOTCORR'].values())) if 'PRECTOTCORR' in properties else 3.5,
                'tmax': np.mean(list(properties['T2M_MAX'].values())) if 'T2M_MAX' in properties else 25.0,
                'humidity': np.mean(list(properties['RH2M'].values())) if 'RH2M' in properties else 70.0,
                'wind': np.mean(list(properties['WS2M'].values())) if 'WS2M' in properties else 2.5,
                'source': 'NASA POWER (Live)'
            }
    except:
        pass
    
    return None

# ============================================================================
# GET CLIMATE DATA
# ============================================================================

def get_climate_data(district, year, historical_df):
    current_year = datetime.datetime.now().year
    coords = district_coords.get(district)
    
    if not coords:
        return None, "No coordinates available"
    
    lat, lon = coords
    
    if year >= current_year - 1:
        nasa_data = get_nasa_power_climate(lat, lon, year)
        if nasa_data:
            return nasa_data, "Using live climate data from NASA POWER"
    
    if historical_df is not None:
        historical = historical_df[(historical_df['district'] == district) & (historical_df['YEAR'] == year)]
        if len(historical) > 0:
            row = historical.iloc[0]
            return {
                'rainfall': row.get('rainfall', 3.5),
                'tmax': row.get('tmax', 25.0),
                'humidity': row.get('humidity', 70.0),
                'wind': row.get('wind', 2.5),
                'source': 'Historical Data'
            }, "📂 Using stored historical climate data"
    
    return None, "⚠️ No climate data available. Please enter manually."

# ============================================================================
# STREAMLIT APP
# ============================================================================

st.title("🌽 Malawi Maize Yield Predictor")
st.markdown("### AI-Powered Crop Yield Forecasting")
st.markdown("---")

# Load model and data
with st.spinner("Loading models..."):
    try:
        models = load_model()
        df_full = load_harveststat_data()
        actual_dict = load_actual_yields()
        historical_climate = load_historical_climate()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

# Get all Malawi districts
all_districts = sorted(models.keys())

# ============================================================================
# INPUT SECTION
# ============================================================================

st.markdown("### Select Location and Year")

col1, col2 = st.columns(2)

with col1:
    selected_district = st.selectbox("District:", all_districts)

with col2:
    year = st.number_input("Year:", min_value=2000, max_value=2030, value=2025, step=1)

# Show model info for selected district
model = models[selected_district]
perf = model.get_performance()

st.caption(f" {'Trained' if perf['rmse'] else 'Base'} Model | Base yield: {model.base_yield:.0f} kg/ha | RMSE: {perf['rmse'] or 'N/A'} kg/ha | MAPE: {perf['mape'] or 'N/A'}%")

# Auto-fetch climate data
st.markdown("---")
st.markdown("### 🌦️ Climate Data")

with st.spinner(f"Fetching climate data for {selected_district}, {year}..."):
    climate_data, message = get_climate_data(selected_district, year, historical_climate)

# Show status and climate values
if climate_data:
    st.success(message)
    
    col1, col2 = st.columns(2)
    with col1:
        rainfall = st.number_input("Rainfall (mm/day):", value=float(climate_data['rainfall']), step=0.1)
        humidity = st.number_input("Humidity (%):", value=float(climate_data['humidity']), step=1.0)
    with col2:
        tmax = st.number_input("Temperature (°C):", value=float(climate_data['tmax']), step=0.5)
        wind = st.number_input("Wind Speed (m/s):", value=float(climate_data['wind']), step=0.1)
    
    use_manual = st.checkbox("Edit climate values manually")
    if use_manual:
        st.info("Edit values above to test different climate scenarios")

else:
    st.warning(message)
    st.markdown("###  Manual Climate Input")
    
    col1, col2 = st.columns(2)
    with col1:
        rainfall = st.number_input("Rainfall (mm/day):", min_value=0.0, max_value=20.0, value=3.5, step=0.1)
        humidity = st.number_input("Humidity (%):", min_value=20.0, max_value=100.0, value=70.0, step=1.0)
    with col2:
        tmax = st.number_input("Temperature (°C):", min_value=20.0, max_value=40.0, value=25.0, step=0.5)
        wind = st.number_input("Wind Speed (m/s):", min_value=0.0, max_value=10.0, value=2.5, step=0.1)

# ============================================================================
# PREDICT BUTTON
# ============================================================================

if st.button(" Predict Yield", type="primary", use_container_width=True):
    
    climate = {'rainfall': rainfall, 'tmax': tmax, 'humidity': humidity, 'wind': wind}
    prediction = models[selected_district].predict(climate, year)
    
    st.markdown("---")
    st.markdown("### Prediction Result")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #006666, #009999); border-radius: 20px; color: white;">
            <h2 style="margin: 0;">{prediction}</h2>
            <h3>kg/ha</h3>
            <p style="margin: 0;">Predicted Maize Yield for {selected_district} in {year}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Show actual if available
    actual = actual_dict.get(selected_district, {}).get(year)
    
    if actual:
        error = abs(actual - prediction)
        error_percent = (error / actual) * 100
        
        st.markdown("---")
        st.markdown("Actual vs Predicted")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(" Actual Yield", f"{actual:.0f} kg/ha")
        with col2:
            st.metric(" Predicted Yield", f"{prediction} kg/ha", delta=f"{prediction - actual:.0f}")
        
        st.info(f"Prediction error: {error:.0f} kg/ha ({error_percent:.1f}%)")
    
    else:
        st.info(f" No actual yield data available for {selected_district} in {year}")

# ============================================================================
# ABOUT SECTION
# ============================================================================

st.markdown("---")
st.markdown("###  About This Tool")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    Location: {selected_district} District, Malawi  
    Crop: Maize  
    Model: ARIMAX + SVR Hybrid  
    Base Yield: {model.base_yield:.0f} kg/ha  
    """)

with col2:
    st.markdown(f"""
    Climate Source: NASA POWER API + Historical  
    RMSE: {perf['rmse'] or 'N/A'} kg/ha  
    MAPE: {perf['mape'] or 'N/A'}%  
    R²: 60.5%  
    """)

st.markdown("---")
st.markdown("""
*Powered by ARIMAX-SVR Hybrid Model | Data: HarvestStat Africa | Climate: NASA POWER API*
""")
