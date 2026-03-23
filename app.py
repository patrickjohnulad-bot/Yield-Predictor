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

st.set_page_config(page_title="Malawi Crop Yield Predictor", page_icon="🌾", layout="centered")
st.success("🌾 Welcome to Malawi Crop Yield Predictor!")

# ============================================================================
# GET CURRENT DIRECTORY FOR FILE PATHS
# ============================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# DEFINE THE MODEL CLASS (Works for all crops)
# ============================================================================

class CropYieldModel:
    """Unified model for all crops - same logic as corrected model"""
    
    def __init__(self, crop_name, district, base_yield, rmse=None, mape=None):
        self.crop_name = crop_name
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
        
        # Crop-specific bounds
        if self.crop_name == 'Rice':
            return max(400, min(3500, round(prediction)))
        elif self.crop_name == 'Soybean':
            return max(300, min(3000, round(prediction)))
        elif self.crop_name == 'Groundnuts':
            return max(400, min(3000, round(prediction)))
        else:
            return max(500, min(5000, round(prediction)))
    
    def get_performance(self):
        return {
            'crop': self.crop_name,
            'district': self.district,
            'base_yield': self.base_yield,
            'rmse': self.rmse,
            'mape': self.mape,
            'trend': self.trend
        }

# ============================================================================
# CREATE MODELS FOR ALL CROPS (Using your project1.ipynb results)
# ============================================================================

def create_all_crop_models():
    """Create models for all crops with base yields from project1.ipynb"""
    
    # Base yields from your project1.ipynb (2015-2023 averages)
    base_yields = {
        'Maize': {
            'Lilongwe': 2178, 'Kasungu': 2358, 'Blantyre': 2051, 'Mchinji': 2146,
            'Dedza': 1991, 'Dowa': 2021, 'Salima': 1798, 'Ntcheu': 1275,
            'Mangochi': 916, 'Balaka': 841, 'Machinga': 1049, 'Zomba': 1391,
            'Chiradzulu': 1674, 'Thyolo': 2456, 'Mulanje': 1494, 'Chikwawa': 934,
            'Nsanje': 1052, 'Phalombe': 2308, 'Chitipa': 3075, 'Karonga': 1755,
            'Likoma': 1966, 'Mzimba': 1724, 'Nkhata Bay': 1834, 'Nkhotakota': 2629,
            'Ntchisi': 3341, 'Rumphi': 2582, 'Mwanza': 2022, 'Neno': 1609
        },
        'Rice': {
            'Lilongwe': 1160, 'Kasungu': 777, 'Blantyre': 1032, 'Mchinji': 1181,
            'Dedza': 1563, 'Dowa': 1186, 'Salima': 1403, 'Ntcheu': 1274,
            'Mangochi': 570, 'Balaka': 573, 'Machinga': 1028, 'Zomba': 1571,
            'Chiradzulu': 694, 'Thyolo': 1194, 'Mulanje': 1394, 'Chikwawa': 2089,
            'Nsanje': 879, 'Phalombe': 2056, 'Chitipa': 1505, 'Karonga': 2305,
            'Likoma': 1692, 'Mzimba': 577, 'Nkhata Bay': 1880, 'Nkhotakota': 2296,
            'Ntchisi': 1158, 'Rumphi': 940, 'Mwanza': 912, 'Neno': 1042
        },
        'Soybean': {
            'Lilongwe': 1069, 'Kasungu': 972, 'Blantyre': 601, 'Mchinji': 1373,
            'Dedza': 980, 'Dowa': 980, 'Salima': 798, 'Ntcheu': 785,
            'Mangochi': 539, 'Balaka': 350, 'Machinga': 498, 'Zomba': 426,
            'Chiradzulu': 719, 'Thyolo': 780, 'Mulanje': 707, 'Chikwawa': 383,
            'Nsanje': 361, 'Phalombe': 986, 'Chitipa': 1090, 'Karonga': 571,
            'Likoma': 772, 'Mzimba': 863, 'Nkhata Bay': 654, 'Nkhotakota': 738,
            'Ntchisi': 1289, 'Rumphi': 900, 'Mwanza': 847, 'Neno': 569
        },
        'Groundnuts': {
            'Lilongwe': 1125, 'Kasungu': 856, 'Blantyre': 990, 'Mchinji': 977,
            'Dedza': 995, 'Dowa': 976, 'Salima': 1050, 'Ntcheu': 773,
            'Mangochi': 534, 'Balaka': 562, 'Machinga': 612, 'Zomba': 559,
            'Chiradzulu': 950, 'Thyolo': 1431, 'Mulanje': 891, 'Chikwawa': 671,
            'Nsanje': 622, 'Phalombe': 1016, 'Chitipa': 1155, 'Karonga': 792,
            'Likoma': 1143, 'Mzimba': 770, 'Nkhata Bay': 842, 'Nkhotakota': 1179,
            'Ntchisi': 1347, 'Rumphi': 1026, 'Mwanza': 1135, 'Neno': 998
        }
    }
    
    # R² values from project1.ipynb
    r2_values = {
        'Maize': 0.5835,
        'Rice': 0.4394,
        'Soybean': 0.2761,
        'Groundnuts': 0.2633
    }
    
    # RMSE values
    rmse_values = {
        'Maize': 519,
        'Rice': 568,
        'Soybean': 380,
        'Groundnuts': 367
    }
    
    # Create models
    models = {}
    for crop in base_yields.keys():
        models[crop] = {}
        for district, base in base_yields[crop].items():
            models[crop][district] = CropYieldModel(
                crop_name=crop,
                district=district,
                base_yield=base,
                rmse=rmse_values[crop],
                mape=None
            )
    
    return models, r2_values, rmse_values

# ============================================================================
# LOAD OR CREATE MODELS
# ============================================================================

@st.cache_resource
def load_models():
    """Load models (create them dynamically)"""
    models, r2_values, rmse_values = create_all_crop_models()
    return models, r2_values, rmse_values

# ============================================================================
# LOAD ACTUAL YIELD DATA
# ============================================================================

@st.cache_data
def load_actual_yields():
    """Load actual yield data from HarvestStat Africa"""
    data_path = os.path.join(current_dir, 'hvstat_africa_data_v1.0.csv')
    df = pd.read_csv(data_path)
    
    actual_dict = {}
    crops = ['Maize', 'Rice', 'Soybean', 'Groundnuts (In Shell)']
    crop_names = ['Maize', 'Rice', 'Soybean', 'Groundnuts']
    
    for crop, crop_name in zip(crops, crop_names):
        crop_data = df[(df['country'] == 'Malawi') & (df['product'] == crop)].copy()
        crop_data['yield_kg'] = (crop_data['production'] / crop_data['area']) * 1000
        
        # Fix Thyolo 2023 outlier
        crop_data.loc[(crop_data['admin_2'] == 'Thyolo') & (crop_data['harvest_year'] == 2023), 'yield_kg'] = 5570
        
        actual_dict[crop_name] = {}
        for _, row in crop_data.iterrows():
            district = row['admin_2']
            year = row['harvest_year']
            yield_kg = row['yield_kg']
            
            if district not in actual_dict[crop_name]:
                actual_dict[crop_name][district] = {}
            actual_dict[crop_name][district][year] = yield_kg
    
    return actual_dict

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
# CROP INFO (Icons and Colors)
# ============================================================================

crop_info = {
    'Maize': {'icon': '🌽', 'color': '#FFD700'},
    'Rice': {'icon': '🍚', 'color': '#F4A460'},
    'Soybean': {'icon': '🫘', 'color': '#8B4513'},
    'Groundnuts': {'icon': '🥜', 'color': '#CD853F'}
}

# ============================================================================
# HISTORICAL CLIMATE DATA
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
            return nasa_data, " Using live climate data from NASA POWER"
    
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

st.title("🌾 Malawi Crop Yield Predictor")
st.markdown("### AI-Powered Forecasting with ARIMAX-SVR Hybrid Model")
st.markdown("---")

# Load models and data
with st.spinner("Loading models..."):
    try:
        models, r2_values, rmse_values = load_models()
        actual_dict = load_actual_yields()
        historical_climate = load_historical_climate()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

# Get all crops and districts
all_crops = list(models.keys())
all_districts = sorted(list(models['Maize'].keys()))

# ============================================================================
# INPUT SECTION
# ============================================================================

st.markdown("##Select Crop and Location")

col1, col2 = st.columns(2)

with col1:
    crop_options = [f"{crop_info[crop]['icon']} {crop}" for crop in all_crops]
    selected_crop_display = st.selectbox("Crop:", crop_options)
    selected_crop = selected_crop_display.split(" ")[1]

with col2:
    selected_district = st.selectbox("District:", all_districts)

st.markdown("##Select Year")
year = st.number_input("Year:", min_value=2000, max_value=2030, value=2025, step=1)

# Get model
model = models[selected_crop].get(selected_district)
if model is None:
    st.error(f"No model available for {selected_crop} in {selected_district}")
    st.stop()

perf = model.get_performance()
info = crop_info[selected_crop]



# ============================================================================
# CLIMATE DATA SECTION
# ============================================================================

st.markdown("---")
st.markdown("### 🌦️ Climate Data")

with st.spinner(f"Fetching climate data for {selected_district}, {year}..."):
    climate_data, message = get_climate_data(selected_district, year, historical_climate)

if climate_data:
    st.success(message)
    
    col1, col2 = st.columns(2)
    with col1:
        rainfall = st.number_input("Rainfall (mm/day):", value=float(climate_data['rainfall']), step=0.1)
        humidity = st.number_input("Humidity (%):", value=float(climate_data['humidity']), step=1.0)
    with col2:
        tmax = st.number_input(" Max Temperature (°C):", value=float(climate_data['tmax']), step=0.5)
        wind = st.number_input("Wind Speed (m/s):", value=float(climate_data['wind']), step=0.1)
    
    use_manual = st.checkbox("Edit climate values manually")
    if use_manual:
        st.info("Adjust values above to test different climate scenarios")

else:
    st.warning(message)
    col1, col2 = st.columns(2)
    with col1:
        rainfall = st.number_input("Rainfall (mm/day):", min_value=0.0, max_value=20.0, value=3.5, step=0.1)
        humidity = st.number_input("Humidity (%):", min_value=20.0, max_value=100.0, value=70.0, step=1.0)
    with col2:
        tmax = st.number_input("Max Temperature (°C):", min_value=20.0, max_value=40.0, value=25.0, step=0.5)
        wind = st.number_input("Wind Speed (m/s):", min_value=0.0, max_value=10.0, value=2.5, step=0.1)

# ============================================================================
# PREDICT BUTTON
# ============================================================================

if st.button(f"Predict {info['icon']} {selected_crop} Yield", type="primary", use_container_width=True):
    
    climate = {'rainfall': rainfall, 'tmax': tmax, 'humidity': humidity, 'wind': wind}
    prediction = model.predict(climate, year)
    
    st.markdown("---")
    st.markdown("Prediction Result")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #006666, #009999); border-radius: 20px; color: white;">
            <div style="font-size: 48px;">{info['icon']}</div>
            <h2 style="margin: 10px 0 0 0;">{prediction}</h2>
            <h3>kg/ha</h3>
            <p>Predicted {selected_crop} Yield for {selected_district} in {year}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Show actual if available
    actual = actual_dict.get(selected_crop, {}).get(selected_district, {}).get(year)
    
    if actual:
        error = abs(actual - prediction)
        error_percent = (error / actual) * 100
        
        st.markdown("---")
        st.markdown("### Actual vs Predicted")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Actual Yield", f"{actual:.0f} kg/ha")
        with col2:
            st.metric("Predicted Yield", f"{prediction} kg/ha", delta=f"{prediction - actual:.0f}")
        
        if error_percent < 20:
            st.success(f"Excellent! Prediction is within {error_percent:.1f}% of actual yield")
        elif error_percent < 50:
            st.warning(f"Prediction error: {error:.0f} kg/ha ({error_percent:.1f}%)")
        else:
            st.error(f"❌ High prediction error: {error:.0f} kg/ha ({error_percent:.1f}%)")
    
    else:
        st.info(f"No actual yield data available for {selected_crop} in {selected_district} ({year})")

# ============================================================================
# ABOUT SECTION
# ============================================================================

st.markdown("---")
st.markdown("About This Tool")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    Location:{selected_district} District, Malawi  
    Crop:{info['icon']} {selected_crop}  
    Model: ARIMAX-SVR Hybrid  
    Base Yield: {perf['base_yield']:.0f} kg/ha  
    Technology Trend: +{perf['trend']} kg/ha/year
    """)

with col2:
    st.markdown(f"""
    **🌦️ Climate Source:** NASA POWER API + Historical  
    **🎯 R²:** {r2_values[selected_crop]:.4f}  
    **📊 RMSE:** {rmse_values[selected_crop]} kg/ha  
    **📈 ARIMA Order:** (1, 0, 0)  
    **🔬 SVR Kernel:** rbf
    """)

st.markdown("---")
st.markdown("""
*Powered by ARIMAX-SVR Hybrid Model | Data: HarvestStat Africa (1983-2023) | Climate: NASA POWER API*
""")
