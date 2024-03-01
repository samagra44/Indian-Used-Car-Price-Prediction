import streamlit as st
import numpy as np
import pickle
import joblib
# Load the trained model using pickle
with open('model.pkl', 'rb') as model_file:
    rfr = pickle.load(model_file)

st.set_page_config(
    page_title='Indian Used Car Price Prediction 🚗',
    page_icon="🚗",
    layout='wide',
)

# Features label encoding
company_labels = {'MARUTI SUZUKI': 12, 'HYUNDAI': 7, 'HONDA': 19, 'MAHINDRA': 5, 'TATA': 13, 'FORD': 21,
                  'TOYOTA': 11, 'RENAULT': 6, 'VOLKSWAGEN': 17, 'KIA': 16, 'SKODA': 9, 'JEEP': 4,
                  'NISSAN': 20, 'MERCEDES BENZ': 10, 'AUDI': 1, 'MG': 3, 'BMW': 18, 'ISUZU': 14,
                  'CHEVROLET': 0, 'DATSUN': 8, 'VOLVO': 22, 'MITSUBISHI': 15, 'FIAT': 2}

fuel_type_labels = {'PETROL': 4, 'DIESEL': 1, 'CNG': 0, 'LPG': 2, 'ELECTRIC': 5, 'HYBRID': 3}

colour_labels = {'Silver': 61, 'Red': 56, 'Grey': 34, 'A Blue': 0, 'Black': 9, 'Blue': 11, 'Steel Grey': 66,
                 'Moondust Silver': 47, 'Orange': 49, 'Marine Blue': 38, 'Brown': 14, 'White': 71,
                 'White Red': 72, 'Gold': 30, 'Yellow': 74, 'Pearl White': 52, 'Maroon': 39, 'Forest Dew': 28,
                 'Silky Silver': 60, 'Beige': 7, 'Purple': 54, 'Silver Grey': 62, 'Metalic Grey': 40,
                 'Bottle Green': 13, 'Cherry Red': 20, 'Warm Grey': 70, 'Smoke Grey': 63,
                 'Bluish Silver Met.': 40, 'Dark Grey': 13, 'Dark Blue': 16, 'Intense Black': 27,
                 'Diamond White': 53, 'G. Red': 25, 'Burgundy': 10, 'Golden': 69, 'A Silver': 51,
                 'Urban Titanium': 17, 'Arizona Grey': 6, 'Beige Blue': 48, 'Wine Red': 59,
                 'Chocolate Brown': 58, 'Metallic Silver': 5, 'Riviera Red': 3, 'Star Silver': 18,
                 'Metallic Black': 45, 'Other': 6, 'Golden Beige': 26, 'Stain Grey': 40, 'Cherry': 55,
                 'Metallic Grey': 2, 'Modern Grey': 37, 'Green': 75, 'C Silver': 41, 'Dust': 27,
                 'Prime Beige': 53, 'Deep Red': 25, 'Black Magic': 10, 'W B Mist': 11, 'Pearl Silver': 66,
                 'C.Blue': 19, 'B.Green': 43, 'Ocean Blue': 46, 'Seafoam Green': 33, 'Sapphire Silver': 16,
                 'B Mist': 27, 'Aquamarine': 53, 'Carbon Bronze': 25, 'Milky White': 10, 'Steel Mist': 69,
                 'L Grey': 51, 'Chill': 17, 'Rasberry Red': 27, 'Adriatic Blue': 53, 'M.Maroon': 30,
                 'other': 8, 'Metallic': 22}

body_style_labels = {'HATCHBACK': 1, 'SUV': 5, 'SEDAN': 3, 'MPV': 6, 'COMPACTSUV': 2, 'Sedan': 9, 'MUV': 4,
                     'HATCHBACK': 0, 'VAN': 8, 'SUV': 7}

owner_labels = {'1st Owner': 0, '2nd Owner': 1, '3rd Owner': 2, '4th Owner': 3}

dealer_state_labels = {'Delhi': 2, 'Maharashtra': 4, 'Karnataka': 0, 'Haryana': 1, 'Uttar Pradesh': 8,
                       'West Bengal': 7, 'Telangana': 3, 'Tamil Nadu': 6, 'Rajasthan': 9, 'Madhya Pradesh': 5}

dealer_name_labels = {'Car Choice Exclusif': 52, 'Car&Bike Superstore Pune': 38, 'Prestige Autoworld Pvt Ltd': 4,
                      'Star Auto India': 1, 'Noida Car Ghar': 56, 'Top Gear Cars': 29, 'Car Estate': 0,
                      'OM Motors': 34, 'Jeen Mata Motors': 47, 'Royal Motors (Prop. Auto Carriage Pvt Ltd)': 51,
                      'Sri Vaishnavi Cars': 11, 'Auto Elite': 21, 'Adharshiya Motors': 9, 'Shree Radha Krishna Motors': 10,
                      'Zippy Automart': 43, 'Max Motors': 33, 'Guru Kripa Motors': 7, 'ACE MOTORS': 16,
                      'Sai Motors': 5, 'Noida Car Point ll': 12, 'Carz Villa': 42, 'MM Motors': 17,
                      'SUSHIL CARS PVT. LTD': 27, 'SK Associates': 50, 'Taneja Fourwheels': 45,
                      'Sireesh Auto Pvt Ltd': 6, 'Vinayak Autolink Private Limited': 20, 'Car Chacha': 36,
                      'Eshwari Caars - Annanagar': 23, 'LUXMI CARS GURGAON': 41, 'Cardiction': 32,
                      'Renew 4 u Automobiles PVT Ltd': 31, 'Simaks Cars': 18, 'DrivUS Motorcorp': 2,
                      'Instant Solutions': 48, 'Shree Automotive (P) Ltd': 15, 'Motor Hut': 54,
                      'Fast Wheels Cars': 40, 'Car&Bike Select - Sahyadri Motors Pune': 55, 'PROPEL MOTORS': 13,
                      'Adeep Motors': 49, 'The True Drive': 25, 'Shiv Auto Wings': 35, 'Universal Wheels': 46,
                      'K.S. Motors': 24, 'Adyah Motors': 14, 'Ikka Motors': 44, 'Pitbox Motors': 19,
                      'Anant Cars Auto Pvt Ltd': 39, 'Rajasthan Car World': 28, 'NK Cars': 26,
                      'Heritage Expo car Sales Pvt Ltd': 3, 'Mamta Motors': 53, 'VVC Motors': 30,
                      'Mahindra First Choice Wheels Ltd': 8, 'NR Autos': 22, 'Harbans Motor Pvt ltd': 37}

city_labels = {'Delhi': 0, 'Bangalore': 10, 'Gurgaon': 2, 'Pune': 3, 'Noida': 9, 'Kolkata': 4, 'Hyderabad': 5,
               'Chennai': 8, 'Jaipur': 1, 'Mumbai': 7, 'Indore': 6}

# Streamlit App
st.title("Car Price Prediction App 🚗")

# Display user input form
st.sidebar.header("User Input")
user_input = {}

user_input_Company = st.sidebar.selectbox("Company", list(company_labels.keys()))
user_input_FuelType = st.sidebar.selectbox("FuelType", list(fuel_type_labels.keys()))
user_input_Colour = st.sidebar.selectbox("Colour", list(colour_labels.keys()))
user_input_BodyStyle = st.sidebar.selectbox("BodyStyle", list(body_style_labels.keys()))
user_input_Age = st.sidebar.slider("Age", min_value=1, max_value=10, value=3)
user_input_Owner = st.sidebar.selectbox("Owner", list(owner_labels.keys()))
user_input_DealerState = st.sidebar.selectbox("DealerState", list(dealer_state_labels.keys()))
user_input_DealerName = st.sidebar.selectbox("DealerName", list(dealer_name_labels.keys()))
user_input_City = st.sidebar.selectbox("City", list(city_labels.keys()))
user_input_Kilometer = st.sidebar.slider("Kilometer", min_value=0, max_value=100000, value=50000)
user_input_Warranty = st.sidebar.slider("Warranty", min_value=0, max_value=5, value=1)
user_input_QualityScore = st.sidebar.slider("QualityScore", min_value=0.0, max_value=10.0, value=7.5)
# Submit button
if st.sidebar.button("Submit"):
    # Make a prediction
    input_array = np.array([[company_labels[user_input_Company], fuel_type_labels[user_input_FuelType],
                             colour_labels[user_input_Colour],
                             body_style_labels[user_input_BodyStyle ],
                             owner_labels[user_input_Owner], dealer_state_labels[user_input_DealerState],
                             dealer_name_labels[user_input_DealerName], city_labels[user_input_City], 
                             user_input_Age,
                             user_input_Warranty, user_input_Kilometer, user_input_QualityScore]])
    
    prediction = rfr.predict(input_array)

    # Display prediction
    with st.spinner("Predicting....."):
        st.header("Model Prediction:")
        predicted_price = prediction[0]
        st.write("Predicted Car Price:", f"{predicted_price:.2f}", font_size=44)
    # st.write("Predicted Car Price:", prediction[0])
