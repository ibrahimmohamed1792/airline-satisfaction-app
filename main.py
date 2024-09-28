import streamlit as st
import pickle
import numpy as np

# Load encoders, scaler, and models
with open("Customer Type_le.pkl", "rb") as f:
    customer_type_encoder = pickle.load(f)

with open("Class_le.pkl", "rb") as f:
    class_encoder = pickle.load(f)

with open("Gender_le.pkl", "rb") as f:
    gender_encoder = pickle.load(f)

with open("satisfaction_le.pkl", "rb") as f:
    satisfaction_encoder = pickle.load(f)

with open("Type of Travel_le.pkl", "rb") as f:
    travel_type_encoder = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("pt.pkl", "rb") as f:
    pt = pickle.load(f)

with open("svc.pkl", "rb") as f:
    svc_model = pickle.load(f)

with open("xgb22.pkl", "rb") as f:
    xgb_model = pickle.load(f)

# Define the Streamlit app
st.title('Airline Satisfaction Prediction')
st.write("Please fill in the following details to predict satisfaction:")

# User inputs
customer_type = st.selectbox('Customer Type', ['Loyal Customer', 'disloyal Customer'])
travel_type = st.selectbox('Type of Travel', ['Business travel', 'Personal Travel'])
class_type = st.selectbox('Class', ['Eco', 'Eco Plus', 'Business'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 18, 100)
flight_distance = st.number_input('Flight Distance (in miles)', min_value=1)

# Change sliders to select boxes for ratings
inflight_wifi = st.selectbox('Inflight Wifi Service', [1, 2, 3, 4, 5])
departure_time_convenience = st.selectbox('Departure Time Convenience', [1, 2, 3, 4, 5])
arrival_time_convenience = st.selectbox('Arrival Time Convenience', [1, 2, 3, 4, 5])
food_and_drink = st.selectbox('Food and Drink', [1, 2, 3, 4, 5])
gate_location = st.selectbox('Gate Location', [1, 2, 3, 4, 5])
online_boarding = st.selectbox('Online Boarding', [1, 2, 3, 4, 5])
seat_comfort = st.selectbox('Seat Comfort', [1, 2, 3, 4, 5])
inflight_entertainment = st.selectbox('Inflight Entertainment', [1, 2, 3, 4, 5])
onboard_service = st.selectbox('Onboard Service', [1, 2, 3, 4, 5])
leg_room_service = st.selectbox('Leg Room Service', [1, 2, 3, 4, 5])
baggage_handling = st.selectbox('Baggage Handling', [1, 2, 3, 4, 5])
checkin_service = st.selectbox('Check-in Service', [1, 2, 3, 4, 5])
inflight_service = st.selectbox('Inflight Service', [1, 2, 3, 4, 5])
cleanliness = st.selectbox('Cleanliness', [1, 2, 3, 4, 5])
departure_delay = st.number_input('Departure Delay in Minutes', min_value=0)
arrival_delay = st.number_input('Arrival Delay in Minutes', min_value=0)

# Add a submit button
if st.button('Submit'):
    with st.spinner('Predicting satisfaction...'):
        # Encode categorical variables
        customer_type_encoded = customer_type_encoder.transform([customer_type])[0]
        travel_type_encoded = travel_type_encoder.transform([travel_type])[0]
        class_encoded = class_encoder.transform([class_type])[0]
        gender_encoded = gender_encoder.transform([gender])[0]

        # Create feature array
        features = np.array([
            customer_type_encoded,
            travel_type_encoded,
            class_encoded,
            gender_encoded,
            age,
            flight_distance,
            inflight_wifi,
            departure_time_convenience,
            arrival_time_convenience,
            gate_location,
            food_and_drink,
            online_boarding,
            seat_comfort,
            inflight_entertainment,
            onboard_service,
            leg_room_service,
            baggage_handling,
            checkin_service,
            inflight_service,
            cleanliness,
            departure_delay,
            arrival_delay
        ]).reshape(1, -1)

        pt_features = pt.transform(features)
        # Apply scaler
        features_scaled = scaler.transform(pt_features)

        # Predict using both models
        svc_pred = svc_model.predict(features_scaled)
        xgb_pred = xgb_model.predict(features_scaled)

        # Take the average of both models
        average_pred = (svc_pred + xgb_pred) / 2

        # Decode the predicted satisfaction
        satisfaction = satisfaction_encoder.inverse_transform([int(round(average_pred[0]))])[0]

        # Output prediction
    st.success(f"The predicted satisfaction is: **{satisfaction}**")
