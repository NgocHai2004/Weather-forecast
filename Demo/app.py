# app.py
import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from PIL import Image
import os


st.set_page_config(
    page_title="Dự đoán Thời tiết bằng mô hình Random Forest",
    page_icon="🌈",
    layout="wide"
)


if 'snow_shown' not in st.session_state:
    st.session_state.snow_shown = False

if not st.session_state.snow_shown:
    st.snow()
    st.session_state.snow_shown = True


custom_css = """
<style>
    h1 { color: #FF5733; text-align: center; font-family: 'Comic Sans MS', cursive; }
    .stButton>button {
        background-color: #3498DB; color: white; border-radius: 8px;
        padding: 10px 24px; font-size: 16px; font-weight: bold; border: none;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2); transition: background-color 0.3s ease;
    }
    .stButton>button:hover { background-color: #2980B9; }
    .stTextInput>div>div>input, .stSelectbox>div>div>div {
        border-radius: 8px; border: 2px solid #3498DB; padding: 8px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


try:
    model_loaded = joblib.load('RandomForest_model.sav')
except FileNotFoundError:
    st.error("❌ Không tìm thấy file 'RandomForest_model.sav'. Vui lòng huấn luyện và lưu mô hình trước.")
    st.stop()

try:
    df = pd.read_csv('weather_classification_data.csv')

    le_cloud_cover = LabelEncoder().fit(df['Cloud Cover'])
    le_season = LabelEncoder().fit(df['Season'])
    le_location = LabelEncoder().fit(df['Location'])
    le_weather_type = LabelEncoder().fit(df['Weather Type'])

    cloud_cover_options = sorted(le_cloud_cover.classes_)
    season_options = sorted(le_season.classes_)
    location_options = sorted(le_location.classes_)
    weather_type_classes = le_weather_type.classes_

    weather_image_mapping = {
        'Sunny': 'sunny.jpg',
        'Rainy': 'rainy.jpg',
        'Cloudy': 'cloudy.jpg',
        'Snowy': 'snowy.jpg',
    }
except FileNotFoundError:
    st.error("❌ Không tìm thấy file 'weather_classification_data.csv'. Vui lòng tải lên file này.")
    st.stop()

# --- Giao diện người dùng ---
st.title('🌈 Dự đoán Thời tiết bằng mô hình Random Forest')
st.markdown("Cùng nhau khám phá kiểu thời tiết bằng cách điền thông tin dưới đây!")

with st.container():
    st.header("Thông tin Khí tượng")
    col1, col2 = st.columns(2)

    with col1:
        temperature = st.number_input('Nhiệt độ (°C)', 
            min_value=float(df['Temperature'].min()), max_value=float(df['Temperature'].max()), value=20.0, step=0.5)
        wind_speed = st.number_input('Tốc độ gió (km/h)', 
            min_value=float(df['Wind Speed'].min()), max_value=float(df['Wind Speed'].max()), value=10.0, step=0.5)
        atmospheric_pressure = st.number_input('Áp suất khí quyển (hPa)', 
            min_value=float(df['Atmospheric Pressure'].min()), max_value=float(df['Atmospheric Pressure'].max()), value=1000.0, step=0.5)
        visibility = st.number_input('Tầm nhìn (km)', 
            min_value=float(df['Visibility (km)'].min()), max_value=float(df['Visibility (km)'].max()), value=5.0, step=0.1)

    with col2:
        humidity = st.number_input('Độ ẩm (%)', 
            min_value=int(df['Humidity'].min()), max_value=int(df['Humidity'].max()), value=50, step=1)
        precipitation = st.number_input('Lượng mưa (%)', 
            min_value=int(df['Precipitation (%)'].min()), max_value=int(df['Precipitation (%)'].max()), value=20, step=1)
        uv_index = st.number_input('Chỉ số UV', 
            min_value=int(df['UV Index'].min()), max_value=int(df['UV Index'].max()), value=5, step=1)

    st.header("Thông tin chung")
    col3, col4, col5 = st.columns(3)
    with col3: cloud_cover_selected = st.selectbox('Lượng mây', cloud_cover_options)
    with col4: season_selected = st.selectbox('Mùa', season_options)
    with col5: location_selected = st.selectbox('Vị trí', location_options)

   
    if st.button('Dự đoán Kiểu Thời tiết', key='predict_button'):
        encoded_cloud_cover = le_cloud_cover.transform([cloud_cover_selected])[0]
        encoded_season = le_season.transform([season_selected])[0]
        encoded_location = le_location.transform([location_selected])[0]

        input_data = np.array([[temperature, humidity, wind_speed, precipitation,
                                encoded_cloud_cover, atmospheric_pressure, uv_index,
                                encoded_season, visibility, encoded_location]])

        prediction_proba = model_loaded.predict_proba(input_data)
        prediction = model_loaded.predict(input_data)
        predicted_weather = le_weather_type.inverse_transform(prediction)[0]

        st.success(f'🌤️ Kiểu thời tiết được dự đoán là: **{predicted_weather}**')

        # Hiển thị ảnh
        if predicted_weather in weather_image_mapping:
            image_filename = weather_image_mapping[predicted_weather]

            base_dir = os.path.dirname(os.path.abspath(__file__))
            image_path = os.path.join(base_dir, "weather_images", image_filename)

            if os.path.exists(image_path):
                img = Image.open(image_path)
                st.image(img, caption=f"Hình minh họa cho {predicted_weather}", use_container_width=True)
            else:
                st.warning(f"⚠️ Không tìm thấy ảnh {image_filename} trong thư mục weather_images")
        else:
            st.info("❔ Không có hình ảnh minh họa cho kiểu thời tiết này.")

        probabilities_df = pd.DataFrame(prediction_proba, columns=weather_type_classes).T
        probabilities_df.columns = ['Xác suất']
        st.bar_chart(probabilities_df)

        st.session_state['result'] = {
            'Temperature': temperature,
            'Humidity': humidity,
            'Wind Speed': wind_speed,
            'Precipitation (%)': precipitation,
            'Cloud Cover': cloud_cover_selected,
            'Atmospheric Pressure': atmospheric_pressure,
            'UV Index': uv_index,
            'Season': season_selected,
            'Visibility (km)': visibility,
            'Location': location_selected,
            'Predicted Weather': predicted_weather
        }

    # Lưu kết quả ra file
    if 'result' in st.session_state:
        if st.button('💾 Lưu Kết Quả vào CSV', key='save_button'):
            try:
                result_df = pd.DataFrame([st.session_state['result']])
                file_exists = os.path.isfile('prediction_results.csv')
                result_df.to_csv('prediction_results.csv', mode='a', index=False, header=not file_exists)
                st.success("✅ Kết quả đã được lưu thành công vào 'prediction_results.csv'")
            except Exception as e:
                st.error(f"❌ Lỗi khi lưu file: {e}")
