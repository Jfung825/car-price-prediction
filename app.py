import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="centered")

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🚗 Car Price Prediction")
page = st.sidebar.radio("เมนู", [
    "📖 อธิบาย ML Model",
    "📖 อธิบาย Neural Network",
    "🧪 ทดสอบ ML Model",
    "🧪 ทดสอบ Neural Network"
])

# ============================================================
# PAGE 1: อธิบาย ML Model
# ============================================================
if page == "📖 อธิบาย ML Model":
    st.title("📖 ML Model — Ensemble (Stacking)")
    st.markdown("---")

    st.subheader("Dataset ที่ใช้")
    st.write("**used_cars.csv** — ข้อมูลรถมือสองจากสหรัฐอเมริกา จำนวน 4,009 แถว")
    st.write("- **Target:** price (ราคาเป็นดอลลาร์)")
    st.write("- **Features:** brand, model_year, milage, fuel_type, transmission, engine, accident, clean_title")

    st.markdown("---")
    st.subheader("การเตรียมข้อมูล (Data Preparation)")
    st.write("1. **แปลง price** — จาก `'$10,300'` → `10300.0`")
    st.write("2. **แปลง milage** — จาก `'51,000 mi.'` → `51000.0`")
    st.write("3. **จัดการ Missing Values** — fuel_type (170), accident (113), clean_title (596) เติมด้วย Mode")
    st.write("4. **ตัด Outliers** — ใช้ IQR กำจัดราคาที่ผิดปกติ")
    st.write("5. **แยกขนาดเครื่องยนต์** — ดึงตัวเลขจาก engine column")
    st.write("6. **One-Hot Encoding** — แปลง categorical columns เป็นตัวเลข")

    st.markdown("---")
    st.subheader("อัลกอริทึม — Stacking Regressor")
    st.write("ประกอบด้วย Base Models 3 ตัว รวมกันด้วย Meta Model:")

    col1, col2, col3 = st.columns(3)
    col1.info("🌲 **Random Forest**\nสร้าง Decision Tree หลายต้น แล้วเฉลี่ยผล")
    col2.info("⚡ **XGBoost**\nสร้าง Tree ทีละต้น โดยแก้ไข Error ของต้นก่อน")
    col3.info("📐 **Ridge Regression**\nLinear Regression + L2 Regularization")
    st.success("🏆 **Meta Model: Ridge** — รับผลจาก 3 โมเดลข้างต้น แล้วรวมเป็นคำตอบสุดท้าย")

    st.markdown("---")
    st.subheader("ผลลัพธ์โมเดล")
    result = pd.DataFrame({
        "Metric": ["R² Score", "MAE", "RMSE"],
        "ค่า": ["0.8264", "5,855.71", "8,724.12"],
        "ความหมาย": [
            "โมเดลอธิบายความแปรปรวนได้ 82.6%",
            "ค่าเฉลี่ยความผิดพลาด $5,856",
            "ค่าความผิดพลาดรวม $8,724"
        ]
    })
    st.table(result)

    st.markdown("---")
    st.subheader("แหล่งอ้างอิง")
    st.write("- Dataset: https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset")
    st.write("- Scikit-learn Ensemble: https://scikit-learn.org/stable/modules/ensemble.html")
    st.write("- XGBoost Docs: https://xgboost.readthedocs.io/")

# ============================================================
# PAGE 2: อธิบาย Neural Network
# ============================================================
elif page == "📖 อธิบาย Neural Network":
    st.title("📖 Neural Network Model — MLP")
    st.markdown("---")

    st.subheader("Dataset ที่ใช้")
    st.write("**car_price_prediction.csv** — ข้อมูลรถจากประเทศจอร์เจีย จำนวน 19,237 แถว")
    st.write("- **Target:** Price")
    st.write("- **Features:** Manufacturer, Prod.year, Category, Fuel type, Engine volume, Mileage, Gear box, Cylinders, Airbags, Color ฯลฯ")

    st.markdown("---")
    st.subheader("การเตรียมข้อมูล (Data Preparation)")
    st.write("1. **แปลง Levy** — ค่า `'-'` แทน null → เติมด้วย Median")
    st.write("2. **แปลง Mileage** — จาก `'186005 km'` → `186005.0`")
    st.write("3. **แปลง Engine volume** — จาก `'3.5 Turbo'` → `3.5`")
    st.write("4. **แปลง Doors** — `'04-May'` → `4`, `'02-Mar'` → `2`, `'>5'` → `6`")
    st.write("5. **ตัด Outliers** — ใช้ IQR")
    st.write("6. **One-Hot Encoding + StandardScaler** — แปลงและปรับขนาด Features")

    st.markdown("---")
    st.subheader("โครงสร้าง Neural Network")
    st.write("ออกแบบเป็น MLP (Multi-Layer Perceptron) สำหรับ Tabular Data:")

    arch = pd.DataFrame({
        "Layer": ["Input", "Dense + ReLU", "BatchNorm + Dropout(0.3)", "Dense + ReLU", "BatchNorm + Dropout(0.2)", "Dense + ReLU", "Output"],
        "ขนาด": ["จำนวน Features", "256 neurons", "-", "128 neurons", "-", "64 neurons", "1 neuron"],
        "หน้าที่": ["รับ input", "เรียนรู้ pattern", "ป้องกัน overfitting", "ลดมิติ", "ป้องกัน overfitting", "compress", "พยากรณ์ราคา"]
    })
    st.table(arch)

    st.subheader("Hyperparameters")
    hp = pd.DataFrame({
        "Parameter": ["Optimizer", "Loss", "Epochs", "Batch Size", "Early Stopping"],
        "ค่า": ["Adam (lr=0.001)", "MSE", "100", "64", "patience=10"]
    })
    st.table(hp)

    st.markdown("---")
    st.subheader("ผลลัพธ์โมเดล")
    result2 = pd.DataFrame({
        "Metric": ["R² Score", "MAE", "RMSE"],
        "ค่า": ["0.6214", "4,739.43", "6,914.82"],
        "ความหมาย": [
            "โมเดลอธิบายความแปรปรวนได้ 62.1%",
            "ค่าเฉลี่ยความผิดพลาด 4,739",
            "ค่าความผิดพลาดรวม 6,915"
        ]
    })
    st.table(result2)

    st.markdown("---")
    st.subheader("แหล่งอ้างอิง")
    st.write("- Dataset: https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge")
    st.write("- TensorFlow/Keras: https://www.tensorflow.org/api_docs/python/tf/keras")
    st.write("- Batch Normalization: Ioffe & Szegedy (2015) https://arxiv.org/abs/1502.03167")

# ============================================================
# PAGE 3: ทดสอบ ML Model
# ============================================================
elif page == "🧪 ทดสอบ ML Model":
    st.title("🧪 ทดสอบ ML Model")
    st.write("กรอกข้อมูลรถมือสอง แล้วกดพยากรณ์ราคา (หน่วย: USD)")
    st.markdown("---")

    @st.cache_resource
    def load_ml():
        model   = joblib.load("ensemble_model_d1.pkl")
        scaler  = joblib.load("scaler1.pkl")
        columns = joblib.load("feature_columns_d1.pkl")
        return model, scaler, columns

    try:
        model, scaler, feature_cols = load_ml()

        col1, col2 = st.columns(2)
        with col1:
            brand       = st.selectbox("ยี่ห้อ", ['Acura','Audi','BMW','Buick','Cadillac','Chevrolet','Chrysler','Dodge','Ford','GMC','Genesis','Honda','Hyundai','INFINITI','Jaguar','Jeep','Kia','Land','Lexus','Lincoln','MINI','Mazda','Mercedes-Benz','Mitsubishi','Nissan','Porsche','RAM','Subaru','Tesla','Toyota','Volkswagen','Volvo'])
            model_year  = st.number_input("ปีรถ", min_value=1990, max_value=2024, value=2018)
            engine_cc   = st.number_input("ขนาดเครื่องยนต์ (ลิตร)", min_value=0.5, max_value=9.0, value=2.0, step=0.1)

        with col2:
            milage      = st.number_input("ระยะทาง (ไมล์)", min_value=0, max_value=500000, value=30000, step=1000)
            fuel_type   = st.selectbox("เชื้อเพลิง", ['Gasoline','Hybrid','Diesel','Plug-In Hybrid','E85 Flex Fuel'])
            transmission = st.selectbox("เกียร์", ['Automatic','Manual'])

        accident    = st.selectbox("ประวัติอุบัติเหตุ", ['None reported','At least 1 accident or damage reported'])
        clean_title = st.selectbox("Clean Title", ['Yes','No'])
        ext_col     = st.selectbox("สีภายนอก", ['Black','White','Silver','Gray','Blue','Red','Green','Brown','Gold','Other'])
        int_col     = st.selectbox("สีภายใน", ['Black','Gray','Brown','Beige','White','Other'])

        if st.button("พยากรณ์ราคา"):
            input_data = {col: 0 for col in feature_cols}

            if 'model_year' in input_data: input_data['model_year'] = model_year
            if 'milage'     in input_data: input_data['milage']     = milage
            if 'engine_cc'  in input_data: input_data['engine_cc']  = engine_cc

            for col_name, val in [
                (f'brand_{brand}', 1),
                (f'fuel_type_{fuel_type}', 1),
                (f'accident_{accident}', 1),
                (f'clean_title_{clean_title}', 1),
                (f'ext_col_{ext_col}', 1),
                (f'int_col_{int_col}', 1),
            ]:
                if col_name in input_data:
                    input_data[col_name] = val

            if transmission == 'Manual':
                for c in feature_cols:
                    if 'transmission_' in c and 'Manual' in c:
                        input_data[c] = 1
                        break

            X = pd.DataFrame([input_data])[feature_cols]
            pred = model.predict(scaler.transform(X))[0]
            pred = max(0, pred)

            st.markdown("---")
            st.success(f"💰 ราคาที่พยากรณ์: **${pred:,.0f} USD**")
            st.caption("พยากรณ์โดย Stacking Ensemble (Random Forest + XGBoost + Ridge)")

    except FileNotFoundError:
        st.error("❌ ไม่พบไฟล์โมเดล กรุณาวางไฟล์ .pkl ในโฟลเดอร์เดียวกับ app.py")

# ============================================================
# PAGE 4: ทดสอบ Neural Network
# ============================================================
elif page == "🧪 ทดสอบ Neural Network":
    st.title("🧪 ทดสอบ Neural Network")
    st.write("กรอกข้อมูลรถ แล้วกดพยากรณ์ราคา")
    st.markdown("---")

    @st.cache_resource
    def load_nn():
        try:
            import tensorflow as tf
            nn = tf.keras.models.load_model("nn_model_d2.keras")
        except Exception:
            import keras
            nn = keras.models.load_model("nn_model_d2.keras")
        scaler  = joblib.load("scaler2.pkl")
        columns = joblib.load("feature_columns_d2.pkl")
        return nn, scaler, columns

    try:
        nn_model, scaler2, feature_cols2 = load_nn()

        col1, col2 = st.columns(2)
        with col1:
            manufacturer = st.selectbox("ยี่ห้อ", ['LEXUS','CHEVROLET','HONDA','FORD','HYUNDAI','TOYOTA','MERCEDES-BENZ','BMW','VOLKSWAGEN','AUDI','NISSAN','KIA','MAZDA','SUBARU','VOLVO','PORSCHE','OPEL','SKODA','RENAULT','PEUGEOT'])
            prod_year    = st.number_input("ปีผลิต", min_value=1990, max_value=2020, value=2015)
            engine_vol   = st.number_input("ขนาดเครื่องยนต์ (ลิตร)", min_value=0.5, max_value=8.0, value=2.0, step=0.1)
            cylinders    = st.selectbox("จำนวนสูบ", [2, 3, 4, 5, 6, 8, 10, 12])

        with col2:
            mileage      = st.number_input("ระยะทาง (km)", min_value=0, max_value=500000, value=50000, step=5000)
            levy         = st.number_input("ภาษี (Levy)", min_value=0, max_value=5000, value=500)
            airbags      = st.number_input("จำนวน Airbags", min_value=0, max_value=16, value=4)
            category     = st.selectbox("ประเภทรถ", ['Sedan','Jeep','Hatchback','Universal','Coupe','Minivan','Cabriolet'])

        col3, col4 = st.columns(2)
        with col3:
            fuel_type2   = st.selectbox("เชื้อเพลิง", ['Petrol','Diesel','Hybrid','Plug-in Hybrid','CNG','LPG'])
            gear_box     = st.selectbox("เกียร์", ['Automatic','Manual','Tiptronic','Variator'])
        with col4:
            drive_wheels = st.selectbox("ขับเคลื่อน", ['Front','4x4','Rear'])
            wheel        = st.selectbox("พวงมาลัย", ['Left wheel','Right-hand drive'])

        leather = st.selectbox("เบาะหนัง", ['Yes','No'])
        color   = st.selectbox("สี", ['Black','White','Silver','Grey','Blue','Red','Green','Brown','Orange','Other'])
        doors   = st.selectbox("จำนวนประตู", [2, 4, 6])

        if st.button("พยากรณ์ราคา"):
            input_data2 = {col: 0 for col in feature_cols2}

            for key, val in [
                ('Levy', levy), ('Prod. year', prod_year),
                ('Cylinders', float(cylinders)), ('Airbags', airbags),
                ('Mileage', mileage), ('Engine volume', engine_vol), ('Doors', doors)
            ]:
                if key in input_data2: input_data2[key] = val

            for prefix, value in [
                ('Manufacturer', manufacturer), ('Category', category),
                ('Fuel type', fuel_type2), ('Gear box type', gear_box),
                ('Drive wheels', drive_wheels), ('Wheel', wheel),
                ('Leather interior', leather), ('Color', color),
            ]:
                k = f"{prefix}_{value}"
                if k in input_data2: input_data2[k] = 1

            X2 = pd.DataFrame([input_data2])[feature_cols2]
            pred2 = float(nn_model.predict(scaler2.transform(X2), verbose=0).flatten()[0])
            pred2 = max(0, pred2)

            st.markdown("---")
            st.success(f"💰 ราคาที่พยากรณ์: **${pred2:,.0f} USD**")
            st.caption("พยากรณ์โดย Neural Network MLP (Dense 256→128→64→1)")

    except FileNotFoundError:
        st.error("❌ ไม่พบไฟล์โมเดล กรุณาวางไฟล์ .keras และ .pkl ในโฟลเดอร์เดียวกับ app.py")
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาด: {e}")