from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Cargar modelo, scaler y nombres de columnas (ya reducidas)
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

NUMERIC_FEATURES = ["age", "thal", "ca", "oldpeak", "slope", "thalach"]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    proba = None

    if request.method == "POST":
        # Inicializar todas las features en 0
        feats = {name: 0.0 for name in feature_names}

        # 1) Numéricas
        for name in NUMERIC_FEATURES:
            val = request.form.get(name, "0")
            try:
                feats[name] = float(val)
            except ValueError:
                feats[name] = 0.0

        # 2) Sexo -> sex_1 (1 = hombre, 0 = mujer)
        sex = request.form.get("sex", "male")
        feats["sex_1"] = 1.0 if sex == "male" else 0.0

        # 3) Dolor torácico -> cp_asymptomatic (1 si es asintomático, 0 en otro caso)
        cp = request.form.get("cp", "typical")
        feats["cp_asymptomatic"] = 1.0 if cp == "asymptomatic" else 0.0

        # 4) Angina inducida por ejercicio -> exang_1
        exang = request.form.get("exang", "no")
        feats["exang_1"] = 1.0 if exang == "yes" else 0.0

        # 5) Pasar al array en el orden correcto
        values = [feats[name] for name in feature_names]
        X_input = np.array([values])
        X_input_scaled = scaler.transform(X_input)

        pred = model.predict(X_input_scaled)[0]
        proba = float(model.predict_proba(X_input_scaled)[0][1])

        prediction = (
            "Alto riesgo de enfermedad cardiaca."
            if pred == 1
            else "Bajo riesgo de enfermedad cardiaca."
        )

    return render_template("index.html", prediction=prediction, proba=proba)

if __name__ == "__main__":
    app.run(debug=True)
