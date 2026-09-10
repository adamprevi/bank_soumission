import streamlit as st
import joblib as jb
import pandas as pd
import numpy as np


# ============================================================
# Chargement des fichiers du modèle
# ============================================================

encoders = jb.load("encoders_bank.joblib")
uniques = jb.load("uniques_bank.joblib")
scaler = jb.load("scaler_bank.joblib")
xgb = jb.load("xgb_model_bank.joblib")

# Noms des classes
clasnames = uniques[9]


# ============================================================
# Fonction de prédiction
# ============================================================

def Pred_func(age, job,  marital,  education, housing, loan,  contact,   month,  day_of_week, duration,    campaign,   pdays, previous,
    poutcome):

    # Encodage des variables catégorielles
    job = encoders[0].transform([job])[0]
    marital = encoders[1].transform([marital])[0]
    education = encoders[2].transform([education])[0]
    housing = encoders[3].transform([housing])[0]
    loan = encoders[4].transform([loan])[0]
    contact = encoders[5].transform([contact])[0]
    month = encoders[6].transform([month])[0]
    day_of_week = encoders[7].transform([day_of_week])[0]
    poutcome = encoders[8].transform([poutcome])[0]

    # Création du vecteur
    x_new = np.array([age, job,  marital,   education, housing,  loan,    contact,   month,  day_of_week,  duration,   campaign, pdays,
        previous, poutcome  ])

    # Mise en forme
    x_new = x_new.reshape(1, -1)

    # Normalisation
    x_new = scaler.transform(x_new)

    # Prédiction
    y_pred = xgb.predict(x_new)

    return clasnames[y_pred[0]]


# ============================================================
# Configuration de la page
# ============================================================

st.set_page_config(
    page_title="Prédiction de souscription",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# Titre
# ============================================================

st.title("🏦 Prédiction de souscription à un dépôt à terme")

st.write(
    "Ce modèle de machine learning permet de prédire si un client "
    "est susceptible de souscrire à un dépôt à terme ou non."
)


# ============================================================
# Onglets
# ============================================================

tab1, tab2 = st.tabs([
    "🔹 Simple Prediction",
    "📂 Prédiction multiple"
])


# ============================================================
# 1. PRÉDICTION SIMPLE
# ============================================================

with tab1:

    st.subheader("Informations du client")

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=30
        )

        job = st.selectbox(
            "Job",
            options=uniques[0]
        )

        marital = st.selectbox(
            "Marital",
            options=uniques[1]
        )

        education = st.selectbox(
            "Education",
            options=uniques[2]
        )

        housing = st.selectbox(
            "Housing",
            options=uniques[3]
        )

    with col2:

        loan = st.selectbox(
            "Loan",
            options=uniques[4]
        )

        contact = st.selectbox(
            "Contact",
            options=uniques[5]
        )

        month = st.selectbox(
            "Month",
            options=uniques[6]
        )

        day_of_week = st.selectbox(
            "Day of week",
            options=uniques[7]
        )

        duration = st.number_input(
            "Duration",
            min_value=0.0,
            value=0.0
        )

    with col3:

        campaign = st.number_input(
            "Campaign",
            min_value=0.0,
            value=1.0
        )

        pdays = st.number_input(
            "Pdays",
            min_value=0.0,
            value=999.0
        )

        previous = st.number_input(
            "Previous",
            min_value=0.0,
            value=0.0
        )

        poutcome = st.selectbox(
            "Poutcome",
            options=uniques[8]
        )


    # Bouton de prédiction

    if st.button(
        "🔮 Prédire",
        type="primary",
        use_container_width=True
    ):

        prediction = Pred_func(
            age,
            job,
            marital,
            education,
            housing,
            loan,
            contact,
            month,
            day_of_week,
            duration,
            campaign,
            pdays,
            previous,
            poutcome
        )

        st.success(
            f"Résultat de la prédiction : **{prediction}**"
        )


# ============================================================
# 2. PRÉDICTION MULTIPLE
# ============================================================

with tab2:

    st.subheader("Prédiction à partir d'un fichier CSV")

    st.write(
        "Importez un fichier CSV contenant les 14 variables "
        "utilisées par le modèle."
    )

    uploaded_file = st.file_uploader(
        "Importer un fichier CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.write("Aperçu du fichier :")
        st.dataframe(df.head())

        if st.button(
            "🔮 Lancer les prédictions",
            type="primary"
        ):

            try:

                predictions = []

                for row in df.iloc[:, :].values:

                    y_pred = Pred_func( row[0], row[1],  row[2],    row[3],  row[4],  row[5],   row[6], row[7],  row[8],  row[9],row[10],
                        row[11], row[12], row[13] )

                    predictions.append(y_pred)

                df["etat"] = predictions

                st.success("Prédictions effectuées avec succès.")

                st.dataframe(df)

                # Préparation du fichier à télécharger
                csv = df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="⬇️ Télécharger le fichier CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )

            except Exception as e:

                st.error(
                    f"Une erreur est survenue : {e}"
                )
```

