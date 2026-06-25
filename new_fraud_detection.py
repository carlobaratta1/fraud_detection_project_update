import streamlit as st
import pandas as pd
import joblib

# Caricamento del modello
# Assicurati che il file .pickle sia nella stessa cartella!
model = joblib.load('fraud_detection_pipeline.pickle')

# Creazione GUI
st.title('Fraud Detection Prediction App (model Traditional ML)')
st.markdown('Inserisci i dettagli della transazione e clicca su Predict')
st.divider()

# Input dell'utente
transaction_type = st.selectbox('Transaction Type', ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEPOSIT', 'CASH_IN'])
amount = st.number_input('Amount', min_value=0.0, value=1000.0)
old_balance_original = st.number_input('Old Balance Sender', min_value=0.0, value=10000.0)
new_balance_original = st.number_input('New Balance Sender', min_value=0.0, value=9000.0)
old_balance_destination = st.number_input('Old Balance Receiver', min_value=0.0, value=0.0)
new_balance_destination = st.number_input('New Balance Receiver', min_value=0.0, value=0.0)

# Bottone di predizione
if st.button('Predict'):
    # Creazione del DataFrame per il modello
    input_data = pd.DataFrame([{
        'type': transaction_type,
        'amount': amount,
        'oldbalanceOrg': old_balance_original,
        'newbalanceOrig': new_balance_original,
        'oldbalanceDest': old_balance_destination,
        'newbalanceDest': new_balance_destination
    }])

 # Probabilità di frode
    prob = model.predict_proba(input_data)[0][1]
    fraud_percentage = prob * 100
    prediction = int(prob >= 0.4)

    st.divider()
    st.subheader('Fraud Prediction:')
    if prediction == 1:
        st.error(f'This transaction can be fraud\n\n**Fraud Probability: {fraud_percentage:.2f}%**')
    else:
        st.success(f'This transaction looks like it is not a fraud\n\n**Fraud Probability: {fraud_percentage:.2f}%**')