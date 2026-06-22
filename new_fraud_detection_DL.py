import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import joblib

# 1. Definiamo la struttura del modello (deve essere identica all'addestramento)
class FraudDetectionNN(nn.Module):
    def __init__(self, input_dim):
        super(FraudDetectionNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2), # Dropout moderato
            
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1), # Dropout più leggero verso la fine
            
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        return self.network(x)

# 2. Carichiamo modello e preprocessore (st.cache_resource evita che l'app sia lenta)
@st.cache_resource
def load_files():
    device = torch.device("cpu")

    try:
        state_dict = torch.load('fraud_detection_pytorch.pth', map_location=device, weights_only=True)
    except TypeError:
        # Compatibilita con versioni PyTorch che non supportano weights_only.
        state_dict = torch.load('fraud_detection_pytorch.pth', map_location=device)

    input_dim = state_dict['network.0.weight'].shape[1]
    model = FraudDetectionNN(input_dim=input_dim)
    model.load_state_dict(state_dict)
    model.eval()
    preprocessor = joblib.load('preprocessor.pkl')
    return model, preprocessor, device

model, preprocessor, device = load_files()

# 3. Interfaccia Grafica 
st.title("Fraud Detection Prediction App")
st.markdown("Please enter the transaction details and use the predict button.")
st.divider()

transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
amount = st.number_input("Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance Sender", min_value=0.0, value=10000.0)
newbalanceOrig = st.number_input("New Balance Sender", min_value=0.0, value=9000.0)
oldbalanceDest = st.number_input("Old Balance Receiver", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance Receiver", min_value=0.0, value=0.0)

# 4. Logica del bottone
if st.button("Predict"):
    # Creiamo il dataset con l'input dell'utente
    input_data = pd.DataFrame([{
        'type': transaction_type,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest
    }])
    
    # Trasformiamo i dati con scikit-learn
    input_processed = preprocessor.transform(input_data)
    if hasattr(input_processed, 'toarray'):
        input_processed = input_processed.toarray()
        
    # Convertiamo per PyTorch
    input_tensor = torch.tensor(input_processed, dtype=torch.float32, device=device)
    
    # Otteniamo la predizione
    with torch.no_grad():
        prediction = model(input_tensor)
        prob = torch.sigmoid(prediction).item()
        
    # Mostriamo il risultato con la percentuale
    st.subheader("Fraud Prediction:")
    fraud_percentage = prob * 100
    
    if prob >= 0.4:
        st.error(f"⚠️ This transaction can be fraud\n\n**Fraud Probability: {fraud_percentage:.2f}%**")
    else:
        st.success(f"✅ This transaction looks like it is not a fraud\n\n**Fraud Probability: {fraud_percentage:.2f}%**")