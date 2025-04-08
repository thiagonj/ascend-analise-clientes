import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Carrega o modelo
@st.cache_resource
def load_model():
    model_name = "pierreguillou/bert-large-cased-squad-v1.1-portuguese"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    print("Configuração do Modelo:", model.config)  # Para ver a configuração no terminal
    return tokenizer, model

# Interpreta os logits para sentimento
def interpretar_logits(logits):
    probabilities = F.softmax(logits, dim=-1)[0].tolist()
    # Supondo que a ordem dos labels seja [negativo, neutro, positivo] - PRECISAMOS CONFIRMAR ISSO
    sentimentos = ["negativo", "neutro", "positivo"]
    predicted_index = probabilities.index(max(probabilities))
    return sentimentos[predicted_index], probabilities[predicted_index]

# Interface
st.title("Análise de Sentimentos - Projeto Ascend")
texto = st.text_area("Digite um comentário de cliente:")

if texto:
    st.write("Debug: Texto recebido:", texto)
    tokenizer, model = load_model()
    st.write("Debug: Modelo carregado")
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    st.write("Debug: Saída obtida - Logits:", logits)
    probabilities = F.softmax(logits, dim=-1)
    st.write("Debug: Saída obtida - Probabilidades:", probabilities)

    sentimento, confianca = interpretar_logits(logits)
    confianca_percent = round(confianca * 100, 2)

    st.write(f"*Sentimento Predito:* {sentimento}")
    st.write(f"*Confiança:* {confianca_percent}%")
