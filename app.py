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
    return tokenizer, model

# Interface
st.title("Análise de Sentimentos - Projeto Ascend")
texto = st.text_area("Digite um comentário de cliente:")

if texto:
    st.write("Debug: Bloco 'if texto' está sendo executado") # Adicionamos esta linha
    tokenizer, model = load_model()
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1)

    with open("output.txt", "w") as f:
        f.write("Logits Brutos:\n")
        f.write(str(logits.tolist()))
        f.write("\n\nProbabilidades Brutas:\n")
        f.write(str(probabilities.tolist()))

    sentimento, confianca = interpretar_logits(logits)
    confianca_percent = round(confianca * 100, 2)

    st.write(f"*Sentimento Predito:* {sentimento}")
    st.write(f"*Confiança:* {confianca_percent}%")

def interpretar_logits(logits):
    probabilities = F.softmax(logits, dim=-1)[0].tolist()
    sentimentos = ["negativo", "neutro", "positivo"] # Supondo esta ordem
    predicted_index = probabilities.index(max(probabilities))
    return sentimentos[predicted_index], probabilities[predicted_index]
