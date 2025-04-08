import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

@st.cache_resource
def load_model():
    model_name = "pierreguillou/bert-base-cased-sentiment-analysis-sst-2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def interpretar_logits(logits):
    probabilities = F.softmax(logits, dim=-1)[0].tolist()
    sentimentos = ["negativo", "positivo"]  # Ordem usada no modelo SST-2
    predicted_index = probabilities.index(max(probabilities))
    return sentimentos[predicted_index], probabilities[predicted_index]

st.title("Análise de Sentimentos - Projeto Ascend")
texto = st.text_area("Digite um comentário de cliente:")

if texto:
    tokenizer, model = load_model()
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits

    sentimento, confianca = interpretar_logits(logits)
    confianca_percent = round(confianca * 100, 2)

    st.write(f"*Sentimento Predito:* {sentimento}")
    st.write(f"*Confiança:* {confianca_percent}%")
