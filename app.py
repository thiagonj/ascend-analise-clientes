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
    tokenizer, model = load_model()
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1)

    st.write("*Logits Brutos:*")
    st.write(logits)
    st.write("*Probabilidades Brutas:*")
    st.write(probabilities)
