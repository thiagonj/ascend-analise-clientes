import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import torch
import re

# Carrega modelos com cache
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    model = AutoModelForSequenceClassification.from_pretrained("neuralmind/bert-base-portuguese-cased")
    keybert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    kw_model = KeyBERT(model=keybert_model)
    return tokenizer, model, kw_model

tokenizer, model, kw_model = load_models()

# Limpeza de texto
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text

# Sentimento
def sentiment_analysis(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    classes = ["negativo", "neutro", "positivo"]
    return classes[torch.argmax(scores)], scores.tolist()

# Palavras-chave
def extract_keywords(text):
    keywords = kw_model.extract_keywords(text, top_n=5, stop_words='portuguese')
    return [kw[0] for kw in keywords]

# App Streamlit
st.title("Ascend - Análise de Avaliações de Clientes")

comentario = st.text_area("Cole o comentário do cliente:")

if st.button("Analisar"):
    if comentario.strip() == "":
        st.warning("Por favor, insira um comentário.")
    else:
        texto = preprocess(comentario)
        sentimento, scores = sentiment_analysis(texto)
        keywords = extract_keywords(texto)

        st.subheader("Resultado da Análise")
        st.markdown(f"*Sentimento identificado:* {sentimento}")
        st.markdown(f"*Palavras-chave extraídas:* {', '.join(keywords)}")

        st.subheader("Insight gerado:")
        if sentimento == "positivo":
            insight = "Este cliente está satisfeito. As palavras-chave sugerem pontos fortes como: " + ", ".join(keywords)
        elif sentimento == "negativo":
            insight = "Este cliente está insatisfeito. Reforce atenção nos pontos: " + ", ".join(keywords)
        else:
            insight = "Comentário neutro. Pode haver oportunidade de melhoria nos aspectos citados: " + ", ".join(keywords)

        st.info(insight)