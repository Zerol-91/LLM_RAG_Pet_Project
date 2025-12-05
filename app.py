import streamlit as st
from openai import OpenAI
from pypdf import PdfReader # Библиотека для чтения PDF
import chromadb 
import os
from chromadb.config import Settings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer 

# --- НАСТРОЙКИ ---
st.set_page_config(page_title="RAG Cloud Chat", page_icon="📄")
st.title("☁️ Чат с PDF (OpenRouter + Local Embeddings)")

load_dotenv() 
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    st.error("Не найден ключ API! Создайте файл .env и впишите туда OPENROUTER_API_KEY")
    st.stop()

# OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

@st.cache_resource# Декоратор для единоразовой загрузки MiniLM
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()


chroma_client = chromadb.PersistentClient(path="my_vector_db")
collection = chroma_client.get_or_create_collection(
    name="my_documents",
    metadata={"hnsw:space": "cosine"} 
)

def get_pdf_text(uploaded_file):
    text = ""
    try:
        pdf_reader = PdfReader(uploaded_file)
        # Читаем каждую страницу
        for page in pdf_reader.pages:
            text += page.extract_text()
    except Exception as e:
        st.error(f"Ошибка чтения PDF: {e}")
    return text


def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 50: # Игнорируем совсем мелкие кусочки
            chunks.append(chunk)
    return chunks

def get_embedding(text):
    response = client.embeddings.create(
        model="all-minilm", 
        input=text
    )
    return response.data[0].embedding


def get_embedding(text):
    return embedding_model.encode(text).tolist()

if "messages" not in st.session_state:
    st.session_state.messages = []



with st.sidebar:
    st.header("Загрузка")
    uploaded_file = st.file_uploader("Выберите PDF файл", type="pdf")
    
    if uploaded_file:
        filename = uploaded_file.name
        existing_docs = collection.get(where={"source": filename})
        
        if len(existing_docs['ids']) > 0:
            st.success(f"Файл '{filename}' уже есть в базе.")
        else:
            with st.spinner("Индексирую новый файл..."):
                text = get_pdf_text(uploaded_file)
                chunks = split_text(text)
                

                ids = []       
                metadatas = [] 
                vectors = []   
                documents_text = [] 
                
                progress = st.progress(0)
                for i, chunk in enumerate(chunks):
                    vec = get_embedding(chunk)
                    
                    ids.append(f"{filename}_chunk{i}")
                    metadatas.append({"source": filename})
                    vectors.append(vec)
                    documents_text.append(chunk)
                    
                    progress.progress((i+1)/len(chunks))
                

                collection.add(
                    ids=ids,
                    embeddings=vectors,
                    documents=documents_text, 
                    metadatas=metadatas
                )
                st.success("Сохранено в базу.")



for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Вопрос..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    query_vec = get_embedding(prompt)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=5
    )

    valid_chunks = []
    # Найденная в базе информация 
    with st.expander("Техническая информация (Что нашла база)"):
        found_chunks = results['documents'][0]
        distances = results['distances'][0]
            
        for i, dist in enumerate(distances):
            chunk_text = found_chunks[i]
            st.write(f"**Кусок {i+1}** (Дистанция: {dist:.4f}):")
            st.caption(chunk_text[:200] + "...") # Показываем начало куска
                
            # Фильтр: берем только если дистанция меньше 0.7 (можно менять)
            if dist < 0.7:
                st.success("Подходит")
                valid_chunks.append(chunk_text)
            else:
                st.warning("Этот кусок отброшен (слишком непохож)")

 
    if not valid_chunks:
        system_prompt = "Ты умный и полезный ассистент."
    else:
        context_text = "\n---\n".join(valid_chunks)
        system_prompt = f"Ответь как умный и полезный ассистент, используя контекст:\n{context_text}"

    # Генерация (OpenRouter)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        

        try:
            stream = client.chat.completions.create(
                model="meta-llama/llama-3.3-70b-instruct:free", # Или "google/gemma-2-9b-it:free"
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                stream=True,
                extra_headers={
                    "HTTP-Referer": "http://localhost:8501",
                    "X-Title": "Local RAG App"
                }
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌") # ▌ - это курсор
            message_placeholder.markdown(full_response) # Финальный текст без курсора
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Ошибка API: {e}")
        
