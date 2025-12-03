import streamlit as st
from openai import OpenAI
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader # Библиотека для чтения PDF
import chromadb 
from chromadb.config import Settings


# --- НАСТРОЙКИ ---
st.set_page_config(page_title="RAG PDF Chat", page_icon="📄")
st.title("📄 Чат с твоим PDF-файлом + память")

#0 Подключение к "Кухне" (Ollama)
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',
)

# Мы говорим: "Храни данные в папке 'my_vector_db' прямо тут, в проекте"
chroma_client = chromadb.PersistentClient(path="my_vector_db")
collection = chroma_client.get_or_create_collection(
    name="my_documents",
    metadata={"hnsw:space": "cosine"} 
)
# 1. Функция чтения PDF
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

# 2. Функция нарезки текста (Чанкинг)
def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 50: # Игнорируем совсем мелкие кусочки
            chunks.append(chunk)
    return chunks
# 3. Функция получения эмбеддингов (Библиотекарь)
def get_embedding(text):
    response = client.embeddings.create(
        model="all-minilm", 
        input=text
    )
    return response.data[0].embedding


# БОКОВАЯ ПАНЕЛЬ: Загрузка файла
with st.sidebar:
    st.header("📂 Загрузка")
    uploaded_file = st.file_uploader("Выберите PDF файл", type="pdf")
    
    if uploaded_file:
        filename = uploaded_file.name
        
        # ПРОВЕРКА: А вдруг мы этот файл уже читали?
        # Мы ищем в базе записи, где source == filename
        existing_docs = collection.get(where={"source": filename})
        
        if len(existing_docs['ids']) > 0:
            st.success(f"Файл '{filename}' уже есть в базе.")
        else:
            with st.spinner("⏳ Индексирую новый файл..."):
                text = get_pdf_text(uploaded_file)
                chunks = split_text(text)
                
                # Подготовка данных для Chroma
                ids = []       # Уникальные ID кусков (например "doc1_chunk0")
                metadatas = [] # Описание (откуда кусок)
                vectors = []   # Сами векторы
                documents_text = [] # Текст кусков
                
                progress = st.progress(0)
                for i, chunk in enumerate(chunks):
                    vec = get_embedding(chunk)
                    
                    ids.append(f"{filename}_chunk{i}")
                    metadatas.append({"source": filename})
                    vectors.append(vec)
                    documents_text.append(chunk)
                    
                    progress.progress((i+1)/len(chunks))
                
                # Загружаем всё в базу одной командой
                collection.add(
                    ids=ids,
                    embeddings=vectors,
                    documents=documents_text, # Chroma умеет хранить и сам текст!
                    metadatas=metadatas
                )
                st.success("Сохранено на диск!")


if "messages" not in st.session_state:
    st.session_state.messages = []
# 4. Отрисовка истории чата
# При обновлении страницы мы пробегаем по памяти и рисуем все прошлые сообщения
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
        n_results=3
    )

    valid_chunks = []
    for i, dist in enumerate(results['distances'][0]):
        if dist < 0.9: # Порог (надо подбирать экспериментально)
            valid_chunks.append(results['documents'][0][i])

    if not valid_chunks:
    # Не отправлять запрос в Mistral вообще!
        st.write("В базе нет ничего похожего.")

    
    context_text = "\n---\n".join(valid_chunks)
    if not valid_chunks:
        system_prompt = "Ты ассистент."
    else:
        system_prompt = f"Ответь, используя контекст:\n{context_text}"

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        stream = client.chat.completions.create(
            model="mistral",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            stream=True,
        )

        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌") # ▌ - это курсор
        
        message_placeholder.markdown(full_response) # Финальный текст без курсора
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
