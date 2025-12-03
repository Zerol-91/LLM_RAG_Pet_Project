import streamlit as st
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader # Библиотека для чтения PDF

# --- НАСТРОЙКИ ---
st.set_page_config(page_title="RAG PDF Chat", page_icon="📄")
st.title("📄 Чат с твоим PDF-файлом")

#0 Подключение к "Кухне" (Ollama)
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',
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



# 4. Инициализация Памяти (Session State)
# Сайт обновляется при каждом клике. Чтобы чат не исчезал,
# мы храним его в специальном хранилище st.session_state.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = [] # Тут будем хранить векторы чанков



# БОКОВАЯ ПАНЕЛЬ: Загрузка файла
with st.sidebar:
    st.header("📂 Загрузка документа")
    uploaded_file = st.file_uploader("Выберите PDF файл", type="pdf")
    
    if uploaded_file and not st.session_state.vector_db:
        with st.spinner("⏳ Читаю и анализирую файл... (это может занять время)"):
            # А. Получаем текст
            raw_text = get_pdf_text(uploaded_file)
            st.success(f"Прочитано символов: {len(raw_text)}")
            
            # Б. Режем на кусочки
            chunks = split_text(raw_text)
            st.info(f"Нарезано на {len(chunks)} фрагментов.")
            
            # В. Создаем эмбеддинги (Самое долгое!)
            # Сохраняем словарь: {"text": кусок_текста, "vector": вектор}
            db = []
            progress_bar = st.progress(0)
            for i, chunk in enumerate(chunks):
                vector = get_embedding(chunk)
                db.append({"text": chunk, "vector": vector})
                progress_bar.progress((i + 1) / len(chunks))
            
            st.session_state.vector_db = db # Сохраняем в память сессии
            st.success("✅ Файл проиндексирован! Можете задавать вопросы.")




# 4. Отрисовка истории чата
# При обновлении страницы мы пробегаем по памяти и рисуем все прошлые сообщения
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Поле ввода (Ждем, пока юзер напишет и нажмет Enter)
if prompt := st.chat_input("Напишите сообщение..."):
    
    # --- ДЕЙСТВИЯ ПОЛЬЗОВАТЕЛЯ ---
    # А. Показываем сообщение пользователя на экране
    with st.chat_message("user"):
        st.markdown(prompt)
    # Б. Сохраняем его в память
    st.session_state.messages.append({"role": "user", "content": prompt})


# 2. RAG: Поиск информации
    if st.session_state.vector_db:
        # А. Векторизуем вопрос
        query_vector = get_embedding(prompt)
        
        # Б. Считаем сходство со всеми чанками
        # Извлекаем все векторы из нашей базы
        db_vectors = [item["vector"] for item in st.session_state.vector_db]
        similarities = cosine_similarity([query_vector], db_vectors)[0]
        
        # В. Берем ТОП-3 лучших куска
        top_indices = np.argsort(similarities)[-3:][::-1] # Сортируем и берем 3 последних (самых больших)
        
        # Собираем контекст из найденных кусков
        context_text = ""
        for idx in top_indices:
            score = similarities[idx]
            if score > 0.25: # Фильтр мусора
                context_text += f"\n---\n{st.session_state.vector_db[idx]['text']}"
        
        # Г. Формируем системный промпт
        system_prompt = f"""
        Ты аналитик. Используй ТОЛЬКО следующий контекст для ответа на вопрос.
        Если в контексте нет информации, скажи "В документе нет информации об этом".
        
        Контекст из документа:
        {context_text}
        """
    else:
        # Если файл не загружен, просто болтаем
        system_prompt = "Ты полезный ассистент."

    # 3. Генерация ответа
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

        
        # Г. Получаем ответ по кусочкам и обновляем текст на лету
        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌") # ▌ - это курсор
        
        message_placeholder.markdown(full_response) # Финальный текст без курсора
    
    # Д. Сохраняем ответ бота в память
    st.session_state.messages.append({"role": "assistant", "content": full_response})
