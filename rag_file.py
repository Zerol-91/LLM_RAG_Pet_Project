# import numpy as np
# from openai import OpenAI
# from sklearn.metrics.pairwise import cosine_similarity

# # Настройка клиента
# client = OpenAI(
#     base_url='http://localhost:11434/v1',
#     api_key='ollama',
# )

# # --- НОВАЯ ФУНКЦИЯ: Читаем файл и режем на кусочки ---
# def load_and_chunk_file(filepath='C:\Users\ASUS\OneDrive\Рабочий стол\Учба\Красная таблетка\DataScience\ПРОЕКТЫ\LLM_Lama\data.txt' , chunk_size=200):
    
#     print(f"📂 Читаю файл {filepath}...")
#     try:
#         with open(filepath, 'r', encoding='utf-8') as f:
#             text = f.read()
#     except FileNotFoundError:
#         print("❌ Ошибка: Файл data.txt не найден! Создай его.")
#         return []

#     # Простая нарезка: делим текст каждые 200 символов
#     # В профи-системах режут умнее (по точкам, абзацам), но для начала хватит и так.
#     chunks = []
#     for i in range(0, len(text), chunk_size):
#         chunk = text[i : i + chunk_size]
#         chunks.append(chunk)
    
#     print(f"🔪 Текст нарезан на {len(chunks)} кусочков (чанков).")
#     return chunks

# # 1. Загружаем данные из файла
# documents = load_and_chunk_file()

# if not documents:
#     exit() # Если файл пустой или не найден - выходим

# print("📚 Создаю индексы...")
# # Функция получения эмбеддинга
# def get_embedding(text):
#     return client.embeddings.create(model="all-minilm", input=text).data[0].embedding

# # Векторизуем чанки
# doc_vectors = [get_embedding(doc) for doc in documents]
# print("✅ Готово к работе!")

# # --- ЦИКЛ ---
# while True:
#     user_query = input("\nВаш вопрос по файлу (или 'выход'): ")
#     if user_query.lower() in ["выход", "exit"]: break

#     # Поиск
#     query_vector = get_embedding(user_query)
#     similarities = cosine_similarity([query_vector], doc_vectors)[0]
    
#     # Берем ТОП-1 лучший кусочек
#     best_idx = np.argmax(similarities)
#     best_doc = documents[best_idx]
#     score = similarities[best_idx]

#     print(f"   (Найден фрагмент с точностью {score:.2f})")

#     # Генерация
#     prompt = f"Используй этот текст для ответа: '{best_doc}'. Вопрос: {user_query}"
    
#     stream = client.chat.completions.create(
#         model="mistral",
#         messages=[{"role": "user", "content": prompt}],
#         stream=True
#     )
    
#     print("🤖 Бот: ", end="")
#     for chunk in stream:
#         if chunk.choices[0].delta.content:
#             print(chunk.choices[0].delta.content, end="", flush=True)
#     print("")