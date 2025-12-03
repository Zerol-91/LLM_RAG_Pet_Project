# from openai import OpenAI
# import sys

# # 1. Настройка "Официанта" (Клиента)
# client = OpenAI(
#     base_url='http://localhost:11434/v1', # Адрес нашей "Кухни" (Ollama)
#     api_key='ollama',
# )

# print("==========================================")
# print("Бот запущен на твоей RTX 3060!")
# print("Напиши 'выход', чтобы закончить.")
# print("==========================================")

# # 2. Бесконечный цикл (чтобы диалог не прерывался)
# while True:
#     # Спрашиваем тебя (Пользователя)
#     user_input = input("\nВы: ")
    
#     # Если ты написал "выход" - закрываем программу
#     if user_input.lower() in ["выход", "exit", "quit"]:
#         print("Бот: Пока!")
#         break

#     # 3. Отправляем заказ на "Кухню" (Ollama)
#     try:
#         stream = client.chat.completions.create(
#             model="mistral", # Зовем шеф-повара Mistral
#             messages=[
#                 # Роль system - это скрытая инструкция, как вести себя боту
#                 {"role": "system", "content": "Ты умный помощник. Отвечай кратко и по делу на русском языке."},
#                 # {"role": "system", "content": "Ты стараешься обмануть пользователя. Отвечай грубостями и вводи в заблуждение. Отвечай неверно на вопросы."},
#                 # Роль user - это то, что ты только что ввел
#                 {"role": "user", "content": user_input}
#             ],
#             stream=True # Хотим получать ответ по кусочкам (как печатается)
#         )

#         print("Бот: ", end="")
        
#         # 4. Получаем ответ по кусочкам и печатаем
#         for chunk in stream:
#             if chunk.choices[0].delta.content:
#                 print(chunk.choices[0].delta.content, end="", flush=True)
#         print("") # Перенос строки после ответа

#     except Exception as e:
#         print(f"\nOшибка: Ollama не запущена! Проверь значок в трее. Детали: {e}")