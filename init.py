# Додати кнопку "Скинути усі значення та щось в такому роді, можна просто перезавантажити бота"  після кнопки "Видалити пост"


import os
import requests
from dotenv import load_dotenv
import asyncio
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from db.postgreSQL import create_connection
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
import base64
import json
from openai import OpenAI, AsyncOpenAI
from together import Together
from datetime import datetime
import re
from video_creator import generate_video_with_effects, create_video_with_audio_and_subtitles

from elevenlabs import ElevenLabs

import torch
from TTS.api import TTS
import traceback

import pysubs2


# Завантаження змінних середовища з файлу .env
load_dotenv()
TOKEN = os.getenv('TOKEN')
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
CHATGPT_API_KEY = os.getenv('CHATGPT_API_KEY')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')

# logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
# logger = logging.getLogger(__name__)

# Глобальна змінна для зберігання підключення до бази даних
global_connection = None

def get_connection():
    global global_connection
    if global_connection is None or global_connection.closed:
        global_connection = create_connection()
    return global_connection

def create_keyboard(context: ContextTypes.DEFAULT_TYPE = None, mode="default"):
    """Створює клавіатуру з кнопками та мітками."""
    whatDb_table_now_info = context.user_data.get('whatDb_table_now_info') if context and context.user_data else "Не вказано"

    keyboards = {
        "default": [
            [KeyboardButton("Пояснення як працювати з ботом")],
            [KeyboardButton("Вибрати з яким каналом працювати (In development)")],
            [KeyboardButton("Перезавантажити бота"), KeyboardButton("Create video (test)")],
            [KeyboardButton("Work with POSTS"), KeyboardButton("Work with VIDEOS")],
        ],
        "work_with_posts": [
            [KeyboardButton(f"Вибір які пости обробляти ({whatDb_table_now_info})")],
            [KeyboardButton("Наступний пост"), KeyboardButton("Почати з початку")],
            [KeyboardButton("Підтвердити пост"), KeyboardButton("Видалити пост")],
            [KeyboardButton("Переробити повністю"), KeyboardButton("Переробити текст"), KeyboardButton("Переробити фото")],
            [KeyboardButton("Перезавантажити бота"), KeyboardButton("Вивести усі пости"), KeyboardButton("Default keyboard")],
        ],
        "work_with_videos": [
            [KeyboardButton(f"Вибір які пости обробляти ({whatDb_table_now_info})"), KeyboardButton("In development")],
            [KeyboardButton("Наступний пост"), KeyboardButton("Почати з початку")],
            [KeyboardButton("Підтвердити пост"), KeyboardButton("Видалити пост")],
            [KeyboardButton("Переробити повністю"), KeyboardButton("Переробити текст"), KeyboardButton("Переробити фото")],
            [KeyboardButton("Вивести усі пости"), KeyboardButton("Default keyboard")],
        ],
    }
        
    return ReplyKeyboardMarkup(keyboards.get(mode, keyboards["default"]), resize_keyboard=True)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обробляє натискання кнопок."""
    text = update.message.text
    if context.user_data is None:
        context.user_data = {}

    if text == "Вивести усі пости":
        await show_all_posts(update, context)
    elif text == "Наступний пост":
        await show_next_post(update, context)
    elif text == "Почати з початку":
        await show_posts_fromStart(update, context)
    elif text == "Видалити пост":
        await delete_post_by_id(update, context)
    elif text.startswith("Вибір які пости обробляти"):
        await choosing_whichPosts_process(update, context)
    elif text.startswith("Підтвердити пост"):
        await confirm_post(update, context)
    elif text == "Default keyboard":
        await set_keyboard(update, context, "default")
    elif text == "Work with POSTS":
        await set_keyboard(update, context, "work_with_posts")
    elif text == "Work with VIDEOS":
        await set_keyboard(update, context, "work_with_videos")
    elif text == "Create video (test)":
        await create_video(update, context)

async def choosing_whichPosts_process(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Виводить в чат можливості вибору і записує вибір користувача в змінну."""
    keyboard = [
        [InlineKeyboardButton("Обробляти нові основи для постів", callback_data='newBases_forPosts')],
        [InlineKeyboardButton("Обробляти нові готові пости для телеграму", callback_data='readyPosts_forTelegram')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text('Виберіть, які пости обробляти:', reply_markup=reply_markup)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обробляє вибір користувача і записує його в змінну."""
    query = update.callback_query
    await query.answer()
    choice = query.data

    if choice == 'newBases_forPosts':
        context.user_data['whatDb_table_now_info'] = "Нові основи для постів"
    elif choice == 'readyPosts_forTelegram':
        context.user_data['whatDb_table_now_info'] = "Готові пости для телеграму"

    # для того щоб працювало два види клавіатур одночасно потрібно оновити клавіатуру
    await query.message.reply_text(f"Ви вибрали ({context.user_data['whatDb_table_now_info']})", reply_markup=create_keyboard(context))

async def set_keyboard(update: Update, context: ContextTypes.DEFAULT_TYPE, mode: str) -> None:
    """Оновлює клавіатуру для заданого режиму."""
    await update.message.reply_text(f"Клавіатуру змінено для роботи з {mode}", reply_markup=create_keyboard(context, mode))

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Відправляє повідомлення при команді /start."""

    # Глобальні значення 
    context.user_data['post_index_for_show_next_post'] = 0
    context.user_data['whatDb_table_now_info'] = "Обробляти нові основи для постів"
    
    await update.message.reply_text('Привіт! Використовуйте кнопки нижче для взаємодії.', reply_markup=create_keyboard(context))
    chat_id = update.effective_chat.id
    logger.info(f"Chat ID: {chat_id}")
    await update.message.reply_text(f'Ваш chat_id: {chat_id}', reply_markup=create_keyboard(context))

def get_posts(context: ContextTypes.DEFAULT_TYPE):
    connection = get_connection()
    
    db_table_name = getDbTableName(context)
        
    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT id, data FROM {db_table_name}")
                posts = cursor.fetchall()
                return [{"id": post[0], "data": post[1]} for post in posts]
        except Exception as e:
            print(f"❌ Failed to fetch posts: {e}")
            return []
    else:
        return []

async def create_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    client = AsyncOpenAI(
        api_key=context.user_data.get('api_key', CHATGPT_API_KEY),
    )
    
    # Отримання поточного часу і дати
    current_time = datetime.now().strftime("%H-%M_%d.%m.%Y")
    video_name = f"{current_time}"
    output_folder = f"videos/{video_name}"
    pathFor_audio = f"{output_folder}/audio.wav"
    pathFor_images = f"{output_folder}/images"
    pathFor_subtitlesASS = f"{output_folder}/subtitlesASS.ass"
    pathFor_subtitlesSRT = f"{output_folder}/subtitlesSRT.srt"
    pathFor_images = f"{output_folder}/images"
    pathFor_outputVideo = f"{output_folder}/{video_name}.mp4"
    
    name_scenariusForImages = f"scenarius_{video_name}.txt"
    name_promptsForImages = f"promptsForImages_{video_name}.txt"
    
    video_base = "Apple стає серйозним гравцем у світі штучного інтелекту, співпрацюючи з компанією Broadcom для розробки свого першого серверного чіпа, призначеного для обробки AI-застосунків. Уявіть лише: у вашій домівці не буде просто 'яблучних' пристроїв, а ціла екосистема, оптимізована під штучний інтелект. Здається, Apple нарешті усвідомила, що без AI майбутнє виглядає, м'яко кажучи, тьмяно. Чи готові ви до нового етапу в технологічній революції, де ваш смартфон не просто дзвонитиме, а й самостійно вирішуватиме ваші проблеми? Цей чіп може стати основою для нових, небачених уявлень про те, як технології можуть взаємодіяти з нашим життям. І так, Apple вкотре доводить, що в їхній ДНК закладено прагнення до інновацій. Часом дивно, як одна компанія здатна зрушити з місця цілу індустрію. Чекаємо на новини!"

    video_length = 30  # Довжина відео за замовчуванням, у секундах

    # Розрахунок таймлайнів
    clickbait_end = 5  # Кінець клікбейту
    main_story_end = video_length - 5  # Кінець основної частини
    call_to_action_start = video_length - 5  # Початок заклику до дії

    await update.message.reply_text('Створення сценарію з video_base.')
    response = await client.chat.completions.create(

        model="gpt-4o-mini",
        messages = [
            {
                "role": "system",
                "content": (
                    "You must create an engaging and compelling script for a shorts video based on the provided text in English. "
                    "The script should have the following structure: "
                    f"Clickbait (start of video, 0-{clickbait_end} seconds) — grabs the viewer's attention immediately. "
                    f"Main Story ({clickbait_end}-{main_story_end} seconds) — explains the topic in a way that keeps the viewer intrigued. "
                    f"Call to Action ({call_to_action_start}-{video_length} seconds) — encourages viewers to take action, such as subscribing or staying tuned for more. "
                    "The text must be written in an interesting and attention-grabbing manner, making viewers want to watch the video until the end. "
                    "Make the script suitable for TTS models. "
                    "Provide only the script text as your response without any labels or additional formatting."
                    "Don't use emoticons and links."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Text for the script: {video_base}. "
                )
            }
        ],
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.7,
    )
    
    ####
    # response_text = response['choices'][0]['message']['content'].strip()
    response_text = response.choices[0].message.content.strip()

    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, name_scenariusForImages)
    # Збереження тексту у файл
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(response_text)
    ####
    print(response_text)
    await update.message.reply_text('Сценарій створенно.')


    # CREATE IMAGES PROMPTS
    await update.message.reply_text('Створення промтів для фотогорафій.')
    prompts_for_images = await create_imagesPromts_with_chatGPT(video_base, context)
    print(prompts_for_images)
    
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, name_promptsForImages)
    # Перетворення списку на рядок, де кожен елемент починається з нового рядка
    prompts_for_images_str = "\n".join(prompts_for_images)
    # Збереження тексту у файл
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(prompts_for_images_str)
    
    
    await update.message.reply_text('Процес створення фотографій.')
    for prompt in prompts_for_images:
        await create_img(update, context, prompt, pathFor_images)
        await asyncio.sleep(10)
    await update.message.reply_text('Фотографії створилися успішно.')
    
    # CREATE AUDIO WITH TTS
    try:
        await update.message.reply_text('Створення audio.')

        # Отримання пристрою (CUDA чи CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Список доступних моделей
        print("Available models:", TTS().list_models())
        # Ініціалізація моделі TTS
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC").to(device)
        tts.tts_to_file(text=response_text, file_path=pathFor_audio)
    except Exception as e:
        print("An error occurred:", e)
        traceback.print_exc()
        
    # CREATE TRANSCRIPTION WITH OPENAI FOR SUBTITLES
    await update.message.reply_text('Створення транскрипції.')

    audio_file = open(pathFor_audio, "rb")
    transcript = await client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
        response_format="verbose_json",
        timestamp_granularities=["word"]
    )
    print(transcript)
    if transcript.words:
        # Збереження у файл формату SRT
        with open(pathFor_subtitlesSRT, "w", encoding="utf-8") as srt_file:
            for i, word in enumerate(transcript.words):
                start = format_timestamp(word.start)
                end = format_timestamp(word.end)
                text = word.word
                srt_file.write(f"{i+1}\n{start} --> {end}\n{text}\n\n")
                
        # Створення файлу субтитрів
        subs = pysubs2.SSAFile()
        # Додавання субтитрів із розпізнаного тексту
        for word in transcript.words:
            start = word.start * 1000  # секунди → мілісекунди
            end = word.end * 1000
            text = word.word
            subs.append(pysubs2.SSAEvent(start=start, end=end, text=text))
        # Налаштування стилю
        style = pysubs2.SSAStyle()
        style.fontname = "Arial"
        style.fontsize = 24
        style.primarycolor = pysubs2.Color(255, 255, 0)  # Жовтий текст
        style.bold = True
        subs.styles["Default"] = style
        # Збереження у файл
        subs.save(pathFor_subtitlesASS)
    else:
        print("Помилка: Відповідь не містить атрибутів 'segments' або 'words'")
        
    # CREATE VIDEO
    await update.message.reply_text('Створення відео.')

    image_folder = pathFor_images
    audio_file = pathFor_audio
    output_file = pathFor_outputVideo
    subtitles = pathFor_subtitlesASS
    
    await create_video_with_audio_and_subtitles(image_folder, audio_file, subtitles, output_file)
    await update.message.reply_text('Відео успішно створилось.')

    # Надсилання відео в чат
    with open(output_file, 'rb') as video_file:
        await update.message.reply_video(video_file)
        
    # tts --list_models
    # tts --text "Apple is launching a revolution in artificial intelligence! Are you ready for it? Apple is becoming a serious player in the world of AI, partnering with Broadcom to create the first server chip designed to handle AI applications. Imagine: your home is not just filled with 'apple' devices, but becomes an ecosystem optimized for artificial intelligence. Without AI, the future looks bleak. Your smartphone will not only make calls, but also solve your problems on its own. This chip can become the basis of new technological ideas, and Apple once again proves that the desire to innovate is in their DNA. We are waiting for news! Subscribe to the channel so you don't miss important updates!" --model_name "tts_models/en/ljspeech/tacotron2-DDC" --vocoder_name "vocoder_models/en/ljspeech/hifigan_v2" --out_path "E:/FromDesktop/ABV_telegram_bot/TTS_test/output/speech.wav"
    # tts --text "Епл запускає революцію в штучному інтелекті! Чи готові ви до цього? Епл стає серйозним гравцем у світі ЕЙАЙ, співпрацюючи з Бродком для створення першого серверного чіпа, призначеного для обробки ЕЙАЙ-застосунків. Уявіть: ваша домівка не просто заповнена 'яблучними' пристроями, а стає екосистемою, оптимізованою під штучний інтелект. Без ЕЙАЙ майбутнє виглядає тьмяно. Ваш смартфон не лише дзвонитиме, а й самостійно вирішуватиме ваші проблеми. Цей чіп може стати основою нових технологічних ідей, а ЕПЛ вкотре доводить, що прагнення до інновацій - у їхній ДНК. Чекаємо на новини! Підписуйтесь на канал, щоб не пропустити важливі оновлення!" --model_name "tts_models/multilingual/multi-dataset/xtts_v2" --vocoder_name "vocoder_models/en/ljspeech/hifigan_v2" --speaker_idx 0 --out_path "E:/FromDesktop/ABV_telegram_bot/TTS_test/output/speechUK.wav"
    # response_text = "Apple запускає революцію в штучному інтелекті! Чи готові ви до цього? Apple стає серйозним гравцем у світі AI, співпрацюючи з Broadcom для створення першого серверного чіпа, призначеного для обробки AI-застосунків. Уявіть: ваша домівка не просто заповнена 'яблучними' пристроями, а стає екосистемою, оптимізованою під штучний інтелект. Без AI майбутнє виглядає тьмяно. Ваш смартфон не лише дзвонитиме, а й самостійно вирішуватиме ваші проблеми. Цей чіп може стати основою нових технологічних ідей, а Apple вкотре доводить, що прагнення до інновацій — у їхній ДНК. Чекаємо на новини! Підписуйтесь на канал, щоб не пропустити важливі оновлення!"
    # response_text = "Apple is launching a revolution in artificial intelligence! Are you ready for it? Apple is becoming a serious player in the world of AI, partnering with Broadcom to create the first server chip designed to handle AI applications. Imagine: your home is not just filled with 'apple' devices, but becomes an ecosystem optimized for artificial intelligence. Without AI, the future looks bleak. Your smartphone will not only make calls, but also solve your problems on its own. This chip can become the basis of new technological ideas, and Apple once again proves that the desire to innovate is in their DNA. We are waiting for news! Subscribe to the channel so you don't miss important updates!"


    # client = ElevenLabs(api_key=context.user_data.get('api_key', ELEVENLABS_API_KEY), )        
    # response = client.text_to_speech.convert_with_timestamps(
    #     voice_id="21m00Tcm4TlvDq8ikWAM",
    #     output_format="mp3_44100_128",
    #     text=response_text,
    #     model_id="eleven_multilingual_v2"
    # )
    # # Перевірка наявності ключів у відповіді
    # if 'audio_base64' in response and 'alignment' in response:
    #     audio_base64 = response['audio_base64']
    #     alignment = response['alignment']
    #     # Декодування аудіо з base64
    #     audio_content = base64.b64decode(audio_base64)
    #     # Збереження озвучки у файл
    #     with open("response_audio.mp3", "wb") as audio_file:
    #         audio_file.write(audio_content)
    #     # Збереження таймстампів у файл
    #     with open("timestamps.json", "w", encoding="utf-8") as ts_file:
    #         json.dump(alignment, ts_file, ensure_ascii=False, indent=4)
    #     # print(alignment)
    #     await update.message.reply_text("Озвучка збережена")
    # else:
    #     await update.message.reply_text("Помилка: Невірна структура відповіді")
        


    # all_files = os.listdir(image_folder)
    # image_files = [f for f in all_files if f.endswith(('.jpg', '.png'))]
    # image_files.sort(key=lambda x: os.path.getmtime(os.path.join(image_folder, x)))
    # image_files = image_files[-10:]
    # image_files = [os.path.join(image_folder, f) for f in image_files]

    # Задайте тривалість для кожного зображення та тривалість переходів
    # durations = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]  # Тривалість для кожного зображення в секундах
    # transition_duration = 2  # Тривалість переходу в секундах

    # await update.message.reply_text("Початок цроцесу створення відео основи.")
    # await generate_video_with_effects(image_files, "output_video.mp4", durations, transition_duration)
    # await update.message.reply_text("Відео основу створено успішно.")
   
# Функція для конвертації часу у формат HH:MM:SS,mmm
def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

async def create_imagesPromts_with_chatGPT(video_base: str, context: ContextTypes.DEFAULT_TYPE) -> list:
    client = AsyncOpenAI(
        api_key=context.user_data.get('api_key', CHATGPT_API_KEY),
    )
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system", 
                "content": (
                    "You are a helpful assistant specializing in creative tasks. "
                    "Your primary role is to generate detailed, high-quality prompts for images that "
                    "will collectively form a cohesive video series. Each prompt should describe a visually "
                    "appealing and meaningful scene, ensuring that the scenes naturally flow from one to the next. "
                    "Use vivid descriptions, focus on storytelling elements, and include essential details like "
                    "composition, colors, and lighting to create an immersive visual narrative."
                )
            },
            {"role": "user", "content": f"Write 10 detailed prompts for pictures that will create a video series. Each prompt should start with the word 'Prompt' and should fit one after the other in terms of meaning. Here is the text: {video_base}"}
        ],
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.7,
    )
    
    # response_text = response['choices'][0]['message']['content'].strip()
    response_text = response.choices[0].message.content.strip()
    prompts = [prompt.strip() for prompt in response_text.split('Prompt') if prompt]
    
    # Видалення спеціальних символів та слова "Prompt" на початку кожного промту
    cleaned_prompts = []
    for prompt in prompts:
        # cleaned_prompt = re.sub(r'\*\*|\n', '', prompt).strip()
        cleaned_prompt = re.sub(r'\*\*|\n|\d+:', '', prompt).strip() # Видалення цифр та двокрапки

        if cleaned_prompt:
            cleaned_prompts.append(cleaned_prompt)
    
    context.user_data['prompts'] = cleaned_prompts
    return cleaned_prompts

async def create_img(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt: str, folder_path: str) -> None:
    """Генерує зображення за допомогою Together API."""
    userAPIKey = context.user_data.get('userAPIKey', TOGETHER_API_KEY)
    iterativeMode = False
    url = "https://api.together.xyz/v1/images/generations"

    payload = {
        "model": "black-forest-labs/FLUX.1-schnell-Free",
        "prompt": prompt,
        "steps": 4, # max 4
        # "seed": 123 if iterativeMode else None,
        "seed": 6385 if iterativeMode else None,
        "n": 1, # Кількість зображень для генерації
        "height": 1792, # max 1792
        "width": 1088,
        "response_format": "base64",
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {userAPIKey}"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Перевірка на HTTP помилки
    except requests.exceptions.RequestException as e:
        print(f"HTTP помилка: {e}")
    except Exception as e:
        print(f"Інша помилка: {e}")
        
    if response.status_code == 200:
        try:
            response_json = response.json()
            
            image_data = response_json['data'][0]['b64_json']
            image_bytes = base64.b64decode(image_data)
            
            # Отримання поточної дати та часу
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
            # Збереження зображення в папку src/images/назва_папки
            # folder_path = f'src/images/{folder_name}'
            os.makedirs(folder_path, exist_ok=True)
            with open(f'{folder_path}/{timestamp}.png', 'wb') as image_file:
                image_file.write(image_bytes)
            
        except (KeyError, TypeError, base64.binascii.Error) as e:
            print(f"Error processing image data: {e}")
            print(f"❌ Failed to process image data: {str(e)}")
        except OSError as e:
            print(f"Error saving image: {e}")
            print(f"❌ Failed to save image: {str(e)}")
    else:
        print(f"❌ Failed to generate image:")
                
def get_all_post_ids(context: ContextTypes.DEFAULT_TYPE):
    connection = get_connection()
    
    db_table_name = getDbTableName(context)
    
    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT id FROM {db_table_name}")

                ids = cursor.fetchall()
                return [id[0] for id in ids]
        except Exception as e:
            print(f"❌ Failed to fetch post ids: {e}")
            return []
    else:
        return []

def print_structure(d, indent=0):
    
    # # Перебір всіх ключів у відповіді, включаючи вкладені ключі
#     try:
#         response_json = response.json()
#         print_structure(response_json)
#     except ValueError:
#         print("Response is not in JSON format")

    """Рекурсивно виводить структуру словника, включаючи типи значень."""
    for key, value in d.items():
        print(' ' * indent + f"{key}: {type(value).__name__}")
        if isinstance(value, dict):
            print_structure(value, indent + 2)
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            print(' ' * (indent + 2) + "[")
            print_structure(value[0], indent + 4)
            print(' ' * (indent + 2) + "]")

def get_post_by_id(context: ContextTypes.DEFAULT_TYPE, post_id):
    connection = get_connection()
    
    db_table_name = getDbTableName(context)
    
    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT id, data FROM {db_table_name} WHERE id = %s", (post_id,))

                post = cursor.fetchone()
                if post:
                    return {"id": post[0], "data": post[1]}
                else:
                    return None
        except Exception as e:
            print(f"❌ Failed to fetch post: {e}")
            return None
    else:
        return None

async def delete_post_by_id(update: Update, context: ContextTypes.DEFAULT_TYPE, post_id=None):
    
    if post_id is None and context.user_data.get('post_index_for_show_next_post') != 0:
        post_ids = get_all_post_ids(context)
        
        post_index = context.user_data.get('post_index_for_show_next_post')

        if post_index < len(post_ids):
            post_id = post_ids[post_index - 1]
        else:
            print("❌ No post to delete.")
            await update.message.reply_text("❌ No post to delete.")
            return False

    connection = get_connection()

    db_table_name = getDbTableName(context)

    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"DELETE FROM {db_table_name} WHERE id = %s", (post_id,))
                connection.commit()
                if cursor.rowcount > 0:
                    context.user_data['post_index_for_show_next_post'] -= 1
                    await update.message.reply_text(f"✅ Post with id {post_id} deleted successfully.")
                    return True
                else:
                    await update.message.reply_text(f"❌ Post with id {post_id} not found.")
                    return False
        except Exception as e:
            print(f"❌ Failed to delete post: {e}")
            await update.message.reply_text(f"❌ Failed to delete post: {e}")
            return False
    else:
        await update.message.reply_text("❌ Failed to connect to the database.")
        return False

async def show_all_posts(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Вивід усіх постів з бази даних."""
    posts = get_posts(context)

    for post in posts:
        id = post['id']
        try:
            image, data = post['data'].split(',', 1)
            if image:
                await context.bot.send_photo(chat_id=update.effective_chat.id, photo=image, caption=data, reply_markup=create_keyboard())
            else:
                await update.message.reply_text(data, reply_markup=create_keyboard())
        except Exception as e:
            print(f"❌ Failed to show post: {e}, post_id: {id}")
        
async def show_next_post(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Вивід наступного поста з бази даних."""
    if 'post_index_for_show_next_post' not in context.user_data:
        context.user_data['post_index_for_show_next_post'] = 0

    post_ids = get_all_post_ids(context)
    post_index = context.user_data['post_index_for_show_next_post']

    if post_index < len(post_ids):
        post = get_post_by_id(context, post_ids[post_index])
        try:
            image, data = post['data'].split(',', 1)
            if image:
                await context.bot.send_photo(chat_id=update.effective_chat.id, photo=image, caption=data, reply_markup=create_keyboard())
            else:
                await update.message.reply_text(data, reply_markup=create_keyboard())
        except Exception as e:
            context.user_data['post_index_for_show_next_post'] += 1
            await update.message.reply_text(f"❌ Failed to show post: {e}, post_id: {id}, \n Пост пропускається і показується наступний.")
            print(f"❌ Failed to show post: {e}, post_id: {id}")

        context.user_data['post_index_for_show_next_post'] += 1
    else:
        await update.message.reply_text("Це був останній пост.", reply_markup=create_keyboard())

async def confirm_post(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.user_data.get('post_index_for_show_next_post') != 0:
        post_ids = get_all_post_ids(context)
        post_id = post_ids[context.user_data['post_index_for_show_next_post'] - 1]
        post = get_post_by_id(context, post_id)
        if post:
            data = post['data']
            try:
                image, text = data.split(',', 1)
                # Отримання ідентифікатора останнього повідомлення
                last_message_id = update.message.message_id - 1

                # Відповідь на останнє повідомлення
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="Confirm the post",
                    reply_to_message_id=last_message_id
                )
            except Exception as e:
                print(f"❌ Failed to confirm post: {e}")
                await update.message.reply_text(f"❌ Failed to confirm post: {e}")
        else:
            await update.message.reply_text("❌ Post not found.")
    else:
        await update.message.reply_text("❌ No post selected for confirmation.")
        
def getDbTableName(context: ContextTypes.DEFAULT_TYPE) -> str:
    """Повертає назву таблиці залежно від вибору користувача."""
    if context.user_data.get('whatDb_table_now_info') == "Нові основи для постів":
        return "basics_posts"
    elif context.user_data.get('whatDb_table_now_info') == "Готові пости для телеграму":
        return "telegram_posts"
    
async def show_posts_fromStart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Початок перегляду постів з початку, індексування глобальної зміної з нуля."""
    context.user_data['post_index_for_show_next_post'] = 0

async def send_startup_message(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Відправляє повідомлення при запуску бота."""
    chat_id = '1311004971'  # Замість YOUR_CHAT_ID вкажіть ваш chat_id
    await context.bot.send_message(chat_id=chat_id, text=f"Привіт! Використовуйте кнопки нижче для взаємодії, chatId: ({chat_id})", reply_markup=create_keyboard(context))
    
def main() -> None:
    """Запускає бота."""
    application = Application.builder().token(TOKEN).build()

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Налаштування JobQueue
    job_queue = application.job_queue
    job_queue.run_once(send_startup_message, 0)

    application.run_polling()

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
# **Сценарій для відео шортс**

# **0-5 секунд (Клікбейт):**  
# (Відео: динамічний монтаж з логотипами Apple та AI технологій)  
# Озвучка: "Apple запускає революцію в штучному інтелекті! Чи готові ви до цього?"

# ---

# **5-25 секунд (Основна розповідь):**  
# (Відео: анімація, що показує співпрацю Apple з Broadcom, серверні чіпи, сценарії використання AI у побуті)  
# Озвучка: "Apple стає серйозним гравцем у світі AI, співпрацюючи з Broadcom для створення першого серверного чіпа, призначеного для обробки AI-застосунків. Уявіть: ваша домівка не просто заповнена 'яблучними' пристроями, а стає екосистемою, оптимізованою під штучний інтелект. Без AI майбутнє виглядає тьмяно. Ваш смартфон не лише дзвонитиме, а й самостійно вирішуватиме ваші проблеми. Цей чіп може стати основою нових технологічних ідей, а Apple вкотре доводить, що прагнення до інновацій — у їхній ДНК."

# ---

# **25-30 секунд (Заклик до дії):**
# (Відео: заключний кадр з логотипом Apple та закликом підписатися)
# Озвучка: "Чекаємо на новини! Підписуйтесь на канал, щоб не пропустити важливі оновлення!"



# виправ якщо приходять наступні дані {'audio_base64': '....=', 'alignment': {'characters': ['t', 'e', 's', 't'], 'character_start_times_seconds': [0.0, 0.232, 0.348, 0.43], 'character_end_times_seconds': [0.232, 0.348, 0.43, 0.929]}, 'normalized_alignment': {'characters': [' ', 't', 'e', 's', 't', ' '], 'character_start_times_seconds': [0.0, 0.151, 0.232, 0.348, 0.43, 0.546], 'character_end_times_seconds': [0.151, 0.232, 0.348, 0.43, 0.546, 0.929]}}


# {'characters': ['A', 'p', 'p', 'l', 'e', ' ', 'з', 'а', 'п', 'у', 'с', 'к', 'а', 'є', ' ', 'р', 'е', 'в', 'о', 'л', 'ю', 'ц', 'і', 'ю', ' ', 'в', ' ', 'ш', 'т',''
# ', '!'], 'character_start_times_seconds': [0.0, 0.29, 0.348, 0.395, 0.488, 0.522, 0.546, 0.58, 0.65, 0.697, 0.755, 0.824, 0.894, 1.01, 1.057, 1.115, 1.149, 1.207, 1.254, 1.312, 1.382, 1.463, 1.544, 1.625, 1.672, 1.707, 1.73