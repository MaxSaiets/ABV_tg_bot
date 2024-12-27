import os
import random
from PIL import Image
import cv2
import numpy as np
from moviepy import *  # noqa
# from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import AudioFileClip, TextClip, CompositeVideoClip, ImageClip, concatenate_videoclips
from moviepy.video.fx.resize import resize
import moviepy.editor as mp
import pysubs2
# Налаштування шляху до ImageMagick
from moviepy.config import change_settings
change_settings({"IMAGEMAGICK_BINARY": "C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"})

def resize_images_to_match(img1, img2):
    """Змінює розмір зображень, щоб вони мали однаковий розмір."""
    width1, height1 = img1.size
    width2, height2 = img2.size
    new_width = min(width1, width2)
    new_height = min(height1, height2)
    img1 = img1.resize((new_width, new_height), Image.LANCZOS)
    img2 = img2.resize((new_width, new_height), Image.LANCZOS)
    return img1, img2

def create_transition_frames_blend(img1, img2, steps=30):
    """Створює перехід через змішування між двома зображеннями."""
    img1, img2 = resize_images_to_match(img1, img2)
    frames = []
    for alpha in range(steps):
        blend = Image.blend(img1, img2, alpha / steps)
        frames.append(blend)
    return frames

def create_transition_frames_slide(img1, img2, steps=30):
    """Створює перехід через зсув між двома зображеннями."""
    img1, img2 = resize_images_to_match(img1, img2)
    frames = []
    width, height = img1.size
    for step in range(steps):
        offset = int(width * step / steps)
        new_frame = Image.new('RGB', (width, height))
        new_frame.paste(img1, (-offset, 0))
        new_frame.paste(img2, (width - offset, 0))
        frames.append(new_frame)
    return frames

def create_transition_frames_wipe(img1, img2, steps=30):
    """Створює перехід через витіснення між двома зображеннями."""
    img1, img2 = resize_images_to_match(img1, img2)
    frames = []
    width, height = img1.size
    for step in range(steps):
        offset = int(width * step / steps)
        new_frame = Image.new('RGB', (width, height))
        new_frame.paste(img1.crop((0, 0, offset, height)), (0, 0))
        new_frame.paste(img2.crop((offset, 0, width, height)), (offset, 0))
        frames.append(new_frame)
    return frames

async def generate_video_with_effects(image_paths, output_path, durations, transition_duration, fps=30):
    """Створює відео з ефектами та переходами."""
    images = [Image.open(img_path) for img_path in image_paths]
    num_images = len(images)
    
    # Розрахунок кількості кадрів для кожного зображення та переходу
    transition_frames = int(transition_duration * fps)
    all_frames = []

    for i in range(num_images):
        img1 = images[i]
        image_duration = durations[i]
        image_frames = int(image_duration * fps)
        
        # Додавання кадрів для поточного зображення
        for _ in range(image_frames):
            all_frames.append(img1)
        
        # Додавання переходу до наступного зображення
        if i < num_images - 1:
            img2 = images[i + 1]
            transition_function = random.choice([create_transition_frames_blend,])
            transition_frames_list = transition_function(img1, img2, transition_frames)
            all_frames.extend(transition_frames_list)
    
    # Зберігаємо кадри як відео
    frame = cv2.cvtColor(np.array(all_frames[0]), cv2.COLOR_RGB2BGR)
    height, width = frame.shape[:2]
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for img in all_frames:
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        video.write(frame)

    video.release()
    


async def create_video_with_audio_and_subtitles(image_folder, audio_file, subtitles_file, output_file):
    # Отримання списку зображень
    # all_files = os.listdir(image_folder)
    # image_files = [f for f in all_files if f.endswith(('.jpg', '.png'))]
    # image_files.sort(key=lambda x: os.path.getmtime(os.path.join(image_folder, x)))
    # image_files = image_files[-10:] # from the last 10 images
    # image_files = [os.path.join(image_folder, f) for f in image_files]

    # Отримання списку зображень
    all_files = os.listdir(image_folder)
    image_files = [f for f in all_files if f.endswith(('.jpg', '.png'))]
    image_files.sort(key=lambda x: os.path.getmtime(os.path.join(image_folder, x)))
    image_files = [os.path.join(image_folder, f) for f in image_files]

    # Задайте тривалість для кожного зображення та тривалість переходів
    # durations = [3] * len(image_files)  # Тривалість для кожного зображення в секундах
    durations = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]  # Тривалість для кожного зображення в секундах
    transition_duration = 2  # Тривалість переходу в секундах

    # # Створення кліпів з зображень
    # clips = []
    # for image_file, duration in zip(image_files, durations):
    #     clip = ImageClip(image_file, duration=duration)
    #     if clip.w / clip.h > 1080 / 1920:
    #         # Якщо зображення ширше, ніж 1080x1920
    #         clip = clip.resize(width=1080)
    #         top_margin = (1920 - clip.h) // 2
    #         bottom_margin = 1920 - clip.h - top_margin
    #         clip = clip.margin(top=top_margin, bottom=bottom_margin, color=(0, 0, 0))
    #     else:
    #         # Якщо зображення вище, ніж 1080x1920
    #         clip = clip.resize(height=1920)
    #         left_margin = (1080 - clip.w) // 2
    #         right_margin = 1080 - clip.w - left_margin
    #         clip = clip.margin(left=left_margin, right=right_margin, color=(0, 0, 0))
    #     clips.append(clip)
    # Створення кліпів з зображень
    clips = []
    for image_file, duration in zip(image_files, durations):
        clip = ImageClip(image_file, duration=duration)
        clip = clip.resize(height=1920, width=1080).set_position(("center", "center"))
        clips.append(clip)
    # Додавання плавних переходів
    # clips_with_transitions = [
    #     clips[i].crossfadeout(transition_duration).crossfadein(transition_duration)
    #     for i in range(len(clips) - 1)
    # ]
    
# clip.crossfadeout(transition_duration)розчинення
# Плавне затемнення та появлення:
# clip = clip.fadein(transition_duration).fadeout(transition_duration)

    # Додавання плавних переходів
    clips_with_transitions = []
    for i in range(len(clips) - 1):
        clips_with_transitions.append(clips[i].crossfadeout(transition_duration))
        clips_with_transitions.append(clips[i + 1].crossfadein(transition_duration))

    # Об'єднання кліпів із переходами
    video = concatenate_videoclips(clips_with_transitions, method="compose")

# clip1 = VideoFileClip("video1.mp4").fx(vfx.speedx, 1.5)  # Прискорений кліп    
# clip2 = VideoFileClip("video2.mp4").fx(vfx.speedx, 0.75)  # Уповільнений кліп
    # Завантаження кліпів
# clip1 = VideoFileClip("video1.mp4")
# clip2 = VideoFileClip("video2.mp4").crossfadein(1)  # Перехід на 1 секунду
    # Збереження об'єднаного відео
# final_clip.write_videofile("final_video_with_transition.mp4", codec="libx264", audio_codec="aac")
    # Додавання переходів між кліпами
    # video = concatenate_videoclips(clips, method="compose", padding=-transition_duration)

    # Додавання аудіо
    audio = AudioFileClip(audio_file)
    video = video.set_audio(audio)

    # Додавання субтитрів
    subs = pysubs2.load(subtitles_file)
    def add_subtitles(clip, subs):
        def make_textclip(event):
            return TextClip(event.text, fontsize=100, color='red', font='Arial-Bold', stroke_color='black', stroke_width=2).set_start(event.start / 1000).set_end(event.end / 1000).set_position(("center", "center"))
        
        subtitle_clips = [make_textclip(event) for event in subs.events]
        return CompositeVideoClip([clip, *subtitle_clips])

    video = add_subtitles(video, subs)

    # Збереження відео
    video.write_videofile(output_file, fps=24, codec='libx264', audio_codec='aac')

def apply_effect(image_path):
    """Додає ефект до зображення (наприклад, змінення кольору)."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Чорно-білий ефект
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Повертаємо до формату кольорів


def resize_video_to_fill(clip, target_size):
    
    # how to use
    # ширина зарез 1920 1080 
    #  # Завантаження відео
    # video_path = output_file
    # output_path = "output_video.mp4"
    # clip = mp.VideoFileClip(video_path)

    # # Масштабування до 1920x1080
    # target_size = (1920, 1080)
    # resized_clip = resize_video_to_fill(clip, target_size)

    # # Збереження результату
    # resized_clip.write_videofile(output_path, codec="libx264", preset="superfast")
    
    
    """
    Масштабування відео до заданого розміру, зберігаючи пропорції
    та обрізаючи надлишки, щоб уникнути чорних ліній.

    :param clip: Відео-кліп MoviePy
    :param target_size: (width, height) цільовий розмір
    :return: Відмасштабований відео-кліп
    """
    target_width, target_height = target_size
    clip_width, clip_height = clip.size

    # Визначення масштабів для ширини та висоти
    scale_w = target_width / clip_width
    scale_h = target_height / clip_height

    # Вибір більшого масштабу для заповнення екрану
    scale = max(scale_w, scale_h)

    # Розрахунок нового розміру та центроване обрізання
    new_width = int(clip_width * scale)
    new_height = int(clip_height * scale)
    resized_clip = clip.resize((new_width, new_height))
    final_clip = resized_clip.crop(
        x_center=new_width / 2,
        y_center=new_height / 2,
        width=target_width,
        height=target_height
    )
    return final_clip
