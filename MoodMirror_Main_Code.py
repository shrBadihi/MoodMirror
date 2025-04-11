# ----------------------------- Imports -----------------------------
from google import genai
from google.genai import types
from kaggle_secrets import UserSecretsClient
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
import json
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
from typing import List, Dict
import speech_recognition as sr
from fer import FER
from PIL import Image
import typing_extensions as typing
from collections import Counter
from pydub import AudioSegment
import string
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# import nltk

# ----------------------------- API Setup -----------------------------
GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)
# /////////////////////////////////////////////////////////////////////////////////////////////////////
# ----------------------------- RAG Documents -----------------------------
RAG_DOCUMENTS = [
    "Practicing gratitude can significantly improve mental well-being by shifting focus from negative thoughts.",
    "CBT techniques help reframe negative thinking patterns into constructive insights.",
    "Social connection and being heard improve emotional regulation and resilience.",
    "Journaling builds self-awareness and emotional processing.",
    "Mindfulness and breathing exercises can reduce anxiety symptoms.",
    "Setting boundaries helps prevent burnout and protects well-being.",
    "Labeling emotions accurately helps regulate them.",
    "Acts of self-care can boost mood and positivity.",
    "A sense of purpose improves long-term well-being.",
    "Self-compassion reduces self-criticism and nurtures kindness.",
    "Building resilience involves embracing challenges and learning from adversity.",
    "Exercise and physical activity have a profound impact on mental health.",
    "Developing a growth mindset helps overcome obstacles and setbacks.",
    "Sleep hygiene and quality rest play a critical role in emotional health.",
    "Accepting imperfections and practicing self-forgiveness can reduce stress.",
    "Positive affirmations can improve self-esteem and mental clarity.",
    "Mindful eating and nutrition impact mental and emotional states.",
    "Visualization techniques can help manage stress and anxiety.",
    "Effective communication skills are essential for managing conflict and building connections.",
    "Grief is a complex emotional experience that requires time, patience, and support."
]
# /////////////////////////////////////////////////////////////////////////////////////////////////////
# ----------------------------- Embed RAG Documents -----------------------------
embedding_response = client.models.embed_content(
    model="models/text-embedding-004",
    contents=RAG_DOCUMENTS,
    config=types.EmbedContentConfig(task_type="retrieval_document")
)
rag_embeddings = [e.values for e in embedding_response.embeddings]
# /////////////////////////////////////////////////////////////////////////////////////////////////////
# ----------------------------- Journal Analysis -----------------------------
def get_top_context(journal_entry: str, top_k=3) -> str:
    """
    Finds the top matching RAG documents for a journal entry based on cosine similarity
    with embedded document vectors.

    Args:
        journal_entry (str): The user journal text to match against RAG documents.
        top_k (int): Number of top documents to retrieve.

    Returns:
        str: Concatenated top-k RAG documents as grounding context.
    """
    query_embedding = client.models.embed_content(
        model="models/text-embedding-004",
        contents=journal_entry,
        config=types.EmbedContentConfig(task_type="retrieval_query")
    ).embeddings[0].values

    scores = cosine_similarity([query_embedding], rag_embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return "\n".join([RAG_DOCUMENTS[i] for i in top_indices])

class JournalAnalysis(typing.TypedDict):
    emotion: str
    themes: List[str]
    suggestion: str
    affirmation: str
    
def analyze_journal_entry(entry: str) -> str:
    """
    Analyzes a journal entry using the Gemini model and returns a structured output
    with emotion, themes, suggestions, and affirmation.
    
    Args:
        entry (str): The user's journal text.
    
    Returns:
        str: Parsed JSON response as a dictionary.
    """
    context = get_top_context(entry)
    prompt = f"""
        You are a supportive AI journaling assistant.
        Analyze the user's journal entry and return a structured JSON with the following:
        1. Primary emotion (e.g., sad, anxious, hopeful)
        2. Up to 3 key themes
        3. One CBT-style reflection suggestion
        4. One daily affirmation

        Use this context to ground your suggestions:
        {context}

        Here are a few examples:

        Journal Entry:
        I feel hopeless. Everything I do seems to go wrong.
        Response:
        {{
          "emotion": "hopeless",
          "themes": ["self-doubt", "negativity"],
          "suggestion": "Try writing down three things that went well each day, no matter how small.",
          "affirmation": "You are resilient and capable of overcoming hard days."
        }}

        Journal Entry:
        I felt better today. I went for a walk and saw some friends.
        Response:
        {{
          "emotion": "grateful",
          "themes": ["connection", "nature"],
          "suggestion": "Continue spending time doing things that bring you joy.",
          "affirmation": "Joy is found in small, simple moments."
        }}

        Now analyze the following entry:
        {entry}
        Respond in JSON format like:
        {{
          "emotion": "...",
          "themes": ["...", "..."],
          "suggestion": "...",
          "affirmation": "..."
        }}
        """

    config = types.GenerateContentConfig(
        temperature=0.9,  # Increased to encourage more randomness
        top_k=5,         # Number of top tokens to consider at each step
        response_mime_type="application/json",
        response_schema=JournalAnalysis
    )
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt],
        config=config
    )

    return response.parsed
# /////////////////////////////////////////////////////////////////////////////////////////////////////
# ----------------------------- Audio Processing -----------------------------
def convert_m4a_to_wav(input_path: str) -> str:
    """
    Converts an M4A audio file to WAV format for processing.
    
    Args:
        input_path (str): Path to the input M4A audio file.
    
    Returns:
        str: Path to the converted WAV file.
    """
    output_path = input_path.rsplit(".", 1)[0] + ".wav"
    sound = AudioSegment.from_file(input_path, format="m4a")
    sound.export(output_path, format="wav")
    return output_path

def recognize_audio(audio_path):
    """
    Transcribes spoken audio to text using Google's Speech Recognition API.
    
    Args:
        audio_path (str): Path to the WAV audio file.
    
    Returns:
        str: Transcribed text or error message.
    """
    recognizer = sr.Recognizer()

    try:
        if audio_path.endswith(".m4a"):
            audio_path = convert_m4a_to_wav(audio_path)

        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)

        return recognizer.recognize_google(audio)

    except Exception as e:
        return f"[Transcription error] {e}"

# /////////////////////////////////////////////////////////////////////////////////////////////////////
# ----------------------------- Image Analysis -----------------------------
def detect_emotion_from_image(img: Image.Image) -> str:
    """
    Detects the dominant emotion in a facial image using the FER library.
    
    Args:
        img (Image.Image): A PIL image containing a human face.
    
    Returns:
        str: The top detected emotion or a message if no face is found.
    """
    detector = FER(mtcnn=True)
    img_array = np.array(img)
    result = detector.detect_emotions(img_array)
    if not result:
        return "unknown"
    emotions = result[0]["emotions"]
    top_emotion = max(emotions, key=emotions.get)
    return top_emotion
# /////////////////////////////////////////////////////////////////////////////////////////////////////
# ----------------------------- Visualizations -----------------------------
def get_emotion_score(emotion: str) -> int:
    """
    Maps an emotion string to a numerical score for trend visualization.
    
    Args:
        emotion (str): Emotion label (e.g., 'happy', 'sad').
    
    Returns:
        int: A numerical score from 1 (low mood) to 5 (high mood).
    """
    emotion = emotion.lower().strip()
    
    mood_categories = {
        1: ["burnt", "overwhelmed", "depressed", "hopeless", "terrible", "exhausted", "down"],
        2: ["anxious", "sad", "stressed", "nervous", "worried"],
        3: ["neutral", "okay", "fine", "meh", "uncertain", "tired"],
        4: ["hopeful", "better", "relieved", "good"],
        5: ["grateful", "happy", "calm", "joyful", "peaceful", "excited"]
    }

    for score, keywords in mood_categories.items():
        if any(re.search(rf"\b{kw}\b", emotion) for kw in keywords):
            return score

    return 3  # Default to neutral if no match
    
def plot_emotional_trend(results: List[Dict]):
    
    """
    Plots a line chart of emotional scores over time based on journal entries.
    
    Args:
        results (List[Dict]): List of journal entries with 'date' and 'emotion'.
    
    Returns:
        matplotlib.figure.Figure: A plot showing mood trends.
    """
    if not results:
        return None

    dates = [entry["date"] for entry in results]
    scores = [get_emotion_score(entry["emotion"]) for entry in results]

    plt.figure(figsize=(8, 4))
    plt.plot(dates, scores, marker='o')
    plt.title("Mood Trend Over Time (All Entry Types)")
    plt.ylabel("Emotional State (1=Low, 5=High)")
    plt.xlabel("Date")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.yticks([1, 2, 3, 4, 5], ["Very Low", "Low", "Neutral", "High", "Very High"])
    plt.tight_layout()
    return plt


def plot_trend_wrapper():
    """
    Wrapper function to plot journal trend chart from global variable `journal_entries`.
    
    Returns:
        matplotlib.figure.Figure: Trend plot figure.
    """
    return plot_emotional_trend(journal_entries)
    
def plot_emotion_distribution(entries: List[Dict]):
    """
    Generates a bar chart showing the distribution of different emotions across entries.
    
    Args:
        entries (List[Dict]): List of journal entries with 'emotion'.
    
    Returns:
        matplotlib.figure.Figure: A bar chart of emotion counts.
    """
    emotion_counts = Counter(entry["emotion"] for entry in entries if entry.get("emotion"))
    labels, values = zip(*emotion_counts.items())
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.title("Emotion Distribution")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

# nltk.download('stopwords')
# def plot_wordcloud(entries: List[Dict]):
#     """
#     Generates a word cloud visualization from all journal entry texts.
    
#     Args:
#         entries (List[Dict]): List of journal entries with 'entry' text.
    
#     Returns:
#         matplotlib.figure.Figure: Word cloud plot.
#     """
#     text = " ".join(entry["entry"] for entry in entries if entry.get("entry"))
#     text = text.lower().translate(str.maketrans('', '', string.punctuation))

#     stop_words = set(stopwords.words("english"))
#     stemmer = PorterStemmer()

#     words = text.split()
#     filtered = [stemmer.stem(w) for w in words if w not in stop_words and len(w) > 2]
#     cleaned_text = " ".join(filtered)

#     wordcloud = WordCloud(
#         width=800, height=400,
#         background_color='white'
#     ).generate(cleaned_text)

#     plt.figure(figsize=(10, 5))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis("off")
#     plt.title("Common Words in Journal Entries")
#     plt.tight_layout()
#     return plt

def reflect_on_week(entries: List[Dict]):
    """
    Summarizes a user's journal entries from the past week and offers supportive advice.

    Args:
        entries (List[Dict]): A list of journal entries, each containing 'date', 'entry', and 'emotion'.

    Returns:
        str: A short reflection generated by Gemini summarizing the week's entries and advice.
    """
    if not entries:
        return "No entries yet."
    journal_texts = "\n\n".join(f"{e['date']}: {e['entry']} ({e['emotion']})" for e in entries[-7:])
    prompt = f"""
    Summarize the following journal logs from the past week and give the user some supportive advice:
    {journal_texts}
    Respond in a paragraph.
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt]
    )
    return response.text

def suggest_tip_toward_goal(goal: str):
    """
    Generates a CBT-style tip and affirmation to support progress toward a specific emotional goal.

    Args:
        goal (str): The desired emotional state, e.g., "calm", "confident".

    Returns:
        str: A formatted suggestion with both a CBT tip and an affirmation.
    """
    prompt = f"""
    Suggest a CBT-style action and affirmation to help someone feel more {goal}.
    Format:
    - Tip: ...
    - Affirmation: ...
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt],
        config=types.GenerateContentConfig(temperature=0.7)
    )
    return response.text

# /////////////////////////////////////////////////////////////////////////////////////////////////////
# ----------------------------- Function Calling Tools -----------------------------
def summarize_week() -> str:
    """
    Summarizes the number of journal entries and the emotions recorded over the past 7 days.

    Returns:
        str: A brief textual summary including number of entries and a list of emotions.
    """
    week_entries = [e for e in journal_entries if datetime.strptime(e['date'], "%Y-%m-%d") >= datetime.now() - timedelta(days=7)]
    emotions = [e["emotion"] for e in week_entries]
    return f"Entries: {len(week_entries)} | Emotions: {', '.join(emotions)}"

def suggest_cbt_tip() -> str:
    """
    Suggests a CBT-style tip based on recent journal entries using similarity search with RAG documents.

    Returns:
        str: The most relevant CBT-style tip from the document set or journaling encouragement if insufficient data.
    """
    if not journal_entries:
        return "Try to start journaling consistently."
    all_entries = [e for e in journal_entries if "emotion" in e and e["emotion"]]
    if not all_entries:
        return "Reflect on how you're feeling daily to get helpful feedback."
    query_text = "Average mood over the past few days has been " + ", ".join(e["emotion"] for e in all_entries[-5:])
    query_embedding = client.models.embed_content(
        model="models/text-embedding-004",
        contents=query_text,
        config=types.EmbedContentConfig(task_type="retrieval_query")
    ).embeddings[0].values
    scores = cosine_similarity([query_embedding], rag_embeddings)[0]
    top_index = int(np.argmax(scores))
    return RAG_DOCUMENTS[top_index]

def describe_emotion_trend() -> str:
    """
    Analyzes the mood trend over time based on emotion scores from journal entries.

    Returns:
        str: A message indicating whether mood has improved, declined, or remained steady.
    """
    if len(journal_entries) < 2:
        return "Not enough entries to assess a trend."
    scores = [get_emotion_score(e["emotion"]) for e in journal_entries if e.get("emotion")]
    if not scores:
        return "No emotion scores available to determine a trend."
    delta = scores[-1] - scores[0]
    if delta > 0:
        return "Your mood seems to be improving over time. 🌱"
    elif delta < 0:
        return "Your mood seems to have dipped recently. Be kind to yourself. 💙"
    else:
        return "Your mood has remained steady. 🧘"

cbt_tools = [summarize_week, suggest_cbt_tip, describe_emotion_trend]

instruction = """You are a helpful assistant that helps users reflect on their emotional patterns and well-being trends. \
You can use available functions to summarize how their week has been or suggest CBT-style tips based on their average emotional state.\
Use summarize_week to summarize mood history. Use suggest_cbt_tip to offer relevant tips grounded in past mood logs. Use describe_emotion_trend to give feedback on how the user's emotional state has evolved over time."""

assistant_chat = client.chats.create(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        tools=cbt_tools,
        system_instruction=instruction
    )
)
# /////////////////////////////////////////////////////////////////////////////////////////////////////
# ----------------------------- Gradio Interface -----------------------------
journal_entries: List[Dict] = []

def analyze_and_display(entry_text: str, entry_date: str) -> str:
    result = analyze_journal_entry(entry_text)
    journal_entries.append({
        "date": entry_date,
        "entry": entry_text,
        **result
    })
    return f"""
**Emotion:** {result['emotion']}

**Themes:** {', '.join(result['themes'])}

**Reflection Suggestion:** {result['suggestion']}

**Daily Affirmation:** {result['affirmation']}"""

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 MoodMirror: Multimodal Journal Tracker")
    date_input = gr.Text(label="Entry Date", value=str(date.today()))

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Tab("📝 Text Entry"):
                entry_input = gr.Textbox(
                    lines=6,
                    label="Journal Entry",
                    # value="For example: I’ve been feeling overwhelmed with work and don’t have time to relax.",
                    placeholder="Write your journal entry here..."
                )
        
                text_analysis = gr.Markdown()
                analyze_text_btn = gr.Button("Analyze Text")
        
                analyze_text_btn.click(analyze_and_display, inputs=[entry_input, date_input], outputs=text_analysis)
    
            with gr.Tab("🎤 Audio Entry"):
                audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath")
                audio_transcript = gr.Textbox(label="Transcript")
                audio_analysis = gr.Markdown()
                transcribe_btn = gr.Button("Transcribe + Analyze")
                transcribe_btn.click(recognize_audio, inputs=audio_input, outputs=audio_transcript).then(
                    analyze_and_display, inputs=[audio_transcript, date_input], outputs=audio_analysis
                )

            with gr.Tab("📸 Image Entry"):
                image_input = gr.Image(type="pil")
                image_result = gr.Textbox(label="Detected Emotion")
                analyze_image_btn = gr.Button("Detect and Add")

                def detect_and_store_image(img, date_str):
                    emotion = detect_emotion_from_image(img)
                    journal_entries.append({
                        "date": date_str,
                        "entry": "Image mood analysis",
                        "emotion": emotion,
                        "themes": [],
                        "suggestion": "",
                        "affirmation": ""
                    })
                    return emotion

                analyze_image_btn.click(detect_and_store_image, inputs=[image_input, date_input], outputs=image_result)

            with gr.Tab("📈 Mood Visualizations"):
                with gr.Row():
                    trend_btn = gr.Button("📈 Mood Trend")
                    dist_btn = gr.Button("📊 Emotion Distribution")
                    # cloud_btn = gr.Button("☁️ Word Cloud")
                viz_plot = gr.Plot()
                trend_btn.click(fn=plot_trend_wrapper, outputs=viz_plot)
                dist_btn.click(lambda: plot_emotion_distribution(journal_entries), outputs=viz_plot)
                # cloud_btn.click(lambda: plot_wordcloud(journal_entries), outputs=viz_plot)

            with gr.Tab("🎯 Suggestions & Reflections"):
                with gr.Row():
                    goal_input = gr.Textbox(label="How do you want to feel?", placeholder="e.g., calm, joyful")
                with gr.Row():
                    goal_btn = gr.Button("🎯 Suggest Tip Toward Goal")
                    goal_output = gr.Textbox(label="Suggestion and Affirmation")
                    goal_btn.click(suggest_tip_toward_goal, inputs=goal_input, outputs=goal_output)
                    
                with gr.Row():
                    reflect_btn = gr.Button("🧘 Weekly Reflection Summary")
                    reflect_output = gr.Textbox(label="Reflection Summary")
                    reflect_btn.click(lambda: reflect_on_week(journal_entries), outputs=reflect_output)
                
        with gr.Column(scale=2):
            gr.Markdown("## 💬 MoodMirror Assistant")
            chatbot = gr.Chatbot()
            chat_input = gr.Textbox(placeholder="Ask for summary or mood trend...")
            send_btn = gr.Button("Send")

            def chatbot_reply(message, history):
                response = assistant_chat.send_message(message)
                return history + [[message, response.text]]

            send_btn.click(fn=chatbot_reply, inputs=[chat_input, chatbot], outputs=chatbot)
            chat_input.submit(fn=chatbot_reply, inputs=[chat_input, chatbot], outputs=chatbot)

demo.launch()