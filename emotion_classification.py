# Import libraries
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load model
model_name = "j-hartmann/emotion-english-distilroberta-base"
emotion_classifier = pipeline("sentiment-analysis", model=model_name, return_all_scores=False)

# Function to classify emotions
def classify_emotion(text):
    result = emotion_classifier(text)
    return result[0]['label']

# Load data
def load_data():
    headlines_df = pd.read_csv('fake_or_real_news.csv')
    return headlines_df

# Perform emotion classification for headline in data and add column for predicted emotions
def perform_emotion_classification(headlines_df):
    headlines_df['predicted_emotion'] = headlines_df['title'].apply(classify_emotion)
    return headlines_df

# Create tables and visualizations to analyze  distribution of emotions
def analyze_emotion_distribution(headlines_df):
    # all data
    all_data_emotion_counts = headlines_df['predicted_emotion'].value_counts()
    print("Distribution of emotions across all data:")
    print(all_data_emotion_counts)
    all_data_emotion_counts.to_csv('out/all_data_emotion_counts.csv', index=False) 
    # real news
    real_news_emotion_counts = headlines_df[headlines_df['label'] == 'REAL']['predicted_emotion'].value_counts()
    print("\nDistribution of emotions across real news:")
    print(real_news_emotion_counts)
    real_news_emotion_counts.to_csv('out/real_news_emotion_counts.csv', index=False)
    # Fake news
    fake_news_emotion_counts = headlines_df[headlines_df['label'] == 'FAKE']['predicted_emotion'].value_counts()
    print("\nDistribution of emotions across fake news:")
    print(fake_news_emotion_counts)
    fake_news_emotion_counts.to_csv('out/fake_news_emotion_counts.csv', index=False)
    # Add colours to the values
    emotion_colours = {
    'anger': 'red',
    'joy': 'yellow',
    'sadness': 'gray',
    'fear': 'purple',
    'surprise': 'pink',
    'disgust' : 'green',
    'neutral' : 'blue'}

    # Create bar charts to visualize the emotions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    all_data_emotion_counts.plot(kind='bar', ax=axes[0], color=[emotion_colours[emotion] for emotion in all_data_emotion_counts.index])
    axes[0].set_title("Emotion Distribution (All Data)")
    real_news_emotion_counts.plot(kind='bar', ax=axes[1], color=[emotion_colours[emotion] for emotion in real_news_emotion_counts.index])
    axes[1].set_title("Emotion Distribution (Real News)")
    fake_news_emotion_counts.plot(kind='bar', ax=axes[2], color=[emotion_colours[emotion] for emotion in fake_news_emotion_counts.index])
    axes[2].set_title("Emotion Distribution (Fake News)")
    plt.savefig('out/FakevsReal_Emotions.png')

# Too much "neutral" so lets generate the same without neutral
def analyze_emotion_distribution_withoutneutral(headlines_df):
    #  All data
    all_data_emotion_counts = headlines_df['predicted_emotion'].value_counts()
    all_data_emotion_counts = all_data_emotion_counts.drop('neutral')
    print("Distribution of emotions across all data(without neutral):")
    print(all_data_emotion_counts)
    all_data_emotion_counts.to_csv('out/all_data_emotion_counts_without_neutral.csv', index=False)
    # Real news
    real_news_emotion_counts = headlines_df[headlines_df['label'] == 'REAL']['predicted_emotion'].value_counts()
    real_news_emotion_counts = real_news_emotion_counts.drop('neutral')
    print("\nDistribution of emotions across real news(without neutral):")
    print(real_news_emotion_counts)
    real_news_emotion_counts.to_csv('out/real_news_emotion_counts_without_neutral.csv', index=False)
    # Fake news
    fake_news_emotion_counts = headlines_df[headlines_df['label'] == 'FAKE']['predicted_emotion'].value_counts()
    fake_news_emotion_counts = fake_news_emotion_counts.drop('neutral')
    print("\nDistribution of emotions across fake news(without neutral):")
    print(fake_news_emotion_counts)
    fake_news_emotion_counts.to_csv('out/fake_news_emotion_counts_without_neutral.csv', index=False)
    
    # Add colours to the values
    emotion_colours = {
    'anger': 'red',
    'joy': 'yellow',
    'sadness': 'gray',
    'fear': 'purple',
    'surprise': 'pink',
    'disgust' : 'green'}

    # Create bar charts to visualize the emotions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    all_data_emotion_counts.plot(kind='bar', ax=axes[0], color=[emotion_colours[emotion] for emotion in all_data_emotion_counts.index])
    axes[0].set_title("Emotion Distribution (All Data)(without neutral)")
    real_news_emotion_counts.plot(kind='bar', ax=axes[1], color=[emotion_colours[emotion] for emotion in real_news_emotion_counts.index])
    axes[1].set_title("Emotion Distribution (Real News)(without neutral)")
    fake_news_emotion_counts.plot(kind='bar', ax=axes[2], color=[emotion_colours[emotion] for emotion in fake_news_emotion_counts.index])
    axes[2].set_title("Emotion Distribution (Fake News)(without neutral)")
    plt.savefig('out/FakevsReal_Emotions_withoutneutral.png')

# Load headlines data
headlines_df = load_data()

# Perform emotion classification for every headline in the data
headlines_df = perform_emotion_classification(headlines_df)

# Aanalyze distribution
analyze_emotion_distribution(headlines_df)

# Same but without neutral
analyze_emotion_distribution_withoutneutral(headlines_df)