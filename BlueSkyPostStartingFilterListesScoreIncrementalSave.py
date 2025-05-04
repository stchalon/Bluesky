from atproto import Client
import csv
import os
from datetime import datetime, timezone

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from textblob import TextBlob
from deep_translator import GoogleTranslator
from langdetect import detect

import json

import getpass

analyzer = SentimentIntensityAnalyzer()

def detect_translate_analyze(text):
    lang = detect(text)
    translated_text = GoogleTranslator(source='auto', target='en').translate(text) if lang != 'en' else text
    blob = TextBlob(translated_text)
    sentiment = blob.sentiment
    return {
        "original_language": lang,
        "translated_text": translated_text if lang != 'en' else None,
        "polarity": sentiment.polarity,
        "subjectivity": sentiment.subjectivity
    }

def fetch_and_save_posts():
    # Prompt for username and password
    print("Please enter your Bluesky credentials.")
    username = input("Bluesky Username: ")
    password = getpass.getpass("Bluesky Password: ")

    client = Client()
    client.login(username, password)

    users = [
        'benzinga.bsky.social',
        'financialtimes.com',
        'cnbc.com',
        'decrypt.co',
        'economist.com',
        'afpfr.bsky.social'
        'reuters.bsky.social',
        'boursier.com',
        'investir.bsky.social',
        'lemonde.fr',
        'lefigaro.fr',
        'latribune.fr',
        'lesechos.bsky.social'
    ]

    keywords = ['Apple', 'Google', 'Microsoft',
        'Nokia', 'Nvidia', 'Rheinmetall', 'Tesla'
        'Tesla',
        'TSLA']
    since_datetime = datetime(2024, 9, 1, tzinfo=timezone.utc)

    csv_filename = f"c:/ml/bluesky_posts_{datetime.now().strftime('%Y%m%d')}.csv"
    file_exists = os.path.isfile(csv_filename)

    # Open file in append mode
    with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write header if file is new
        if not file_exists:
            writer.writerow([
                'Username', 'Post Text', 'Translated Text', 'URL', 'Timestamp',
                'Labels', 'Tags', 'Sentiment Score', 'Neutral', 'Positive', 'Negative'
            ])

        for user in users:
            print(f"\nFetching posts from @{user}...")

            try:
                profile = client.get_profile(user)
            except Exception as e:
                print(f"⚠️ Skipping @{user} — profile not found or inaccessible. Reason: {e}")
                continue
            
            cursor = None
            more_posts = True

            while more_posts:
                feed = client.get_author_feed(profile.did, cursor=cursor)
                if not feed.feed:
                    break

                for item in feed.feed:
                    if not hasattr(item, 'post') or not hasattr(item.post, 'record'):
                        continue

                    created_at_str = item.post.record.created_at
                    post_time = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))

                    if post_time < since_datetime:
                        more_posts = False
                        break

                    if not any(keyword.lower() in item.post.record.text.lower() for keyword in keywords):
                        continue

                    post_url = f"https://bsky.app/profile/{user}/post/{item.post.uri.split('/')[-1]}"
                    print(f"🗨️ {item.post.record.text}")
                    print(f"🔗 {post_url}")
                    print("---")

                    result = detect_translate_analyze(item.post.record.text)
                    translated_text = result['translated_text'] if result['original_language'] != 'en' else item.post.record.text
                    scores = analyzer.polarity_scores(translated_text)

                    writer.writerow([
                        user,
                        item.post.record.text,
                        translated_text,
                        post_url,
                        post_time.strftime('%Y-%m-%d %H:%M:%S'),
                        getattr(item.post.record, 'labels', None),
                        getattr(item.post.record, 'tags', None),
                        scores['compound'],
                        scores['neu'],
                        scores['pos'],
                        scores['neg']
                    ])

                    # Save to JSONL
                    json_post = {
                        'username': user,
                        'text': item.post.record.text,
                        'translated_text': translated_text,
                        'url': post_url,
                        'timestamp': post_time.isoformat(),
                        'labels': getattr(item.post.record, 'labels', None),
                        'tags': getattr(item.post.record, 'tags', None),
                        'original_language': result['original_language'],
                        'polarity': result['polarity'],
                        'subjectivity': result['subjectivity'],
                        'vader_scores': scores
                    }

                    json_filename = f"c:/ml/bluesky_posts_{datetime.now().strftime('%Y%m%d')}.jsonl"
                    with open(json_filename, 'a', encoding='utf-8') as jf:
                        jf.write(json.dumps(json_post, ensure_ascii=False) + '\n')

                cursor = feed.cursor
                if not cursor:
                    break

    print(f"\n✅ Posts from all users saved to {csv_filename}")

fetch_and_save_posts()
