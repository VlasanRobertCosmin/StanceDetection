from googleapiclient.discovery import build
import pandas as pd

# ------------------ 🔑 SETUP ------------------
API_KEY = ''  # Înlocuiește cu cheia ta YouTube Data API
youtube = build('youtube', 'v3', developerKey=API_KEY)

# ------------------ 🎯 DEFINIRE VIDEO ------------------
VIDEO_ID = '96kGq_nFzuM'  # Înlocuiește cu ID-ul unui video relevant despre vaccinuri
comments = []
next_page_token = None

# ------------------ 💬 EXTRAGERE COMENTARII ------------------
while True:
    request = youtube.commentThreads().list(
        part='snippet',
        videoId=VIDEO_ID,
        maxResults=100,
        pageToken=next_page_token,
        textFormat='plainText'
    )
    response = request.execute()

    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        comments.append({
            'author': comment['authorDisplayName'],
            'text': comment['textDisplay'],
            'published_at': comment['publishedAt'],
            'like_count': comment['likeCount']
        })

    next_page_token = response.get('nextPageToken')
    if not next_page_token:
        break

# ------------------ 💾 SALVARE ------------------
df = pd.DataFrame(comments)
df.to_csv('youtube_data_new.csv', index=False, encoding='utf-8')
print("✅ Comentariile au fost salvate în 'youtube_vaccine_comments.csv'")
