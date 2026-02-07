import pandas as pd
import random
import streamlit as st
import urllib.parse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    from langdetect import detect, DetectorFactory
except ImportError:
    st.error("The language library is still installing. Please refresh in 1 minute!")

DetectorFactory.seed = 0

st.set_page_config(page_icon="🌷", page_title="Playlist Generator")

st.markdown("""
    <style>
    [data-testid="stHeader"]
    {
        display: none;
    }
    
    .stApp
    {
        background-color: #FFF5F7;
        font-family: 'Georgia', serif;
    }
    
    h1, h2, h3, p, span, div, label, input, textarea, select
    {
        font-family: 'Georgia', serif !important;
        color: #7A5C5C;
    }
    
    h1, h2, h3
    {
        color: #D4A5A5;
        text-align: center;
    }
    
    .stButton>button
    {
        background-color: #FFC0CB;
        color: white;
        border-radius: 20px;
        border: 2px solid #F8ADBA;
        padding: 10px 24px;
        font-family: 'Georgia', serif !important;
        transition: 0.3s;
    }
    
    .stButton>button:hover        
    {
        background-color: #F8ADBA;
        border: 2px solid #FFC0CB;
        color: white;
    }

    .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>select
    {
        background-color: #FFFFFF;
        border: 1px solid #FFD1DC;
        border-radius: 15px;
        color: #7A5C5C;
        font-family: 'Georgia', serif !important;
    }
    
    code, .stCodeBlock, [data-testid="stCodeBlock"]
    {
        background-color: #FFF0F3 !important;
        color: #7A5C5C !important;
        border: 2px dashed #FFD1DC !important;
        border-radius: 15px !important;
        font-family: 'Georgia', serif !important;
    }

    .stCodeBlock div
    {
        background-color: transparent !important;
    }
            
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dataset.csv")
        df = df.drop_duplicates(subset=['track_name', 'artists'], keep='first')
        df['artists'] = df['artists'].str.replace(';', ', ')
        df['metadata'] = (df['artists'].fillna('') + " " + 
                         df['track_genre'].fillna(''))
        return df
    except FileNotFoundError:
        st.error("Missing 'dataset.csv'!")
        return None

def get_global_recommendations(user_query, df, lang_code):
    tfidf = TfidfVectorizer(stop_words='english')
    
    df = df.sample(frac=1).reset_index(drop=True)
    
    temp_df = pd.concat([df['metadata'], pd.Series([user_query])], ignore_index=True)
    matrix = tfidf.fit_transform(temp_df)
    sim_scores = cosine_similarity(matrix[-1], matrix[:-1])[0]
    
    selected_indices = []
    artist_counts = {}
    
    potential_indices = sim_scores.argsort()[-5000:][::-1]
    
    for idx in potential_indices:
        if len(selected_indices) >= 30:
            break
            
        song_name = df.iloc[idx]['track_name']
        current_artist = df.iloc[idx]['artists']

        if lang_code != "any":
            try:
                if detect(song_name) != lang_code:
                    continue
            except:
                continue 
        
        count = artist_counts.get(current_artist, 0)
        if count >= 3: 
            continue
    
        selected_indices.append(idx)
        artist_counts[current_artist] = count + 1

    random.shuffle(selected_indices)
    return df.iloc[selected_indices]

st.markdown("<h1>🌷🎧 Playlist Generator</h1>", unsafe_allow_html=True)

languages = {
    "Any Language": "any",
    "English 🇺🇸/🇬🇧": "en",
    "Tamil 🇮🇳/🇱🇰": "ta",
    "Spanish 🇪🇸": "es",
    "French 🇫🇷": "fr",
    "German 🇩🇪": "de",
    "Japanese 🇯🇵": "ja",
    "Korean 🇰🇷": "ko",
    "Portuguese 🇧🇷": "pt",
    "Italian 🇮🇹": "it",
    "Hindi 🇮🇳": "hi",
    "Arabic 🇦🇪": "ar",
    "Turkish 🇹🇷": "tr"
}

lang_label = st.selectbox("Select Target Language", list(languages.keys()))
lang_code = languages[lang_label]

col1, col2 = st.columns(2)
with col1:
    user_artists = st.text_input("Artists you love")
with col2:
    user_genres = st.text_input("Genres to include")

user_mood = st.text_area("Mood or vibe")

if st.button("🩷 Generate playlist 🩷"):
    data = load_data()
    if data is not None:
        with st.spinner("Finding your playlist..."):
            query = f"{user_artists} {user_genres} {user_mood}"
            results = get_global_recommendations(query, data, lang_code)
            
            if len(results) == 0:
                st.error("No matches found. Try again!")
            else:
                tracklist = [f"{row['track_name']} - {row['artists']}" for _, row in results.iterrows()]
                encoded_list = urllib.parse.quote("\n".join(tracklist))
                export_url = f"https://www.spotlistr.com/search/textbox?data={encoded_list}"
                
                st.markdown("<h3>Your Curated Tracks</h3>", unsafe_allow_html=True)
                st.link_button("🌸 Export to Spotify 🌸", export_url, use_container_width=True)
                
                playlist_html = f"""
                <div style="
                    background-color: #FDF2F4; 
                    border: 2px dashed #FFD1DC; 
                    padding: 20px; 
                    border-radius: 15px; 
                    color: #7A5C5C; 
                    font-family: 'Georgia', serif;
                    white-space: pre-wrap;
                ">{'<br>'.join(tracklist)}</div>
                """
                st.markdown(playlist_html, unsafe_allow_html=True)
