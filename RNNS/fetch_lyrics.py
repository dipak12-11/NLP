import lyricsgenius
from dotenv import load_dotenv
import os

load_dotenv()
genius_token=os.getenv("GENIUS_ACESS_TOKEN")
genius=lyricsgenius.Genius(genius_token)
artist=genius.search_artist("charlie Puth",max_songs=3,sort="popularity")

with open ("charlie_puth.txt",'w',encoding='utf-8')as f :
     for song in artist.songs:
         f.write(song.lyrics+"\n\n")