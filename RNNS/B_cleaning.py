import re

def clean_lyrics(raw_text):
    # Remove [Verse 1], [Chorus], etc.
    text = re.sub(r"\[.*?\]", "", raw_text)

    # Remove contributor counts or metadata lines (lines with numbers and translations)
    text = re.sub(r"^\d+ Contributors.*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Translations.*", "", text, flags=re.MULTILINE)
    
    # Remove 'Read More' and description summaries
    text = re.sub(r"Read More.*", "", text)
    
    # Remove extra newlines and whitespace
    text = re.sub(r"\n{2,}", "\n", text)  # Remove multiple newlines
    text = re.sub(r" +", " ", text)       # Remove extra spaces

    # Remove leading/trailing whitespace
    return text.strip()

# Example usage
with open(r"D:\ML\learn_and_practice\NLP\RNNS\charlie_puth.txt", "r", encoding='utf-8') as f:
    raw_data = f.read()

cleaned_lyrics = clean_lyrics(raw_data)

with open("charlie_cleaned_lyrics.txt", "w", encoding='utf-8') as f:
    f.write(cleaned_lyrics)
