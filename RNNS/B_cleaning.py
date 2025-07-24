import re
import contractions
import string

# --- Step 1: Clean Metadata ---
def clean_lyrics(raw_text):
    text = re.sub(r"\[.*?\]", "", raw_text)  # Remove [Verse 1], [Chorus], etc.
    text = re.sub(r"^\d+ Contributors.*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Translations.*", "", text, flags=re.MULTILINE)
    text = re.sub(r"Read More.*", "", text)
    text = re.sub(r"\n{2,}", "\n", text)  # Collapse multiple newlines
    text = re.sub(r" +", " ", text)       # Collapse extra spaces
    return text.strip()

# --- Step 2: Fix "in'" endings like throwin' → throwing ---
def fix_in_apostrophe(text):
    return re.sub(r"(\b\w+)in'", r"\1ing", text, flags=re.IGNORECASE)

# --- Step 3: Replace slang like 'round → around ---
slang_dict = {
    r"'round": "around",
    r"'bout": "about",
    r"'cause": "because",
    r"'em": "them",
    r"'gonna": "going to",
    r"'wanna": "want to",
    r"'gotta": "got to",
    r"ain't": "is not",
    r"ya": "you",
    r"'ready": "already"
}

def replace_slang(text, slang_dict):
    for pattern, replacement in slang_dict.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def fully_clean_text(text):
    # Remove text in parentheses
    text = re.sub(r'\([^)]*\)', '', text)
    # Remove punctuation
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
    # Collapse multiple spaces
    # text = re.sub(r'\s+', ' ', text)
    return text.strip()
                       # Trim start/end

# --- Step 4: Final pipeline ---
def full_preprocess(text):
    text = clean_lyrics(text)
    text = contractions.fix(text)
    text = fix_in_apostrophe(text)
    text = replace_slang(text, slang_dict)
    text = fully_clean_text(text)
    return text

# --- Run it ---
with open(r"D:\ML\learn_and_practice\NLP\charlie_puth.txt", "r", encoding='utf-8') as f:
    raw_data = f.read()

cleaned_lyrics = full_preprocess(raw_data)

with open("charlie_cleaned_lyrics.txt", "w", encoding='utf-8') as f:
    f.write(cleaned_lyrics)

print("✅ Lyrics cleaned and saved successfully!")
