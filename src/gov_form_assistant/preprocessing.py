from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


TOKEN_PATTERN = r"(?u)[\w\u0900-\u097f]+"

# Multilingual stop words for English, Hindi, and Marathi
ENGLISH_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him",
    "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't",
    "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
    "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves"
}

HINDI_STOPWORDS = {
    "है", "हैं", "हो", "था", "थे", "थी", "का", "के", "की", "को", "में", "से", "पर", "ने", "और", "या",
    "एवं", "तथा", "यह", "वह", "इस", "उस", "इन", "उन", "जो", "तो", "ही", "भी", "द्वारा", "किया", "किए",
    "करें", "लिए", "आदि", "इत्यादि", "कृपया", "तक", "दे", "दी", "गया", "गई", "गए", "कर", "रहा", "रही",
    "रहे", "हुए", "हुआ", "हुई", "जब", "तब", "कहा", "जाए", "जाता", "जाती", "होता", "होती", "होते"
}

MARATHI_STOPWORDS = {
    "आहे", "आहेत", "आणि", "किंवा", "व", "चे", "च्या", "ची", "चा", "ला", "स", "ने", "नी", "वर", "खाली",
    "मध्ये", "त", "या", "ती", "तो", "ते", "त्या", "जो", "जी", "जे", "ज्या", "साठी", "मुळे", "कडून", "पैकी",
    "केले", "करावे", "करा", "इत्यादि", "कृपया", "होते", "होता", "होती", "नाही", "नाहीत", "एक", "दोन", "सर्व",
    "तसेच", "इतर", "केली", "झाले", "झाली", "असेल", "असावे"
}

MULTILINGUAL_STOPWORDS = sorted(ENGLISH_STOPWORDS | HINDI_STOPWORDS | MARATHI_STOPWORDS)


@dataclass
class NLPFeatures:
    normalized_text: str
    sentences: list[str]
    tokens: list[str]
    bag_of_words: list[tuple[str, int]]
    tfidf_terms: list[tuple[str, float]]
    vocabulary_size: int = 0
    token_count: int = 0
    lexical_diversity: float = 0.0


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\x00", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def split_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?।॥])\s+|\n+", normalize_text(text))
        if sentence.strip()
    ]


def tokenize_words(text: str) -> list[str]:
    return re.findall(TOKEN_PATTERN, normalize_text(text), flags=re.UNICODE)


def _rank_terms(sentences: list[str], top_k: int) -> tuple[list[tuple[str, int]], list[tuple[str, float]]]:
    if not sentences:
        return [], []

    count_vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        token_pattern=TOKEN_PATTERN,
        lowercase=True,
        stop_words=MULTILINGUAL_STOPWORDS,
        min_df=1,
    )
    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        token_pattern=TOKEN_PATTERN,
        lowercase=True,
        stop_words=MULTILINGUAL_STOPWORDS,
        min_df=1,
    )
    try:
        counts = count_vectorizer.fit_transform(sentences)
        count_values = counts.sum(axis=0).A1
        count_terms = count_vectorizer.get_feature_names_out()
        bag_of_words = sorted(
            zip(count_terms, count_values.tolist()), key=lambda item: (-item[1], item[0])
        )[:top_k]
    except ValueError:
        bag_of_words = []

    try:
        weights = tfidf_vectorizer.fit_transform(sentences).sum(axis=0).A1
        tfidf_terms = tfidf_vectorizer.get_feature_names_out()
        ranked_tfidf = sorted(
            zip(tfidf_terms, weights.tolist()), key=lambda item: (-item[1], item[0])
        )[:top_k]
        return bag_of_words, [(term, float(weight)) for term, weight in ranked_tfidf]
    except ValueError:
        return bag_of_words, []


def preprocess_document(text: str, top_k: int = 25) -> NLPFeatures:
    normalized_text = normalize_text(text)
    sentences = split_sentences(normalized_text)
    tokens = tokenize_words(normalized_text)
    unique_vocab = set(w.lower() for w in tokens)
    vocabulary_size = len(unique_vocab)
    token_count = len(tokens)
    lexical_diversity = (vocabulary_size / token_count) if token_count > 0 else 0.0

    bag_of_words, tfidf_terms = _rank_terms(sentences, top_k)
    return NLPFeatures(
        normalized_text=normalized_text,
        sentences=sentences,
        tokens=tokens,
        bag_of_words=bag_of_words,
        tfidf_terms=tfidf_terms,
        vocabulary_size=vocabulary_size,
        token_count=token_count,
        lexical_diversity=lexical_diversity,
    )


def select_relevant_context(question: str, context: str, max_chars: int = 3000) -> str:
    normalized_context = normalize_text(context)
    sentences = split_sentences(normalized_context)
    if not sentences or max_chars <= 0:
        return normalized_context[:max(0, max_chars)]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        token_pattern=TOKEN_PATTERN,
        lowercase=True,
        stop_words=MULTILINGUAL_STOPWORDS,
    )
    try:
        matrix = vectorizer.fit_transform([normalize_text(question), *sentences])
        scores = (matrix[1:] @ matrix[0].T).toarray().ravel()
    except ValueError:
        return normalized_context[:max_chars]

    if not len(scores) or max(scores) <= 0:
        return normalized_context[:max_chars]

    selected: list[int] = []
    for index in sorted(range(len(sentences)), key=lambda item: (-scores[item], item)):
        if scores[index] <= 0:
            break
        for nearby_index in (index - 1, index, index + 1):
            if 0 <= nearby_index < len(sentences) and nearby_index not in selected:
                selected.append(nearby_index)
        if len(selected) >= 18:
            break

    chosen: list[str] = []
    used_chars = 0
    for index in sorted(selected):
        sentence = sentences[index]
        separator_size = 1 if chosen else 0
        available = max_chars - used_chars - separator_size
        if available <= 0:
            break
        if len(sentence) > available:
            if not chosen:
                chosen.append(sentence[:available])
            break
        chosen.append(sentence)
        used_chars += len(sentence) + separator_size
    return "\n".join(chosen) if chosen else normalized_context[:max_chars]