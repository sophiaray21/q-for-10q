import re
import numpy as np

'''
Sentiment feature pipeline (ticket #3). Turns the MD&A section of a downloaded 10-Q
into one scalar in roughly [-1, 1] that goes into the state vector (ticket #4).

How it works:
1. extract_mda() pulls the text of "Item 2. Management's Discussion and Analysis"
   out of the filing HTML. MD&A is where management actually editorializes about the
   quarter, which is why we score this section and not the whole document (the rest
   is mostly boilerplate and tables).
2. mda_sentiment() embeds each sentence with a sentence-transformers model and compares
   it against two sets of anchor sentences - one written in clearly positive financial
   language, one clearly negative. Each sentence scores
   (cosine sim to positive anchors) - (cosine sim to negative anchors), and the
   feature is the average over all sentences.

Why anchors instead of a sentiment classifier: sentence-transformers gives us
embeddings, not labels. Comparing against fixed anchor sentences turns similarity
into a sentiment axis with no training data needed, and the anchors are written in
filing-speak ("revenue increased...", "we recorded an impairment...") so the axis
matches the domain. The raw differences are small (cosine sims live close together),
so we rescale by ~10x and tanh-squash to spread useful values across [-1, 1].

The model (all-MiniLM-L6-v2, ~80MB) downloads from HuggingFace on first use and is
cached locally after that. Loading is lazy so importing this module stays cheap.
'''

POSITIVE_ANCHORS = [
    "Revenue increased significantly compared to the prior year period.",
    "We reported record net income and strong margin expansion this quarter.",
    "Demand for our products remains strong and we are gaining market share.",
    "Cash flow from operations improved and our balance sheet remains strong.",
    "We raised our full year guidance based on better than expected results.",
]

NEGATIVE_ANCHORS = [
    "Revenue declined significantly compared to the prior year period.",
    "We reported a net loss and our margins contracted this quarter.",
    "Demand for our products weakened and we are losing market share.",
    "We recorded an impairment charge and our liquidity position deteriorated.",
    "We lowered our full year guidance due to worse than expected results.",
]

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def extract_mda(filepath):
    '''
    Returns the MD&A section text from a 10-Q HTML file (str), or None if not found.

    Gotcha this handles: "Item 2. Management's Discussion..." appears twice in most
    filings - once in the table of contents near the top and once at the real section.
    We take the LAST occurrence of the Item 2 heading, then cut at the next
    "Item 3" / "Item 4" heading after it.
    '''
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    import warnings
    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
    # modern SEC filings are XBRL (XML-flavored HTML); the HTML parser handles them
    # fine for text extraction, so silence bs4's complaint about it
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(" ")
    text = re.sub(r"\s+", " ", text)

    item2 = re.compile(
        r"item\s*2[\.\:\s]*management[\u2019\u2018']?s?\s+discussion\s+and\s+analysis",
        re.IGNORECASE)
    item34 = re.compile(
        r"item\s*3[\.\:\s]*quantitative\s+and\s+qualitative|item\s*4[\.\:\s]*controls\s+and\s+procedures",
        re.IGNORECASE)

    starts = [m.start() for m in item2.finditer(text)]
    if not starts:
        return None
    start = starts[-1]

    end_match = item34.search(text, start + 1)
    end = end_match.start() if end_match else len(text)

    mda = text[start:end].strip()
    # if the "last" match was actually the table of contents (tiny section), fall back
    # to the first match instead
    if len(mda) < 500 and len(starts) > 1:
        start = starts[0]
        end_match = item34.search(text, start + 1)
        end = end_match.start() if end_match else len(text)
        mda = text[start:end].strip()
    return mda if mda else None


def split_sentences(text, min_words=6, max_sentences=300):
    '''Crude sentence splitter, good enough for filing prose. Drops fragments shorter
    than min_words (mostly table debris and headings) and caps the count so one giant
    filing can't make embedding take forever.'''
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    sentences = [p.strip() for p in parts if len(p.split()) >= min_words]
    return sentences[:max_sentences]


def mda_sentiment(text):
    '''Scalar sentiment of a block of MD&A text, tanh-squashed into (-1, 1).
    Positive = the language reads like good news, negative = bad news.'''
    sentences = split_sentences(text)
    if not sentences:
        return 0.0
    model = _get_model()
    # normalized embeddings -> dot product below IS cosine similarity
    emb = model.encode(sentences, normalize_embeddings=True)
    pos = model.encode(POSITIVE_ANCHORS, normalize_embeddings=True).mean(axis=0)
    neg = model.encode(NEGATIVE_ANCHORS, normalize_embeddings=True).mean(axis=0)
    per_sentence = emb @ pos - emb @ neg
    return float(np.tanh(10.0 * per_sentence.mean()))


def sentiment_from_file(filepath):
    '''filepath of a downloaded 10-Q .htm -> (sentiment scalar, n chars of MD&A found).
    Returns (0.0, 0) if no MD&A section could be located.'''
    mda = extract_mda(filepath)
    if mda is None:
        return 0.0, 0
    return mda_sentiment(mda), len(mda)


if __name__ == "__main__":
    import sys

    # sanity check the anchor axis on text where the right answer is obvious
    good = ("Revenue grew 15 percent driven by strong customer demand. "
            "Gross margin expanded and we generated record operating cash flow. "
            "We are raising our outlook for the remainder of the year.")
    bad = ("Net sales decreased 20 percent due to weak demand. "
           "We recorded significant impairment charges and our operating loss widened. "
           "We expect continued pressure on margins next quarter.")
    print(f"obviously positive text -> {mda_sentiment(good):+.3f} (want > 0)")
    print(f"obviously negative text -> {mda_sentiment(bad):+.3f} (want < 0)")

    # score real filings if any paths were passed in
    for path in sys.argv[1:]:
        score, n_chars = sentiment_from_file(path)
        print(f"{path}: sentiment {score:+.3f} (MD&A section: {n_chars} chars)")
