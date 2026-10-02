# Text Preprocessing in NLP — Detailed Notes with Python

Text preprocessing is a fundamental step in Natural Language Processing (NLP). Before applying machine learning or deep learning algorithms to text, we often need to clean, normalize, and transform the raw text into a format suitable for the model.

Let's study the concepts step by step, from basic text cleaning to modern transformer tokenization, with examples and Python code.

## 1. What is text preprocessing?

Text preprocessing is the process of converting raw text into a cleaner, more consistent, and usable form for NLP tasks.

Suppose we have this raw text:

```
"OMG!!! I am LOVING this movie 😍😍. It's sooo amazing!!!
Visit https://example.com for more info."
```

A possible cleaned representation for a traditional text-classification task could be:

```
omg love movie amazing
```

However, this is not necessarily the best result for every task. Removing the emoji, URL, or words such as `not` may remove useful information.

The objective is not to remove as much text as possible. It is to prepare text while preserving the information needed by the model.

### Why is text preprocessing necessary?

- Consistency: Handle differences in capitalization and whitespace.
- Noise reduction: Remove unwanted HTML markup and formatting.
- Vocabulary management: Normalize different forms of words where appropriate.
- Feature extraction: Prepare text for techniques such as Bag of Words and TF-IDF.
- Model compatibility: Prepare text for tokenizers and neural network models.
- Data quality: Handle malformed or inconsistent text.

## 2. The complete text preprocessing pipeline

### 1

Raw text

Text from websites, documents, reviews, etc.

### 2

Text cleaning

HTML, unwanted markup, whitespace and noise

### 3

Normalization

Case, Unicode, contractions and spelling when appropriate

### 4

Tokenization

Split text into words, sentences or subwords

### 5

Linguistic processing

Optional stop-word handling, stemming, lemmatization, POS tagging

### 6

Numerical representation

BoW, TF-IDF, token IDs or embeddings

### 7

NLP model

Classification, sentiment analysis, retrieval, translation

This pipeline is a general guide. You don't need to apply every step to every project. Transformer models, for example, generally use their own pretrained tokenizers and often do not require stemming or stop-word removal.

## 3. Text cleaning

Text cleaning removes or repairs unwanted elements in raw text.

### 3.1 Removing HTML tags

Raw text:

```
<p>I love <b>Natural Language Processing</b>.</p>
```

Expected result:

```
I love Natural Language Processing.
```

Python implementation:

```
from bs4 import BeautifulSouphtml = "<p>I love <b>Natural Language Processing</b>.</p>"soup = BeautifulSoup(html, "html.parser")clean_text = soup.get_text(" ", strip=True)print(clean_text)
```

Output:

```
I love Natural Language Processing.
```

Install the library using:

```
pip install beautifulsoup4
```

Beautiful Soup is useful for extracting readable text from HTML. It is generally more reliable than trying to remove all HTML tags using a single regular expression.

### 3.2 Removing URLs

URLs can be irrelevant in some classification tasks, but useful in spam detection or website analysis.

```
import retext = "Learn NLP at https://example.com/tutorial"clean_text = re.sub(    r"https?://\S+|www\.\S+",    " URL ",    text).strip()print(clean_text)
```

Output:

```
Learn NLP at URL
```

Replacing a URL with a placeholder preserves the fact that a URL was present.

### 3.3 Removing email addresses

```
import retext = "Contact us at support@example.com"clean_text = re.sub(    r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",    " EMAIL ",    text)print(clean_text.strip())
```

Output:

```
Contact us at  EMAIL
```

This is useful when the actual email address is irrelevant but its presence may be meaningful.

### 3.4 Removing punctuation

```
import stringtext = "Hello!!! NLP, Python & AI."translator = str.maketrans("", "", string.punctuation)clean_text = text.translate(translator)print(clean_text)
```

Output:

```
Hello NLP Python  AI
```

Be careful with this operation. Removing all punctuation can damage:

- `don't` → `dont`
- `C++` → `C`
- `3.14` → `314`
- `#NLP` → `NLP`

For sentiment analysis, punctuation such as `!!!` may convey emotional intensity. For code processing, punctuation may be essential.

### 3.5 Removing extra whitespace

```
import retext = "Natural    Language\n\nProcessing\tis useful."clean_text = re.sub(r"\s+", " ", text).strip()print(clean_text)
```

Output:

```
Natural Language Processing is useful.
```

This is often a useful general-purpose cleaning step, although line breaks should be preserved when document structure matters.

## 4. Lowercasing and text normalization

Normalization transforms different representations into a consistent form.

For example:

```
Natural Language Processing
NATURAL LANGUAGE PROCESSING
natural language processing
```

After lowercasing:

```
natural language processing
```

Python:

```
text = "Natural Language Processing"print(text.lower())
```

Output:

```
natural language processing
```

### Why do we lowercase text?

Traditional machine learning models may otherwise treat `Movie`, `movie`, and `MOVIE` as separate features. Lowercasing can reduce vocabulary size and improve consistency.

However, capitalization can also carry meaning:

- `US` and `us`
- `Apple` as a company and `apple` as a fruit
- Names and abbreviations
- Case-sensitive technical identifiers

For transformer models, preserve the casing expected by the model's tokenizer. Some models are uncased, while others are case-sensitive.

### Unicode normalization

Unicode normalization helps standardize text representations that may look identical but have different underlying character sequences.

```
import unicodedatatext = "café"normalized = unicodedata.normalize("NFC", text)print(normalized)
```

This is particularly relevant to multilingual text. Compatibility normalization such as NFKC can be useful for matching text, but it can also erase distinctions, so choose it carefully.

## 5. Tokenization

Tokenization is the process of splitting text into smaller units called tokens.

Consider:

```
I am learning NLP with Python.
```

Word-level tokens might be:

```
["I", "am", "learning", "NLP", "with", "Python", "."]
```

Tokens may be words, sentences, subwords, or characters.

### 5.1 Word tokenization using NLTK

Install NLTK:

```
pip install nltk
```

Example:

```
import nltkfrom nltk.tokenize import word_tokenizenltk.download("punkt_tab")text = "I am learning NLP with Python."tokens = word_tokenize(text)print(tokens)
```

Typical output:

```
['I', 'am', 'learning', 'NLP', 'with', 'Python', '.']
```

### 5.2 Sentence tokenization

```
from nltk.tokenize import sent_tokenizetext = (    "NLP is fascinating. "    "Preprocessing is important. "    "Let's learn it!")sentences = sent_tokenize(text)print(sentences)
```

Output:

```
[    'NLP is fascinating.',    'Preprocessing is important.',    "Let's learn it!"]
```

Sentence tokenization is useful when processing documents sentence by sentence.

### 5.3 Subword tokenization

Modern transformer models frequently use subword tokenization methods such as:

- Byte Pair Encoding (BPE)
- WordPiece
- Unigram tokenization
- SentencePiece-based tokenization

An uncommon word such as `unhappiness` might be divided into smaller pieces. The exact split depends on the tokenizer's vocabulary.

Subword tokenization helps models handle rare words and new word forms without needing an independent vocabulary entry for every possible word.

Important: When using pretrained transformer models, use the tokenizer associated with the model instead of manually splitting words and assuming the results will match.
