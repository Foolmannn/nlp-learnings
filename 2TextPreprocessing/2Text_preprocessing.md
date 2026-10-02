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

## 6. Stop-word removal

Stop words are frequent words that some NLP applications treat as less informative.

Examples include:

```
the, is, am, are, of, in, to, and, a
```

Original tokens:

```
["this", "is", "a", "very", "useful", "book"]
```

After removing selected stop words:

```
["very", "useful", "book"]
```

### Python implementation using NLTK

```
import nltkfrom nltk.corpus import stopwordsfrom nltk.tokenize import word_tokenizenltk.download("stopwords")nltk.download("punkt_tab")text = "This is a very useful book about NLP."tokens = word_tokenize(text)stop_words = set(stopwords.words("english"))filtered_tokens = [    word for word in tokens    if word.lower() not in stop_words]print(filtered_tokens)
```

### Why can stop-word removal be dangerous?

Consider these sentences:

```
I like this movie.
I do not like this movie.
```

If `not` is removed, the second sentence loses its negation and may be interpreted incorrectly.

Therefore:

- Preserve negation for sentiment analysis.
- Avoid indiscriminate stop-word removal for question answering.
- Evaluate whether stop-word removal helps traditional classification.
- Generally avoid manual stop-word removal when using pretrained transformers.

## 7. Stemming

Stemming reduces a word to an approximate stem, often by removing suffixes according to predefined rules.

| Word       | Possible stem |
| ---------- | ------------- |
| playing    | play          |
| played     | play          |
| studies    | studi         |
| connected  | connect       |
| relational | relat         |

A stem does not necessarily need to be a valid dictionary word.

### Porter Stemmer

```
from nltk.stem import PorterStemmerstemmer = PorterStemmer()words = [    "playing",    "played",    "plays",    "studies",    "connected"]for word in words:    print(word, "->", stemmer.stem(word))
```

The Porter stemmer applies a sequence of rules to transform word forms.

### Advantages

- Computationally inexpensive.
- Reduces vocabulary size.
- Can group related word forms.
- May be useful for search and traditional text classification.

### Disadvantages

- May generate invalid words.
- May merge words that differ in meaning.
- Does not fully understand grammar or context.

## 8. Lemmatization

Lemmatization converts a word to its dictionary base form, called its lemma.

| Word     | Lemma                           |
| -------- | ------------------------------- |
| running  | run                             |
| ran      | run                             |
| children | child                           |
| mice     | mouse                           |
| studies  | study                           |
| was      | be                              |
| better   | good, when used as an adjective |

Unlike stemming, lemmatization attempts to produce a valid lexical form.

### Python implementation using spaCy

Install spaCy and its English language model:

```
pip install spacy
python -m spacy download en_core_web_sm
```

Code:

```
import spacynlp = spacy.load("en_core_web_sm")text = "The children were running and playing games."doc = nlp(text)for token in doc:    print(token.text, "->", token.lemma_)
```

Typical output:

```
The -> the
children -> child
were -> be
running -> run
and -> and
playing -> play
games -> game
```

Exact output depends on the installed model.

### Stemming versus lemmatization

| Feature                | Stemming                   | Lemmatization                            |
| ---------------------- | -------------------------- | ---------------------------------------- |
| Method                 | Rule-based transformations | Linguistic analysis                      |
| Output                 | Approximate stem           | Dictionary lemma                         |
| Valid word guaranteed? | No                         | Aims for valid lexical form              |
| Grammatical context    | Usually unnecessary        | Often helpful                            |
| Speed                  | Generally faster           | Generally more computationally expensive |
| Example                | studies → studi            | studies → study                          |

Use stemming when lightweight normalization is sufficient. Consider lemmatization when meaningful word forms matter. Neither method is required for every NLP task.

## 9. Handling contractions and spelling variations

Contractions are shortened word forms:

| Contraction | Expanded form |
| ----------- | ------------- |
| don't       | do not        |
| isn't       | is not        |
| I'm         | I am          |
| they're     | they are      |
| we've       | we have       |
| can't       | cannot        |

Example:

```
text = "I don't think it's a bad idea."text = text.replace("don't", "do not")text = text.replace("it's", "it is")print(text)
```

Output:

```
I do not think it is a bad idea.
```

This is a simple example, not a complete contraction-expansion system. Some contractions are ambiguous, and replacing them without considering context can introduce errors.

Informal text can also contain repeated characters:

```
soooo happy
goooood
```

A simple rule for reducing repeated characters:

```
import retext = "This movie is sooo goooood!"normalized = re.sub(r"(.)\1{2,}", r"\1\1", text)print(normalized)
```

Output:

```
This movie is soo good!
```

Spell correction and repeated-character normalization can help with some social media data, but they may damage names, slang, technical words, and multilingual text.

## 10. Handling emojis, hashtags, and social media text

Consider:

```
I love this movie 😍
This is terrible 😡
#MachineLearning
```

Emojis may communicate sentiment, so removing them may reduce useful information.

Hashtags can sometimes be segmented into words:

```
#NaturalLanguageProcessing
```

Possible normalized form:

```
Natural Language Processing
```

For social media preprocessing, you might replace usernames and URLs with placeholders:

```
import redef clean_social_text(text):    text = re.sub(        r"https?://\S+|www\.\S+",        " URL ",        text    )    text = re.sub(        r"@\w+",        " USER_MENTION ",        text    )    text = re.sub(r"\s+", " ", text).strip()    return texttext = (    "Amazing tutorial! Follow @someone "    "at https://example.com 😊")print(clean_social_text(text))
```

Output:

```
Amazing tutorial! Follow USER_MENTION at URL 😊
```

Notice that the emoji and punctuation are preserved.

## 11. Complete traditional preprocessing pipeline in Python

The following implementation combines several techniques into one reusable function.

It is designed as an illustrative English text-classification pipeline, not as a universal solution.

```
import reimport unicodedataimport spacyfrom bs4 import BeautifulSoupnlp = spacy.load("en_core_web_sm")# Preserve negation wordsSTOP_WORDS = nlp.Defaults.stop_words - {    "no", "not", "never"}def preprocess_text(text):    # 1. Normalize Unicode    text = unicodedata.normalize("NFC", text)    # 2. Extract text from HTML    text = BeautifulSoup(        text, "html.parser"    ).get_text(" ")    # 3. Replace URLs    text = re.sub(        r"https?://\S+|www\.\S+",        " URL ",        text    )    # 4. Replace email addresses    text = re.sub(        r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",        " EMAIL ",        text    )    # 5. Normalize case    text = text.lower()    # 6. Normalize whitespace    text = re.sub(r"\s+", " ", text).strip()    # 7. Tokenize and lemmatize    doc = nlp(text)    tokens = []    for token in doc:        # Remove punctuation and whitespace        if token.is_punct or token.is_space:            continue        # Remove selected stop words        if token.text in STOP_WORDS:            continue        # Append lemma        tokens.append(token.lemma_)    return tokenstext = """<p>I am NOT enjoying this movie!!!</p>Visit https://example.com for more information."""print(preprocess_text(text))
```

Install the required packages:

```
pip install spacy beautifulsoup4
python -m spacy download en_core_web_sm
```

### Understanding the pipeline

1. Unicode normalization standardizes certain character representations.
2. HTML extraction removes markup while retaining readable content.
3. URLs and emails are replaced with placeholders.
4. Lowercasing standardizes capitalization.
5. Whitespace normalization removes repeated spacing.
6. spaCy performs tokenization and linguistic processing.
7. Selected stop words are removed, while negation is preserved.
8. Lemmatization converts word forms to base forms.

One limitation is that the example removes punctuation, including repeated exclamation marks. That may not be desirable for sentiment analysis. You should compare this pipeline with a less aggressive version that preserves punctuation and emojis.

## 12. Converting text into numerical features

After preprocessing, traditional machine learning models need a numerical representation.

Three important approaches are:

[14. Natural Language Processing — Machine Learning for Socio-Economic and Georeferenced Data](bow.jpg)

Bag of Words (BoW)

Represents documents using word counts. It is simple and useful for traditional text classification.

[Introduction to Term Frequency — Inverse Document Frequency(TF-IDF) in Natural Language Processing (NLP) | by Dinesh Chandra Kumawat | Medium](tf_idf.jpg)

TF-IDF

Weights words according to their frequency in a document and how common they are across the document collection.

[Embeddings - Comprehensive Guide](tokenization.jpg)

Token IDs and embeddings

Neural NLP models convert tokenizer output into numerical IDs and learned vector representations.

### TF-IDF example

```
from sklearn.feature_extraction.text import TfidfVectorizerdocuments = [    "NLP is interesting",    "Machine learning is interesting",    "I am learning NLP"]vectorizer = TfidfVectorizer()X = vectorizer.fit_transform(documents)print(vectorizer.get_feature_names_out())print(X.shape)print(X.toarray())
```

The resulting matrix can be passed to a traditional machine learning model, such as logistic regression or Naive Bayes.

For a real project, split your dataset first, fit the vectorizer only on the training data, and transform the validation and test data with the fitted vectorizer.

## 13. Traditional NLP versus transformer preprocessing

| Operation              | Traditional ML with TF-IDF  | Transformer models              |
| ---------------------- | --------------------------- | ------------------------------- |
| Lowercasing            | Often useful                | Depends on the model            |
| Stop-word removal      | Sometimes useful            | Usually avoid                   |
| Stemming               | Sometimes useful            | Usually avoid                   |
| Lemmatization          | Sometimes useful            | Usually avoid                   |
| Word tokenization      | Common                      | Use the model's tokenizer       |
| Subword tokenization   | Not needed for basic TF-IDF | Common                          |
| Padding and truncation | Not needed for basic TF-IDF | Common for neural inputs        |
| Attention masks        | Not applicable              | Used by many transformer models |

For BERT, for example, use the tokenizer associated with the pretrained checkpoint. It may generate token IDs, special tokens, and attention masks. Do not assume that manually removing stop words or stemming words will improve the model.

## 14. Common mistakes to avoid

- Removing `not`, `never`, and other meaningful words.
- Removing all punctuation without considering the task.
- Assuming stemming and lemmatization are interchangeable.
- Applying English-specific rules to Nepali or other languages.
- Applying transformations that remove information needed by the model.
- Fitting vocabulary or feature statistics on the full dataset before splitting.
- Assuming more preprocessing always improves accuracy.
- Applying traditional preprocessing blindly to pretrained transformers.

## 15. Quick revision table

| Technique             | Purpose                                     | Example                            |
| --------------------- | ------------------------------------------- | ---------------------------------- |
| HTML removal          | Extract readable text                       | `<b>NLP</b>` → `NLP`               |
| URL handling          | Remove or normalize links                   | URL → `URL`                        |
| Lowercasing           | Standardize case                            | `NLP` → `nlp`                      |
| Tokenization          | Split text into units                       | `learn NLP` → `["learn", "NLP"]`   |
| Stop-word removal     | Remove selected frequent words              | Remove `the` where appropriate     |
| Stemming              | Produce approximate stems                   | `studies` → `studi`                |
| Lemmatization         | Find dictionary base forms                  | `was` → `be`                       |
| Unicode normalization | Standardize character representations       | Normalize equivalent Unicode forms |
| TF-IDF                | Convert text to weighted numerical features | Documents → feature matrix         |
| Subword tokenization  | Split words into model vocabulary pieces    | Word → model-specific subwords     |

# Chat Words Conversion in NLP

Chat words conversion is a text preprocessing technique that converts informal words, abbreviations, slang, and short forms used in chats or social media into their standard forms.

People frequently use shortened words when texting, such as `u`, `ur`, `btw`, `idk`, and `gonna`. These forms can create vocabulary inconsistencies in traditional NLP systems.

For example:

| Chat word | Standard form                       |
| --------- | ----------------------------------- |
| `u`       | you                                 |
| `ur`      | your / you're, depending on context |
| `r`       | are                                 |
| `u r`     | you are                             |
| `btw`     | by the way                          |
| `idk`     | I don't know                        |
| `imo`     | in my opinion                       |
| `omg`     | oh my God                           |
| `thx`     | thanks                              |
| `pls`     | please                              |
| `b4`      | before                              |
| `gr8`     | great                               |
| `gonna`   | going to                            |
| `wanna`   | want to                             |
| `kinda`   | kind of                             |
| `asap`    | as soon as possible                 |
| `brb`     | be right back                       |
| `ttyl`    | talk to you later                   |
| `lol`     | laughing out loud                   |
| `smh`     | shaking my head                     |

The exact meaning of some abbreviations depends on context. For example, `ur` could mean your or you're, so replacing it blindly can introduce errors.

## 1. Why is chat word conversion important?

Consider these sentences:

```
Original:  "I luv NLP. U r gr8!"
Converted: "I love NLP. You are great!"
```

Conversion can help traditional NLP systems recognize that different spellings express similar meanings.

It can be useful for:

- Sentiment analysis of social media posts.
- Spam detection.
- Chatbot development.
- Text classification.
- Analysis of customer reviews.
- Processing informal messages.

However, conversion is optional. Modern transformer models may already understand many common abbreviations, and preserving the original text can sometimes produce better results.

## 2. Implementing chat word conversion in Python

A simple and practical approach is to create a dictionary mapping informal expressions to standard forms.

```
import rechat_words = {    "u": "you",    "r": "are",    "ur": "your",    "btw": "by the way",    "idk": "i do not know",    "imo": "in my opinion",    "omg": "oh my god",    "thx": "thanks",    "pls": "please",    "plz": "please",    "b4": "before",    "gr8": "great",    "asap": "as soon as possible",    "brb": "be right back",    "ttyl": "talk to you later",    "gonna": "going to",    "wanna": "want to",    "kinda": "kind of"}def convert_chat_words(text):    words = text.split()    converted_words = []    for word in words:        # Separate surrounding punctuation        match = re.fullmatch(            r"([^\w]*)([\w']+)([^\w]*)",            word        )        if match:            prefix, core, suffix = match.groups()            replacement = chat_words.get(core.lower(), core)            converted_words.append(                prefix + replacement + suffix            )        else:            converted_words.append(word)    return " ".join(converted_words)text = "BTW, u r gr8! Pls help me ASAP."print(convert_chat_words(text))
```

Output:

```
by the way, you are great! please help me as soon as possible.
```

This example illustrates dictionary-based conversion. For production use, punctuation, capitalization, contractions, and ambiguous abbreviations should be handled more carefully.

## 3. Handling context-dependent chat words

Some abbreviations have multiple meanings.

| Chat word | Possible meanings                                     |
| --------- | ----------------------------------------------------- |
| `ur`      | your / you're                                         |
| `rn`      | right now / registered nurse                          |
| `jk`      | just kidding / initials                               |
| `bc`      | because / British Columbia                            |
| `lol`     | laughing out loud / a general conversational reaction |

A simple dictionary cannot always choose the correct expansion. For example:

- `ur phone` → `your phone`
- `ur coming` → `you're coming`

For accurate conversion, use context-aware rules or a language model rather than relying exclusively on direct replacements.

## 4. A better preprocessing order

For a traditional NLP pipeline, you can generally follow this sequence:

Raw chat text

"OMG, u r gr8!"

Chat word conversion

"oh my god, you are great!"

Normalization and cleaning

Standardize case and whitespace

Tokenization

["oh", "my", "god", "you", "are", "great"]

Feature extraction or model input

The exact order can vary. For example, tokenization may be needed before applying a more sophisticated chat normalization dictionary.

## 5. Practice: convert these chat messages

Write the standard form of each message.

1\. \`idk what u mean\`

2\. \`brb, ttyl\`

3\. \`thx, pls reply asap\`

Show answers

Key takeaway: Chat word conversion uses a dictionary, rules, or contextual language models to transform informal expressions into standard text. It is particularly useful for traditional NLP preprocessing, but ambiguous expressions should not be expanded without considering context.