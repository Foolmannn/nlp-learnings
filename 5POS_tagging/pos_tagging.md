
# Part-of-Speech (POS) Tagging in NLP — Detailed Guide

## 1. What is POS Tagging?

**Part-of-Speech (POS) tagging** is an Natural Language Processing (NLP) technique used to assign a grammatical category to each word in a sentence based on its meaning, grammatical role, and surrounding context.

These categories include\:

- Noun (NN)
- Verb (VB)
- Adjective (JJ)
- Adverb (RB)
- Pronoun (PRP)
- Determiner (DT)
- Preposition (IN)
- Conjunction (CC)
- Interjection (UH)

For example, consider this sentence\:

> The clever student solves problems quickly.

After POS tagging, it might look like this\:

| Word | POS tag | Meaning |
|--------|--------|--------|
| The | DT | Determiner |
| clever | JJ | Adjective |
| student | NN | Noun |
| solves | VBZ | Verb, third-person singular present |
| problems | NNS | Plural noun |
| quickly | RB | Adverb |
| \. | \. | Punctuation |

The POS tagger identifies the grammatical role of every token in the sentence.

## How POS tagging works

**Input sentence**

The clever student solves problems quickly.

**Tokenization**

The &#124; clever &#124; student &#124; solves &#124; problems &#124; quickly &#124; .

**POS tagger**

Analyzes each word using its form and context

**Tagged output**

`The/DT clever/JJ student/NN solves/VBZ problems/NNS quickly/RB`

POS tagging is also called **grammatical tagging** or **morphosyntactic tagging**\.

## 2. Why is POS tagging important?

Words do not always have the same grammatical role. POS tagging helps an NLP system understand how words function in a sentence.

For example\:

- "I **book** a ticket." — `book` is a verb.
- "I read a **book**\." — `book` is a noun.

The spelling is identical, but the grammatical role changes with context.

POS tagging is useful in\:

1. **Text classification\:** Helps extract grammatical features from documents.
2. **Named Entity Recognition (NER)\:** Provides grammatical clues for detecting names, places, and organizations.
3. **Information extraction\:** Helps identify actions, entities, and relationships.
4. **Machine translation\:** Helps determine how words should be interpreted and translated.
5. **Question answering\:** Helps identify subjects, verbs, and objects.
6. **Text-to-speech\:** Helps distinguish words whose pronunciation depends on their grammatical role.
7. **Sentiment analysis\:** Can help identify adjectives and adverbs, such as "excellent," "terrible," and "extremely."
8. **Lemmatization\:** POS information improves the conversion of words to their dictionary forms.

POS tagging is not a complete understanding of language, but it provides useful grammatical information for downstream NLP tasks.

---

## 3. Different parts of speech and their tags

Let's understand the main grammatical categories and common English POS tags.

### 3.1 Noun (NN)

A noun represents a person, place, thing, concept, or idea.

Examples: student, Kathmandu, computer, happiness.

| Tag | Meaning | Example |
|--------|--------|--------|
| NN | Singular or mass noun | computer |
| NNS | Plural noun | computers |
| NNP | Proper singular noun | Suman |
| NNPS | Proper plural noun | Americans |

Example\:

> Suman bought a laptop.

- `Suman/NNP`
- `laptop/NN`

### 3.2 Verb (VB)

A verb represents an action, occurrence, or state.

Examples: run, eat, think, learn, be.

| Tag | Meaning | Example |
|--------|--------|--------|
| VB | Base form | learn |
| VBD | Past tense | learned |
| VBG | Gerund or present participle | learning |
| VBN | Past participle | learned |
| VBP | Non-third-person singular present | learn |
| VBZ | Third-person singular present | learns |

Example\:

> She is learning NLP.

- `is/VBZ`
- `learning/VBG`

Notice that `is` is tagged `VBZ` in the Penn Treebank tagset because it is a present-tense verb used with a third-person singular subject.

### 3.3 Adjective (JJ)

An adjective describes a noun or pronoun.

Examples: beautiful, intelligent, large, interesting.

| Tag | Meaning | Example |
|--------|--------|--------|
| JJ | Adjective | intelligent |
| JJR | Comparative adjective | smarter |
| JJS | Superlative adjective | smartest |

Example\:

> This is a beautiful house.

`beautiful/JJ` describes `house`\.

### 3.4 Adverb (RB)

An adverb modifies a verb, adjective, another adverb, or sometimes a whole sentence.

Examples: quickly, slowly, very, carefully.

| Tag | Meaning | Example |
|--------|--------|--------|
| RB | Adverb | quickly |
| RBR | Comparative adverb | faster |
| RBS | Superlative adverb | fastest |
| WRB | Wh-adverb | where |

Example\:

> He runs very quickly.

- `very/RB`
- `quickly/RB`

### 3.5 Pronoun (PRP)

A pronoun replaces or refers to a noun or noun phrase.

| Tag | Meaning | Example |
|--------|--------|--------|
| PRP | Personal pronoun | I, you, they |
| PRP\$ | Possessive pronoun/determiner | my, their |
| WP | Wh-pronoun | who, what |
| WP\$ | Possessive wh-pronoun | whose |

Example\:

> They completed their project.

- `They/PRP`
- `their/PRP$`

### 3.6 Determiner (DT)

A determiner introduces or specifies a noun.

Examples: the, a, an, this, some, each.

Example\:

> The student bought a book.

- `The/DT`
- `a/DT`

### 3.7 Preposition or subordinating conjunction (IN)

The Penn Treebank tag `IN` is used for prepositions and subordinating conjunctions.

Examples: in, on, at, because, although.

> The laptop is on the table.

`on/IN` indicates the relationship between the laptop and the table.

### 3.8 Conjunction (CC)

A conjunction connects words, phrases, or clauses.

Examples: and, but, or.

> I study Python and NLP.

`and/CC`

### 3.9 Other useful tags

| Tag | Meaning | Example |
|--------|--------|--------|
| CD | Cardinal number | five |
| MD | Modal verb | can, should |
| TO | The word "to" | to learn |
| UH | Interjection | wow |
| RP | Particle | give up |
| POS | Possessive ending | John's |
| WDT | Wh-determiner | which |
| PDT | Predeterminer | all (in "all the students") |
| EX | Existential "there" | There is a problem |
| FW | Foreign word | Depends on context |
| SYM | Symbol | Mathematical or other symbol |
| \. | Sentence-ending punctuation | \. \! ? |
| , | Comma | , |

**Important\:** These tags are from the Penn Treebank-style English tagset, widely used by tools such as NLTK. Other libraries, including spaCy, may use different tag conventions or provide both coarse-grained POS tags and fine-grained tags.

---

## 4. How POS tagging handles ambiguous words

This is one of the most important concepts in POS tagging.

A word may belong to different grammatical categories depending on its context. This is called **POS ambiguity**\.

Consider the word `watch`\.

Sentence 1

I **watch** movies every day.

(watch) Verb (VBP)

---

Sentence 2

My **watch** is expensive.

(watch) Noun (NN)

---

Sentence 3

Keep a **watch** on the door.

(watch) Noun (NN)

A POS tagger cannot always determine the correct tag from a word in isolation. It uses the surrounding words and learned language patterns.

Other examples include\:

| Word | Sentence | POS |
|--------|--------|--------|
| run | I run daily. | Verb |
| run | I went for a run. | Noun |
| light | Turn on the light. | Noun |
| light | This bag is light. | Adjective |
| light | Light the candle. | Verb |
| book | Read the book. | Noun |
| book | Book a room. | Verb |
| fast | He drives fast. | Adverb |
| fast | It is a fast car. | Adjective |

This is why POS tagging is more than assigning a fixed tag to every word in a dictionary.

## 5. How does a POS tagger work?

A POS tagger generally performs the following operations\:

1. **Receive the text\:** The system receives a sentence or document.
2. **Tokenize the text\:** The sentence is split into tokens, usually words and punctuation.
3. **Extract useful information\:** Depending on the model, it considers word endings, capitalization, neighboring words, word embeddings, and other features.
4. **Predict POS tags\:** It estimates which grammatical tag fits each token.
5. **Resolve ambiguity\:** The model uses context and sentence-level patterns to select a plausible sequence of tags.
6. **Return tagged tokens\:** Each token is paired with its predicted grammatical category.

For example\:

Input\:

> The small cat is sleeping.

Possible intermediate representation\:

```python
[
    "The",
    "small",
    "cat",
    "is",
    "sleeping",
    "."
]
```

Output\:

```python
[
    ("The", "DT"),
    ("small", "JJ"),
    ("cat", "NN"),
    ("is", "VBZ"),
    ("sleeping", "VBG"),
    (".", ".")
]
```

The exact tags may vary slightly by model and tagset.

### POS tagging as a prediction problem

Let a sentence contain tokens\:

$$
X = (x_1, x_2, \ldots, x_n)
$$

The tagger predicts a corresponding sequence\:

$$
Y = (y_1, y_2, \ldots, y_n)
$$

Here\:

- $x_i$ is the $i$\-th token.
- $y_i$ is its POS tag.
- $n$ is the number of tokens.

A statistical sequence tagger aims to find the most probable tag sequence given the sentence\:

$$
\hat{Y} = \arg\max_Y P(Y \mid X)
$$

This equation is a general way to express POS tagging as a sequence-prediction task. Different algorithms calculate or approximate this probability differently.

---

## 6. Different approaches to POS tagging

POS tagging methods have evolved from manually written grammar rules to machine learning and neural networks.

**1\. Rule-based POS tagging**

Uses manually defined linguistic rules and dictionaries.

Example: a word ending in `-ly` may be an adverb.

`quickly` → RB

Limitation: exceptions are common; not every `-ly` word is an adverb.

---

**2\. Statistical POS tagging**

Learns patterns from previously tagged sentences using probability-based models.

Common examples: Hidden Markov Models (HMMs) and Maximum Entropy taggers.

Limitation: performance depends on the training data, model assumptions, and features.

---

**3\. Classical supervised machine learning**

Trains a classifier using labeled examples and features such as the current word, suffixes, capitalization, and neighboring words.

Examples: Decision Trees, Support Vector Machines (SVMs), and Conditional Random Fields (CRFs).

Limitation: feature engineering may be necessary, especially for older models.

---

**4\. Deep learning POS tagging**

Uses neural networks to learn representations of words and their contexts.

Examples: BiLSTM models, BiLSTM-CRF models, and Transformer-based models such as BERT.

Limitation: can require more computation and suitable labeled data; model quality depends on domain and language.

### 6.1 Rule-based tagging

A rule-based system relies on grammar rules, word dictionaries, and contextual conditions.

Illustrative rules\:

```text
If a word follows a determiner
and commonly behaves as a noun:
    consider a noun tag

If a word ends in "-ly":
    consider an adverb tag

If a word follows "to" in an infinitive construction:
    consider a base-form verb tag
```

For example\:

> She quickly opened the door.

The rule system may identify\:

- `quickly` as an adverb because of its ending.
- `opened` as a verb based on its form and sentence context.
- `door` as a noun after the determiner `the`\.

These are only simplified rules. Real systems need exceptions and context-sensitive rules to handle irregular grammar.

### 6.2 Statistical tagging using an HMM

A Hidden Markov Model (HMM) treats the POS tags as hidden states and the observed words as emissions.

It uses two key probabilities\:

**Transition probability**

$$
P(y_i \mid y_{i-1})
$$

This measures how likely one POS tag is to follow another.

**Emission probability**

$$
P(x_i \mid y_i)
$$

This measures how likely a word is to occur with a particular POS tag.

For example, in\:

> The dog barks.

A model may learn that\:

- A determiner is often followed by a noun.
- A noun is often followed by a verb.
- The word `dog` frequently occurs as a noun.

A simplified HMM scores a tag sequence using\:

$$
P(Y,X) \approx
P(y_1)\prod_{i=2}^{n}P(y_i\mid y_{i-1})
\prod_{i=1}^{n}P(x_i\mid y_i)
$$

The best sequence can be found using the **Viterbi algorithm**, which efficiently searches for the highest-scoring path through possible tags.

An HMM is useful for understanding probabilistic sequence labeling, although modern taggers often use richer contextual representations.

### 6.3 Neural POS tagging

A neural tagger represents words as vectors and uses surrounding context to predict tags.

A typical architecture is\:

Input tokens

Token embeddings

Context encoder (BiLSTM or Transformer)

Tag prediction layer

POS tag for each token

For example, a Transformer can use both preceding and following words to determine whether `book` is a noun or a verb.

Unlike a simple suffix rule, a contextual model can learn that the word's role depends on the entire sentence.

---

## 7. Implement POS tagging using Python

Since you're learning NLP practically, let's implement POS tagging using **spaCy and NLTK**, then inspect the output and understand how to use it in a dataset.

### 7.1 Install the libraries

Run these commands in your terminal or notebook\:

```bash
pip install spacy nltk pandas
```

For spaCy, download an English language model\:

```bash
python -m spacy download en_core_web_sm
```

You only need to download the model once per Python environment.

### 7.2 POS tagging with spaCy

spaCy is a practical choice for modern NLP projects because it provides tokenization, POS tagging, lemmatization, dependency parsing, and named entity recognition in one pipeline.

```python
import spacy

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

text = "The clever student solves problems quickly."

# Process the sentence
doc = nlp(text)

# Print each token and its POS information
for token in doc:
    print(
        token.text,
        token.pos_,
        token.tag_,
        token.lemma_
    )
```

Example output\:

```text
The       DET   DT    the
clever    ADJ   JJ    clever
student   NOUN  NN    student
solves    VERB  VBZ   solve
problems  NOUN  NNS   problem
quickly   ADV   RB    quickly
.         PUNCT .     .
```

The precise output can vary with model versions.

Let's understand the code.

- `token.text`\: The original token.
- `token.pos_`\: The coarse-grained POS category, such as `NOUN`, `VERB`, or `ADJ`\.
- `token.tag_`\: The fine-grained tag, such as `NN`, `VBZ`, or `JJR`\.
- `token.lemma_`\: The base or dictionary form of the token.

For example\:

| Token | `pos_` | `tag_` | `lemma_` |
|--------|--------|--------|--------|
| solves | VERB | VBZ | solve |
| problems | NOUN | NNS | problem |
| quickly | ADV | RB | quickly |

**Important distinction\:** POS tagging and lemmatization are separate operations. POS tagging assigns a grammatical label, while lemmatization converts a word into its base form. spaCy can perform both in the same pipeline.

### 7.3 Inspect the difference between `pos_` and `tag_`

```python
import spacy

nlp = spacy.load("en_core_web_sm")

text = "I booked a room and read a book."
doc = nlp(text)

for token in doc:
    if not token.is_punct and not token.is_space:
        print(f"Word: {token.text}")
        print(f"POS: {token.pos_}")
        print(f"Fine tag: {token.tag_}")
        print("-" * 30)
```

The word `book` can be a verb in `booked a room` or a noun in `read a book`\. The model considers context to assign the appropriate tags.

### 7.4 POS tagging with NLTK

NLTK is especially useful for studying traditional NLP algorithms and experimenting with different taggers.

```python
import nltk

nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")

text = "The clever student solves problems quickly."

# Tokenize the sentence
tokens = nltk.word_tokenize(text)

# Assign POS tags
tagged_tokens = nltk.pos_tag(tokens)

print(tagged_tokens)
```

Example output\:

```python
[
    ('The', 'DT'),
    ('clever', 'JJ'),
    ('student', 'NN'),
    ('solves', 'VBZ'),
    ('problems', 'NNS'),
    ('quickly', 'RB'),
    ('.', '.')
]
```

NLTK's `pos_tag()` uses a pretrained English tagger. Its output uses a Penn Treebank-style tagset.

Depending on the NLTK version and tokenizer configuration, you may need different tokenizer resources. The commands above are suitable for recent NLTK versions.

### 7.5 Which library should you use?

| Requirement | Recommended choice |
|--------|--------|
| Learn NLP fundamentals | NLTK |
| Build an NLP pipeline | spaCy |
| Process large text datasets | spaCy with `nlp.pipe()` |
| Experiment with traditional taggers | NLTK |
| Build a Transformer-based tagger | Hugging Face Transformers |
| Get POS tags and lemmas together | spaCy |

For your NLP projects, I'd recommend starting with spaCy for practical preprocessing and using NLTK to understand classical methods.

---

## 8. POS tagging on a Pandas DataFrame

Suppose you have a movie-review dataset with 1,000 rows\:

| review |
|--------|
| This movie was excellent. |
| The story was boring. |
| Actors performed brilliantly. |

You can generate POS tags for every review.

### Method 1: Store tagged tokens in a new column

```python
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

df = pd.DataFrame({
    "review": [
        "This movie was excellent.",
        "The story was boring.",
        "Actors performed brilliantly."
    ]
})

def pos_tag_text(text):
    doc = nlp(str(text))

    return [
        (token.text, token.pos_)
        for token in doc
    ]

df["pos_tags"] = df["review"].apply(pos_tag_text)

print(df[["review", "pos_tags"]])
```

Possible output for the first review\:

```python
[
    ("This", "PRON"),
    ("movie", "NOUN"),
    ("was", "AUX"),
    ("excellent", "ADJ"),
    (".", "PUNCT")
]
```

The exact categories depend on the model. In spaCy, `AUX` distinguishes auxiliary verbs from other verbs.

### Method 2: Store fine-grained POS tags

If you want more detailed grammatical labels\:

```python
def detailed_pos_tag(text):
    doc = nlp(str(text))

    return [
        (token.text, token.tag_)
        for token in doc
    ]

df["detailed_pos"] = df["review"].apply(detailed_pos_tag)
```

This will distinguish categories such as plural nouns (`NNS`\) and third-person singular present-tense verbs (`VBZ`\).

### Method 3: Process a large dataset efficiently

If your dataset contains thousands or millions of text records, calling `nlp()` separately for each row can introduce unnecessary overhead.

Use `nlp.pipe()` to process texts in batches\:

```python
texts = df["review"].fillna("").astype(str)

df["pos_tags"] = [
    [
        (token.text, token.pos_)
        for token in doc
    ]
    for doc in nlp.pipe(texts, batch_size=128)
]
```

This is usually more efficient than repeatedly invoking `nlp()` on individual rows. Actual speed depends on document length, hardware, pipeline components, and batch size.

---

## 9. POS tagging and text preprocessing

POS tagging is often used alongside other NLP preprocessing techniques, but each technique has a different purpose.

| Technique | Purpose | Example |
|--------|--------|--------|
| Tokenization | Split text into tokens | `"cats run"` → `["cats", "run"]` |
| POS tagging | Assign grammatical labels | `cats/NNS`, `run/VBP` |
| Stemming | Reduce words to crude stems | `studies` → `studi` |
| Lemmatization | Find dictionary base forms | `studies` → `study` |
| Stopword removal | Remove selected common words | Remove `the`, if appropriate |
| Named Entity Recognition | Identify entities | `Kathmandu` → location |
| Dependency parsing | Identify grammatical relationships | `student` → subject of `learns` |

A typical NLP pipeline could be\:

The ordering is not universal. For example, spaCy performs tagging and lemmatization within its pipeline, and many Transformer models work directly on tokenized text without manually removing stopwords or lemmatizing every word.

### POS-aware lemmatization with NLTK

NLTK's `WordNetLemmatizer` defaults to treating a word as a noun. Supplying the correct POS can improve the result.

```python
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet")
nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("running"))
# running: default noun interpretation

print(lemmatizer.lemmatize("running", pos="v"))
# run
```

Why does the result differ? Without a verb tag, the lemmatizer interprets `running` as a noun by default. With the verb POS, it returns `run`\.

To connect NLTK POS tags to WordNet, you need to map tag prefixes to WordNet categories.

```python
from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV

    return wordnet.NOUN
```

This mapping is useful when you want to lemmatize a sentence using NLTK's POS tagger. Other tagsets, such as spaCy's, require a different mapping.

---

## 10. POS tagging in sentiment analysis

Let's connect POS tagging with your NLP text-classification learning.

Consider these movie reviews\:

1. "The movie was excellent."
2. "The movie was extremely boring."
3. "The acting was surprisingly good."

POS tagging can identify adjectives and adverbs that are useful for sentiment analysis.

| Word | POS | Potential role in sentiment |
|--------|--------|--------|
| excellent | ADJ | Positive opinion |
| extremely | ADV | Intensifies the following adjective |
| boring | ADJ | Negative opinion |
| surprisingly | ADV | Modifies the evaluation |
| good | ADJ | Positive opinion |

You can extract adjectives and adverbs with spaCy\:

```python
import spacy

nlp = spacy.load("en_core_web_sm")

review = "The movie was extremely boring but visually beautiful."
doc = nlp(review)

descriptive_words = [
    token.text
    for token in doc
    if token.pos_ in {"ADJ", "ADV"}
]

print(descriptive_words)
```

Example output\:

```python
["extremely", "boring", "visually", "beautiful"]
```

However, **POS tagging alone does not determine sentiment**\. For example, "not good" contains a positive adjective but expresses a negative evaluation. A sentiment model must account for negation and context.

POS features can complement TF-IDF, n-grams, embeddings, and other features, but they do not necessarily improve every classification task.

## 11. How is POS tagging evaluated?

POS tagging is typically evaluated against a manually annotated dataset called a *gold-standard corpus*\.

Suppose the correct tags are\:

```text
The/DT dog/NN runs/VBZ
```

And the model predicts\:

```text
The/DT dog/VB runs/VBZ
```

Two of the three tags are correct.

The standard metric is token-level accuracy\:

$$
\text{Accuracy} =
\frac{\text{Number of correctly tagged tokens}}
{\text{Total number of evaluated tokens}}
$$

In this example\:

$$
\text{Accuracy} = \frac{2}{3} \approx 66.67\%
$$

You can also measure precision, recall, and F1-score for individual POS categories, especially when you want to understand which tags the model confuses.

For example, a model may perform well on common nouns but struggle to distinguish verbs from nouns when the same word can function as either.

---

## 12. Common challenges in POS tagging

**1\. Ambiguous words**

The same word can be a noun, verb, or adjective depending on context.

**2\. Unknown words**

New words, product names, slang, and misspellings may not appear in training data.

**3\. Informal text**

Social media posts contain abbreviations, emojis, incomplete sentences, and nonstandard spelling.

**4\. Domain differences**

A tagger trained on news articles may perform differently on medical reports, scientific papers, or movie reviews.

**5\. Multiple languages**

Different languages have different grammatical systems, word orders, and morphological patterns. English models should not automatically be assumed to work well for Nepali or other languages.

**6\. Tokenization difficulties**

Contractions, hyphenated words, punctuation, and multiword expressions can complicate token-level tagging.

**7\. Context and sentence structure**

Long sentences and unusual grammatical constructions can make it difficult to select the correct tag sequence.

---

## 13. Test your understanding

Try predicting the POS tags for the following sentences before revealing the answers.

## 14. What you should learn next

Since you're progressing through NLP from preprocessing and text representation to classification, I recommend this order\:

1. **POS tagging\:** Understand tokens, tags, ambiguity, and spaCy implementation.
2. **Lemmatization\:** Use POS information to find correct dictionary forms.
3. **Dependency parsing\:** Learn grammatical relationships, such as subject, object, and modifier.
4. **Named Entity Recognition (NER)\:** Identify people, organizations, locations, and other entities.
5. **Sequence labeling models\:** Study HMM, Viterbi decoding, CRF, BiLSTM, and Transformer-based tagging.

The key takeaway is that **POS tagging assigns a grammatical category to each token using its context**\. Start by implementing it with spaCy, experiment with ambiguous words, and then study HMMs and neural sequence-tagging models to understand how the predictions are made.

![Why Many Thai Learners Study Grammar for Years — Yet Still Struggle to Read | by Aura Literary | Mar, 2026 | Medium](https://images.openai.com/static-rsc-4/Xoe0nSD-Z2G3wCC408XXfbPYf2aq_jDjXlIVC3FoIhrMKyan6qWdrKRpRWPUjHx-zONCz16wRfghBHsHM6g4BqePoQG4b3slQdIvPFC9TZTS_INsfNqm9ghI9QbRdAO0FgJ11fea05M0VZyVsYx935U8dCcl9nc8PAMdhgeiZfdnwaqb9G7fG76veKMb1KgJ?purpose=fullsize)

![고등학생의 친구 같은 챗봇 이루다 만들기 프로젝트를 소개합니다! 🎉 – 노마드 코더 Nomad Coders](https://images.openai.com/static-rsc-4/Gn4ESW3Ff0wxZfaDQIKwJoobW-oBhoMvnD_XN1yNDxlpd_jAV4OCn860oh5_GX-TMG6GcEuusx1UrUNDeOLFUfzK7VtzfaFqIwJgdEXenGGIm-E50l5xYSxfLyMB7ILLNuaaAOUSCme2RXScvTORMBMuzQHYl6FpS8OZq861zgXs7gnQLaLvbplP3ppb8p0h?purpose=fullsize)

![Understanding the mystique of sklearn’s DictVectorizer | by Crystal X | Geek Culture | Medium](https://images.openai.com/static-rsc-4/IjiQK2_fYgLBAVjW1syqKzrMCzo-Hr71MF0DHDGzD_l-4tnRPBZUyxRJGU9mR46ryxti0VIl0CZRPY7OR2h8j6ewW_I-aYVPiXrxqrVTN9mKLPPJ-ZRMzeqaa5XtbMRtgBQlbdghKVrV7hUAGh-iTphpLwXDkgODko74dnn6YHAFcTQHppa7WyesIceXa_lb?purpose=fullsize)

![Chapter 7: The Attention Mechanism and Transformers | Python Deep Learning](https://images.openai.com/static-rsc-4/HOAnLctqemFJiLQ03Wu3xICKEumuH8JQlwJH4oEyxnXQC0tJDafGaTDJFCPkB0W_nc6EM7D7JE7jGb5_Gu4H58ku5S0PfdWjLOAxzZEXIEiLSk5LRiTm8GECMSzVn-skEZQoD8uu3TRsEoKYsDHII1lH1i_9L3SXGgFa9Px94QX4raO3Q4843wKI4FvL9jwM?purpose=fullsize)

---
