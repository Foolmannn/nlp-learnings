# NLP Pipeline in Detail

### Basic Steps :

##### 1.Data Acquisition 

                From where would you acquire the data?
##### 2.Text Preparation

                What kind of cleaning steps would you perform?
                What text preprocessing step would you apply?
                Is advanced text preprocessing required?
##### 3.Feature Engineering

                What kind of features would you create?
##### 4.Modelling

                What algorithm would you use to solve the problem at hand?
                What intrinsic evaluation metrics would you use?
                What extrinsic evaluation metrics would you use?
##### 5.Deployment

                How would you deploy your solution into the entire product?
                How and what things will you monitor?
                What would be your model update strategy?


An **NLP pipeline** is a sequence of steps through which raw human language is transformed into a form that a computer can process, analyze, understand, or generate.

A general NLP pipeline looks like:

```text
                    RAW TEXT
                       │
                       ▼
              ┌─────────────────┐
              │ 1. Text Input   │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ 2. Cleaning &   │
              │ Normalization   │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ 3. Sentence     │
              │ Segmentation    │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ 4. Tokenization │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ 5. Linguistic   │
              │ Processing      │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ 6. Text         │
              │ Representation  │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ 7. Feature      │
              │ Extraction      │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ 8. NLP Model    │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ 9. Prediction / │
              │ Understanding   │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ 10. Evaluation  │
              └────────┬────────┘
                       ▼
                    OUTPUT
```

However, **modern NLP does not always use every step**. A classical TF-IDF system and a Transformer-based system have very different pipelines.

---

# 1. What is an NLP Pipeline?

Suppose we give an NLP system:

> **"I absolutely loved this movie! The acting was amazing."**

The computer cannot directly process this sentence as meaningful information.

We need to transform it:

```text
Human Language
      ↓
Raw Text
      ↓
Clean / Normalize
      ↓
Split into sentences
      ↓
Tokenize
      ↓
Represent numerically
      ↓
Machine Learning / Deep Learning Model
      ↓
Prediction
```

For example, for sentiment analysis:

```text
"I absolutely loved this movie!"
                 ↓
          NLP Pipeline
                 ↓
           Positive 😊
```

---

# 2. Two Major Types of NLP Pipelines

Before studying individual stages, understand that there are two broad styles.

## Classical NLP Pipeline

Usually:

```text
Raw Text
   ↓
Cleaning
   ↓
Tokenization
   ↓
Stopword Removal
   ↓
Stemming/Lemmatization
   ↓
BoW / TF-IDF
   ↓
ML Model
   ↓
Prediction
```

Example:

```text
Review
 ↓
TF-IDF
 ↓
Logistic Regression
 ↓
Positive/Negative
```

---

## Modern Transformer Pipeline

Modern NLP often looks more like:

```text
Raw Text
   ↓
Tokenizer
   ↓
Token IDs
   ↓
Attention Mask
   ↓
Embedding
   ↓
Transformer
   ↓
Contextual Representation
   ↓
Task Head / Generation
   ↓
Output
```

For an LLM:

```text
Prompt
   ↓
Tokenizer
   ↓
Token IDs
   ↓
Embeddings
   ↓
Transformer
   ↓
Next-token probabilities
   ↓
Token selection
   ↓
Repeat
   ↓
Generated text
```

This distinction is **very important**.

---

# 3. Stage 1 — Text Input

The first stage is obtaining the raw language.

Input may come from:

- user messages
- documents
- websites
- PDFs
- emails
- social media
- speech-to-text systems
- databases
- chat applications
- customer reviews

Example:

```text
"I really enjoyed this movie. The acting was excellent!"
```

At this point it is simply a string.

```python
text = "I really enjoyed this movie. The acting was excellent!"
```

---

# 4. Stage 2 — Text Cleaning

Raw text often contains unnecessary or problematic information.

Example:

```text
"OMG!!! I LOVED this movie 😍😍😍 Visit https://example.com"
```

Depending on the task, we might want to process:

- URLs
- HTML
- extra whitespace
- repeated characters
- special symbols
- unwanted markup
- encoding problems

For example:

```text
"I     love     NLP"
```

could become:

```text
"I love NLP"
```

---

## But Don't Over-Clean

This is a very important NLP principle.

You should **not automatically remove everything that looks unusual**.

Consider:

```text
"I don't like this."
```

If you remove:

```text
don't
```

you might end up with:

```text
"I like this."
```

which completely changes the sentiment.

Similarly:

```text
not good
```

is different from:

```text
good
```

Therefore:

> **Preprocessing should be task-dependent.**

This becomes especially important in modern NLP.

---

# 5. Stage 3 — Text Normalization

Normalization makes text more consistent.

Common operations include:

### Lowercasing

```text
"Hello WORLD"
```

↓

```text
"hello world"
```

But this isn't always appropriate.

For example:

```text
US
```

and:

```text
us
```

can have different meanings.

---

### Unicode normalization

Different Unicode representations can sometimes represent visually similar text.

This matters particularly for multilingual NLP.

---

### Spelling normalization

For noisy data:

```text
goooood
```

might be normalized to:

```text
good
```

But again, this depends on the application.

---

# 6. Stage 4 — Sentence Segmentation

A document may contain multiple sentences.

Example:

```text
"I love NLP. It is very interesting. I want to learn Transformers."
```

We divide it into:

```text
Sentence 1:
"I love NLP."

Sentence 2:
"It is very interesting."

Sentence 3:
"I want to learn Transformers."
```

This is called:

**Sentence Segmentation** or **Sentence Boundary Detection**.

---

## Why is it important?

Different NLP tasks operate at different levels.

```text
Document
   ↓
Sentences
   ↓
Words/Tokens
```

For example, sentiment might be calculated:

```text
Document-level
Sentence-level
Aspect-level
```

---

# 7. Stage 5 — Tokenization

This is one of the most fundamental NLP operations.

**Tokenization** means breaking text into smaller units called **tokens**.

Example:

```text
"I love NLP."
```

might become:

```text
["I", "love", "NLP", "."]
```

These are tokens.

---

# 8. Word Tokenization

Traditional NLP often uses words as tokens.

```text
"I love machine learning."
```

↓

```text
["I", "love", "machine", "learning", "."]
```

This is easy to understand.

But modern NLP usually uses **subword tokenization**.

---

# 9. Subword Tokenization

Suppose we have:

```text
unhappiness
```

Instead of treating the entire word as one token, a tokenizer might divide it into pieces such as:

```text
un
happiness
```

or even smaller pieces depending on the tokenizer.

Why?

Because natural language contains an enormous number of possible words.

Subword tokenization helps models handle:

- rare words
- new words
- spelling variations
- morphology
- unknown words

Common approaches include:

- BPE
- WordPiece
- SentencePiece

These become extremely important when studying Transformers.

---

# 10. Stage 6 — Stopword Processing

Stopwords are very common words such as:

```text
the
is
a
an
of
to
in
```

In some classical NLP applications, we remove them.

Example:

```text
"The cat is sitting on the table."
```

could become:

```text
["cat", "sitting", "table"]
```

---

## Should Stopwords Always Be Removed?

**No.**

Modern Transformer models generally don't use traditional stopword removal.

Consider:

```text
"This is not good."
```

Removing:

```text
not
```

would be harmful.

So:

```text
Stopword removal
```

is mainly a **classical NLP preprocessing technique**, not a mandatory NLP step.

---

# 11. Stage 7 — Stemming

Stemming attempts to reduce words to a root form.

Example:

```text
playing
played
plays
player
```

may be reduced to something like:

```text
play
```

A stemmer generally uses rules rather than deep linguistic understanding.

Therefore, stems can sometimes be unnatural.

---

# 12. Stage 8 — Lemmatization

Lemmatization attempts to convert a word into its linguistically valid base form.

For example:

```text
running → run
better → good
children → child
```

It generally requires more linguistic information than stemming.

---

## Stemming vs Lemmatization

| Stemming | Lemmatization |
|---|---|
| Rule-based reduction | Linguistic normalization |
| Faster | Usually slower |
| May produce invalid words | Produces valid lemmas |
| Less linguistically accurate | More linguistically meaningful |
| Common in classical NLP | Useful when linguistic form matters |

Example:

```text
studies
```

A stemmer might produce:

```text
studi
```

while lemmatization gives:

```text
study
```

---

# 13. Stage 9 — Part-of-Speech Tagging

Now we can identify the grammatical role of each word.

Example:

```text
"Suman writes Python."
```

could be:

```text
Suman      → PROPN
writes     → VERB
Python     → PROPN
```

Common POS categories:

```text
NOUN
VERB
ADJECTIVE
ADVERB
PRONOUN
PREPOSITION
CONJUNCTION
DETERMINER
```

POS tagging helps with linguistic understanding.

---

# 14. Stage 10 — Named Entity Recognition

**Named Entity Recognition (NER)** identifies important entities.

Example:

> "Suman lives in Kathmandu and studies at Tribhuvan University."

NER could identify:

```text
Suman
    → PERSON

Kathmandu
    → LOCATION

Tribhuvan University
    → ORGANIZATION
```

Common entity types:

- PERSON
- ORGANIZATION
- LOCATION
- DATE
- TIME
- MONEY
- PRODUCT
- EVENT

---

# 15. Stage 11 — Dependency Parsing

Dependency parsing tries to determine relationships between words.

Example:

> "The cat chased the mouse."

Conceptually:

```text
          chased
         /      \
       cat      mouse
      subject    object
```

This provides information about grammatical relationships.

---

# 16. Stage 12 — Semantic Analysis

Now we move beyond grammatical structure toward **meaning**.

Consider:

```text
"I deposited money in the bank."
```

versus:

```text
"The boat reached the river bank."
```

The word:

```text
bank
```

has different meanings depending on context.

This is why modern NLP uses **contextual representations**.

---

# 17. Stage 13 — Text Representation

This is one of the most important stages.

Computers need numerical representations.

There are several generations of representations.

---

## Level 1 — One-Hot Encoding

Vocabulary:

```text
["cat", "dog", "car"]
```

Represent:

```text
cat → [1,0,0]

dog → [0,1,0]

car → [0,0,1]
```

Problem:

There is no semantic relationship.

The vectors are also high-dimensional and sparse for large vocabularies.

---

# 18. Bag of Words

Example:

```text
"I love NLP"

"I love AI"
```

Vocabulary:

```text
I
love
NLP
AI
```

Represent:

```text
I love NLP
→ [1,1,1,0]

I love AI
→ [1,1,0,1]
```

This captures word frequency but not deep meaning.

---

# 19. TF-IDF

TF-IDF gives different weights to words depending on:

- frequency within a document
- rarity across documents

Conceptually:

```text
Document
    ↓
Words
    ↓
TF
    ↓
IDF
    ↓
TF-IDF vector
```

This is widely useful for classical text classification.

---

# 20. Word Embeddings

Instead of sparse vectors:

```text
[0,0,0,0,0,1,0,0,...]
```

we learn dense vectors:

```text
cat → [0.21, -0.42, 0.73, ...]
```

Examples:

- Word2Vec
- GloVe
- FastText

These representations can capture semantic relationships.

---

# 21. Contextual Embeddings

Modern models go further.

The representation of a word depends on its context.

Consider:

```text
I went to the bank to deposit money.
```

and:

```text
The fisherman sat on the river bank.
```

The word:

```text
bank
```

can receive different contextual representations.

Models such as BERT and Transformer-based systems do this.

---

# 22. Stage 14 — Feature Extraction

After converting language into numerical representations, we can extract useful features.

Classical features might include:

```text
TF-IDF
N-grams
word frequencies
POS counts
named entities
document length
```

Example:

```text
Review
   ↓
TF-IDF
   ↓
10,000-dimensional vector
```

Then:

```text
Vector
 ↓
Logistic Regression
```

---

# 23. Stage 15 — NLP Model

Now we apply a machine-learning or deep-learning model.

Different tasks require different models.

### Classification

```text
Text
 ↓
Representation
 ↓
Classifier
 ↓
Class
```

Example:

```text
"I love this movie!"
        ↓
   Sentiment Model
        ↓
     Positive
```

---

### Named Entity Recognition

```text
Text
 ↓
Transformer
 ↓
Token-level predictions
 ↓
PERSON / LOCATION / ORG
```

---

### Machine Translation

```text
English
 ↓
Encoder
 ↓
Decoder
 ↓
Nepali
```

---

### Text Generation

```text
Prompt
 ↓
Language Model
 ↓
Next-token prediction
 ↓
Generated sequence
```

---

# 24. Stage 16 — Prediction

Suppose we have:

```text
"I absolutely loved this movie."
```

The classifier might output:

```text
Positive: 0.96
Negative: 0.04
```

Then:

```text
Predicted class = Positive
```

For a multi-class problem:

```text
Sports       0.05
Technology   0.80
Politics     0.10
Business     0.05
```

Prediction:

```text
Technology
```

---

# 25. Stage 17 — Post-processing

The raw model output may not be the final user-facing result.

For example:

```text
Model output
      ↓
Decode
      ↓
Remove special tokens
      ↓
Format
      ↓
Final answer
```

For generation:

```text
Token IDs
 ↓
Tokenizer decoder
 ↓
Text
```

For NER:

```text
Token predictions
 ↓
Merge entities
 ↓
Human-readable output
```

---

# 26. Stage 18 — Evaluation

An NLP pipeline must be evaluated.

Different tasks use different metrics.

### Classification

Use:

- accuracy
- precision
- recall
- F1-score
- confusion matrix

---

### Machine Translation

Traditionally:

- BLEU

---

### Summarization

Commonly:

- ROUGE

---

### Language Modeling

A key metric is:

**Perplexity**

Conceptually, lower perplexity generally indicates the model is less surprised by the observed sequence, though it should be interpreted carefully across different tokenizers/models.

---

### Retrieval

For RAG systems:

- Recall@K
- Precision@K
- MRR
- nDCG

And then separately evaluate:

- answer correctness
- faithfulness/groundedness
- citation quality

---

# 27. Complete Classical NLP Pipeline

Let's take a sentiment analysis example.

Input:

> "The movie was absolutely fantastic!"

### Step 1 — Raw text

```text
"The movie was absolutely fantastic!"
```

### Step 2 — Normalize

```text
"the movie was absolutely fantastic"
```

### Step 3 — Tokenize

```text
["the", "movie", "was", "absolutely", "fantastic"]
```

### Step 4 — Stopword processing

Potentially:

```text
["movie", "absolutely", "fantastic"]
```

### Step 5 — Lemmatization

```text
["movie", "absolutely", "fantastic"]
```

### Step 6 — TF-IDF

```text
[0.00, 0.43, 0.72, ...]
```

### Step 7 — Model

```text
Logistic Regression
```

### Step 8 — Prediction

```text
Positive
```

Complete:

```text
Raw Text
   ↓
Cleaning
   ↓
Tokenization
   ↓
Stopword Removal
   ↓
Lemmatization
   ↓
TF-IDF
   ↓
Logistic Regression
   ↓
Positive
```

---

# 28. Modern Transformer NLP Pipeline

Now consider the same sentiment problem using BERT.

Input:

```text
"The movie was absolutely fantastic!"
```

### Step 1

Raw text.

```text
"The movie was absolutely fantastic!"
```

### Step 2 — Tokenizer

A Transformer tokenizer converts text into subword tokens.

Conceptually:

```text
[CLS]
The
movie
was
absolutely
fantastic
!
[SEP]
```

Exact tokenization depends on the model.

---

### Step 3 — Token IDs

Tokens are converted into integers:

```text
[101, 1996, 3185, ...]
```

The exact IDs depend on the tokenizer vocabulary.

---

### Step 4 — Attention Mask

The model receives information about which positions contain valid input tokens.

Conceptually:

```text
1 1 1 1 1 1 1 1
```

---

### Step 5 — Embeddings

Token IDs are converted into vectors.

```text
Token IDs
    ↓
Embedding Layer
    ↓
Dense vectors
```

---

### Step 6 — Transformer

The vectors pass through Transformer blocks.

Inside each block:

```text
Input
 ↓
Multi-Head Self-Attention
 ↓
Add & Norm
 ↓
Feed Forward Network
 ↓
Add & Norm
 ↓
Output
```

---

### Step 7 — Contextual Representation

The model builds representations based on surrounding words.

For example:

```text
fantastic
```

is interpreted in the context of:

```text
movie
absolutely
was
```

---

### Step 8 — Classification Head

A classification layer produces logits:

```text
Positive → 4.8
Negative → -2.1
```

After softmax:

```text
Positive → 0.998
Negative → 0.002
```

---

### Step 9 — Final Prediction

```text
Positive
```

The pipeline:

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Attention Mask
 ↓
Embeddings
 ↓
Transformer
 ↓
Contextual Representation
 ↓
Classification Head
 ↓
Softmax
 ↓
Prediction
```

---

# 29. Modern LLM Pipeline

A generative LLM is slightly different.

Suppose the prompt is:

> "Explain machine learning in simple terms."

The process is roughly:

```text
Prompt
   ↓
Tokenizer
   ↓
Token IDs
   ↓
Token Embeddings
   ↓
Positional Information
   ↓
Transformer Blocks
   ↓
Logits
   ↓
Probability Distribution
   ↓
Token Selection
   ↓
New Token
   ↓
Repeat
   ↓
Generated Text
```

The crucial idea is:

> A decoder-style language model repeatedly predicts the next token based on the preceding context.

---

# 30. Token Generation Example

Suppose the prompt is:

```text
"Machine learning is"
```

The model produces probabilities such as:

```text
a        0.30
a method 0.20
the      0.10
...
```

Suppose:

```text
"a"
```

is selected.

Now:

```text
"Machine learning is a"
```

The model predicts the next token again.

Perhaps:

```text
method
```

Then:

```text
"Machine learning is a method"
```

This repeats until:

```text
EOS / stopping condition
```

---

# 31. RAG NLP Pipeline

Since you're moving toward modern NLP, this pipeline is especially important.

Suppose you ask:

> "What is the refund policy in this PDF?"

The system might work like:

```text
                 DOCUMENTS
                     │
                     ▼
               Document Loader
                     │
                     ▼
                Text Extraction
                     │
                     ▼
                  Chunking
                     │
                     ▼
                Embeddings
                     │
                     ▼
                Vector Store
                     │
                     │
USER QUERY ──────────┘
     │
     ▼
 Query Embedding
     │
     ▼
 Similarity Search
     │
     ▼
 Top-K Documents
     │
     ▼
 Reranking / Filtering
     │
     ▼
 Context
     │
     ▼
     LLM
     │
     ▼
 Generated Answer
```

This is an **NLP pipeline combined with information retrieval and generative AI**.

---

# 32. NLP Pipeline: Classical vs Modern

| Stage | Classical NLP | Transformer NLP |
|---|---|---|
| Cleaning | Common | Task-dependent |
| Tokenization | Word-based | Subword-based |
| Stopword removal | Common | Usually unnecessary |
| Stemming | Common | Usually unnecessary |
| Lemmatization | Common | Usually unnecessary |
| BoW | Common | No |
| TF-IDF | Common | Usually no |
| Word2Vec | Common | Sometimes |
| Embeddings | Static | Contextual |
| RNN/LSTM | Sometimes | Usually replaced |
| Attention | No/limited | Core component |
| Transformer | No | Core architecture |
| BERT/GPT | No | Common |
| Fine-tuning | Classical ML training | Common |
| RAG | No | Common in LLM applications |

---

# 33. The Most Important Concept: Pipeline Depends on the Task

There is **no single universal NLP pipeline**.

For example:

### Sentiment Analysis

```text
Text
 ↓
Tokenizer
 ↓
Representation
 ↓
Classifier
 ↓
Sentiment
```

### NER

```text
Text
 ↓
Tokenizer
 ↓
Transformer
 ↓
Token Classification
 ↓
Entities
```

### Translation

```text
Source Language
 ↓
Tokenizer
 ↓
Encoder
 ↓
Decoder
 ↓
Target Language
```

### Summarization

```text
Document
 ↓
Tokenizer
 ↓
Transformer
 ↓
Generation
 ↓
Summary
```

### RAG

```text
Documents
 ↓
Chunking
 ↓
Embeddings
 ↓
Vector DB
 ↓
Retrieval
 ↓
LLM
 ↓
Answer
```

---

# 34. The NLP Pipeline You Should Memorize

For your learning, remember these three versions.

## Classical

```text
RAW TEXT
   ↓
CLEANING
   ↓
TOKENIZATION
   ↓
NORMALIZATION
   ↓
STOPWORDS
   ↓
STEMMING / LEMMATIZATION
   ↓
BoW / N-GRAM / TF-IDF
   ↓
ML MODEL
   ↓
PREDICTION
```

---

## Transformer

```text
RAW TEXT
   ↓
TOKENIZER
   ↓
TOKEN IDs
   ↓
EMBEDDINGS
   ↓
POSITIONAL INFORMATION
   ↓
SELF-ATTENTION
   ↓
TRANSFORMER BLOCKS
   ↓
CONTEXTUAL REPRESENTATION
   ↓
TASK HEAD
   ↓
OUTPUT
```

---

## LLM/RAG

```text
              DOCUMENTS
                  ↓
              CHUNKING
                  ↓
              EMBEDDING
                  ↓
             VECTOR STORE
                  ↑
                  │
USER → QUERY → RETRIEVAL
                  ↓
               CONTEXT
                  ↓
                 LLM
                  ↓
              GENERATION
                  ↓
               ANSWER
```

---

# 35. What You Should Learn Next

Since you're starting NLP, I recommend we go **in this exact order**:

```text
1. NLP Fundamentals
       ↓
2. Text Preprocessing
       ↓
3. Tokenization
       ↓
4. Stemming & Lemmatization
       ↓
5. Bag of Words
       ↓
6. N-Grams
       ↓
7. TF-IDF
       ↓
8. Text Classification
       ↓
9. Word Embeddings
       ↓
10. Word2Vec
       ↓
11. RNN
       ↓
12. LSTM
       ↓
13. Attention
       ↓
14. Transformers
       ↓
15. BERT
       ↓
16. Hugging Face
       ↓
17. LLMs
       ↓
18. Embeddings + Vector Search
       ↓
19. RAG
       ↓
20. Advanced RAG / Agents
```

**Your next topic should be _Text Preprocessing in NLP in detail_**, where we can go deeply into normalization, tokenization, stopwords, stemming, lemmatization, regex, handling URLs/emails/emojis, and implement each technique in Python before moving to **Bag of Words and TF-IDF**.