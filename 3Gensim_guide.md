
Gensim is a Python library designed around **unsupervised NLP and large-scale text representations**, with a strong emphasis on processing corpora without requiring the entire corpus to be represented as a huge in-memory matrix. [PII Tools](https://radimrehurek.com/lrec2010_final.pdf?utm_source=chatgpt.com)

Below is a beginner-to-advanced guide tailored to where you are now.

---

# Gensim — Detailed Beginner Guide

## 1. What is Gensim?

**Gensim** is a Python library for working with text and learning semantic representations from text.

It is particularly known for algorithms such as:

- Word2Vec
- FastText
- Doc2Vec
- LDA
- TF-IDF
- LSI
- similarity/search-related functionality
- topic modelling

Its core idea is that text can be transformed into numerical representations that machine-learning algorithms can work with.

A simplified NLP pipeline looks like:

```text
Raw Text
   ↓
Text preprocessing
   ↓
Tokenization
   ↓
Corpus
   ↓
Numerical representation
   ↓
Gensim model
   ↓
Vectors / Topics / Similarity
```

For your current learning:

```text
Raw text
   ↓
sent_tokenize()
   ↓
simple_preprocess()
   ↓
[['machine','learning',...], [...], ...]
   ↓
Word2Vec
   ↓
Word vectors
```

---

# 2. Installation

Install Gensim with:

```bash
pip install gensim
```

Check the installation:

```python
import gensim

print(gensim.__version__)
```

You can also check:

```python
import gensim
print(gensim)
```

---

# 3. Understand the Gensim ecosystem

This is the most important conceptual overview.

Gensim has several major components:

```text
gensim
│
├── models
│   ├── Word2Vec
│   ├── FastText
│   ├── Doc2Vec
│   ├── LdaModel
│   ├── LsiModel
│   └── TfidfModel
│
├── corpora
│   ├── Dictionary
│   └── MmCorpus
│
├── utils
│   ├── simple_preprocess
│   └── other utilities
│
└── similarities
    └── similarity/search functionality
```

You don't need to learn everything immediately.

For your current NLP journey, I'd learn in this order:

```text
1. gensim.utils
        ↓
2. gensim.corpora.Dictionary
        ↓
3. Word2Vec
        ↓
4. FastText
        ↓
5. Doc2Vec
        ↓
6. TF-IDF
        ↓
7. LDA
        ↓
8. LSI
        ↓
9. Similarity/search
```

---

# 4. `gensim.utils`

You are already using:

```python
from gensim.utils import simple_preprocess
```

This is a utility function for basic text preprocessing/tokenization.

Example:

```python
from gensim.utils import simple_preprocess

text = "Machine Learning is AMAZING!"

tokens = simple_preprocess(text)

print(tokens)
```

Output:

```python
['machine', 'learning', 'is', 'amazing']
```

So:

```python
simple_preprocess()
```

is doing roughly:

```text
Text
 ↓
lowercase
 ↓
tokenization
 ↓
basic cleaning
 ↓
tokens
```

---

# 5. `simple_preprocess()` parameters

You can specify:

```python
simple_preprocess(
    text,
    deacc=False,
    min_len=2,
    max_len=15
)
```

For example:

```python
tokens = simple_preprocess(
    "Machine learning is powerful!",
    deacc=True
)
```

`deacc=True` removes accent marks.

For example, text containing accented characters can be normalized.

You can also change token length:

```python
simple_preprocess(
    text,
    min_len=1,
    max_len=20
)
```

---

# 6. Your current code

You currently have:

```python
from nltk import sent_tokenize
from gensim.utils import simple_preprocess

story = []

for filename in os.listdir('data'):
    
    f = open(os.path.join('data',filename))
    corpus = f.read()

    raw_sent = sent_tokenize(corpus)

    for sent in raw_sent:
        story.append(simple_preprocess(sent))
```

Conceptually:

```text
                  data/
                    │
          ┌─────────┼─────────┐
          ↓         ↓         ↓
       file1.txt file2.txt file3.txt
          │         │         │
          └─────────┼─────────┘
                    ↓
                 corpus
                    ↓
             sent_tokenize
                    ↓
              individual
               sentences
                    ↓
          simple_preprocess
                    ↓
             tokenized sentences
                    ↓
                  story
```

Your final `story` looks like:

```python
[
    ['machine', 'learning', 'is', 'powerful'],
    ['deep', 'learning', 'uses', 'neural', 'networks'],
    ['natural', 'language', 'processing', 'is', 'interesting'],
    ...
]
```

This is an extremely important structure.

---

# 7. Why does Gensim like this structure?

Gensim has an important concept called a **corpus**.

A corpus can essentially be thought of as:

```text
collection of documents
```

For Word2Vec:

```text
corpus
   ↓
sentences
   ↓
words
```

For example:

```python
corpus = [
    ['i', 'love', 'machine', 'learning'],
    ['machine', 'learning', 'is', 'interesting'],
    ['deep', 'learning', 'is', 'powerful']
]
```

Each inner list is a document/sentence.

Gensim's design emphasizes iterable/streaming corpora, so the entire corpus does not necessarily need to be materialized in memory for every workflow. [PII Tools](https://radimrehurek.com/lrec2010_final.pdf?utm_source=chatgpt.com)

This becomes very useful with huge datasets.

---

# 8. `gensim.corpora.Dictionary`

This is the next Gensim concept you should learn.

Import:

```python
from gensim.corpora import Dictionary
```

Suppose:

```python
documents = [
    ['machine', 'learning'],
    ['deep', 'learning'],
    ['machine', 'intelligence']
]
```

Create a dictionary:

```python
dictionary = Dictionary(documents)
```

Now:

```python
print(dictionary)
```

You might get:

```text
Dictionary<4 unique tokens: ['deep', 'intelligence', 'learning', 'machine']>
```

The dictionary assigns an integer ID to each unique word.

Conceptually:

```text
machine      → 0
learning     → 1
deep         → 2
intelligence → 3
```

You can inspect it:

```python
print(dictionary.token2id)
```

Example:

```python
{
    'deep': 0,
    'intelligence': 1,
    'learning': 2,
    'machine': 3
}
```

The exact IDs depend on the data/order.

---

# 9. Why do we need a Dictionary?

Computers ultimately need numerical representations.

Suppose:

```python
[
    ['machine', 'learning'],
    ['deep', 'learning']
]
```

The computer can convert the vocabulary into IDs:

```text
machine  → 0
learning → 1
deep     → 2
```

Then:

```text
machine learning
     ↓
0 1
```

and:

```text
deep learning
     ↓
2 1
```

This is the foundation for several Gensim corpus representations.

---

# 10. Bag of Words — `doc2bow()`

Now:

```python
dictionary.doc2bow()
```

This is one of the most important Gensim methods.

Example:

```python
documents = [
    ['machine', 'learning'],
    ['deep', 'learning'],
    ['machine', 'learning', 'machine']
]

dictionary = Dictionary(documents)
```

Then:

```python
bow = dictionary.doc2bow(
    ['machine', 'learning', 'machine']
)

print(bow)
```

You get something conceptually like:

```python
[(0, 2), (1, 1)]
```

This means:

```text
word ID 0 → appears 2 times
word ID 1 → appears 1 time
```

The structure is:

```text
(word_id, frequency)
```

So:

```python
[(0, 2), (1, 1)]
```

means:

```text
word 0 → 2 occurrences
word 1 → 1 occurrence
```

---

# 11. Why is this called Bag of Words?

Because the representation ignores word order.

Consider:

```text
I love Python
```

and:

```text
Python love I
```

Bag of Words essentially sees the same word counts.

```text
I       → 1
love    → 1
Python  → 1
```

This is different from Word2Vec.

---

# 12. Bag of Words vs Word2Vec

This distinction is very important.

### Bag of Words

```text
word
 ↓
integer/count
```

Example:

```text
machine → 2
learning → 3
python → 1
```

### Word2Vec

```text
word
 ↓
dense vector
 ↓
[0.23, -0.81, 0.17, ...]
```

For example:

```python
model.wv['machine']
```

might return:

```text
[ 0.12, -0.43, 0.87, ...]
```

So:

```text
BoW
→ frequency representation

Word2Vec
→ semantic/dense vector representation
```

---

# 13. Word2Vec

This is the part you are currently learning.

Import:

```python
from gensim.models import Word2Vec
```

Then:

```python
model = Word2Vec(
    sentences=story,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)
```

Let's understand every parameter.

---

# 14. `sentences`

```python
sentences=story
```

This is your training corpus.

For example:

```python
story = [
    ['machine', 'learning', 'is', 'powerful'],
    ['deep', 'learning', 'is', 'popular'],
    ['machine', 'learning', 'uses', 'data']
]
```

Word2Vec learns from these sentences.

---

# 15. `vector_size`

```python
vector_size=100
```

This determines the number of dimensions in each word vector.

For:

```python
vector_size=100
```

each word gets:

```text
100-dimensional vector
```

For example:

```python
model.wv['machine']
```

returns something conceptually like:

```text
[
  0.12,
 -0.45,
  0.83,
  ...
]
```

with 100 numbers.

You can verify:

```python
print(model.wv['machine'].shape)
```

Output:

```text
(100,)
```

---

# 16. What does `vector_size` actually mean?

This is a **learned semantic space**.

Suppose:

```text
king
queen
man
woman
```

are represented as:

```text
king  → vector
queen → vector
man   → vector
woman → vector
```

Words used in similar contexts tend to have similar vector representations.

That is why Word2Vec is called a **word embedding** technique.

---

# 17. `window`

```python
window=5
```

This controls the context size.

Suppose:

```text
I love machine learning very much
```

If:

```python
window=2
```

then a word learns mainly from nearby words within a context window.

Conceptually:

```text
I       love    machine    learning    very    much
                 ↑
               target
```

The window determines how far around the target word Gensim considers context during training.

Larger window:

```text
more context
```

Smaller window:

```text
more local context
```

---

# 18. `min_count`

```python
min_count=2
```

This means:

> Ignore words appearing fewer than 2 times.

Suppose:

```text
machine → 100 occurrences
learning → 80
python → 50
elephant → 1
```

With:

```python
min_count=2
```

`elephant` may be removed from the vocabulary.

This is extremely useful for large corpora because rare words can create noise and unnecessarily increase the vocabulary.

For small learning datasets:

```python
min_count=1
```

is useful.

For larger corpora:

```python
min_count=2
```

or:

```python
min_count=5
```

is common depending on the task.

---

# 19. `workers`

```python
workers=4
```

This determines the number of worker threads used for training.

If your CPU has multiple cores:

```python
workers=4
```

can make training considerably faster than:

```python
workers=1
```

You can inspect your CPU count:

```python
import os

print(os.cpu_count())
```

---

# 20. `sg`

Another important Word2Vec parameter:

```python
sg=0
```

or:

```python
sg=1
```

It determines the training architecture.

```text
sg=0
→ CBOW

sg=1
→ Skip-gram
```

### CBOW

```text
context words
      ↓
   target word
```

Example:

```text
I love [machine] learning
```

Context:

```text
love
learning
```

predicts:

```text
machine
```

### Skip-gram

Opposite direction:

```text
target word
    ↓
context words
```

For:

```text
I love machine learning
```

the model uses:

```text
machine
```

to predict surrounding words.

---

# 21. `epochs`

You can specify:

```python
epochs=10
```

This determines how many times the training algorithm goes through the training corpus.

Example:

```python
model = Word2Vec(
    sentences=story,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    epochs=10
)
```

Conceptually:

```text
Corpus
 ↓
epoch 1
 ↓
epoch 2
 ↓
epoch 3
 ↓
...
 ↓
epoch 10
```

---

# 22. Complete Word2Vec example

```python
from gensim.models import Word2Vec

model = Word2Vec(
    sentences=story,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=1,
    epochs=10
)
```

This means:

```text
story
 ↓
Word2Vec
 ↓
100-dimensional embeddings
 ↓
window = 5
 ↓
ignore words occurring < 2 times
 ↓
Skip-gram
 ↓
10 training epochs
```

---
