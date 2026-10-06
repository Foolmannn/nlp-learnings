# Word2Vec in NLP — Detailed Explanation

**Word2Vec** is one of the most important techniques for learning **word embeddings** in NLP.

Unlike **One-Hot Encoding, Bag of Words, and TF-IDF**, where words are represented mainly as independent features, Word2Vec tries to represent words based on their **meaning and context**.

For example:

```text
king
queen
man
woman
```

Word2Vec learns numerical vectors such that semantically related words tend to have similar vectors.

---

# 1. Why do we need Word2Vec?

Consider these sentences:

```text
The dog is eating food.
The cat is eating food.
```

A traditional Bag of Words representation might create:

```text
dog → [1, 0, 0, 0, ...]
cat → [0, 1, 0, 0, ...]
```

The problem is that the representation doesn't inherently tell us that:

```text
dog
cat
```

are semantically related.

They are simply different columns.

Similarly, One-Hot Encoding gives:

```text
dog → [0, 1, 0, 0, 0]
cat → [0, 0, 1, 0, 0]
```

The vectors are completely independent.

But Word2Vec tries to learn:

```text
dog → [0.21, -0.45, 0.72, ...]
cat → [0.19, -0.41, 0.68, ...]
```

The vectors are similar because **dog and cat occur in similar contexts**.

---

# 2. What is Word Embedding?

A **word embedding** is a numerical vector representation of a word.

For example:

```text
"king"
      ↓
[0.23, -0.51, 0.71, 0.18, ...]
```

Maybe the vector has 100 dimensions:

```text
king → [0.23, -0.51, 0.71, ..., 0.44]
```

Instead of representing a word with:

```text
[0, 0, 0, 0, 1, 0, 0, 0, ...]
```

we represent it with **dense numerical values**.

---

# 3. One-Hot vs Word2Vec

Suppose vocabulary:

```text
["dog", "cat", "king", "queen", "car"]
```

### One-Hot

```text
dog   → [1, 0, 0, 0, 0]
cat   → [0, 1, 0, 0, 0]
king  → [0, 0, 1, 0, 0]
queen → [0, 0, 0, 1, 0]
car   → [0, 0, 0, 0, 1]
```

The vectors are:

- sparse
- high-dimensional for large vocabularies
- unrelated words have no meaningful relationship
- no semantic information is directly encoded

### Word2Vec

Could learn something like:

```text
dog   → [0.72, 0.31, -0.15, 0.66]
cat   → [0.69, 0.35, -0.12, 0.63]
king  → [0.81, -0.42, 0.73, 0.25]
queen → [0.78, -0.39, 0.75, 0.28]
car   → [-0.21, 0.84, 0.12, -0.63]
```

The actual numbers are learned from data.

---

# 4. The main idea behind Word2Vec

The fundamental idea is:

> **Words that occur in similar contexts tend to have similar meanings.**

This is called the **distributional hypothesis**.

For example:

```text
I drank coffee in the morning.
I drank tea in the morning.
```

`coffee` and `tea` occur in similar contexts:

```text
drank ___ in the morning
```

Therefore Word2Vec learns similar representations for them.

Similarly:

```text
The dog is running.
The cat is running.
The puppy is running.
```

The words:

```text
dog
cat
puppy
```

appear in similar contexts.

Their embeddings become similar.

---

# 5. How Word2Vec actually works

Word2Vec is based on a small neural network.

But importantly:

> **The neural network is mainly used to learn the word embeddings.**

The learned weights become the word vectors.

There are two major Word2Vec architectures:

```text
                Word2Vec
                   │
          ┌────────┴────────┐
          ↓                 ↓
       CBOW               Skip-Gram
       │                    │
Predict target          Predict context
from context            from target
```

---

# 6. CBOW

**CBOW = Continuous Bag of Words**

CBOW predicts the **center/target word from surrounding context words**.

Suppose:

```text
"The cat is sitting on the mat"
```

Take:

```text
"The cat is sitting"
```

and target:

```text
"on"
```

depending on the chosen context window.

For example:

```text
The cat is [sitting] on
```

If `sitting` is the target:

```text
Context:
The
cat
is
on

Target:
sitting
```

The model learns:

```text
context words → target word
```

---

# 7. CBOW Example

Sentence:

```text
I love machine learning
```

Suppose window size = 1.

For:

```text
I love machine
```

target:

```text
love
```

Context:

```text
I
machine
```

Training example:

```text
Context             Target

I + machine   →     love
```

Another example:

```text
love + learning → machine
```

The model learns to predict the missing/center word.

---

# 8. CBOW architecture

Conceptually:

```text
Context Words
     │
     ├──────────────┐
     │              │
     ▼              ▼
   Word            Word
   Vector          Vector
     │              │
     └──────┬───────┘
            ↓
       Average/Sum
            ↓
      Hidden Layer
            ↓
       Output Layer
            ↓
      Probability
            ↓
      Target Word
```

Suppose:

```text
Context:
["I", "machine"]
```

Their embeddings are:

```text
I       → [0.2, 0.5, 0.1]
machine → [0.4, 0.3, 0.7]
```

CBOW might combine them:

```text
[0.2, 0.5, 0.1]
+
[0.4, 0.3, 0.7]
----------------
[0.6, 0.8, 0.8]
```

or use their average.

Then it predicts:

```text
love
```

---

# 9. Skip-Gram

Skip-Gram works in the opposite direction.

Instead of:

```text
Context → Target
```

it uses:

```text
Target → Context
```

For example:

```text
I love machine learning
```

Target:

```text
machine
```

Context:

```text
love
learning
```

Training examples become:

```text
machine → love
machine → learning
```

With a larger context window:

```text
I love machine learning with Python
```

Target:

```text
machine
```

could produce:

```text
machine → I
machine → love
machine → learning
machine → with
```

depending on the window size.

---

# 10. CBOW vs Skip-Gram

| Feature | CBOW | Skip-Gram |
|---|---|---|
| Full form | Continuous Bag of Words | Skip-Gram |
| Input | Context words | Target word |
| Output | Target word | Context words |
| Training | Faster | Usually slower |
| Common words | Works well | Works well |
| Rare words | Less effective | Often better |
| Small datasets | Often useful | Can work well |
| Basic idea | Context → word | Word → context |

A useful memory trick:

```text
CBOW:

Context → Word


Skip-Gram:

Word → Context
```

---

# 11. How does Word2Vec learn the vectors?

This is the most important part.

Suppose:

```text
Vocabulary:

["I", "love", "cats", "dogs"]
```

Each word initially gets a random vector.

For example:

```text
cat → [0.13, -0.42, 0.71]
dog → [-0.31, 0.25, 0.19]
```

These random vectors have no useful meaning initially.

The model trains on many examples.

For example:

```text
cat → animal
cat → pet
cat → cute
dog → animal
dog → pet
dog → cute
```

The neural network repeatedly adjusts its weights so that it becomes better at predicting context words.

During this process, the word vectors change.

Eventually:

```text
cat → [0.72, 0.31, 0.55]
dog → [0.69, 0.34, 0.52]
```

They become similar because they have similar contexts.

---

# 12. The neural network

A simplified Word2Vec model looks like:

```text
          Input
            │
            ▼
       One-Hot Vector
            │
            ▼
      Hidden Layer
       (Embeddings)
            │
            ▼
       Output Layer
            │
            ▼
          Softmax
            │
            ▼
    Word probabilities
```

Suppose vocabulary size is:

```text
10,000
```

The input could be:

```text
[0, 0, 0, 1, 0, ..., 0]
```

This is a one-hot vector.

The model has a weight matrix:

```text
W
```

with dimensions:

```text
10,000 × 300
```

if embedding dimension = 300.

The multiplication:

```text
one_hot × W
```

essentially selects one row of `W`.

That row becomes the word's embedding.

This is a beautiful insight:

> **The Word2Vec embedding is essentially learned from the neural network's weight matrix.**

---

# 13. Word2Vec training mathematically

For Skip-Gram, suppose we have:

```text
center word = w
context word = c
```

We want:

\[
P(c|w)
\]

to be high.

The basic objective is:

\[
\max \sum_{t=1}^{T}
\sum_{-c \leq j \leq c,j\neq0}
\log P(w_{t+j}|w_t)
\]

where:

- \(T\) = number of words
- \(c\) = context window
- \(w_t\) = current word
- \(w_{t+j}\) = surrounding word

In simple terms:

> Make the probability of actual context words high.

---

# 14. Softmax

The traditional Word2Vec model can use Softmax to calculate:

\[
P(context|word)
\]

For vocabulary \(V\):

\[
P(w_O|w_I)
=
\frac{e^{v_{w_O}^{T}v_{w_I}}}
{\sum_{w=1}^{V}e^{v_w^Tv_{w_I}}}
\]

Where:

- \(w_I\) = input word
- \(w_O\) = output/context word
- \(v_{w_I}\) = input embedding
- \(v_{w_O}\) = output embedding
- \(V\) = vocabulary size

The problem is that if:

```text
Vocabulary = 1,000,000 words
```

we would need to calculate probabilities for a million words for every training example.

That's expensive.

Word2Vec therefore introduced important optimization techniques.

---

# 15. Negative Sampling

**Negative Sampling** is one of the most important Word2Vec concepts.

Instead of predicting the probability of every word in the vocabulary, we train the model to distinguish:

```text
real context
```

from:

```text
random/negative context
```

Suppose:

```text
"The cat drinks milk"
```

Positive example:

```text
cat → milk
```

Negative examples could be:

```text
cat → computer
cat → airplane
cat → mountain
cat → database
```

The model learns:

```text
cat + milk       → 1
cat + computer   → 0
cat + airplane   → 0
cat + mountain   → 0
```

So instead of calculating a huge Softmax over the entire vocabulary, it only needs to evaluate a small number of positive and negative examples.

---

# 16. Why negative sampling works

The model is effectively learning:

> "Which words are likely to appear together?"

For example:

```text
        cat
       /   \
      /     \
   milk     pet
     ✓       ✓
```

while:

```text
cat → database
cat → airplane
```

are unlikely in the same context.

Repeated training causes meaningful relationships to emerge.

---

# 17. Word2Vec and semantic relationships

One of the famous observations about Word2Vec is that vector arithmetic can capture relationships.

For example:

\[
king - man + woman \approx queen
\]

Conceptually:

```text
king
 -
man
 +
woman
 =
queen
```

This happens because the learned vector space captures certain linguistic relationships.

Another example might be:

```text
Paris - France + Italy ≈ Rome
```

The exact quality depends heavily on the training corpus and model.

---

# 18. Similarity between words

Once you have word embeddings, you can calculate similarity.

The most common metric is **cosine similarity**.

\[
\cos(\theta)
=
\frac{A\cdot B}
{\|A\|\|B\|}
\]

For example:

```text
vector("dog")
vector("cat")
```

might have:

```text
Cosine similarity = 0.82
```

while:

```text
vector("dog")
vector("database")
```

might have:

```text
Cosine similarity = 0.10
```

So:

```text
higher cosine similarity
        ↓
more similar direction
        ↓
often more semantically/contextually related
```

---

# 19. Training Word2Vec with Gensim

The easiest way to experiment with Word2Vec in Python is usually **Gensim**.

Install:

```bash
pip install gensim
```

Then:

```python
from gensim.models import Word2Vec
```

Suppose your data is:

```python
sentences = [
    ["i", "love", "machine", "learning"],
    ["i", "love", "deep", "learning"],
    ["machine", "learning", "is", "interesting"],
    ["deep", "learning", "is", "powerful"]
]
```

Train:

```python
model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)
```

---

# 20. Important parameters

### `vector_size`

```python
vector_size=100
```

Determines the number of dimensions of each word vector.

For example:

```text
vector_size = 100

"machine" →
[0.12, -0.43, 0.51, ..., 0.27]
```

Common values:

```text
50
100
200
300
```

Larger dimensions can capture more information but require more computation and data.

---

### `window`

```python
window=5
```

Determines how many words around the target word are considered as context.

For:

```text
I love machine learning with Python
```

a larger window captures broader context.

Small window:

```text
local/syntactic relationships
```

Larger window:

```text
broader semantic relationships
```

---

### `min_count`

```python
min_count=2
```

Ignores words that appear fewer than 2 times.

For a large corpus, this is useful because extremely rare words may not provide enough information.

For a tiny demonstration dataset:

```python
min_count=1
```

is appropriate.

---

### `workers`

```python
workers=4
```

Number of CPU threads used for training.

---

### `sg`

This determines CBOW vs Skip-Gram.

```python
sg=0
```

means:

```text
CBOW
```

while:

```python
sg=1
```

means:

```text
Skip-Gram
```

For example:

```python
model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=2,
    sg=1
)
```

uses Skip-Gram.

---

# 21. Getting the vector of a word

After training:

```python
vector = model.wv["machine"]
```

Then:

```python
print(vector)
```

You might see:

```text
[ 0.012, -0.431, 0.217, ...]
```

Check its dimensions:

```python
print(vector.shape)
```

Output:

```text
(100,)
```

because:

```python
vector_size=100
```

---

# 22. Finding similar words

This is one of the coolest things you can do.

```python
model.wv.most_similar("machine")
```

You might get:

```text
[
    ("learning", 0.82),
    ("computer", 0.74),
    ("deep", 0.71),
    ...
]
```

The second value is similarity.

---

# 23. Word similarity

You can directly calculate:

```python
model.wv.similarity(
    "machine",
    "learning"
)
```

Example:

```text
0.82
```

And:

```python
model.wv.similarity(
    "machine",
    "banana"
)
```

might be:

```text
0.05
```

Again, the actual results depend on the corpus.

---

# 24. Word2Vec on your review dataset

Since you're currently working with **review data**, suppose:

```python
df["review"]
```

contains:

```text
"I loved this movie"
"This movie was terrible"
"Great acting and story"
...
```

First tokenize the reviews.

For example:

```python
sentences = df["review"].apply(
    lambda x: x.lower().split()
).tolist()
```

Now:

```python
sentences
```

looks like:

```python
[
    ["i", "loved", "this", "movie"],
    ["this", "movie", "was", "terrible"],
    ["great", "acting", "and", "story"]
]
```

Then:

```python
from gensim.models import Word2Vec

word2vec_model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=1
)
```

Now you have learned embeddings from your reviews.

---

# 25. But Word2Vec doesn't directly give one vector for a whole review

This is a very important distinction.

Word2Vec gives:

```text
word → vector
```

not:

```text
review → vector
```

For example:

```text
"I love this movie"
```

becomes:

```text
I       → vector
love    → vector
this    → vector
movie   → vector
```

So you have multiple vectors.

You need another method to combine them into one review vector.

---

# 26. Simple way: Average Word2Vec vectors

Suppose:

```text
I      → [0.1, 0.2, 0.3]
love   → [0.7, 0.8, 0.2]
movie  → [0.4, 0.1, 0.6]
```

You can calculate:

\[
review\ vector =
\frac{v_1+v_2+v_3}{3}
\]

giving:

```text
[0.4, 0.37, 0.37]
```

In Python:

```python
import numpy as np

def document_vector(doc):
    vectors = []

    for word in doc:
        if word in word2vec_model.wv:
            vectors.append(word2vec_model.wv[word])

    if len(vectors) == 0:
        return np.zeros(word2vec_model.vector_size)

    return np.mean(vectors, axis=0)
```

Then:

```python
review_vector = document_vector(
    ["i", "love", "this", "movie"]
)

print(review_vector)
```

Now:

```text
review
   ↓
words
   ↓
Word2Vec
   ↓
word vectors
   ↓
average
   ↓
one review vector
```

---

# 27. Using Word2Vec vectors for ML

Now you can create:

```python
X = np.array([
    document_vector(review)
    for review in sentences
])
```

Suppose you have:

```text
10,000 reviews
```

and:

```python
vector_size=100
```

Then:

```python
X.shape
```

will be approximately:

```text
(10000, 100)
```

Now this becomes your ML feature matrix.

You can do:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    df["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"]
)
```

Then:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)
```

Now Word2Vec has become a **feature extraction method** for your sentiment classifier.

---

# 28. Word2Vec vs TF-IDF

This is particularly relevant to what you were learning earlier.

| Feature | TF-IDF | Word2Vec |
|---|---|---|
| Representation | Sparse | Dense |
| Vector for | Documents/words | Words |
| Semantic relationships | Limited | Much better |
| Vector dimensions | Vocabulary size | User-defined |
| Context understanding | No | Context-based |
| Word similarity | Limited | Strong |
| Unknown words | Problem | Problem |
| Training required | No | Yes |
| Typical use | Classical ML | Embeddings/Deep Learning |

For example:

### TF-IDF

```text
review
 ↓
[0, 0.32, 0, 0.82, 0, ...]
 ↓
Logistic Regression
```

### Word2Vec

```text
review
 ↓
words
 ↓
word embeddings
 ↓
combine embeddings
 ↓
[0.21, -0.43, 0.67, ...]
 ↓
Logistic Regression / Neural Network
```

---

# 29. Word2Vec's biggest limitation

Word2Vec creates **one fixed vector for each word**.

This means the word:

```text
bank
```

has one vector.

But consider:

```text
I deposited money in the bank.
```

and:

```text
We sat beside the river bank.
```

The meaning of `bank` is different.

Traditional Word2Vec doesn't dynamically change the vector based on the sentence.

It essentially learns:

```text
bank → one vector
```

This is called a **static embedding**.

Modern models such as:

```text
ELMo
BERT
RoBERTa
GPT-style models
```

produce **contextual representations**, where the representation of a word can change according to its surrounding words.

---

# 30. Word2Vec → GloVe → FastText → BERT

A useful progression for your NLP learning is:

```text
One-Hot Encoding
       ↓
Bag of Words
       ↓
TF-IDF
       ↓
Word2Vec
       ↓
GloVe
       ↓
FastText
       ↓
ELMo
       ↓
BERT
       ↓
Transformers
       ↓
Modern LLMs
```

Conceptually, you're moving from:

```text
"Does this word exist?"
```

toward:

```text
"What does this word mean based on its context?"
```

---

# 31. The most important things to remember

If you're preparing this as an NLP topic, remember these **7 points**:

### 1. Word2Vec is a word embedding technique

It converts:

```text
word → dense numerical vector
```

### 2. It learns from context

```text
Similar contexts
       ↓
Similar vectors
```

### 3. It has two architectures

```text
CBOW:
Context → Target


Skip-Gram:
Target → Context
```

### 4. Word vectors are learned

They are not manually assigned.

### 5. Negative Sampling makes training efficient

Instead of calculating probabilities for the entire vocabulary, it trains using a small number of positive and negative examples.

### 6. Word2Vec captures semantic relationships

For example:

```text
king - man + woman ≈ queen
```

### 7. Word2Vec gives word vectors, not automatically sentence vectors

For reviews:

```text
Review
 ↓
Tokenize
 ↓
Word2Vec
 ↓
Individual word vectors
 ↓
Average/weighted combination
 ↓
Review vector
 ↓
ML model
```

---

## The key difference from what you just learned

You previously had:

```text
Review
   ↓
TF-IDF
   ↓
X_train
   ↓
Logistic Regression
```

With Word2Vec:

```text
Review
   ↓
Tokenization
   ↓
Word2Vec
   ↓
Word vectors
   ↓
Combine word vectors
   ↓
X_train
   ↓
Logistic Regression / Neural Network
```

**That distinction—Word2Vec produces vectors for individual words, while your classifier needs a representation for the entire review—is the most important concept to understand before moving to FastText or BERT.**