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
