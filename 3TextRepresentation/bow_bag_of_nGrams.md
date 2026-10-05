# Bag of Words (BoW) and Bag of N-grams in NLP

After **text preprocessing**, one of the most important steps in NLP is converting text into a **numerical representation** so that machine-learning algorithms can process it.

Two fundamental approaches are:

1. **Bag of Words (BoW)**
2. **Bag of N-grams**

Both are **frequency-based text representation techniques**.

---

# 1. Why Do We Need Text Representation?

Machine-learning algorithms cannot directly understand text such as:

> "I love machine learning"

A computer needs numbers.

So we transform:

```text
"I love machine learning"
```

into something like:

```text
[1, 1, 1, 1]
```

or a larger numerical vector.

The general process is:

```text
Raw Text
   ↓
Text Preprocessing
   ↓
Tokenization
   ↓
Vocabulary Creation
   ↓
Text Vectorization
   ↓
Numerical Vectors
   ↓
Machine Learning Model
```

For example:

```text
"I love NLP"
```

could become:

```text
[1, 1, 1]
```

where each position corresponds to a word in the vocabulary.

---

# 2. Bag of Words

## Definition

**Bag of Words (BoW)** is a text representation technique that represents a document using the **frequency or presence of words** in a vocabulary.

The important idea is:

> BoW represents text based on which words occur and how many times they occur, while ignoring grammar and word order.

It is called a **"bag"** because we conceptually throw all words into a bag and don't care about their original order.

For example:

```text
"I love NLP and I love machine learning"
```

The words are:

```text
I
love
NLP
and
I
love
machine
learning
```

BoW mainly cares about:

```text
I       → 2
love    → 2
NLP     → 1
and     → 1
machine → 1
learning→ 1
```

---

# 3. Simple BoW Example

Suppose we have three sentences:

```text
Document 1: I love NLP
Document 2: I love machine learning
Document 3: NLP is interesting
```

## Step 1: Create Vocabulary

Collect unique words:

```text
I
love
NLP
machine
learning
is
interesting
```

Vocabulary:

| Index | Word |
|---:|---|
| 0 | I |
| 1 | love |
| 2 | NLP |
| 3 | machine |
| 4 | learning |
| 5 | is |
| 6 | interesting |

---

## Step 2: Represent Each Document

### Document 1

```text
I love NLP
```

Word counts:

```text
I       = 1
love    = 1
NLP     = 1
machine = 0
learning= 0
is      = 0
interesting = 0
```

Vector:

```text
[1, 1, 1, 0, 0, 0, 0]
```

### Document 2

```text
I love machine learning
```

Vector:

```text
[1, 1, 0, 1, 1, 0, 0]
```

### Document 3

```text
NLP is interesting
```

Vector:

```text
[0, 0, 1, 0, 0, 1, 1]
```

Therefore our dataset becomes:

```text
[
 [1, 1, 1, 0, 0, 0, 0],
 [1, 1, 0, 1, 1, 0, 0],
 [0, 0, 1, 0, 0, 1, 1]
]
```

This is the **Bag-of-Words representation**.

---

# 4. Count Vectorization

The most common implementation of BoW is **CountVectorizer**.

Using scikit-learn:

```python
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "I love NLP",
    "I love machine learning",
    "NLP is interesting"
]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(X.toarray())
```

Possible vocabulary:

```text
['interesting' 'is' 'learning' 'love' 'machine' 'nlp']
```

Output:

```text
[
 [0, 0, 0, 1, 0, 1],
 [0, 0, 1, 1, 1, 0],
 [1, 1, 0, 0, 0, 1]
]
```

The exact column order is determined by the vectorizer.

---

# 5. How CountVectorizer Works

The process is essentially:

```text
Documents
    ↓
Tokenization
    ↓
Vocabulary
    ↓
Count occurrences
    ↓
Create matrix
```

For example:

```text
"I love NLP"
"I love machine learning"
```

Vocabulary:

```text
I
love
NLP
machine
learning
```

Matrix:

| Document | I | love | NLP | machine | learning |
|---|---:|---:|---:|---:|---:|
| D1 | 1 | 1 | 1 | 0 | 0 |
| D2 | 1 | 1 | 0 | 1 | 1 |

This matrix is called a **Document-Term Matrix (DTM)**.

---

# 6. Binary Bag of Words

BoW doesn't necessarily have to represent the number of occurrences.

We can represent only whether a word exists.

For example:

```text
"I love love NLP"
```

Normal count representation:

```text
I     → 1
love  → 2
NLP   → 1
```

Vector:

```text
[1, 2, 1]
```

Binary representation:

```text
I     → 1
love  → 1
NLP   → 1
```

Vector:

```text
[1, 1, 1]
```

Here:

```text
1 = word exists
0 = word doesn't exist
```

In scikit-learn:

```python
vectorizer = CountVectorizer(binary=True)

X = vectorizer.fit_transform(documents)
```

---

# 7. Frequency-Based BoW

There are several ways to represent word information.

### 1. Count

```text
love → 3
```

### 2. Binary

```text
love → 1
```

if it exists.

### 3. TF-IDF

Instead of simply counting words, assign importance based on how common/rare the word is.

```text
Count
   ↓
TF-IDF
```

TF-IDF is another important text representation technique.

---

# 8. Major Problem with Bag of Words

The biggest problem is:

> **BoW completely ignores word order.**

Consider:

```text
Dog bites man.
```

and:

```text
Man bites dog.
```

Their words are:

```text
dog
bites
man
```

Both have exactly the same word counts.

Therefore BoW produces essentially the same representation:

```text
[1, 1, 1]
```

But their meanings are completely different.

```text
Dog bites man
      ≠
Man bites dog
```

This is one of the major limitations of BoW.

---

# 9. Another Problem: Vocabulary Size

Suppose we have:

```text
1,000,000 unique words
```

Then every document may be represented using:

```text
1,000,000 dimensions
```

For example:

```text
Document → [0, 0, 0, 1, 0, 0, ..., 0]
```

Most values will usually be zero.

This is called a **sparse vector**.

---

# 10. What is a Sparse Vector?

Suppose vocabulary contains:

```text
10,000 words
```

but a document contains only:

```text
20 words
```

The vector may look like:

```text
[0, 0, 0, 0, 1, 0, 0, ..., 0, 1, 0]
```

There are thousands of dimensions but only a few non-zero values.

Therefore BoW creates **high-dimensional sparse representations**.

---

# 11. Bag of N-grams

Bag of N-grams extends the Bag-of-Words idea.

Instead of considering only individual words, we consider groups of consecutive words.

These groups are called **N-grams**.

---

# 12. What is an N-gram?

An **N-gram** is a sequence of `N` consecutive tokens.

### Unigram

```text
N = 1
```

Individual words:

```text
I
love
machine
learning
```

### Bigram

```text
N = 2
```

Two consecutive words:

```text
I love
love machine
machine learning
```

### Trigram

```text
N = 3
```

Three consecutive words:

```text
I love machine
love machine learning
```

---

# 13. Example of N-grams

Consider:

```text
"I love machine learning"
```

## Unigrams

```text
I
love
machine
learning
```

## Bigrams

```text
I love
love machine
machine learning
```

## Trigrams

```text
I love machine
love machine learning
```

## 4-grams

```text
I love machine learning
```

---

# 14. Bag of Unigrams

If:

```text
N = 1
```

then Bag of N-grams becomes essentially:

```text
Bag of Words
```

For:

```text
"I love machine learning"
```

we have:

```text
I
love
machine
learning
```

---

# 15. Bag of Bigrams

Suppose our documents are:

```text
D1 = "I love machine learning"
D2 = "I love deep learning"
```

Bigrams:

### D1

```text
I love
love machine
machine learning
```

### D2

```text
I love
love deep
deep learning
```

Vocabulary:

```text
I love
love machine
machine learning
love deep
deep learning
```

Vectors:

| Document | I love | love machine | machine learning | love deep | deep learning |
|---|---:|---:|---:|---:|---:|
| D1 | 1 | 1 | 1 | 0 | 0 |
| D2 | 1 | 0 | 0 | 1 | 1 |

---

# 16. N-grams Capture Some Word Order

This is the major advantage over basic BoW.

Consider:

```text
"The movie is good"
```

Bigrams:

```text
The movie
movie is
is good
```

Now consider:

```text
"The movie is not good"
```

Bigrams:

```text
The movie
movie is
is not
not good
```

The phrase:

```text
not good
```

can provide strong information about sentiment.

A unigram representation sees:

```text
not
good
```

But a bigram representation can capture:

```text
not good
```

as a single feature.

---

# 17. Sentiment Analysis Example

Consider:

```text
"The movie was not good"
```

Unigrams:

```text
the
movie
was
not
good
```

A model may learn:

```text
good → positive
not  → negative
```

But the actual meaning is:

```text
not good → negative
```

Using bigrams gives:

```text
the movie
movie was
was not
not good
```

The model can learn that:

```text
"not good"
```

is associated with negative sentiment.

Similarly:

```text
very good
really good
not bad
very bad
```

can become useful features.

---

# 18. Implementing N-grams with Scikit-learn

`CountVectorizer` supports n-grams using:

```python
ngram_range
```

For unigrams:

```python
vectorizer = CountVectorizer(
    ngram_range=(1, 1)
)
```

For bigrams:

```python
vectorizer = CountVectorizer(
    ngram_range=(2, 2)
)
```

For unigrams + bigrams:

```python
vectorizer = CountVectorizer(
    ngram_range=(1, 2)
)
```

For unigrams + bigrams + trigrams:

```python
vectorizer = CountVectorizer(
    ngram_range=(1, 3)
)
```

---

# 19. Example in Python

```python
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "I love machine learning",
    "I love deep learning"
]

vectorizer = CountVectorizer(
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(X.toarray())
```

The features will include both:

```text
Unigrams:
I
love
machine
learning
deep
```

and:

```text
Bigrams:
I love
love machine
machine learning
love deep
deep learning
```

---

# 20. `ngram_range` Explained

This parameter is extremely important.

```python
ngram_range=(1, 1)
```

means:

```text
Only unigrams
```

---

```python
ngram_range=(2, 2)
```

means:

```text
Only bigrams
```

---

```python
ngram_range=(3, 3)
```

means:

```text
Only trigrams
```

---

```python
ngram_range=(1, 2)
```

means:

```text
Unigrams + Bigrams
```

---

```python
ngram_range=(1, 3)
```

means:

```text
Unigrams + Bigrams + Trigrams
```

---
