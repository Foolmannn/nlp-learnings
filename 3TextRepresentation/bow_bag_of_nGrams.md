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

# 21. CountVectorizer with N-grams

Let's look at a practical example:

```python
from sklearn.feature_extraction.text import CountVectorizer

text = [
    "this movie is very good",
    "this movie is not good",
    "this movie is very bad"
]

vectorizer = CountVectorizer(
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(text)

print(vectorizer.get_feature_names_out())
```

Features will include things such as:

```text
this
movie
is
very
good
not
bad

this movie
movie is
is very
very good
is not
not good
very bad
```

This gives the model more contextual information.

---

# 22. BoW vs Bag of N-grams

| Feature | Bag of Words | Bag of N-grams |
|---|---|---|
| Basic unit | Words | Word sequences |
| N | Usually 1 | 1, 2, 3, ... |
| Word order | Mostly ignored | Partially captured |
| Context | Low | Better |
| Features | Fewer | More |
| Dimensionality | Lower | Higher |
| Sparsity | High | Usually higher |
| Computational cost | Lower | Higher |
| Captures phrases | No | Yes |
| Example | `good` | `very good` |

---

# 23. Important Difference

Consider:

```text
"I do not like this movie"
```

### Bag of Words

Features:

```text
I
do
not
like
this
movie
```

It does not directly represent:

```text
do not like
```

### Bag of Bigrams

Features:

```text
I do
do not
not like
like this
this movie
```

Now the model gets information about local word relationships.

---

# 24. BoW vs Bigram: Important Example

Consider:

```text
"The food is good"
"The food is not good"
```

With unigrams:

```text
good
not
```

With bigrams:

```text
is good
is not
not good
```

The bigram:

```text
not good
```

can be highly useful for sentiment classification.

Therefore:

```text
Unigram → individual word information
Bigram  → two-word contextual information
Trigram → three-word contextual information
```

---

# 25. Advantages of Bag of Words

### 1. Simple

Very easy to understand and implement.

### 2. Fast

Compared with many modern NLP representations, BoW is computationally inexpensive.

### 3. Interpretable

You can directly see which words correspond to which features.

### 4. Works well for many classical ML tasks

For example:

- Spam detection
- Sentiment analysis
- Document classification
- Topic classification
- News classification

### 5. Works with traditional ML algorithms

For example:

```text
BoW
 ↓
Logistic Regression
 ↓
Classification
```

or:

```text
BoW
 ↓
Naive Bayes
 ↓
Spam Detection
```

---

# 26. Disadvantages of Bag of Words

### 1. Ignores word order

```text
Dog bites man
```

and

```text
Man bites dog
```

can have the same representation.

### 2. Doesn't understand semantics

These words are treated as independent features:

```text
car
automobile
vehicle
```

BoW doesn't inherently know that they are semantically related.

### 3. High dimensionality

Large vocabulary means large vectors.

### 4. Sparse representation

Most elements are zero.

### 5. Unknown words

Words not present in the training vocabulary may be ignored or mapped to no feature.

---

# 27. Advantages of Bag of N-grams

### 1. Captures local word order

For example:

```text
not good
```

### 2. Captures phrases

Examples:

```text
machine learning
deep learning
natural language
New York
```

### 3. Often improves classification

Especially in:

- Sentiment analysis
- Spam detection
- Text classification
- Intent classification

### 4. Still relatively simple

You can implement it using:

```python
CountVectorizer
```

---

# 28. Disadvantages of N-grams

### 1. Very large vocabulary

Suppose you have:

```text
100,000 words
```

The number of possible bigrams can become extremely large.

Theoretical possibilities are roughly:

```text
100,000²
```

although only a fraction actually occurs.

### 2. More sparse

More features generally means more zeros.

### 3. More memory

The feature matrix becomes larger.

### 4. More computationally expensive

Training and vectorization can become slower.

### 5. Limited context

A trigram only captures three consecutive words.

For example:

```text
The movie that I watched yesterday was excellent
```

A small n-gram cannot fully understand the relationship between:

```text
movie
```

and:

```text
excellent
```

when they are far apart.

---

# 29. N-gram Size Trade-off

Increasing N gives more context but also increases complexity.

```text
Unigram
   ↓
less context
   ↓
fewer features
   ↓
faster
```

while:

```text
Bigram
   ↓
more context
   ↓
more features
```

and:

```text
Trigram
   ↓
even more context
   ↓
even more features
```

Eventually:

```text
large N
   ↓
huge vocabulary
   ↓
sparse matrix
   ↓
high computational cost
```

Therefore, we don't simply choose the largest possible N.

---

# 30. Word N-grams vs Character N-grams

N-grams can be created from **words** or **characters**.

## Word N-gram

Sentence:

```text
I love NLP
```

Bigrams:

```text
I love
love NLP
```

---

## Character N-gram

For:

```text
hello
```

Character bigrams might be:

```text
he
el
ll
lo
```

Character trigrams:

```text
hel
ell
llo
```

Character n-grams are useful for:

- Spelling variations
- Morphology
- Short text
- Noisy text
- Social-media text
- Languages with complex word formation

In scikit-learn:

```python
CountVectorizer(
    analyzer="char",
    ngram_range=(2, 3)
)
```

---

# 31. Word N-gram vs Character N-gram

| Feature | Word N-gram | Character N-gram |
|---|---|---|
| Unit | Words | Characters |
| Example | `not good` | `no`, `ot`, `t ` |
| Captures | Word context | Subword patterns |
| Vocabulary | Word-based | Character-based |
| Handles spelling variation | Poorer | Better |
| Interpretability | Higher | Lower |
| Common use | Text classification | Noisy/morphological text |

---

# 32. BoW, N-grams and TF-IDF

These concepts are related but shouldn't be confused.

### Bag of Words

Defines the **features**:

```text
words
```

### N-grams

Defines the **features** as sequences:

```text
word
word word
word word word
```

### TF-IDF

Defines **how important each feature is**.

For example:

```text
Feature          Weight
-----------------------
machine          0.31
learning         0.25
machine learning 0.67
```

So you can have:

```text
TF-IDF + unigrams
```

or:

```text
TF-IDF + unigrams + bigrams
```

using:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2)
)
```

---

# 33. Complete Text Representation Pipeline

For classical NLP, a common pipeline is:

```text
Raw Text
   ↓
Lowercase
   ↓
Remove unnecessary characters
   ↓
Tokenization
   ↓
Stopword handling
   ↓
Stemming / Lemmatization
   ↓
N-gram generation
   ↓
Vectorization
   ↓
Machine Learning Model
```

For example:

```text
"I really love this amazing movie!"
```

After preprocessing:

```text
"really love amazing movie"
```

Generate unigrams:

```text
really
love
amazing
movie
```

Generate bigrams:

```text
really love
love amazing
amazing movie
```

Then vectorize:

```text
[1, 1, 1, 1, 1, 1, 1]
```

The actual values depend on whether you're using counts, binary values, TF-IDF, etc.

---

# 34. When Should You Use What?

A useful rule of thumb:

### Use Unigram/BoW when:

- Dataset is large
- You need simplicity
- Interpretability is important
- Word order isn't extremely important
- You want a strong classical baseline

### Use Unigram + Bigram when:

- Context matters
- Sentiment analysis
- Text classification
- Phrase detection
- Intent classification

### Use Trigrams when:

- Specific phrases matter
- You have enough training data
- Computational resources are sufficient

Don't automatically use:

```python
ngram_range=(1, 5)
```

because this can dramatically increase the feature space.

---

# 35. A Practical NLP Example

Suppose we're building a movie-review classifier.

Training data:

```text
"This movie is excellent"
"This movie is amazing"
"This movie is not good"
"This movie is very boring"
```

### Unigram features

```text
movie
excellent
amazing
not
good
very
boring
```

The model can learn:

```text
excellent → positive
amazing   → positive
boring    → negative
```

But the phrase:

```text
not good
```

is important.

Using bigrams:

```text
this movie
movie is
is excellent
is amazing
is not
not good
very boring
```

Now:

```text
not good → negative
very boring → negative
```

can be learned directly.

---

# 36. Important Exam Definition

If asked:

### "What is Bag of Words?"

You can write:

> **Bag of Words (BoW)** is a text representation technique in NLP that converts a collection of documents into numerical vectors based on the occurrence or frequency of words in a predefined vocabulary. It ignores grammatical structure and word order and represents each document using a vector of word counts or binary indicators.

---

### "What is Bag of N-grams?"

> **Bag of N-grams** is an extension of the Bag-of-Words model in which text is represented using sequences of N consecutive tokens rather than only individual words. Unigrams represent single words, bigrams represent pairs of consecutive words, and trigrams represent three consecutive words. N-grams capture limited local word-order and contextual information but generally increase the dimensionality and sparsity of the feature space.

---

# 37. The Big Picture

You can remember the progression like this:

```text
                 TEXT
                   │
                   ▼
          ┌─────────────────┐
          │   Tokenization  │
          └────────┬────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ Text Representation │
        └──────────┬──────────┘
                   │
       ┌───────────┼────────────┐
       │           │            │
       ▼           ▼            ▼
     BoW        N-grams       TF-IDF
       │           │
       │     ┌─────┼─────┐
       │     │     │     │
       │  Unigram Bigram Trigram
       │
       └───────────┬────────────
                   ▼
             Numerical Matrix
                   │
                   ▼
          Machine Learning Model
```

The key idea is:

> **BoW asks: "Which words are present and how frequently?"**

> **N-grams ask: "Which consecutive word combinations are present and how frequently?"**

And the most important limitation to remember is:

```text
BoW
→ loses word order

N-grams
→ captures only LOCAL word order

Modern embeddings / Transformers
→ capture much richer semantic and contextual relationships
```

So **Bag of Words → N-grams → TF-IDF → Word Embeddings → Contextual Embeddings/Transformers** is a useful progression to understand the evolution of NLP text representation.