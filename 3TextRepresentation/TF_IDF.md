# TF-IDF in NLP — Detailed Explanation

**TF-IDF (Term Frequency–Inverse Document Frequency)** is one of the most important classical techniques for **text representation** in NLP.

It converts text into numerical vectors while trying to answer:

> **"How important is this word to this particular document compared with the entire collection of documents?"**

Unlike simple Bag of Words, TF-IDF does **not treat every word equally**. A word that appears many times in one document but rarely across the whole collection receives a higher weight.

---

# 1. Why Do We Need TF-IDF?

Consider these documents:

```text
D1: I love machine learning
D2: I love deep learning
D3: I love natural language processing
```

Words such as:

```text
I
love
```

appear in many documents.

They don't tell us much about what makes a particular document different.

But:

```text
machine
deep
natural
language
processing
```

are more specific.

A simple Bag of Words might produce:

| Document | I | love | machine | learning | deep |
|---|---:|---:|---:|---:|---:|
| D1 | 1 | 1 | 1 | 1 | 0 |
| D2 | 1 | 1 | 0 | 1 | 1 |
| D3 | 1 | 1 | 0 | 0 | 0 |

BoW only tells us **frequency**.

TF-IDF tries to determine **importance**.

---

# 2. Main Idea of TF-IDF

TF-IDF combines two concepts:

### TF

**Term Frequency**

> How frequently does a term occur in a particular document?

### IDF

**Inverse Document Frequency**

> How rare is that term across the entire collection of documents?

Then:

\[
\boxed{TFIDF = TF \times IDF}
\]

Therefore:

```text
High TF + Rare across documents
             ↓
        High TF-IDF
```

while:

```text
Common across documents
             ↓
         Low IDF
             ↓
        Low TF-IDF
```

---

# 3. Understanding Term Frequency (TF)

Term Frequency measures how often a word occurs in a document.

Suppose:

```text
D1 = "machine learning is useful and machine learning is powerful"
```

The word:

```text
machine
```

occurs twice.

The word:

```text
learning
```

also occurs twice.

The word:

```text
powerful
```

occurs once.

---

## 3.1 Basic TF

The simplest definition is:

\[
TF(t,d) = \text{number of times term }t\text{ occurs in document }d
\]

For example:

```text
machine → 2
learning → 2
powerful → 1
```

However, raw counts can be problematic because longer documents naturally contain more words.

---

# 4. Normalized Term Frequency

A common TF formula is:

\[
TF(t,d)=
\frac{\text{Number of occurrences of term }t}
{\text{Total number of terms in document }d}
\]

Suppose:

```text
Document = 10 words
```

and:

```text
machine = 2 times
```

Then:

\[
TF(machine,D)=\frac{2}{10}=0.2
\]

If:

```text
learning = 1
```

then:

\[
TF(learning,D)=\frac{1}{10}=0.1
\]

---

# 5. Why Normalize TF?

Suppose we have two documents.

### Document 1

```text
100 words
machine appears 10 times
```

### Document 2

```text
10 words
machine appears 3 times
```

Raw frequency:

```text
D1 → 10
D2 → 3
```

It looks like `machine` is more important in D1.

But normalized TF:

\[
TF(D1)=\frac{10}{100}=0.10
\]

\[
TF(D2)=\frac{3}{10}=0.30
\]

So relative to document size, `machine` is actually more frequent in D2.

---

# 6. Inverse Document Frequency (IDF)

TF alone isn't enough.

Suppose the word:

```text
the
```

appears 500 times in one document.

That doesn't necessarily make it an important word.

If `the` appears in almost every document, it doesn't help distinguish documents.

This is where **IDF** comes in.

---

# 7. What is Document Frequency?

Suppose we have 5 documents:

```text
D1
D2
D3
D4
D5
```

and the word:

```text
learning
```

appears in:

```text
D1
D2
D4
```

Then:

\[
DF(learning)=3
\]

The **document frequency** is the number of documents containing the term.

---

# 8. Inverse Document Frequency

The basic idea is:

\[
IDF(t)=\log\left(\frac{N}{DF(t)}\right)
\]

where:

- \(N\) = total number of documents
- \(DF(t)\) = number of documents containing term \(t\)

---

# 9. Why "Inverse"?

Because we want:

```text
Common word
     ↓
High DF
     ↓
Low IDF
```

and:

```text
Rare word
     ↓
Low DF
     ↓
High IDF
```

For example, suppose there are:

```text
N = 100 documents
```

### Word A

Appears in:

```text
100 documents
```

\[
IDF(A)=\log(100/100)
\]

\[
=\log(1)=0
\]

So its IDF is:

```text
0
```

It provides almost no discriminating information.

---

### Word B

Appears in:

```text
10 documents
```

\[
IDF(B)=\log(100/10)
\]

\[
=\log(10)
\]

This is much higher.

Therefore, word B receives more importance.

---

# 10. Complete TF-IDF Formula

The basic formula is:

\[
\boxed{
TFIDF(t,d)=TF(t,d)\times IDF(t)
}
\]

with:

\[
TF(t,d)=
\frac{f_{t,d}}
{\sum_k f_{k,d}}
\]

and:

\[
IDF(t)=
\log\left(\frac{N}{DF(t)}\right)
\]

Therefore:

\[
\boxed{
TFIDF(t,d)=
\frac{f_{t,d}}{\sum_k f_{k,d}}
\times
\log\left(\frac{N}{DF(t)}\right)
}
\]

---

# 11. Complete Numerical Example

Let's use three documents.

```text
D1 = "cat likes milk"
D2 = "cat likes fish"
D3 = "dog likes fish"
```

Our vocabulary is:

```text
cat
likes
milk
fish
dog
```

---

## Step 1: Calculate TF

Each document contains 3 words.

For D1:

```text
cat   = 1
likes = 1
milk  = 1
```

Normalized TF:

\[
TF=\frac{1}{3}=0.333
\]

For words not present:

\[
TF=0
\]

So:

| Word | D1 TF |
|---|---:|
| cat | 0.333 |
| likes | 0.333 |
| milk | 0.333 |
| fish | 0 |
| dog | 0 |

---

# 12. Calculate DF

We have:

```text
cat
```

appears in:

```text
D1
D2
```

Therefore:

\[
DF(cat)=2
\]

---

`likes` appears in:

```text
D1
D2
D3
```

Therefore:

\[
DF(likes)=3
\]

---

`milk` appears only in:

```text
D1
```

Therefore:

\[
DF(milk)=1
\]

---

`fish` appears in:

```text
D2
D3
```

Therefore:

\[
DF(fish)=2
\]

---

`dog` appears only in:

```text
D3
```

Therefore:

\[
DF(dog)=1
\]

---

# 13. Calculate IDF

Total documents:

\[
N=3
\]

Using:

\[
IDF(t)=\log\left(\frac{N}{DF(t)}\right)
\]

### cat

\[
IDF(cat)=\log(3/2)
\]

Approximately:

\[
0.405
\]

### likes

\[
IDF(likes)=\log(3/3)
\]

\[
=\log(1)=0
\]

### milk

\[
IDF(milk)=\log(3/1)
\]

\[
\approx1.099
\]

### fish

\[
IDF(fish)=\log(3/2)
\]

\[
\approx0.405
\]

### dog

\[
IDF(dog)=\log(3/1)
\]

\[
\approx1.099
\]

So:

| Word | DF | IDF |
|---|---:|---:|
| cat | 2 | 0.405 |
| likes | 3 | 0 |
| milk | 1 | 1.099 |
| fish | 2 | 0.405 |
| dog | 1 | 1.099 |

Notice something important:

```text
likes
```

appears in **every document**, so:

\[
IDF(likes)=0
\]

Therefore its TF-IDF becomes zero.

This is exactly what we want: a word that occurs everywhere isn't useful for distinguishing documents.

---

# 14. Calculate TF-IDF for D1

D1:

```text
cat likes milk
```

### cat

\[
TFIDF(cat,D1)
=
0.333\times0.405
\]

\[
\approx0.135
\]

### likes

\[
TFIDF(likes,D1)
=
0.333\times0
\]

\[
=0
\]

### milk

\[
TFIDF(milk,D1)
=
0.333\times1.099
\]

\[
\approx0.366
\]

Therefore:

| Word | TF | IDF | TF-IDF |
|---|---:|---:|---:|
| cat | 0.333 | 0.405 | 0.135 |
| likes | 0.333 | 0 | 0 |
| milk | 0.333 | 1.099 | 0.366 |

So the vector for D1 is approximately:

```text
[0.135, 0, 0.366, 0, 0]
```

depending on the vocabulary ordering.

Notice:

```text
milk → 0.366
cat  → 0.135
likes → 0
```

`milk` gets a higher weight because it is more specific to D1.

---

# 15. Intuition Behind TF-IDF

The most important intuition is:

### High TF + High IDF

```text
Important word
```

### High TF + Low IDF

```text
Common word
```

### Low TF + High IDF

```text
Potentially useful but occurs infrequently
```

### Low TF + Low IDF

```text
Usually not important
```

You can remember:

> **TF asks "Is this word important inside this document?"**

> **IDF asks "Is this word rare across documents?"**

> **TF-IDF combines both.**

---

# 16. Why Use Log in IDF?

Suppose:

```text
N = 1,000,000
```

and a rare word appears in only:

```text
1 document
```

Raw inverse frequency:

\[
\frac{1,000,000}{1}
=
1,000,000
\]

That would create an extremely large weight.

Using logarithm:

\[
\log(1,000,000)
\]

gives a much smaller value.

Therefore, logarithm **compresses the range of IDF values**.

---

# 17. Smoothing in TF-IDF

In practical implementations, a smoothed IDF formula is often used:

\[
\boxed{
IDF(t)=
\log
\left(
\frac{1+N}{1+DF(t)}
\right)+1
}
\]

Why?

Suppose a term occurs in every document.

Without smoothing:

\[
\log(N/N)=0
\]

With smoothing:

\[
\log\left(\frac{1+N}{1+N}\right)+1
\]

\[
=1
\]

The exact behavior depends on the implementation and normalization.

Scikit-learn's `TfidfVectorizer` uses a smoothed IDF by default.

---

# 18. TF-IDF Normalization

After calculating TF-IDF values, vectors are often normalized.

For example, **L2 normalization**:

\[
\|v\|_2 =
\sqrt{x_1^2+x_2^2+\cdots+x_n^2}
\]

Then:

\[
x_i'=
\frac{x_i}{\|v\|_2}
\]

This makes the vector length equal to approximately:

\[
1
\]

This is particularly useful when comparing documents using **cosine similarity**.

---

# 19. TF-IDF and Cosine Similarity

This is especially important for your movie recommender/NLP learning path.

Suppose we have:

```text
Movie A:
"action superhero adventure"

Movie B:
"superhero action adventure"

Movie C:
"romantic comedy relationship"
```

After TF-IDF:

```text
Movie A → vector
Movie B → vector
Movie C → vector
```

Then we can calculate:

\[
\text{Cosine Similarity}(A,B)
\]

Since A and B contain similar terms, their cosine similarity will be high.

But:

\[
\text{Cosine Similarity}(A,C)
\]

will be low.

So a common classical NLP pipeline is:

```text
Text
 ↓
TF-IDF
 ↓
Numerical vectors
 ↓
Cosine Similarity
 ↓
Similarity / Recommendation
```

---

# 20. TF-IDF with Scikit-learn

The easiest implementation is:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```

Example:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "I love machine learning",
    "I love deep learning",
    "Machine learning is powerful"
]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(X.toarray())
```

---

# 21. Understanding `fit_transform()`

This line:

```python
X = vectorizer.fit_transform(documents)
```

does two things.

### `fit()`

Learns the vocabulary and document statistics:

```text
Vocabulary
DF
IDF
```

### `transform()`

Converts documents into TF-IDF vectors.

So:

```text
fit()
 ↓
Learn vocabulary + IDF
```

and:

```text
transform()
 ↓
Convert text to vectors
```

Together:

```python
fit_transform()
```

---

# 22. Training and Testing Data

This distinction is **very important in machine learning**.

Suppose:

```python
train_documents = [...]
test_documents = [...]
```

Do:

```python
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(train_documents)

X_test = vectorizer.transform(test_documents)
```

Do **not** do:

```python
X_test = vectorizer.fit_transform(test_documents)
```

Why?

Because the vectorizer should learn vocabulary and IDF statistics **only from the training data**.

Otherwise, information from the test set leaks into the model.

This is called:

> **Data leakage**

---
