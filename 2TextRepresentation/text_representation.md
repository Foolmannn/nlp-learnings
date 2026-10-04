# Text Representation, Text Vectorization, and Feature Extraction in NLP — Detailed Notes

In Natural Language Processing (NLP), text representation, text vectorization, and feature extraction are fundamental techniques that convert human language into numerical data that machine learning algorithms can understand and process.

Since you are starting NLP, we'll build the concepts from the basics to advanced methods, including mathematical intuition, Python implementations, examples, advantages, disadvantages, and practical applications.

# 1. Why do we need text representation in NLP?

Computers and most traditional machine learning algorithms cannot directly perform mathematical operations on raw text.

Suppose we have the following dataset:

| Text                   | Label    |
| ---------------------- | -------- |
| I love this movie      | Positive |
| This movie is terrible | Negative |
| The movie is amazing   | Positive |
| I hate this film       | Negative |

A human can understand the sentiment of each sentence. However, a machine learning model generally needs numerical features such as integers, floating-point numbers, or vectors.

We must transform the text into numbers while preserving as much useful information about its meaning as possible.

Raw text

"I love this movie"

Text preprocessing

Tokenization, normalization, optional stop-word removal

Vectorization / feature extraction

[0.2, 0.8, 0.0, 0.4, ...]

Machine learning model

Sentiment classification, spam detection, topic prediction, etc.

For example, after vectorization, a model may learn that words such as love, amazing, and excellent often appear in positive reviews, while terrible, hate, and awful often appear in negative reviews.

## 1.1 What is text representation?

Text representation is the general process of representing text in a form that a computer can process.

Different representations preserve different kinds of information:

- Word identity: which words occur?
- Word frequency: how often does each word occur?
- Importance: which words are particularly useful?
- Word order: which words come before or after others?
- Semantic meaning: what does the text mean?
- Context: how does a word's meaning change depending on its surroundings?

For example, the word bank can refer to a financial institution or the side of a river. A simple representation treats both uses as the same word, whereas contextual representations can distinguish them.

## 1.2 What is text vectorization?

Text vectorization is the process of converting text into numerical vectors.

A vector is an ordered collection of numbers.

For example:

\\[ \text{"good movie"} \longrightarrow [1,1,0,0] \\]

The numbers might represent whether certain vocabulary words occur in the sentence.

A more sophisticated representation might be:

\\[ \text{"good movie"} \longrightarrow [0.21,-0.35,0.74,0.12,\ldots] \\]

Here, the numbers may encode learned semantic information rather than simple word counts.

## 1.3 What is feature extraction?

Feature extraction is the process of obtaining useful, informative characteristics from raw data and representing them as features that a model can use.

In NLP, features may include:

- Number of positive or negative words.
- Frequency of individual words.
- TF-IDF scores.
- Number of words or characters.
- Word or sentence embeddings.
- Semantic relationships between words.
- Contextual representations produced by a transformer.

For example, from the sentence:

> "This movie is extremely good!"

We could extract:

| Feature                | Value                 |
| ---------------------- | --------------------- |
| Word count             | 5                     |
| Exclamation marks      | 1                     |
| Positive-word count    | 1                     |
| Contains "good"        | 1                     |
| TF-IDF score of "good" | Depends on the corpus |
| Sentence embedding     | A numerical vector    |

### Text representation vs. vectorization vs. feature extraction

These terms overlap, but they emphasize different aspects.

| Term                | Main focus                               | Example                                                     |
| ------------------- | ---------------------------------------- | ----------------------------------------------------------- |
| Text representation | How text is encoded                      | One-hot vector, embedding                                   |
| Text vectorization  | Converting text into numerical vectors   | Bag of Words, TF-IDF                                        |
| Feature extraction  | Obtaining useful information for a model | Word counts, TF-IDF, embeddings                             |
| Feature engineering | Designing or transforming features       | Add text length, punctuation count, sentiment lexicon score |

In practical NLP work, these terms are sometimes used interchangeably. The exact distinction depends on the context.

# 2. Major types of text representation

NLP representations can be divided into several broad categories.

### 1. Basic numerical representations

Represent word identity or occurrence.

One-hot encoding, Bag of Words, n-grams.

### 2. Statistical representations

Represent word importance within a document and corpus.

TF-IDF, frequency-based features.

### 3. Dense word representations

Learn relationships between words in a continuous vector space.

Word2Vec, GloVe, FastText.

### 4. Contextual representations

Represent a word according to its context.

ELMo, BERT, transformer-based embeddings.

### 5. Document and sentence representations

Represent larger pieces of text as vectors.

Averaged word embeddings, Doc2Vec, sentence-transformer embeddings.

Let's examine each method step by step.

# 3. One-Hot Encoding

One-hot encoding is one of the simplest ways to represent words numerically.

The basic idea is:

- Create a vocabulary of unique words.
- Assign one position to each word.
- Represent each word using a vector containing zeros and a single one.
- The position containing `1` identifies the word.

## 3.1 Example

Consider these sentences:

1. `I love NLP`
2. `I love Python`
3. `Python is useful`

The vocabulary is:

\\[ V=[I,\ love,\ NLP,\ Python,\ is,\ useful] \\]

The vocabulary size is \\(6\\).

Each word receives a vector of length \\(6\\).

| Word   | One-hot vector       |
| ------ | -------------------- |
| I      | `[1, 0, 0, 0, 0, 0]` |
| love   | `[0, 1, 0, 0, 0, 0]` |
| NLP    | `[0, 0, 1, 0, 0, 0]` |
| Python | `[0, 0, 0, 1, 0, 0]` |
| is     | `[0, 0, 0, 0, 1, 0]` |
| useful | `[0, 0, 0, 0, 0, 1]` |

The sentence `I love NLP` can be represented by the sequence of vectors:

\\[ [[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0]] \\]

Alternatively, if we aggregate the word indicators, we obtain a sentence-level vector:

\\[ [1,1,1,0,0,0] \\]

The second representation loses word order.

## 3.2 Python implementation

```
from sklearn.preprocessing import OneHotEncoderimport numpy as npwords = np.array([    ["I"],    ["love"],    ["NLP"],    ["Python"]])encoder = OneHotEncoder(sparse_output=False)encoded = encoder.fit_transform(words)print(encoder.categories_)print(encoded)
```

For these four words, the output is a \\(4 \times 4\\) matrix with one `1` per row. The exact column order follows the encoder's learned categories.

## 3.3 Advantages

- Simple and easy to understand.
- Each word has an identifiable representation.
- Useful for categorical variables and small vocabularies.
- Can work well for simple illustrative models.

## 3.4 Disadvantages

1\. High dimensionality

If the vocabulary contains 100,000 words, each one-hot vector has 100,000 elements.

2\. Sparse representation

Most elements are zero.

3\. No semantic similarity

The vectors for `king` and `queen` are as unrelated geometrically as the vectors for `king` and `banana`.

4\. No contextual meaning

The word `bank` gets the same vector in every sentence.

Important: One-hot encoding identifies a word but does not learn its meaning.

# 4. Bag of Words (BoW)

Bag of Words is one of the most important traditional text vectorization techniques.

Instead of representing every word separately, BoW represents an entire document using the frequencies of vocabulary words.

It is called a bag because it generally ignores word order and grammar.

## 4.1 How Bag of Words works

The process is:

1. Collect the documents.
2. Tokenize the text.
3. Build a vocabulary of unique words.
4. Assign each vocabulary word a column.
5. Count how often each word appears in each document.
6. Construct a document-term matrix.

Consider three documents:

- D1: `I love NLP`
- D2: `I love Python`
- D3: `Python is useful`

After lowercasing, the vocabulary is:

\\[ V=[i,love,nlp,python,is,useful] \\]

The resulting document-term matrix is:

| Document | i | love | nlp | python | is | useful |
| -------- | - | ---- | --- | ------ | -- | ------ |
| D1       | 1 | 1    | 1   | 0      | 0  | 0      |
| D2       | 1 | 1    | 0   | 1      | 0  | 0      |
| D3       | 0 | 0    | 0   | 1      | 1  | 1      |

Each row represents a document, and each column represents a word.

For D1, the vector is:

\\[ D_1=[1,1,1,0,0,0] \\]

## 4.2 Binary BoW vs. count-based BoW

BoW can use different values for its features.

Suppose the document is:

`good movie good acting good`

Vocabulary:

`good`, `movie`, `acting`

Count-based representation:

\\[ [3,1,1] \\]

Binary representation:

\\[ [1,1,1] \\]

The count-based representation records frequency. The binary representation records only whether a word occurs.

Count-based BoW is the default behavior of scikit-learn's `CountVectorizer`.

## 4.3 Implementing BoW using Python

```
from sklearn.feature_extraction.text import CountVectorizerdocuments = [    "I love NLP",    "I love Python",    "Python is useful"]vectorizer = CountVectorizer()X = vectorizer.fit_transform(documents)print(vectorizer.get_feature_names_out())print(X.toarray())
```

Conceptually, the output is:

```
['is' 'love' 'nlp' 'python' 'useful']
```

The exact vocabulary and column order are determined by the vectorizer. In particular, `CountVectorizer` normally ignores single-character tokens under its default token pattern, so `i` is excluded.

The matrix is:

```
[[0 1 1 0 0]
 [0 1 0 1 0]
 [1 0 0 1 1]]
```

The result has three rows because there are three documents and five columns because the learned vocabulary has five terms.

Notice that we use `fit_transform()`:

- `fit()` learns the vocabulary.
- `transform()` converts documents using that vocabulary.
- `fit_transform()` does both on the supplied training documents.

## 4.4 Advantages of BoW

- Simple and interpretable.
- Efficient for many traditional machine learning models.
- Works well for spam detection, topic classification, and text categorization.
- Easy to implement using scikit-learn.
- Sparse matrices avoid storing every zero explicitly.

## 4.5 Disadvantages of BoW

### A. Loss of word order

Compare:

- `dog bites man`
- `man bites dog`

With unigram BoW, the two documents contain exactly the same word counts.

Thus, BoW cannot distinguish these meanings.

### B. Loss of semantic relationships

The words `excellent` and `wonderful` are treated as separate features without any built-in knowledge that they have similar meanings.

### C. Large vocabulary

Large text collections can produce thousands or millions of features.

### D. Common words may dominate

Very frequent words can consume feature space without providing much discriminatory information. Stop-word filtering and TF-IDF can help.

### E. Unseen words

If a word does not exist in the training vocabulary, a standard BoW vectorizer cannot assign it a new feature during ordinary transformation.

# 5. N-gram representation

N-grams extend BoW by capturing short sequences of neighboring words or characters.

This is useful when word order matters.

## 5.1 What is an n-gram?

An n-gram is a contiguous sequence of \\(n\\) tokens.

| Type             | Meaning               | Example sentence: "I love NLP" |
| ---------------- | --------------------- | ------------------------------ |
| Unigram (1-gram) | One token             | `I`, `love`, `NLP`             |
| Bigram (2-gram)  | Two adjacent tokens   | `I love`, `love NLP`           |
| Trigram (3-gram) | Three adjacent tokens | `I love NLP`                   |
| Four-gram        | Four adjacent tokens  | Requires at least four tokens  |

For the sentence:

`I love natural language processing`

The unigrams are:

`I`, `love`, `natural`, `language`, `processing`

The bigrams are:

`I love`, `love natural`, `natural language`, `language processing`

The trigrams are:

`I love natural`, `love natural language`, `natural language processing`

## 5.2 Why are n-grams useful?

Consider these sentences:

- `The movie is good`
- `The movie is not good`

A unigram representation contains the words `movie`, `is`, `not`, and `good` in the second sentence. A classifier may learn that `good` is associated with positive sentiment but struggle with the negation.

Bigrams can represent:

- `is good`
- `not good`

This gives the model a useful feature associated with the negative phrase.

However, n-grams do not fully understand language. They only capture fixed-length local sequences.

## 5.3 Python implementation

```
from sklearn.feature_extraction.text import CountVectorizerdocuments = [    "the movie is good",    "the movie is not good"]# Unigram modelunigram_vectorizer = CountVectorizer(ngram_range=(1, 1))X_uni = unigram_vectorizer.fit_transform(documents)# Unigram + bigram modelngram_vectorizer = CountVectorizer(ngram_range=(1, 2))X_ngram = ngram_vectorizer.fit_transform(documents)print("Unigrams:")print(unigram_vectorizer.get_feature_names_out())print("\nUnigrams and bigrams:")print(ngram_vectorizer.get_feature_names_out())
```

The parameter `ngram_range=(1, 2)` generates both unigrams and bigrams.

## 5.4 Character n-grams

Instead of splitting text into words, we can extract character sequences.

For example, the word `playing` may produce character trigrams such as:

- `pla`
- `lay`
- `ayi`
- `yin`
- `ing`

Character n-grams are useful for:

- Spelling variation.
- Misspelled text.
- Social media text.
- Morphologically rich languages.
- Language identification.
- Text containing many rare or unseen words.

In scikit-learn, `analyzer="char"` produces character n-grams, while `analyzer="char_wb"` produces character n-grams within word boundaries.

## 5.5 Advantages and disadvantages

Advantages

- Captures some word order.
- Helps model short phrases.
- Useful for sentiment classification.
- Character n-grams can handle spelling variation.

Disadvantages

- The feature space grows quickly.
- Longer n-grams are less frequent.
- Unseen phrases may not be represented.
- It captures local sequences, not long-distance dependencies.

# 6. TF-IDF (Term Frequency–Inverse Document Frequency)

TF-IDF is one of the most important text vectorization techniques in traditional NLP.

BoW counts how often a word occurs. TF-IDF attempts to measure how informative a word is in a document relative to a collection of documents.

The central idea is:

> A word is useful when it occurs frequently in a particular document but is not equally common across the entire collection.

For example, in a collection of movie reviews, words such as `movie` and `film` may occur almost everywhere. Words such as `masterpiece`, `predictable`, or `heartbreaking` might be more useful for distinguishing particular reviews, depending on the corpus.

## 6.1 The two components of TF-IDF

TF-IDF combines two scores:

1. Term Frequency (TF).
2. Inverse Document Frequency (IDF).

\\[ \operatorname{TFIDF}(t,d)=\operatorname{TF}(t,d)\times\operatorname{IDF}(t) \\]

Here:

- \\(t\\) is a term or word.
- \\(d\\) is a document.
- \\(\operatorname{TF}(t,d)\\) measures the term's frequency in that document.
- \\(\operatorname{IDF}(t)\\) measures how rare the term is across the document collection.

## 6.2 Term Frequency (TF)

A simple definition of term frequency is:

\\[ TF(t,d)=\frac{\text{Number of occurrences of }t\text{ in }d} {\text{Total number of terms in }d} \\]

Suppose a document contains 10 words and the word `excellent` occurs 2 times.

\\[ TF(\text{excellent},d)=\frac{2}{10}=0.2 \\]

Some implementations use raw counts or logarithmically scaled frequencies instead. The exact definition depends on the TF-IDF variant.

## 6.3 Inverse Document Frequency (IDF)

Document frequency is the number of documents containing a term.

A common smoothed IDF definition is:

\\[ IDF(t)=\log\left(\frac{N+1}{df(t)+1}\right)+1 \\]

Where:

- \\(N\\) = total number of documents.
- \\(df(t)\\) = number of documents containing term \\(t\\).
- The added ones smooth the calculation and prevent a zero denominator.

A term appearing in nearly every document gets a lower IDF score than a term appearing in only a few documents.

### Example

Suppose a corpus contains 100 documents.

- The word `movie` appears in 90 documents.
- The word `masterpiece` appears in 5 documents.

Using the smoothed formula above:

\\[ IDF(\text{movie})=\log\left(\frac{101}{91}\right)+1 \\]

\\[ IDF(\text{masterpiece})=\log\left(\frac{101}{6}\right)+1 \\]

Therefore, `masterpiece` receives a higher IDF score.

The base of the logarithm can vary between implementations; the comparison remains the same here.

## 6.4 Calculating TF-IDF step by step

Consider three documents:

- D1: `cat sat mat`
- D2: `cat ate fish`
- D3: `dog ate fish`

We want to calculate the TF-IDF of `cat` in D1 using the smoothed IDF formula.

Step 1: Calculate TF.

`cat` appears once in D1, which contains three words.

\\[ TF(\text{cat},D1)=\frac{1}{3} \\]

Step 2: Calculate document frequency.

The word `cat` occurs in D1 and D2.

\\[ df(\text{cat})=2 \\]

Step 3: Calculate IDF.

There are three documents.

\\[ IDF(\text{cat})= \log\left(\frac{3+1}{2+1}\right)+1 \\]

\\[ IDF(\text{cat})=\log(4/3)+1 \approx 1.288 \\]

Step 4: Calculate TF-IDF.

\\[ TFIDF(\text{cat},D1)=\frac{1}{3}\times1.288 \\]

\\[ \boxed{TFIDF(\text{cat},D1)\approx0.429} \\]

This is an illustrative calculation using normalized term frequency and smoothed IDF. Scikit-learn uses its own documented TF and IDF conventions, then normally applies L2 normalization to each document vector.

## 6.5 Implementing TF-IDF in Python

```
from sklearn.feature_extraction.text import TfidfVectorizerdocuments = [    "cat sat mat",    "cat ate fish",    "dog ate fish"]vectorizer = TfidfVectorizer()X = vectorizer.fit_transform(documents)print("Vocabulary:")print(vectorizer.get_feature_names_out())print("\nTF-IDF matrix:")print(X.toarray())
```

The result is a document-term matrix in which each row represents a document and each column represents a vocabulary term.

The values are not simply counts. They depend on the term frequency, document frequency, smoothing settings, and normalization.

### Important methods and parameters

| Parameter           | Purpose                                           |
| ------------------- | ------------------------------------------------- |
| `max_features`      | Limit the vocabulary size                         |
| `min_df`            | Ignore terms below a document-frequency threshold |
| `max_df`            | Ignore terms above a document-frequency threshold |
| `ngram_range`       | Include word n-grams                              |
| `stop_words`        | Remove specified stop words                       |
| `sublinear_tf=True` | Apply logarithmic term-frequency scaling          |
| `norm="l2"`         | Normalize each document vector                    |

## 6.6 BoW vs. TF-IDF

| Property                   | BoW                                   | TF-IDF                                            |
| -------------------------- | ------------------------------------- | ------------------------------------------------- |
| Main value                 | Word count or presence                | Weighted word importance                          |
| Frequent corpus-wide words | May dominate counts                   | Often receive lower weights                       |
| Rare but informative words | Counted normally                      | Often receive higher weights                      |
| Semantic understanding     | No                                    | No                                                |
| Word order                 | No, unless n-grams are added          | No, unless n-grams are added                      |
| Typical models             | Naive Bayes, logistic regression, SVM | Logistic regression, linear SVM, retrieval models |

Important: TF-IDF does not understand a word's meaning. It is a statistical weighting method.

## 6.7 Applications of TF-IDF

- Search engines and information retrieval.
- Document ranking.
- Text classification.
- Spam detection.
- Keyword extraction.
- News categorization.
- Finding similar documents using cosine similarity.

For a movie recommender such as CineMind, TF-IDF can be applied to movie overviews, genres, keywords, or combined text descriptions. Cosine similarity can then compare the resulting document vectors.

# 7. Word embeddings

One-hot encoding, BoW, and TF-IDF are useful, but they do not inherently encode semantic similarity.

Word embeddings address this limitation by mapping words into dense, continuous numerical vectors.

For example:

\\[ \text{king}\rightarrow[0.21,-0.43,0.18,\ldots] \\]

\\[ \text{queen}\rightarrow[0.19,-0.38,0.23,\ldots] \\]

The exact values are learned from data and depend on the model. Similar words may occupy nearby locations in the vector space.

## 7.1 What is an embedding?

A word embedding is a learned, dense numerical vector that represents a word's patterns of usage and relationships with other words.

Unlike a one-hot vector with one active element, an embedding may have dozens or hundreds of dimensions, with many nonzero values.

For example, a word embedding could have 100 dimensions:

\\[ v\_{\text{word}}\in\mathbb{R}^{100} \\]

This means that each word is represented by 100 learned numbers.

Some embedding models use 50, 100, 300, or other dimensions. Modern transformer models can use much larger hidden dimensions.

## 7.2 Why are embeddings better than one-hot encoding?

Suppose a model learns that:

- `king` and `queen` occur in similar contexts.
- `cat` and `dog` occur in similar contexts.
- `doctor` and `nurse` share some contextual patterns.

The resulting vectors may reflect these relationships.

By contrast, one-hot vectors have no meaningful geometric relationship between related words.

Dense embeddings also use far fewer dimensions than a vocabulary-sized one-hot representation.

However, embeddings do not automatically encode every relationship correctly. Their properties depend on training data, model architecture, and objective.

# 8. Word2Vec

Word2Vec is a popular method for learning dense word embeddings from text.

It was introduced by researchers at Google and became influential because it learns useful word representations from surrounding word contexts.

There are two main Word2Vec architectures:

1. CBOW — Continuous Bag of Words.
2. Skip-gram.

## 8.1 CBOW (Continuous Bag of Words)

CBOW predicts a target word using its surrounding context words.

Example sentence:

`The cat sits on the mat`

Suppose the target word is `sits`.

The context may be:

`The cat on the mat`

The model uses the context to predict:

`sits`
