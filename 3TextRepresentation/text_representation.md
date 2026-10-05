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

### CBOW architecture

Context words

The · cat · on · the · mat

Embedding / context aggregation

Predicted target

## sits

The architecture learns word vectors that help predict words from their contexts.

## 8.2 Skip-gram

Skip-gram does the opposite: it uses a target word to predict surrounding context words.

For the same sentence, if the target word is `sits`, the model tries to predict context words such as:

- `The`
- `cat`
- `on`
- `the`
- `mat`

### Skip-gram architecture

Target word

## sits

Learned word embedding

Predicted context

The · cat · on · the · mat

### CBOW vs. Skip-gram

| Property   | CBOW                               | Skip-gram                               |
| ---------- | ---------------------------------- | --------------------------------------- |
| Input      | Surrounding words                  | Target word                             |
| Prediction | Target word                        | Context words                           |
| Training   | Often efficient                    | Can be more computationally intensive   |
| Common use | Frequent words, efficient training | Learning useful vectors for rarer words |
| Main goal  | Learn embeddings from context      | Learn embeddings from context           |

These are general tendencies, not guarantees for every dataset or implementation.

## 8.3 How Word2Vec learns relationships

Word2Vec learns to predict words from context using an objective function. During training, its vector parameters are updated to improve those predictions.

Words that occur in similar contexts tend to acquire similar vectors.

For example:

- `Paris` and `London` may have similar contexts.
- `running` and `walking` may have similar contexts.
- `happy` and `joyful` may have similar contexts.

A famous property of some Word2Vec models is that vector arithmetic can reflect certain relationships, such as the approximate analogy:

\\[ v\_{\text{king}}-v\_{\text{man}}+v\_{\text{woman}} \approx v\_{\text{queen}} \\]

This is an empirical tendency, not a guaranteed mathematical law.

## 8.4 Implementing Word2Vec

You can use the `gensim` library.

Install it if needed:

```
pip install gensim
```

Example:

```
from gensim.models import Word2Vecsentences = [    ["i", "love", "nlp"],    ["i", "love", "python"],    ["python", "is", "useful"],    ["nlp", "is", "interesting"],    ["machine", "learning", "is", "useful"]]model = Word2Vec(    sentences=sentences,    vector_size=50,    window=3,    min_count=1,    workers=1,    sg=1,    epochs=100,    seed=42)# Vector representation of a wordvector = model.wv["python"]print(vector)print(vector.shape)# Similar wordsprint(model.wv.most_similar("python", topn=3))
```

### Explanation of important parameters

- `sentences`: tokenized training sentences.
- `vector_size=50`: each word embedding has 50 dimensions.
- `window=3`: context window size.
- `min_count=1`: include words appearing at least once.
- `sg=1`: use Skip-gram; `sg=0` selects CBOW.
- `epochs=100`: number of training passes.
- `seed=42`: helps reproducibility.

Important: This is a tiny demonstration corpus, not enough to train reliable real-world embeddings. The similarity results will be unstable or uninformative because the dataset is so small.

## 8.5 Limitations of Word2Vec

- Each word generally receives one fixed vector.
- It struggles with unseen words.
- It does not naturally represent multiple meanings of a word in different contexts.
- Results depend on training data.
- Training from scratch requires sufficient text.

For example, traditional Word2Vec assigns the same vector to `bank` in:

- `I deposited money in the bank.`
- `We sat on the river bank.`

It does not produce a different embedding for each use.

# 9. GloVe (Global Vectors for Word Representation)

GloVe is another method for learning word embeddings.

Unlike the predictive Word2Vec objectives, GloVe uses global word co-occurrence statistics.

It builds a word-word co-occurrence matrix that records how often words appear near one another throughout a corpus. It then learns vectors whose relationships capture useful patterns in these statistics.

### Example

Suppose a large corpus frequently contains:

- `doctor` near `hospital`, `patient`, `nurse`.
- `school` near `student`, `teacher`, `classroom`.
- `football` near `player`, `goal`, `team`.

GloVe uses these global co-occurrence patterns to learn word vectors.

## Word2Vec vs. GloVe

| Property             | Word2Vec                                        | GloVe                                           |
| -------------------- | ----------------------------------------------- | ----------------------------------------------- |
| Main learning signal | Predict context or target words                 | Global co-occurrence statistics                 |
| Representation       | Dense word vectors                              | Dense word vectors                              |
| Contextual?          | No, conventional model uses one vector per word | No, conventional model uses one vector per word |
| Unseen words         | Usually problematic                             | Usually problematic                             |
| Common applications  | Semantic similarity, NLP features               | Semantic similarity, NLP features               |

Both are static word embedding techniques. Neither dynamically changes a word's representation based on its sentence.

# 10. FastText

FastText extends the idea of word embeddings by representing words using character-level subword information.

Instead of relying only on a whole-word vector, it uses character n-grams to construct word representations.

For example, the word:

`playing`

might be associated with character pieces such as:

- `pla`
- `lay`
- `ayi`
- `yin`
- `ing`

FastText learns vectors for these subword units and combines them to represent a word.

## Why is FastText useful?

Consider these words:

- play
- playing
- played
- player

They share related character patterns. FastText can use these shared subword units to help learn related representations.

It can also construct representations for many words that were not seen as complete words during training, provided their character n-grams are available.

### Advantages

- Better handling of rare words in many settings.
- Can generate vectors for many out-of-vocabulary words.
- Useful for morphologically rich languages.
- Captures some word-form similarities.

### Limitations

- Shared spelling does not always mean shared meaning.
- It remains a static embedding method.
- It does not inherently understand full sentence context.

### Summary of classic embeddings

| Method   | Main idea                                  | Handles subwords?    | Context-specific word vectors? |
| -------- | ------------------------------------------ | -------------------- | ------------------------------ |
| Word2Vec | Predictive context learning                | Not in standard form | No                             |
| GloVe    | Global co-occurrence statistics            | Not in standard form | No                             |
| FastText | Word vectors built using character n-grams | Yes                  | No                             |

# 11. Document embeddings and sentence embeddings

Word embeddings represent individual words. But many NLP applications need one vector for an entire sentence, paragraph, or document.

For example, in semantic search, we may need to compare:

- `How do I reset my password?`
- `I forgot my login credentials.`

The wording differs, but the meanings are similar.

A document or sentence representation can help compare them.

## 11.1 Averaging word embeddings

The simplest approach is to average the vectors of the words in a sentence.

Suppose:

\\[ v\_{\text{good}}=[1,0] \\]

\\[ v\_{\text{movie}}=[0,2] \\]

Then the sentence vector for `good movie` is:

\\[ v\_{\text{sentence}}= \frac{v\_{\text{good}}+v\_{\text{movie}}}{2} \\]

\\[ v\_{\text{sentence}}=[0.5,1] \\]

This method produces one fixed-size vector per sentence.

### Python example

```
import numpy as npword_vectors = {    "good": np.array([1.0, 0.0]),    "movie": np.array([0.0, 2.0])}words = ["good", "movie"]sentence_vector = np.mean(    [word_vectors[word] for word in words],    axis=0)print(sentence_vector)
```

Output:

```
[0.5 1. ]
```

Real embeddings have many more dimensions.

### Limitation

Averaging loses word order. It may also dilute the effect of important words and handle negation poorly.

For example, `good movie` and `not good movie` may end up with vectors that are too similar.

## 11.2 Doc2Vec

Doc2Vec extends word embedding approaches to learn fixed-size representations of documents.

It was designed to learn document vectors that capture useful information about document content.

Potential uses include:

- Document similarity.
- Document clustering.
- Document classification.
- Retrieval and organization.

Unlike simple averaging, Doc2Vec learns document-specific parameters during training.

## 11.3 Sentence Transformers

Sentence-transformer models generate embeddings intended to capture sentence-level semantic similarity.

They are particularly useful for:

- Semantic search.
- Finding similar movie descriptions.
- FAQ matching.
- Duplicate question detection.
- Recommendation systems.
- Clustering similar documents.

A simplified example using the `sentence-transformers` library:

```
pip install sentence-transformers
```

```
from sentence_transformers import SentenceTransformerfrom sklearn.metrics.pairwise import cosine_similaritysentences = [    "A man travels through space.",    "An astronaut explores the universe.",    "A chef prepares delicious food."]model = SentenceTransformer(    "sentence-transformers/all-MiniLM-L6-v2")embeddings = model.encode(sentences)print(embeddings.shape)similarity = cosine_similarity(embeddings)print(similarity)
```

This model typically produces a 384-dimensional embedding for each input sentence.

The output matrix has three rows because there are three sentences. The cosine similarity matrix is \\(3\times3\\), with diagonal values of 1 for nonzero vectors.

The exact off-diagonal values depend on the model and input text.

### Why this matters for your movie recommender

You can represent each movie using its overview, genres, and keywords. Then compare the resulting sentence or document embeddings.

For example, a movie about a space mission and another about astronauts exploring distant planets may receive similar embeddings even if they use different words.

That is an important advantage over raw word counts alone.

# 12. Contextual embeddings and transformer models

Traditional word embeddings such as Word2Vec, GloVe, and FastText generally assign one fixed vector to each word.

Modern contextual models generate representations based on the surrounding text.

This is a major development in NLP.

## 12.1 The problem of multiple meanings

Consider these two sentences:

1. `She deposited money in the bank.`
2. `They sat on the bank of the river.`

The word `bank` has different meanings.

A static embedding model gives `bank` the same learned word vector in both sentences.

A contextual model can produce different representations because it considers the surrounding words.

## 12.2 ELMo

ELMo stands for Embeddings from Language Models.

It generates contextual word representations using a deep bidirectional language model, historically based on LSTMs.

It considers context from both directions in the sentence.

For example:

- `bank` near `money`, `deposit`, and `account`.
- `bank` near `river`, `water`, and `shore`.

These contexts influence the representation produced for the word.

## 12.3 BERT

BERT stands for Bidirectional Encoder Representations from Transformers.

BERT uses a transformer encoder to learn contextual representations.

Unlike conventional static embeddings, BERT's output for a token depends on other tokens in the input sequence.

Example:

`The animal did not cross the road because it was tired.`

The representation of `it` can use information from the surrounding words. This allows contextual models to learn relationships that simple word-frequency methods cannot directly express.

BERT is used for tasks such as:

- Text classification.
- Named entity recognition.
- Question answering.
- Sentence-pair classification.
- Extracting contextual token features.

BERT's original pretraining included masked language modeling and next-sentence prediction; many later models use different objectives.

## 12.4 How a transformer produces representations

A simplified transformer-based NLP pipeline is:

Input text

"The movie was surprisingly good."

Tokenizer

Convert text into tokens and token IDs

Input embeddings

Token, positional, and sometimes segment information

Transformer layers

Self-attention and feed-forward processing

Contextual representations

Vectors influenced by the surrounding tokens

### What is self-attention?

Self-attention allows each token's representation to incorporate information from other tokens in the input sequence.

For example, in:

`The movie was not good`

the representation of `good` can be influenced by `not`.

This helps a transformer model learn relationships that are not captured by a unigram bag of words.

However, the model's final representation does not automatically guarantee correct sentiment understanding; it depends on the model, task, and training.

## 12.5 Token embeddings vs. sentence embeddings

These are not the same thing.

Token embeddings: one vector for each input token.

If the tokenizer produces 5 tokens and the hidden dimension is 768, the last hidden layer may have shape:

\\[ (5,768) \\]

Batch dimensions and special tokens may add additional positions.

Sentence embeddings: one vector representing the whole sentence, often obtained by pooling token representations or using a model trained for sentence-level similarity.

For example, a sentence-transformer model may output:

\\[ (384,) \\]

for one sentence.

A standard BERT model does not automatically produce the best semantic sentence embedding simply by taking an arbitrary token vector. Pooling strategy and training objective matter.

# 13. Tokenization and subword tokenization

Before many text vectorization methods can work, the text must be divided into tokens.

A token may be a word, part of a word, punctuation, or another unit depending on the tokenizer.

Consider:

`I am learning NLP!`

Word-level tokenization might produce:

```
["I", "am", "learning", "NLP", "!"]
```

A transformer tokenizer may split some words into subword pieces instead.

For example, an unfamiliar word might be divided into smaller pieces that the tokenizer already knows.

## Why do modern models use subword tokenization?

Suppose the model encounters a rare word such as:

`unbelievability`

A subword tokenizer may represent it using familiar pieces rather than requiring a unique vocabulary entry for the entire word.

Common subword methods include:

- Byte Pair Encoding (BPE).
- WordPiece.
- SentencePiece-based tokenization.

Subword tokenization helps manage vocabulary size and represent many rare or previously unseen word forms.

Important distinction: Tokenization creates token units or token IDs. Embedding layers then map token IDs into vectors. These are related steps, but they are not the same operation.

# 14. Other useful feature extraction techniques

Not all NLP features must be learned embeddings. In traditional ML pipelines, manually designed and statistical features can be very useful.

## 14.1 Text length features

Possible features include:

- Number of characters.
- Number of words.
- Average word length.
- Number of sentences.
- Number of uppercase characters.
- Number of punctuation marks.
- Number of exclamation marks.
- Number of digits.

For example, a spam detection model may find that message length, URLs, and unusual punctuation provide useful additional information.

```
import redef extract_text_features(text):    words = text.split()    return {        "character_count": len(text),        "word_count": len(words),        "average_word_length": (            sum(len(word) for word in words) / len(words)            if words else 0        ),        "exclamation_count": text.count("!"),        "question_count": text.count("?"),        "digit_count": sum(char.isdigit() for char in text),        "uppercase_count": sum(char.isupper() for char in text),        "url_count": len(            re.findall(r"https?://\S+|www\.\S+", text)        )    }text = "WIN a prize now!!! Visit https://example.com"features = extract_text_features(text)print(features)
```

These features can be combined with TF-IDF features using an appropriate scikit-learn pipeline or feature union.

## 14.2 Lexicon-based sentiment features

A sentiment lexicon associates words with sentiment values or categories.

For a simple lexicon:

```
sentiment_lexicon = {    "good": 1,    "excellent": 2,    "happy": 1,    "bad": -1,    "terrible": -2,    "sad": -1}def sentiment_score(text):    words = text.lower().split()    return sum(        sentiment_lexicon.get(word.strip(".,!?"), 0)        for word in words    )print(sentiment_score("The movie was excellent"))print(sentiment_score("The movie was terrible"))
```

This produces positive and negative numerical scores, respectively.

However, this simple method does not reliably handle negation, sarcasm, or context:

`The movie was not good.`

A lexicon-based score may incorrectly classify this sentence as positive because it sees `good` but does not account for `not`.

More sophisticated lexicon methods and learned models can handle some of these cases better.

## 14.3 POS tags and syntactic features

Part-of-speech (POS) tagging identifies grammatical categories such as:

- Noun.
- Verb.
- Adjective.
- Adverb.
- Pronoun.

For example:

`The beautiful girl sings well.`

A POS tagger might identify `beautiful` as an adjective and `sings` as a verb.

Possible features include:

- Number of nouns.
- Number of verbs.
- Adjective frequency.
- Specific grammatical patterns.
- Dependency relations between words.

These can be useful in linguistic analysis, information extraction, and specialized classification tasks.

## 14.4 Named entities

Named entity recognition (NER) identifies items such as:

- Person names.
- Organizations.
- Locations.
- Dates.
- Monetary amounts.

For example:

`Suman visited Kathmandu in October.`

A named entity model may extract `Kathmandu` as a location and `October` as a date-related entity.

These extracted values can become structured features for downstream systems.

# 15. Feature extraction vs. feature selection

These two terms are frequently confused.

Feature extraction creates or derives features from the raw input.

Examples:

- Converting documents into TF-IDF vectors.
- Generating embeddings.
- Extracting text length and punctuation counts.
- Generating character n-grams.

Feature selection chooses a subset of existing features based on their usefulness.

Examples:

- Removing rare terms.
- Selecting the highest-scoring features using chi-square.
- Keeping the top \\(k\\) features according to a statistical criterion.

For example, a TF-IDF vectorizer may generate 50,000 features. A feature-selection method may keep only the 5,000 most useful features for a classification task.

Feature selection can reduce computational cost and sometimes improve generalization.

## Example using scikit-learn

```
from sklearn.feature_extraction.text import TfidfVectorizerfrom sklearn.feature_selection import SelectKBest, chi2documents = [    "free prize claim now",    "win money today",    "meeting scheduled tomorrow",    "project meeting tomorrow",    "claim your free reward",    "the project meeting is ready"]labels = [1, 1, 0, 0, 1, 0]  # 1 = spam, 0 = not spamvectorizer = TfidfVectorizer()X = vectorizer.fit_transform(documents)selector = SelectKBest(score_func=chi2, k=3)X_selected = selector.fit_transform(X, labels)selected_features = vectorizer.get_feature_names_out()[    selector.get_support()]print("Original shape:", X.shape)print("Selected shape:", X_selected.shape)print("Selected features:", selected_features)
```

This illustrates feature selection using chi-square scores. The selected terms depend on the supplied data.

In a real evaluation workflow, fit both the vectorizer and feature selector on the training set only. Do not use the test labels to select features.

# 16. Comparison of the main text representation methods

The following table is useful for revision and choosing an approach for a project.

| Method               | Representation                 | Captures word frequency?         | Captures semantic similarity inherently? | Context-dependent?                          |
| -------------------- | ------------------------------ | -------------------------------- | ---------------------------------------- | ------------------------------------------- |
| One-hot              | Sparse binary vector           | No                               | No                                       | No                                          |
| BoW                  | Sparse count vector            | Yes                              | No                                       | No                                          |
| N-grams              | Sparse sequence-feature vector | Yes, for sequences               | No                                       | No                                          |
| TF-IDF               | Sparse weighted vector         | Yes, weighted                    | No                                       | No                                          |
| Word2Vec             | Dense word vector              | Indirectly through training      | Often captures useful similarity         | No                                          |
| GloVe                | Dense word vector              | Through co-occurrence statistics | Often captures useful similarity         | No                                          |
| FastText             | Dense subword-informed vector  | Through training                 | Often captures useful similarity         | No                                          |
| Averaged embeddings  | Dense sentence/document vector | Indirectly                       | Often                                    | No, not inherently                          |
| Doc2Vec              | Dense document vector          | Through training                 | Potentially                              | Document-specific, but not token-contextual |
| BERT                 | Contextual token vectors       | Learned from pretraining         | Can capture semantic relationships       | Yes                                         |
| Sentence Transformer | Dense sentence vector          | Learned from training            | Designed to capture sentence similarity  | Yes, during encoding                        |

“Captures semantic similarity” does not mean a method will always understand meaning correctly. Performance depends on the vocabulary, corpus, architecture, and task.

# 17. Practical comparison using one example

Consider three movie descriptions:

Movie A

A space explorer travels to a distant planet to save humanity.

Movie B

An astronaut journeys across the galaxy to protect the human race.

Movie C

A chef opens a restaurant and learns to cook traditional food.

We want to find which movie is most similar to Movie A.

## Using BoW

Movie A and Movie B share some words, such as `to`, while many semantically related words differ:

- `space explorer` vs. `astronaut`
- `distant planet` vs. `galaxy`
- `save humanity` vs. `protect the human race`

Their BoW similarity may be limited because the actual word overlap is limited.

## Using TF-IDF

TF-IDF adjusts word importance across the movie descriptions. It may improve comparisons when informative terms overlap, but it still does not inherently recognize synonyms.

## Using Word2Vec or FastText

A model trained on a suitable corpus may represent words such as `astronaut` and `space` in ways that reflect related contexts.

However, aggregating static word vectors into a single movie vector can still lose word order and some sentence-level meaning.

## Using a sentence transformer

A suitable sentence embedding model can place Movie A and Movie B closer together because it is designed to capture semantic similarity across sentences.

Movie C should generally be less similar because its subject is different.

The actual ranking depends on the model and text. These are illustrative expectations, not measured scores.

# 18. Measuring similarity between text vectors

Once we have vectorized documents, we often need to compare their similarity.

A common metric is cosine similarity.

It measures the cosine of the angle between two nonzero vectors.

\\[ \operatorname{cosine\\\_similarity}(A,B)= \frac{A\cdot B}{\\|A\\|\\|B\\|} \\]

Where:

- \\(A\cdot B\\) is the dot product.
- \\(\\|A\\|\\) is the magnitude of vector \\(A\\).
- \\(\\|B\\|\\) is the magnitude of vector \\(B\\).

For normalized, nonzero vectors, a similarity near 1 indicates similar directions, a value near 0 indicates orthogonality, and a value near -1 indicates opposite directions.

TF-IDF and sentence embeddings are both commonly compared using cosine similarity, although their similarity scores have different interpretations.

## Python example

```
from sklearn.feature_extraction.text import TfidfVectorizerfrom sklearn.metrics.pairwise import cosine_similaritydocuments = [    "space explorer saves humanity",    "astronaut protects the human race",    "chef cooks traditional food"]vectorizer = TfidfVectorizer()X = vectorizer.fit_transform(documents)similarity_matrix = cosine_similarity(X)print(similarity_matrix)
```

The result is a \\(3\times3\\) matrix.

- The diagonal entries are 1 for these nonempty document vectors.
- The off-diagonal entries show pairwise cosine similarity.
- Documents with no overlapping TF-IDF terms can have a similarity of zero.

For this example, the similarity between the first two descriptions may remain low because the descriptions share few exact vocabulary terms. That limitation is a reason to investigate semantic embeddings.

# 19. Common mistakes in NLP vectorization

These are important when implementing real ML projects.

## 19.1 Fitting the vectorizer on the complete dataset

Incorrect:

```
X = vectorizer.fit_transform(all_documents)# Split after vectorization
```

This allows information from the test corpus to influence vocabulary construction, feature selection, and potentially IDF statistics.

Correct approach:

```
from sklearn.model_selection import train_test_splitfrom sklearn.feature_extraction.text import TfidfVectorizerX_train_text, X_test_text, y_train, y_test = train_test_split(    documents,    labels,    test_size=0.2,    random_state=42,    stratify=labels)vectorizer = TfidfVectorizer()X_train = vectorizer.fit_transform(X_train_text)X_test = vectorizer.transform(X_test_text)
```

The vectorizer is fitted on training documents and then used to transform test documents.

## 19.2 Removing every stop word

Words such as `the`, `is`, and `and` are often removed because they occur frequently.

However, words such as `not` can be essential for sentiment.

For example:

- `good`
- `not good`

Removing `not` can destroy an important distinction.

Stop-word removal should depend on the task.

## 19.3 Ignoring text cleaning requirements

Depending on the dataset, preprocessing may include:

- Unicode normalization.
- Lowercasing.
- Removing HTML tags.
- Normalizing URLs.
- Handling emojis.
- Expanding contractions.
- Fixing repeated characters.

But preprocessing should not be applied blindly. Removing punctuation, emojis, or special tokens can destroy valuable information.

## 19.4 Assuming more dimensions always means better performance

A vocabulary with 100,000 TF-IDF features is not automatically better than one with 20,000 features.

Likewise, a larger embedding model is not guaranteed to outperform a smaller model for every task.

Model choice should be evaluated using a suitable validation strategy and task-specific metrics.

## 19.5 Using static word embeddings when context matters

If a word has different meanings in different sentences, static embeddings may be insufficient.

Contextual embeddings may be more appropriate when the task requires interpreting sentence meaning, negation, or word relationships.

# 20. End-to-end NLP example: text classification

Let's combine preprocessing, TF-IDF, and machine learning into one practical pipeline.

Our goal is to classify movie reviews as positive or negative.

```
from sklearn.model_selection import train_test_splitfrom sklearn.pipeline import Pipelinefrom sklearn.feature_extraction.text import TfidfVectorizerfrom sklearn.linear_model import LogisticRegressionfrom sklearn.metrics import accuracy_score, classification_reportreviews = [    "This movie is amazing and wonderful",    "I love this excellent movie",    "The film is fantastic",    "This movie is terrible and boring",    "I hate this awful film",    "The movie is disappointing",    "What a great and enjoyable story",    "The worst film I have ever watched",    "Brilliant acting and excellent story",    "Poor acting and a boring story"]labels = [    "positive",    "positive",    "positive",    "negative",    "negative",    "negative",    "positive",    "negative",    "positive",    "negative"]X_train, X_test, y_train, y_test = train_test_split(    reviews,    labels,    test_size=0.3,    random_state=42,    stratify=labels)model = Pipeline([    (        "tfidf",        TfidfVectorizer(            ngram_range=(1, 2),            max_features=5000        )    ),    (        "classifier",        LogisticRegression(max_iter=1000)    )])model.fit(X_train, y_train)predictions = model.predict(X_test)print("Accuracy:", accuracy_score(y_test, predictions))print(classification_report(y_test, predictions, zero_division=0))new_reviews = [    "The movie was fantastic",    "The film was boring and terrible"]print(model.predict(new_reviews))
```

### How this pipeline works

1. `train_test_split()` separates training data from test data.
2. `TfidfVectorizer` learns the training vocabulary and TF-IDF statistics.
3. `ngram_range=(1, 2)` allows unigrams and bigrams.
4. `LogisticRegression` learns a classification boundary from the vectors.
5. `Pipeline` ensures the vectorizer is fitted only on the training portion and reused when making predictions.
6. `predict()` converts new text using the fitted vectorizer and predicts the label.

Important: This dataset is intentionally tiny for demonstration. Its test accuracy is not a meaningful estimate of real-world performance. For a real sentiment classifier, use a larger, representative dataset and a suitable evaluation strategy.

# 21. How to choose the right method

The best representation depends on your task, dataset size, computational resources, and performance requirements.

BoW / TF-IDF

Best starting point for many traditional NLP classification and retrieval tasks.

Examples: spam detection, sentiment classification, document categorization, keyword matching.

Word or character n-grams

Useful when phrases, word order, spelling patterns, or morphology matter.

Examples: negation-sensitive text classification, noisy social media text, language identification.

Word2Vec / GloVe / FastText

Useful when you want dense word-level features, especially when using a suitable pretrained embedding.

Examples: similarity, linguistic features, lightweight NLP models.

BERT and other contextual models

Useful when language meaning depends strongly on context and complex relationships.

Examples: question answering, named entity recognition, contextual classification.

Sentence-transformer embeddings

A strong starting point for semantic similarity and retrieval.

Examples: semantic search, FAQ matching, duplicate detection, movie recommendation.

For your current NLP learning path, I recommend mastering the methods in this order:

1. Tokenization and vocabulary construction.
2. One-hot encoding.
3. Bag of Words using `CountVectorizer`.
4. N-gram vectorization.
5. TF-IDF using `TfidfVectorizer`.
6. Cosine similarity and document retrieval.
7. Word2Vec, GloVe, and FastText.
8. Sentence embeddings.
9. Transformer tokenization and BERT contextual representations.
10. Combining text features with traditional ML models and evaluating the results.

This order helps you understand why each more sophisticated technique was developed.

# 22. Quick revision notes

| Concept               | Remember this                                                  |
| --------------------- | -------------------------------------------------------------- |
| Text representation   | Encoding text into a computer-processable form                 |
| Vectorization         | Converting text into numerical vectors                         |
| Feature extraction    | Deriving useful numerical features from text                   |
| One-hot encoding      | One unique vector position per category or word                |
| BoW                   | Counts vocabulary terms in each document                       |
| N-grams               | Capture short sequences of adjacent tokens                     |
| TF-IDF                | Weights terms by their frequency and corpus-wide rarity        |
| Word2Vec              | Learns dense word vectors using predictive context objectives  |
| GloVe                 | Learns dense word vectors from global co-occurrence statistics |
| FastText              | Uses character n-grams to represent words                      |
| Doc2Vec               | Learns document-level vectors                                  |
| Contextual embeddings | Word/token representations depend on surrounding text          |
| BERT                  | Transformer encoder producing contextual token representations |
| Sentence embeddings   | Fixed-size vectors designed to represent sentence meaning      |
| Cosine similarity     | Measures the angular similarity between two nonzero vectors    |
| Feature selection     | Selects useful features from an existing feature set           |


Final takeaway: Traditional methods such as BoW and TF-IDF primarily represent word occurrence and statistical importance. Word embeddings learn useful relationships between words, while contextual and sentence embeddings provide richer representations that depend on the surrounding text. Understanding these differences is the foundation for building effective NLP applications.