# Text Classification in NLP — Complete Detailed Guide

Text classification is one of the most important tasks in Natural Language Processing (NLP). It allows a machine learning model to understand a piece of text and assign it to one or more predefined categories.

Since you're learning NLP and have already been exploring text preprocessing, Bag of Words, text vectorization, and Word2Vec, we'll connect those concepts to build a complete understanding of text classification, from fundamentals to practical implementation using Python and scikit-learn.

# 1. What is text classification?

Text classification is the process of assigning predefined labels or categories to textual data automatically.

For example, consider these messages:

Input: "Congratulations! You have won a free iPhone. Click here!"

Spam

Predicted class

Input: "Your order will arrive tomorrow."

Not Spam

Predicted class

Input: "The movie was amazing and the acting was brilliant."

Positive Sentiment

Predicted class

The model learns patterns from previously labeled examples and uses those patterns to classify new, unseen text.

## 1.1 Real-world applications

| Application             | Input text              | Classification labels        |
| ----------------------- | ----------------------- | ---------------------------- |
| Spam detection          | Email or SMS            | Spam, not spam               |
| Sentiment analysis      | Product review          | Positive, negative, neutral  |
| News categorization     | News article            | Politics, sports, technology |
| Intent detection        | Chatbot message         | Greeting, booking, complaint |
| Toxicity detection      | Online comment          | Toxic, non-toxic             |
| Language identification | A sentence              | English, Nepali, Hindi       |
| Support ticket routing  | Customer complaint      | Billing, technical, delivery |
| Topic classification    | Research paper abstract | AI, healthcare, finance      |

Notice that text classification is not limited to short sentences. The input can be a word, sentence, review, email, document, or entire article.

# 2. How text classification works

A machine learning algorithm cannot directly understand raw text the way humans do. Traditional machine learning models require numerical input.

Therefore, text classification typically consists of these steps:

1\. Collect labeled text data

Reviews, messages, emails, documents

2\. Preprocess the text

Clean, normalize, tokenize as needed

3\. Convert text into numerical features

Bag of Words, TF-IDF, embeddings

4\. Train a classification model

Naive Bayes, Logistic Regression, SVM, etc.

5\. Evaluate the model

Precision, recall, F1-score, confusion matrix

6\. Predict on new text

Return the predicted category

Let's examine every stage in detail.

# 3. Types of text classification

There are three major classification setups: binary classification, multiclass classification, and multilabel classification.

## 3.1 Binary classification

Binary classification assigns one of exactly two classes to each input.

Examples:

- Spam vs. not spam
- Positive vs. negative sentiment
- Fake news vs. real news
- Toxic vs. non-toxic content

Suppose a model predicts whether a review is positive or negative.

| Review                      | Label    |
| --------------------------- | -------- |
| "Excellent product."        | Positive |
| "Terrible quality."         | Negative |
| "Very useful and reliable." | Positive |
| "It stopped working."       | Negative |

The model learns a decision rule that separates the two classes.

Many binary classifiers internally estimate a probability such as:

\\[ P(y=1\mid x) \\]

where \\(x\\) represents the text features and \\(y=1\\) represents the positive class.

For example:

\\[ P(\text{positive}\mid x)=0.87 \\]

With a decision threshold of 0.5, this would be classified as positive.

## 3.2 Multiclass classification

Multiclass classification assigns one class from three or more mutually exclusive classes.

For example, a news classifier may have four categories:

- Sports
- Politics
- Technology
- Entertainment

A news article about a new AI model might be classified as Technology.

The model may calculate:

| Class         | Predicted probability |
| ------------- | --------------------- |
| Sports        | 0.02                  |
| Politics      | 0.08                  |
| Technology    | 0.82                  |
| Entertainment | 0.08                  |

The probabilities sum to 1 in this example. The predicted class is Technology because it has the highest probability.

\\[ \hat y=\arg\max\_{c}P(y=c\mid x) \\]

Here, \\(\hat y\\) is the predicted label and \\(c\\) represents a possible class.

Common multiclass algorithms include Multinomial Naive Bayes, Logistic Regression, and Support Vector Machines.

## 3.3 Multilabel classification

Multilabel classification allows a single text to belong to multiple categories simultaneously.

Imagine classifying a movie review or movie description into genres.

| Text                                          | Labels                       |
| --------------------------------------------- | ---------------------------- |
| "A detective investigates a murder in space." | Sci-Fi, Mystery, Crime       |
| "A couple falls in love during a war."        | Romance, Drama, War          |
| "A superhero saves the world."                | Action, Adventure, Superhero |

Unlike multiclass classification, multiple labels can be correct for the same input.

A multilabel model might produce:

\\[ \hat y=[1,0,1,1,0] \\]

If the label order is Action, Comedy, Mystery, Romance, Drama, then this vector indicates that Action, Mystery, and Romance are predicted.

Each label may have its own probability and threshold. The probabilities do not have to sum to 1.

Important distinction: Multiclass means one class among many; multilabel means zero, one, or multiple labels can apply.

# 4. Preparing a dataset for text classification

Before training any model, we need a dataset containing text and its correct category.

For example, consider a sentiment analysis dataset.

```
import pandas as pddata = {    "review": [        "This movie is amazing",        "The movie is terrible",        "I really enjoyed the film",        "The acting was very bad",        "Absolutely wonderful experience",        "I hated this movie",        "The story was fantastic",        "Waste of time"    ],    "sentiment": [        "positive",        "negative",        "positive",        "negative",        "positive",        "negative",        "positive",        "negative"    ]}df = pd.DataFrame(data)print(df.head())
```

Output:

| review                          | sentiment |
| ------------------------------- | --------- |
| This movie is amazing           | positive  |
| The movie is terrible           | negative  |
| I really enjoyed the film       | positive  |
| The acting was very bad         | negative  |
| Absolutely wonderful experience | positive  |

Here:

- `review` is the input text, also called the feature or independent variable.
- `sentiment` is the target variable or label.
- Each row is one training example.

## 4.1 Inspect the dataset

```
print(df.shape)print(df.info())print(df.isnull().sum())print(df["sentiment"].value_counts())print(df.duplicated().sum())
```

These operations help us understand:

- How many samples and columns exist.
- Whether any reviews or labels are missing.
- Whether one class has significantly more examples than another.
- Whether duplicate records exist.

For real datasets, class imbalance can matter significantly. For instance, a spam dataset might contain 95% legitimate messages and only 5% spam.

A model that predicts every message as legitimate would obtain 95% accuracy while failing to detect spam.

## 4.2 Separate features and labels

```
X = df["review"]y = df["sentiment"]print(X.head())print(y.head())
```

Notice that `X` is a pandas Series of text, not yet a matrix of numerical features.

We will transform the text into numerical features later.

## 4.3 Split into training and testing data

The training set is used to learn patterns, while the test set measures how well the trained model performs on unseen examples.

```
from sklearn.model_selection import train_test_splitX_train, X_test, y_train, y_test = train_test_split(    X,    y,    test_size=0.25,    random_state=42,    stratify=y)print("Training samples:", len(X_train))print("Testing samples:", len(X_test))
```

Explanation:

- `test_size=0.25`: 25% of the data is used for testing.
- `random_state=42`: makes the split reproducible.
- `stratify=y`: attempts to preserve the class proportions in both sets.

For a dataset containing 10,000 examples, a common starting point is 80% training and 20% testing. The ideal split depends on dataset size and the evaluation setup.

Critical rule: Split your dataset before fitting a vectorizer. Otherwise, information from the test set may leak into the training process.

For a small dataset, stratification may be impossible if some classes have too few examples. In that case, more labeled data may be needed.

# 5. Text preprocessing for classification

Text preprocessing converts inconsistent raw text into a representation that is easier for a model to use.

For example:

Original:

`"THIS Movie was AMAZING!!!"`

Possible normalized version:

`"this movie was amazing"`

However, preprocessing depends on the problem and the model. More cleaning does not always mean better performance.

## 5.1 Lowercasing

```
text = "This Movie Is AMAZING"text = text.lower()print(text)
```

Output:

```
this movie is amazing
```

Why is it useful?

Without lowercasing, a basic case-sensitive vocabulary might treat `Movie`, `movie`, and `MOVIE` as separate features.

Lowercasing reduces vocabulary size, although it can remove meaningful distinctions in some tasks.

## 5.2 Removing punctuation

```
import redef remove_punctuation(text):    return re.sub(r"[^\w\s]", "", text)print(remove_punctuation("Amazing movie!!!"))
```

Output:

```
Amazing movie
```

This can help when punctuation does not carry useful information. But punctuation can be important in tasks such as sentiment analysis, where `!!!` may express strong emotion, or in code and social-media classification.

## 5.3 Removing stop words

Stop words are common words such as `the`, `is`, `a`, and `an`. They often provide limited value in traditional bag-of-words models.

Using NLTK:

```
import nltkfrom nltk.corpus import stopwordsnltk.download("stopwords")stop_words = set(stopwords.words("english"))def remove_stopwords(text):    words = text.split()    return " ".join(        word for word in words        if word not in stop_words    )print(remove_stopwords("This is a very good movie"))
```

Possible output:

```
This good movie
```

The exact result depends on the stop-word list and case handling.

Important warning: Stop-word removal can be harmful to sentiment analysis. Removing `not` from `"not good"` leaves `"good"` and reverses the intended meaning.

For sentiment tasks, keep negation words such as `not`, `no`, and `never`.

## 5.4 Tokenization

Tokenization divides text into smaller units called tokens.

```
text = "I really love natural language processing"tokens = text.split()print(tokens)
```

Output:

```
[    "I", "really", "love",    "natural", "language", "processing"]
```

More sophisticated tokenizers handle punctuation, contractions, and language-specific structures.

For many scikit-learn text classification models, you do not need to tokenize the entire dataset manually. `CountVectorizer` and `TfidfVectorizer` perform tokenization and feature extraction internally.

## 5.5 Stemming and lemmatization

Stemming reduces words using heuristic rules.

For example:

- `playing` → `play`
- `connected` → `connect`
- `studies` → `studi` (depending on the stemmer)

Lemmatization aims to return a linguistically valid base form.

- `running` → `run`
- `better` → `good` in some contexts
- `studies` → `study`

These transformations can reduce vocabulary size, but they are not always necessary. TF-IDF with word and phrase features often works well without either.

## 5.6 An optional preprocessing function

Here is a simple preprocessing function for experimentation:

```
import redef preprocess_text(text):    text = text.lower()    text = re.sub(r"http\S+|www\.\S+", " ", text)    text = re.sub(r"<[^>]+>", " ", text)    text = re.sub(r"[^a-z\s]", " ", text)    text = re.sub(r"\s+", " ", text).strip()    return textprint(    preprocess_text(        "This MOVIE is AMAZING!!! Visit https://example.com"    ))
```

Output:

```
this movie is amazing visit
```

This function is suitable only as a basic example. It removes digits, non-English characters, and punctuation, which may be inappropriate for many datasets. For Nepali text, multilingual text, URLs as features, or emoji-heavy reviews, use preprocessing designed for that data.

For most of the following examples, we'll let the scikit-learn vectorizer handle tokenization and lowercase normalization.

# 6. Converting text into numerical features

This is one of the most important parts of text classification.

A conventional classifier such as Logistic Regression or SVM expects a numerical feature matrix. It cannot directly use an ordinary Python string as its input feature vector.

Consider three documents:

- Document 1: `I love NLP`
- Document 2: `I love Python`
- Document 3: `Python is amazing`

## 6.1 Bag of Words (BoW)

Bag of Words represents each document by counting the occurrences of words from a vocabulary.

First, construct the vocabulary:

```
["I", "love", "NLP", "Python", "is", "amazing"]
```

The corresponding document-term matrix might look like this:

| Document          | I | love | NLP | Python | is | amazing |
| ----------------- | - | ---- | --- | ------ | -- | ------- |
| I love NLP        | 1 | 1    | 1   | 0      | 0  | 0       |
| I love Python     | 1 | 1    | 0   | 1      | 0  | 0       |
| Python is amazing | 0 | 0    | 0   | 1      | 1  | 1       |

Each document becomes a numerical vector.

For example:

\\[ d_1=[1,1,1,0,0,0] \\]

A classifier can learn that words such as `love` may correlate with positive sentiment, while `terrible` may correlate with negative sentiment.

### Implementing BoW

```
from sklearn.feature_extraction.text import CountVectorizercorpus = [    "I love NLP",    "I love Python",    "Python is amazing"]vectorizer = CountVectorizer()X_bow = vectorizer.fit_transform(corpus)print(vectorizer.get_feature_names_out())print(X_bow.toarray())
```

Notice that `CountVectorizer` lowercases words by default and generally ignores single-character tokens under its default token pattern. Consequently, `I` will not appear as a feature in this particular example.

`fit_transform()` learns the vocabulary from the supplied documents and converts them to a sparse numerical matrix.

Advantages:

- Simple and fast.
- Works well with traditional classification algorithms.
- Easy to inspect which words contribute to predictions.

Limitations:

- Does not understand word meaning.
- Ignores word order by default.
- Common words may receive high counts even if they are not informative.

## 6.2 N-grams

An n-gram is a sequence of \\(n\\) consecutive tokens.

Consider:

`"The movie was not good"`

Unigrams:

```
the, movie, was, not, good
```

Bigrams:

```
the movie
movie was
was not
not good
```

The phrase `not good` can be more informative for sentiment classification than the individual word `good`.

You can include both unigrams and bigrams:

```
from sklearn.feature_extraction.text import CountVectorizervectorizer = CountVectorizer(    ngram_range=(1, 2))X_bow = vectorizer.fit_transform(corpus)
```

`ngram_range=(1, 2)` includes unigrams and bigrams. You can use `(1, 3)` to include trigrams as well, but increasing the n-gram range can greatly increase the vocabulary size.

## 6.3 TF-IDF

TF-IDF stands for Term Frequency–Inverse Document Frequency.

It gives words a score based on how frequently they occur in a document and how informative they are across the document collection.

The intuition is:

- A word appearing frequently in a particular document may be important to that document.
- A word appearing in almost every document may be less useful for distinguishing documents.

### Term Frequency (TF)

One possible definition is:

\\[ TF(t,d)=\frac{f\_{t,d}}{\sum\_{w\in d}f\_{w,d}} \\]

Here:

- \\(t\\) is a term.
- \\(d\\) is a document.
- \\(f\_{t,d}\\) is the count of term \\(t\\) in document \\(d\\).

### Inverse Document Frequency (IDF)

A common smoothed IDF formulation used by scikit-learn is:

\\[ IDF(t)=\log\left(\frac{1+N}{1+df(t)}\right)+1 \\]

where:

- \\(N\\) is the number of documents.
- \\(df(t)\\) is the number of documents containing the term.

The TF-IDF score is:

\\[ TFIDF(t,d)=TF(t,d)\times IDF(t) \\]

Scikit-learn normally applies L2 normalization to the resulting document vectors as well.

### Example

Suppose a collection has 1,000 documents:

- `the` appears in 950 documents.
- `excellent` appears in 40 documents.
- `photosynthesis` appears in 3 documents.

The latter terms may be more useful for distinguishing particular documents because they are rarer. Rarity alone does not guarantee predictive value, however.

### Implementing TF-IDF

```
from sklearn.feature_extraction.text import TfidfVectorizercorpus = [    "The movie was excellent",    "The movie was terrible",    "Excellent acting and excellent story"]tfidf = TfidfVectorizer()X_tfidf = tfidf.fit_transform(corpus)print(tfidf.get_feature_names_out())print(X_tfidf.toarray())
```

The matrix contains TF-IDF weights rather than raw word counts.

Why TF-IDF is a good starting point: It produces sparse features, is computationally efficient for many datasets, and often works well with Logistic Regression, Linear SVM, and Naive Bayes.

## 6.4 Word embeddings and Word2Vec

Unlike BoW and TF-IDF, word embeddings represent words as dense numerical vectors that can encode semantic and contextual relationships.

For example, an embedding model may place `king` and `queen` relatively close together in vector space compared with unrelated words.

Word2Vec learns word vectors using surrounding words in a corpus. Its two main training approaches are:

- CBOW: Predict a target word from its surrounding context.
- Skip-gram: Predict surrounding words from a target word.

A major distinction: standard Word2Vec produces a vector for each vocabulary word, not automatically one complete vector for an entire review.

To classify a review with traditional machine learning, you might average its word vectors:

\\[ v_d=\frac{1}{n}\sum\_{i=1}^{n}v\_{w_i} \\]

This creates one fixed-size vector per document, but averaging loses word order and can weaken negation or compositional meaning.

Other options include Doc2Vec, pretrained sentence embeddings, and contextual transformer embeddings such as BERT.

| Representation                   | Feature type                      | Captures semantics?                     | Typical use                  |
| -------------------------------- | --------------------------------- | --------------------------------------- | ---------------------------- |
| BoW                              | Sparse word counts                | Limited                                 | Simple text classifiers      |
| TF-IDF                           | Sparse weighted terms             | Limited                                 | Strong traditional baseline  |
| Word2Vec                         | Dense word vectors                | Some semantic relationships             | Embedding-based features     |
| Sentence embeddings              | Dense document vectors            | Often stronger sentence-level semantics | Semantic classification      |
| BERT-style contextual embeddings | Context-dependent representations | Rich contextual information             | Complex classification tasks |

The best representation depends on dataset size, language, domain, available computing resources, and the importance of context.

# 7. Machine learning algorithms for text classification

Once text has been converted into numerical features, we need a classifier to learn the relationship between those features and their labels.

## 7.1 Multinomial Naive Bayes

Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem.

Bayes' theorem:

\\[ P(C\mid X)=\frac{P(X\mid C)P(C)}{P(X)} \\]

Where:

- \\(C\\) is a class, such as spam.
- \\(X\\) is the observed text representation.
- \\(P(C)\\) is the prior probability of the class.
- \\(P(X\mid C)\\) is the likelihood of observing the features given the class.
- \\(P(C\mid X)\\) is the posterior probability of the class given the features.

For text classification, Multinomial Naive Bayes commonly models word counts. It assumes that word features are conditionally independent given the class, which is a simplifying assumption rather than a realistic description of language.

For a document containing words \\(w_1,\ldots,w_n\\), a simplified scoring expression is:

\\[ P(C\mid d)\propto P(C)\prod\_{i=1}^{n}P(w_i\mid C) \\]

In practice, the model uses logarithms to avoid numerical underflow.

### Example

Suppose the training data contains many spam messages with the words:

- Free
- Prize
- Winner
- Claim

When a new message contains `free`, `prize`, and `claim`, the model may assign a high spam probability based on their learned class-conditional frequencies.

### Training Naive Bayes

```
from sklearn.naive_bayes import MultinomialNBmodel = MultinomialNB()model.fit(X_train_tfidf, y_train)predictions = model.predict(X_test_tfidf)
```

Here, `X_train_tfidf` and `X_test_tfidf` must already have been generated by a fitted TF-IDF vectorizer.

Advantages

- Fast to train and predict.
- Often performs well on spam detection and topic classification.
- Works well with sparse word-count features.
- Requires relatively little computational power.

Limitations

- Its independence assumption is unrealistic for language.
- It may struggle to capture relationships between words.
- It is not automatically the best choice for every dataset.

Multinomial Naive Bayes generally works with nonnegative features such as counts or TF-IDF values. If you use centered features that can contain negative values, this is not the appropriate variant.

## 7.2 Logistic Regression

Despite its name, Logistic Regression is widely used for classification.

For binary classification, it calculates a linear score:

\\[ z=w^Tx+b \\]

Then applies the sigmoid function:

\\[ P(y=1\mid x)=\sigma(z)=\frac{1}{1+e^{-z}} \\]

The resulting probability lies between 0 and 1.

For example:

\\[ P(\text{positive}\mid x)=0.91 \\]

With a threshold of 0.5, the predicted class is positive.

Logistic Regression learns weights that make the observed training labels more likely, usually by minimizing logistic loss with regularization.

For TF-IDF classification, it can learn positive weights for words associated with positive reviews and negative weights for words associated with negative reviews.

### Training Logistic Regression

```
from sklearn.linear_model import LogisticRegressionmodel = LogisticRegression(    max_iter=1000,    random_state=42)model.fit(X_train_tfidf, y_train)predictions = model.predict(X_test_tfidf)
```

Advantages

- Strong baseline for TF-IDF.
- Efficient for many sparse text datasets.
- Supports multiclass classification.
- Provides probabilities through `predict_proba()` in its standard configurations.
- Its coefficients can help explain word-level contributions.

Limitations

- The decision boundary is linear in the feature space.
- Complex relationships may require richer features or a different model.
- Probability estimates may need calibration if accurate probabilities are important.

## 7.3 Support Vector Machine (SVM)

Support Vector Machines seek a decision boundary that separates classes with a large margin.

For binary linear classification, the decision function is:

\\[ f(x)=w^Tx+b \\]

A linear SVM seeks a boundary that separates the classes while maximizing the margin, subject to the chosen regularization and loss.

Text datasets often have many thousands of features, and Linear SVMs frequently perform very well on sparse TF-IDF matrices.

### Training Linear SVM

```
from sklearn.svm import LinearSVCmodel = LinearSVC(    C=1.0,    random_state=42)model.fit(X_train_tfidf, y_train)predictions = model.predict(X_test_tfidf)
```

`LinearSVC` is often a practical choice for large sparse text datasets.

Important: `LinearSVC` does not directly provide `predict_proba()`. Its `decision_function()` returns decision scores rather than calibrated probabilities.

## 7.4 Decision Tree and Random Forest

Decision Trees classify data by applying a sequence of feature-based rules.

For example, a simplified tree might learn:

```
Does the text contain "free"?
          |
       Yes / No
        /     \
 Does it     Check other
 contain      features
 "prize"?
   /   \
 Spam  Other
```

Actual trees learn splits according to their training objective rather than using a manually written rule like this.

Random Forest trains multiple decision trees using randomized samples and features, then combines their predictions.

These algorithms can capture nonlinear interactions, but they are not always competitive with linear models on very high-dimensional sparse text features. They can also become computationally expensive.

## 7.5 Deep learning models

Deep learning becomes especially useful when language structure and contextual meaning are important.

![Alt Text](RNN.jpg)

RNN (Recurrent Neural Network)

Processes sequences step by step, maintaining a hidden state that summarizes previous tokens. Traditional RNNs can struggle with long-range dependencies.

![Frontiers | Chiller Fault Diagnosis Based on Automatic Machine Learning](LSTM_GRU.jpg)

LSTM / GRU

Gated recurrent architectures designed to retain useful information across a sequence. They can model word order and dependencies better than a simple RNN.

![Transformers: The Engine Powering ChatGPT and Beyond - DEV Community](transformers_BERT.jpg)

Transformers and BERT

Use attention to model relationships between tokens. Pretrained models can be fine-tuned for sentiment analysis, topic classification, intent detection, and other NLP tasks.

Deep learning models typically require more computation than traditional TF-IDF classifiers. Pretrained transformers can be particularly useful when the meaning of a word depends heavily on its context.

For example:

- "The battery is light."
- "Turn on the light."

A bag-of-words representation cannot fully distinguish the contextual meanings. Contextual models can use surrounding words to produce different representations.

### Which algorithm should you choose?

| Situation                                          | Good starting point                                        |
| -------------------------------------------------- | ---------------------------------------------------------- |
| Spam detection, simple dataset                     | Multinomial Naive Bayes                                    |
| General-purpose TF-IDF classification              | Logistic Regression                                        |
| Large, sparse text dataset                         | Linear SVM                                                 |
| Small compute budget                               | TF-IDF + Logistic Regression or Naive Bayes                |
| Long-range sequence patterns                       | LSTM/GRU, depending on the task                            |
| Rich context and pretrained language understanding | Fine-tuned transformer                                     |
| Multiple labels per document                       | One-vs-rest or classifier chains; neural multilabel models |

These are starting points, not guarantees. Evaluate several appropriate models on the same validation split.

# 8. Complete practical project: Movie review sentiment classification

Let's build a proper text classification pipeline with TF-IDF and Logistic Regression.

This example is particularly relevant to movie reviews and can later be extended to a larger dataset.

## Step 1: Install the libraries

```
pip install pandas scikit-learn
```

## Step 2: Prepare the data

For this demonstration, we'll use a small example dataset. A real model needs substantially more diverse labeled examples.

```
import pandas as pddata = {    "review": [        "This movie was absolutely fantastic",        "I loved the story and acting",        "An amazing and wonderful film",        "The movie was brilliant",        "Excellent direction and great performances",        "I really enjoyed this film",        "This is one of the best movies",        "A beautiful and entertaining story",        "The movie was terrible",        "I hated this film",        "The story was boring and predictable",        "Very disappointing acting",        "The worst movie I have ever watched",        "A complete waste of time",        "The film was dull and poorly made",        "I did not enjoy this movie"    ],    "sentiment": [        "positive", "positive", "positive", "positive",        "positive", "positive", "positive", "positive",        "negative", "negative", "negative", "negative",        "negative", "negative", "negative", "negative"    ]}df = pd.DataFrame(data)print(df.head())print(df["sentiment"].value_counts())
```

We have two columns:

- `review`: text input.
- `sentiment`: target label.

## Step 3: Split the data

```
from sklearn.model_selection import train_test_splitX = df["review"]y = df["sentiment"]X_train, X_test, y_train, y_test = train_test_split(    X,    y,    test_size=0.25,    random_state=42,    stratify=y)
```

We split the raw text before fitting TF-IDF. This is essential for preventing vocabulary leakage from the test set.

## Step 4: Create a TF-IDF vectorizer

```
from sklearn.feature_extraction.text import TfidfVectorizertfidf = TfidfVectorizer(    lowercase=True,    ngram_range=(1, 2),    min_df=1,    max_df=1.0,    sublinear_tf=True)X_train_tfidf = tfidf.fit_transform(X_train)X_test_tfidf = tfidf.transform(X_test)print("Training matrix:", X_train_tfidf.shape)print("Testing matrix:", X_test_tfidf.shape)
```

Explanation:

- `lowercase=True`: normalizes case.
- `ngram_range=(1, 2)`: includes individual words and two-word phrases.
- `min_df=1`: retains features that occur in at least one training document.
- `max_df=1.0`: does not remove terms based on document frequency.
- `sublinear_tf=True`: uses a logarithmic adjustment to term frequency.

The vectorizer learns its vocabulary only from the training set. The test set is transformed using that same vocabulary.

## Step 5: Train the classifier

```
from sklearn.linear_model import LogisticRegressionmodel = LogisticRegression(    max_iter=1000,    random_state=42)model.fit(X_train_tfidf, y_train)
```

The `fit()` method learns the relationship between the TF-IDF features and the sentiment labels.

## Step 6: Make predictions

```
y_pred = model.predict(X_test_tfidf)print("Actual labels:")print(y_test.to_list())print("Predicted labels:")print(y_pred)
```

The predictions can then be compared with the actual labels.

Because this example has only 16 samples, its performance is not a reliable estimate of real-world accuracy. Its purpose is to demonstrate the workflow.

## Step 7: Evaluate the model

```
from sklearn.metrics import (    accuracy_score,    classification_report,    confusion_matrix)print("Accuracy:", accuracy_score(y_test, y_pred))print("\nClassification report:")print(classification_report(    y_test,    y_pred,    zero_division=0))print("\nConfusion matrix:")print(confusion_matrix(    y_test,    y_pred,    labels=["negative", "positive"]))
```

This reports accuracy, precision, recall, F1-score, and the confusion matrix.

We'll explain each of these evaluation metrics in Section 10.

## Step 8: Predict on an individual review

A common beginner mistake is manually creating a new vectorizer for a new review. You should instead reuse the vectorizer that was fitted on the training data.

```
new_reviews = [    "The movie was wonderful and entertaining",    "The film was boring and terrible",    "The acting was excellent"]new_reviews_tfidf = tfidf.transform(new_reviews)predictions = model.predict(new_reviews_tfidf)for review, prediction in zip(new_reviews, predictions):    print(f"Review: {review}")    print(f"Predicted sentiment: {prediction}")    print("-" * 50)
```

The new reviews are transformed into the same feature space as the training data before classification.

## Step 9: Predict probabilities

Logistic Regression can also return probabilities.

```
probabilities = model.predict_proba(new_reviews_tfidf)for review, probs, prediction in zip(    new_reviews,    probabilities,    predictions):    print("Review:", review)    print("Prediction:", prediction)    for label, probability in zip(        model.classes_,        probs    ):        print(f"{label}: {probability:.4f}")    print()
```

For each review, `predict_proba()` returns a probability for each class. In this binary setup, the two probabilities sum to approximately 1.

Do not interpret these numbers as guaranteed measures of correctness. A classifier can be overconfident or poorly calibrated, particularly when training data is limited or differs from the data encountered in production.

# 9. The recommended implementation: use a Pipeline

The previous implementation works, but scikit-learn provides a cleaner way to combine feature extraction and classification.

A `Pipeline` ensures that the vectorizer is fitted as part of the training process and that the same transformation is applied during prediction.

This is the implementation I recommend for your real NLP projects.

```
from sklearn.pipeline import Pipelinefrom sklearn.feature_extraction.text import TfidfVectorizerfrom sklearn.linear_model import LogisticRegressionfrom sklearn.model_selection import train_test_splitfrom sklearn.metrics import classification_reportX = df["review"]y = df["sentiment"]X_train, X_test, y_train, y_test = train_test_split(    X,    y,    test_size=0.25,    random_state=42,    stratify=y)text_classifier = Pipeline([    (        "tfidf",        TfidfVectorizer(            ngram_range=(1, 2),            sublinear_tf=True        )    ),    (        "classifier",        LogisticRegression(            max_iter=1000,            random_state=42        )    )])# Train both steps togethertext_classifier.fit(X_train, y_train)# Evaluate on unseen texty_pred = text_classifier.predict(X_test)print(classification_report(    y_test,    y_pred,    zero_division=0))# Predict directly from raw textnew_text = [    "The movie was brilliant and entertaining"]print(text_classifier.predict(new_text))
```

Notice the major difference:

```
# Without a PipelineX_new = tfidf.transform(new_text)prediction = model.predict(X_new)# With a Pipelineprediction = text_classifier.predict(new_text)
```

The Pipeline performs the transformation internally.

### Why the Pipeline approach is better

- It prevents accidental fitting of the vectorizer on test data when used correctly.
- It keeps preprocessing and classification together.
- It simplifies deployment.
- It works naturally with cross-validation and hyperparameter tuning.
- It reduces the chance of applying different transformations during training and prediction.

A Pipeline does not automatically prevent every kind of leakage. For example, you must still avoid splitting duplicated reviews across train and test, and time-dependent data may require a chronological split rather than a random split.

# 10. Evaluating a text classification model

Training a classifier is only half the work. We also need to measure how well it classifies unseen examples.

The most important evaluation metrics are accuracy, precision, recall, F1-score, and the confusion matrix.

## 10.1 Confusion matrix

For binary classification, the confusion matrix contains four outcomes.

| Actual / Predicted | Predicted positive  | Predicted negative  |
| ------------------ | ------------------- | ------------------- |
| Actual positive    | True Positive (TP)  | False Negative (FN) |
| Actual negative    | False Positive (FP) | True Negative (TN)  |

Imagine that a spam classifier processes 100 emails:

- 30 are actually spam.
- 70 are legitimate.
- The model correctly identifies 24 spam messages.
- It incorrectly flags 5 legitimate messages as spam.
- It misses 6 spam messages.
- It correctly recognizes 65 legitimate messages.

The confusion matrix is:

| Actual / Predicted | Spam | Not spam |
| ------------------ | ---- | -------- |
| Spam               | 24   | 6        |
| Not spam           | 5    | 65       |

This matrix makes it possible to see which kinds of mistakes the model makes.

## 10.2 Accuracy

Accuracy measures the proportion of all predictions that are correct.

\\[ Accuracy=\frac{TP+TN}{TP+TN+FP+FN} \\]

Using the example:

\\[ Accuracy=\frac{24+65}{100}=0.89 \\]

The model has 89% accuracy.

Accuracy is useful when classes are reasonably balanced and the costs of different mistakes are comparable. It can be misleading when the dataset is highly imbalanced.

## 10.3 Precision

Precision measures the proportion of predicted-positive examples that are actually positive.

\\[ Precision=\frac{TP}{TP+FP} \\]

In the spam example:

\\[ Precision=\frac{24}{24+5}\approx0.828 \\]

Approximately 82.8% of emails flagged as spam were actually spam.

When is precision important?

Consider an email system that might accidentally send important customer emails to the spam folder. High precision helps reduce false spam alerts.

## 10.4 Recall

Recall measures the proportion of actual positive examples that the model successfully identifies.

\\[ Recall=\frac{TP}{TP+FN} \\]

For the example:

\\[ Recall=\frac{24}{24+6}=0.80 \\]

The model detects 80% of all spam emails.

When is recall important?

Recall is important when missing a positive case is costly, such as failing to detect harmful content or a fraudulent message.

## 10.5 F1-score

The F1-score is the harmonic mean of precision and recall.

\\[ F1=2\cdot\frac{Precision\cdot Recall}{Precision+Recall} \\]

Using the example:

\\[ F1\approx0.814 \\]

F1 balances precision and recall, but it does not incorporate true negatives directly.

For imbalanced multiclass datasets, you should inspect class-specific scores and choose a suitable averaging method:

- Macro F1: Calculates the F1-score for each class and gives every class equal weight.
- Weighted F1: Weights each class's F1-score by its support, or number of true examples.
- Micro F1: Aggregates counts across classes before computing the score.

For ordinary single-label multiclass classification, micro F1 equals accuracy when evaluated over the same complete set of examples.

## 10.6 ROC-AUC

ROC-AUC evaluates how well a model ranks positive examples above negative examples across different decision thresholds.

The ROC curve plots:

\\[ TPR=\frac{TP}{TP+FN} \\]

against

\\[ FPR=\frac{FP}{FP+TN} \\]

AUC is the area under this curve.

A higher AUC generally indicates better ranking performance. For highly imbalanced datasets, precision-recall curves can also provide valuable insight.

AUC does not tell you which threshold is best for your application. You must select a threshold based on the costs of false positives and false negatives.

## 10.7 Choosing the right metric

| Goal                                           | Useful metrics                       |
| ---------------------------------------------- | ------------------------------------ |
| General balanced classification                | Accuracy, macro F1                   |
| Spam detection with few false alarms           | Precision                            |
| Detecting as many harmful messages as possible | Recall                               |
| Imbalanced classification                      | Macro F1, per-class precision/recall |
| Comparing ranking quality                      | ROC-AUC or average precision         |
| Understanding specific mistakes                | Confusion matrix                     |

For your projects, always inspect more than accuracy. A classifier with 95% accuracy can still perform poorly on the category you care about most.

# 11. Improving a text classification model

A baseline model is only the starting point. Here are the major techniques used to improve classification performance.

## 11.1 Improve the training dataset

High-quality labeled data is often more valuable than switching to a more complicated algorithm.

Check for:

- Incorrect labels.
- Duplicate examples.
- Missing or empty text.
- Inconsistent labeling rules.
- Very rare classes.
- Training data that does not resemble the intended real-world input.

For sentiment analysis, make sure that examples involving sarcasm, negation, mixed opinions, and neutral sentiment are represented when they matter to your use case.

## 11.2 Tune TF-IDF parameters

You can experiment with:

- `ngram_range`: include word sequences.
- `min_df`: remove very rare features.
- `max_df`: remove terms appearing in a large proportion of documents.
- `max_features`: limit vocabulary size.
- `sublinear_tf`: apply logarithmic term-frequency scaling.
- `max_features` and `min_df` together: manage the size of the feature space.

Do not assume that larger vocabularies or more n-grams always improve performance.

## 11.3 Tune the classification algorithm

For Logistic Regression, `C` controls the inverse strength of regularization:

- Smaller `C`: stronger regularization.
- Larger `C`: weaker regularization.

For Linear SVM, `C` also controls the trade-off between margin size and training errors.

A grid search can help select these parameters using cross-validation.

```
from sklearn.model_selection import GridSearchCV, StratifiedKFoldpipeline = Pipeline([    (        "tfidf",        TfidfVectorizer()    ),    (        "classifier",        LogisticRegression(            max_iter=1000,            random_state=42        )    )])param_grid = {    "tfidf__ngram_range": [(1, 1), (1, 2)],    "tfidf__min_df": [1, 2],    "classifier__C": [0.1, 1.0, 10.0]}cv = StratifiedKFold(    n_splits=5,    shuffle=True,    random_state=42)grid_search = GridSearchCV(    estimator=pipeline,    param_grid=param_grid,    scoring="f1_macro",    cv=cv,    n_jobs=-1)grid_search.fit(X_train, y_train)print("Best parameters:")print(grid_search.best_params_)print("Best cross-validation F1:")print(grid_search.best_score_)best_model = grid_search.best_estimator_print(classification_report(    y_test,    best_model.predict(X_test),    zero_division=0))
```

This evaluates different combinations using the training data, then evaluates the selected model on the held-out test set.

For this example, the smallest dataset is far too small for reliable five-fold tuning. With a real dataset, use enough examples per class to support the folds. Also, do not repeatedly tune against the test set: keep it as a final evaluation set.

## 11.4 Handle class imbalance

If some classes have far fewer examples, possible approaches include:

- Collecting more examples for minority classes.
- Using class weights.
- Oversampling or undersampling where appropriate.
- Adjusting decision thresholds.
- Evaluating macro F1 and minority-class recall.

For Logistic Regression:

```
LogisticRegression(    max_iter=1000,    class_weight="balanced")
```

This increases the relative influence of underrepresented classes during training. It does not guarantee better performance, so validate the effect on the appropriate metrics.

## 11.5 Use pretrained embeddings or transformers

If TF-IDF models fail to capture contextual meaning, try sentence embeddings or a pretrained transformer.

For instance, a pretrained BERT model can produce contextual representations that account for surrounding words. Fine-tuning the model on labeled examples can adapt it to your classification task.

However, transformers are not automatically better. A TF-IDF + Linear SVM baseline may be faster, cheaper, and competitive on smaller, domain-specific datasets.

# 12. Multiclass classification using scikit-learn

Let's consider classifying news articles into three categories: sports, technology, and politics.

```
import pandas as pdnews_df = pd.DataFrame({    "text": [        "The team won the football championship",        "The player scored a brilliant goal",        "The government announced a new policy",        "The minister addressed parliament",        "The new processor improves computer performance",        "Researchers developed a new AI model"    ],    "category": [        "sports",        "sports",        "politics",        "politics",        "technology",        "technology"    ]})
```

The model structure is almost identical to binary classification.

```
from sklearn.model_selection import train_test_splitfrom sklearn.pipeline import Pipelinefrom sklearn.feature_extraction.text import TfidfVectorizerfrom sklearn.linear_model import LogisticRegressionX = news_df["text"]y = news_df["category"]X_train, X_test, y_train, y_test = train_test_split(    X,    y,    test_size=0.33,    random_state=42,    stratify=y)news_classifier = Pipeline([    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),    ("model", LogisticRegression(max_iter=1000))])news_classifier.fit(X_train, y_train)print(    news_classifier.predict([        "Scientists created a powerful artificial intelligence system"    ]))
```

Logistic Regression automatically handles multiclass classification using a suitable multiclass strategy.

This tiny dataset only demonstrates the syntax. It is not sufficient to train a useful news classifier, and the test score would have high uncertainty.

# 13. Multilabel classification using scikit-learn

Suppose each movie description can have several genres. We want to predict whether the description belongs to Action, Comedy, or Drama.

A convenient traditional approach is `OneVsRestClassifier` with Logistic Regression.

```
from sklearn.preprocessing import MultiLabelBinarizerfrom sklearn.feature_extraction.text import TfidfVectorizerfrom sklearn.multiclass import OneVsRestClassifierfrom sklearn.linear_model import LogisticRegressionfrom sklearn.pipeline import Pipelinedescriptions = [    "A superhero fights criminals in an action packed adventure",    "Friends get into funny situations in a comedy",    "A family faces emotional struggles and drama",    "A superhero comedy about friends saving the world",    "An emotional action film about a family"]genres = [    ["Action"],    ["Comedy"],    ["Drama"],    ["Action", "Comedy"],    ["Action", "Drama"]]mlb = MultiLabelBinarizer()Y = mlb.fit_transform(genres)X_train, X_test, Y_train, Y_test = train_test_split(    descriptions,    Y,    test_size=0.4,    random_state=42)multilabel_model = Pipeline([    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),    (        "classifier",        OneVsRestClassifier(            LogisticRegression(max_iter=1000)        )    )])multilabel_model.fit(X_train, Y_train)predicted_labels = multilabel_model.predict(X_test)print(mlb.classes_)print(predicted_labels)
```

Here, `MultiLabelBinarizer` converts genre lists into a binary matrix.

For example:

| Movie                 | Action | Comedy | Drama |
| --------------------- | ------ | ------ | ----- |
| Superhero comedy      | 1      | 1      | 0     |
| Emotional action film | 1      | 0      | 1     |
| Family drama          | 0      | 0      | 1     |

Unlike multiclass classification, the model does not need to choose only one category.

In a real application, evaluate multilabel predictions with metrics such as micro/macro F1, per-label precision and recall, and Hamming loss. The default threshold and the model's output behavior should be considered carefully.

# 14. Using text classification with deep learning and transformers

Traditional machine learning is an excellent starting point, but let's understand how a modern NLP classification system works.

A typical neural text classification pipeline is:

Raw text

"I really enjoyed this movie"

Tokenizer

Convert text into token IDs

Embedding and encoder layers

Learn contextual representations

Classification head

Convert representation into class scores

Predicted label

Positive sentiment

## 14.1 Using a pretrained sentiment classifier

The Hugging Face Transformers library provides pretrained classification pipelines.

Install the packages:

```
pip install transformers torch
```

Example:

```
from transformers import pipelineclassifier = pipeline(    "sentiment-analysis",    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")results = classifier([    "This movie was absolutely brilliant!",    "I disliked the story and the acting."])print(results)
```

The results contain a predicted label and a score for each input.

This particular pretrained model is intended for English positive/negative sentiment classification. It does not automatically solve arbitrary tasks such as classifying news categories or movie genres.

For a custom task, you generally need a suitable pretrained model and labeled data for fine-tuning, or embeddings paired with a separate classifier.

When to use transformers: Consider them when contextual meaning is important, a suitable pretrained model exists, and you have enough computing resources. For a first practical baseline, TF-IDF and Logistic Regression remain excellent choices.

# 15. Saving and loading your trained classifier

When your model is ready, you can save it and use it later without retraining every time.

Because the Pipeline contains both TF-IDF and Logistic Regression, you only need to save the complete Pipeline.

```
import joblibjoblib.dump(    text_classifier,    "text_classifier.joblib")
```

Load it in another Python session:

```
import joblibloaded_model = joblib.load(    "text_classifier.joblib")reviews = [    "The movie was excellent",    "The movie was disappointing"]predictions = loaded_model.predict(reviews)print(predictions)
```

The loaded Pipeline automatically converts raw text into TF-IDF features before making predictions.

Security note: Only load joblib or pickle files from trusted sources because loading them can execute arbitrary code. Also preserve your dependency versions and validate the model before deployment.

# 16. Deploying a text classification model

A trained classifier can be exposed through an API so that a frontend application can send text and receive predictions.

For example, using FastAPI:

```
pip install fastapi uvicorn joblib
```

Create `main.py`:

```
import joblibfrom fastapi import FastAPIfrom pydantic import BaseModel, Fieldapp = FastAPI()model = joblib.load("text_classifier.joblib")class TextRequest(BaseModel):    text: str = Field(min_length=1, max_length=10000)@app.post("/predict")def predict(request: TextRequest):    prediction = model.predict([request.text])[0]    result = {        "text": request.text,        "prediction": str(prediction)    }    if hasattr(model, "predict_proba"):        probabilities = model.predict_proba([request.text])[0]        result["probabilities"] = {            str(label): float(probability)            for label, probability in zip(                model.classes_,                probabilities            )        }    return result
```

Run the application:

```
uvicorn main:app --reload
```

Then open `http://127.0.0.1:8000/docs` to test the endpoint through FastAPI's interactive documentation.

This is a basic local demonstration. A production API also needs suitable error handling, input validation, authentication where required, resource limits, logging, and monitoring.

# 17. Common mistakes in text classification

These are particularly important when building your first NLP projects.

1\. Fitting the vectorizer before splitting the data

This leaks information about the test vocabulary into training. Split first, then fit the vectorizer on training text only. A Pipeline helps enforce this separation.

2\. Removing stop words indiscriminately

Removing `not` from "not good" can reverse sentiment. Preserve negation and other meaningful features.

3\. Evaluating only with accuracy

A model can perform well on the majority class while failing on minority classes. Check per-class precision, recall, F1, and the confusion matrix.

4\. Using too little training data

A tiny demonstration dataset cannot represent the language patterns found in real reviews, emails, or documents. Use sufficient, diverse, accurately labeled examples.

5\. Creating a new vectorizer for every prediction

The vocabulary and feature ordering must remain consistent. Reuse the fitted vectorizer or, preferably, save the complete Pipeline.

6\. Allowing duplicate samples across the split

Duplicate or near-duplicate texts in training and testing can inflate evaluation scores. Deduplicate where appropriate, and consider group-based splitting when multiple records come from the same source.

7\. Using a random split for every problem

If you predict future messages or news, a chronological split may better represent actual deployment. The splitting strategy should match how the model will be used.

# 18. A practical learning roadmap

Since you are learning NLP progressively, I recommend working through text classification in this order.

## Your progress checklist

0/9 complete

1\. Understand classification types

Binary, multiclass and multilabel classification.

2\. Implement Bag of Words

Use CountVectorizer and inspect the document-term matrix.

3\. Implement TF-IDF

Experiment with n-grams, min_df and max_df.

4\. Train traditional classifiers

Compare Multinomial Naive Bayes, Logistic Regression and Linear SVM.

5\. Evaluate performance

Study precision, recall, F1, confusion matrices and ROC-AUC.

6\. Build a complete project

Train and evaluate a movie-review sentiment classifier.

7\. Tune the model

Use cross-validation, GridSearchCV and appropriate metrics.

8\. Learn embeddings and transformers

Move from Word2Vec to sentence embeddings and BERT fine-tuning.

9\. Deploy your model

Save the Pipeline and expose predictions through FastAPI.

## Final summary

Text classification converts text into categories using a combination of text representation and a learning algorithm.

The fundamental workflow is:

1. Collect labeled text.
2. Split it into training and testing sets.
3. Preprocess the text when necessary.
4. Convert the text into features using BoW, TF-IDF, or embeddings.
5. Train a classifier.
6. Evaluate it using appropriate metrics.
7. Tune, save, and deploy the model.

The most useful first implementation to master is TF-IDF + Logistic Regression in a scikit-learn Pipeline. Once you understand that workflow thoroughly, compare it against Naive Bayes and Linear SVM, then move to Word2Vec, sentence embeddings, and transformer-based classification.