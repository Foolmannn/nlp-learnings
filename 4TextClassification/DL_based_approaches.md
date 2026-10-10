# Text Classification Using Deep Learning (DL) in NLP

Since you've already studied traditional machine learning approaches for text classification, such as Naive Bayes, Logistic Regression, SVM, and TF-IDF, let's move to deep learning approaches in detail.

The main difference is that traditional ML usually depends on manually choosing a text representation such as Bag of Words or TF-IDF, whereas deep learning can learn useful text representations during training using neural networks.

We'll cover the concepts, mathematics, Python implementations, model evaluation, and when to use each approach.

## 1. What is deep learning-based text classification?

Text classification is the task of assigning a predefined category or label to a piece of text.

Examples:

| Text                                         | Classification     |
| -------------------------------------------- | ------------------ |
| "This movie was absolutely fantastic!"       | Positive sentiment |
| "The product stopped working after one day." | Negative sentiment |
| "Your account has won a prize. Click here!"  | Spam               |
| "The government announced a new policy."     | Politics / News    |
| "I need to reset my password."               | Technical support  |

A deep learning model learns patterns from labeled examples and uses those patterns to predict labels for unseen text.

Raw text

"This movie is amazing"

Tokenization

["This", "movie", "is", "amazing"]

Numerical representation

Token IDs → embedding vectors

Deep learning model

CNN, RNN, LSTM, GRU, or Transformer

Classification output

## Positive — 96%

Illustrative probability, not a trained model result

The key point is that a neural network does not directly understand raw text. Text must first be converted into numerical inputs.

## 2. How deep learning differs from traditional ML

| Feature               | Traditional ML                        | Deep learning                                        |
| --------------------- | ------------------------------------- | ---------------------------------------------------- |
| Common representation | BoW, TF-IDF                           | Learned embeddings, contextual embeddings            |
| Feature engineering   | Often more manual                     | Can learn useful features automatically              |
| Models                | Naive Bayes, SVM, Logistic Regression | CNN, RNN, LSTM, Transformer                          |
| Training data         | Often effective with small datasets   | Neural networks may need more data                   |
| Computing needs       | Often low                             | Can be high, especially for Transformers             |
| Context understanding | Limited with simple TF-IDF            | Can model word order and context                     |
| Best starting point   | TF-IDF + Logistic Regression          | Embedding + neural network or pretrained Transformer |

Important: Deep learning is not automatically better. For a small dataset, TF-IDF with Logistic Regression or Linear SVM can outperform a neural network, particularly when the neural network is trained from scratch.

## 3. The major deep learning approaches


1\. Feedforward Neural Network (MLP)
![Shallow Neural Networks](feed_forward_nn.jpg)

A basic neural network that classifies text using TF-IDF features or embeddings. A useful starting point for understanding neural classification.


2\. CNN for text
![Chapter 5: Text Classification and Regression Using AutoKeras | Automated Machine Learning with AutoKeras](CNN.jpg)

Uses convolutional filters to detect local patterns, such as "not good" or "highly recommended."


3\. RNN
![Language Models and Contextualised Word Embeddings](RNN_text.jpg)

Processes tokens sequentially, maintaining a hidden state that carries information from earlier tokens.


4\. LSTM and GRU
![Stock Price Prediction using Machine Learning with Source Code](LSTM.jpg)

Improved recurrent networks designed to learn longer dependencies in text more effectively than a basic RNN.


5\. Transformers and BERT

![Sentiment Analysis in Portuguese Restaurant Reviews: Application of Transformer Models in Edge Computing](dl_transformer_bert.jpg)
Use attention to relate different tokens to one another. Pretrained Transformer models are often the strongest practical starting point for high-quality text classification.

We'll start with the foundation—embeddings and a simple neural network—before moving to sequential models and Transformers.

## 4. Text embeddings: the foundation of neural text classification

Before building a neural network, you need to understand word embeddings.

In traditional ML, TF-IDF represents a document using a vector whose dimensions correspond to vocabulary terms. For example:

Vocabulary:

`["movie", "good", "bad", "amazing"]`

A document might be represented as:

\\[ x=[1,1,0,1] \\]

This representation records term occurrence or importance, but it doesn't inherently represent semantic similarity. The words good and excellent are separate features.

An embedding instead represents each token as a dense, learned vector.

For example, using imaginary two-dimensional embeddings:

| Word      | Embedding     |
| --------- | ------------- |
| movie     | `[0.2, 0.7]`  |
| good      | `[0.8, 0.6]`  |
| excellent | `[0.9, 0.7]`  |
| terrible  | `[-0.7, 0.1]` |

These numbers are illustrative. In a real model, embeddings usually have many more dimensions.

Words used in similar contexts may develop similar representations. The model learns these representations from training data or starts with embeddings learned during pretraining.

### How does an embedding layer work?

Suppose your vocabulary contains 10,000 tokens and each embedding has 128 dimensions.

The embedding matrix has shape:

\\[ E\in\mathbb{R}^{10000\times128} \\]

Each token ID selects one row of this matrix.

If a review has 100 tokens, the embedding layer produces:

\\[ (100,128) \\]

For a batch of 32 reviews, the output shape is:

\\[ (32,100,128) \\]

The dimensions mean:

- `32`: number of reviews in the batch.
- `100`: maximum number of tokens per review.
- `128`: embedding dimensions per token.

Unlike TF-IDF, embeddings preserve the sequence of tokens, allowing later neural network layers to learn word order and patterns.

### Padding and truncation

Neural networks commonly process batches of equal-length sequences. Real reviews, however, have different lengths.

Suppose the maximum length is five tokens:

```
Review 1: [12, 45, 78]
Review 2: [11, 22, 33, 44, 55, 66]
```

After padding and truncation:

```
Review 1: [12, 45, 78,  0,  0]
Review 2: [11, 22, 33, 44, 55]
```

Here, `0` represents padding, and the second review has been truncated to five tokens.

Choose a maximum length based on your dataset's text-length distribution. Excessively long sequences increase computational cost; excessively short sequences discard useful information.

## 5. Approach 1: Feedforward Neural Network (MLP)

A Multilayer Perceptron (MLP) is one of the simplest deep learning models for text classification.

It takes a numerical feature vector, passes it through one or more hidden layers, and predicts a class.

\\[ \text{Text}\rightarrow\text{Features}\rightarrow \text{Dense layers}\rightarrow\text{Output} \\]

### Architecture

Input layer

TF-IDF vector or flattened/pooled embeddings

Dense layer: 128 neurons

ReLU activation

Dropout

Regularization to reduce overfitting

Output layer

Sigmoid for binary classification; softmax for multiclass

### Mathematical understanding

For an input vector \\(x\\), a hidden layer computes:

\\[ h=\operatorname{ReLU}(Wx+b) \\]

Here:

- \\(x\\) is the input feature vector.
- \\(W\\) is the weight matrix.
- \\(b\\) is the bias vector.
- ReLU is the activation function.

\\[ \operatorname{ReLU}(z)=\max(0,z) \\]

The output layer then converts the hidden representation into a prediction. For binary classification, the sigmoid function is commonly used:

\\[ \hat y=\sigma(z)=\frac{1}{1+e^{-z}} \\]

The resulting value lies between zero and one and can be interpreted as the model's estimated probability for the positive class.

### Complete implementation using TensorFlow/Keras

Install the required libraries in your environment:

```
pip install tensorflow pandas scikit-learn
```

Assume you have a CSV file called `reviews.csv` with these columns:

| review                   | sentiment |
| ------------------------ | --------- |
| This movie was fantastic | positive  |
| The story was boring     | negative  |
| Acting was excellent     | positive  |

The code below trains an MLP on learned word embeddings. It assumes a binary classification problem with `positive` and `negative` labels.

```
import pandas as pdimport numpy as npimport tensorflow as tffrom sklearn.model_selection import train_test_splitfrom sklearn.preprocessing import LabelEncoderfrom sklearn.metrics import (    classification_report,    confusion_matrix,    accuracy_score)from tensorflow.keras import layers, models# 1. Load the datasetdf = pd.read_csv("reviews.csv")df = df.dropna(subset=["review", "sentiment"]).copy()df["review"] = df["review"].astype(str)df["sentiment"] = df["sentiment"].astype(str).str.strip()# 2. Encode labels as 0 and 1encoder = LabelEncoder()y = encoder.fit_transform(df["sentiment"])X = df["review"].to_numpy()# 3. Split before fitting text preprocessingX_train, X_test, y_train, y_test = train_test_split(    X,    y,    test_size=0.2,    random_state=42,    stratify=y)
```

A subtle point about this architecture: `GlobalAveragePooling1D` averages token embeddings into one document vector. It reduces computational complexity, but it largely discards explicit word order. This makes it a good baseline, not a model that fully understands sentence structure.

Also, `max_tokens` is a vocabulary limit, not a guarantee that every possible token will be retained. The vectorizer handles out-of-vocabulary tokens using its reserved vocabulary entries.

### What happens during training?

1. The vectorizer converts text to token IDs.
2. The embedding layer looks up a vector for each token.
3. Global average pooling creates one vector for each review.
4. Dense layers learn nonlinear patterns.
5. The sigmoid produces a probability.
6. Binary cross-entropy measures the prediction error.
7. Backpropagation calculates gradients, and Adam updates the trainable weights.

For binary classification, binary cross-entropy is:

\\[ L=-\frac{1}{N}\sum\_{i=1}^{N} \left[y_i\log(\hat y_i)+(1-y_i)\log(1-\hat y_i)\right] \\]

The model learns by reducing this loss across training examples.

When to use an MLP: As a first neural baseline, especially when you want to understand embeddings, loss functions, backpropagation, and training before moving to more complex architectures.

## 6. Approach 2: CNN for text classification

CNNs (Convolutional Neural Networks) are known for image processing, but they can also classify text.

A text CNN applies filters over sequences of word embeddings to detect local word patterns.

For example, consider:

> "The movie was not good at all."

A CNN can learn useful patterns involving adjacent words, such as:

- "not good"
- "very good"
- "really disappointing"
- "highly recommended"

The filters learn which patterns are useful for predicting the class.

### CNN architecture

Token IDs: `[12, 45, 78, 91, ...]`

Embedding layer

Sequence of word vectors

1D convolution

Detect local patterns across neighboring tokens

Global max pooling

Keep the strongest detected features

Dense layer → class prediction

### How does convolution work?

Suppose each token has an embedding of dimension 64. A sequence of 100 tokens becomes a matrix of shape:

\\[ X\in\mathbb{R}^{100\times64} \\]

A convolutional filter with a kernel size of 3 examines three neighboring tokens at a time. It slides over the sequence and computes a feature at each position.

Conceptually, it detects patterns of the form:

\\[ [\text{word}\_{i},\text{word}\_{i+1},\text{word}\_{i+2}] \\]

The filter weights are learned during training. They are not manually programmed to recognize particular phrases.

### CNN implementation

This example uses the same `X_train`, `X_test`, `y_train`, `y_test`, and `vectorizer` from the preceding example.

```
    layers.Input(shape=(), dtype=tf.string),    vectorizer,    layers.Embedding(        input_dim=max_tokens,        output_dim=128,        mask_zero=False    ),    # Detect local patterns across 3 neighboring tokens    layers.Conv1D(        filters=128,        kernel_size=3,        activation="relu",        padding="valid"    ),    # Keep the strongest response from each filter    layers.GlobalMaxPooling1D(),    layers.Dense(64, activation="relu"),    layers.Dropout(0.4),    layers.Dense(1, activation="sigmoid")])cnn_model.compile(    optimizer="adam",    loss="binary_crossentropy",    metrics=["accuracy"])cnn_history = cnn_model.fit(    X_train,    y_train,    validation_split=0.2,    epochs=10,    batch_size=32,    callbacks=[early_stopping])cnn_model.evaluate(X_test, y_test)
```

Notice that `mask_zero=False` is used here. Standard convolution does not automatically interpret token ID zero as padding to be ignored. Padding can therefore influence CNN features. For a simple baseline this may be acceptable, but for more careful modeling you can use valid-length handling, masking-compatible architectures, or pretrained models with attention masks.

Advantages of CNNs

- Can train in parallel across token positions.
- Effective at learning local phrase patterns.
- Often computationally lighter than recurrent networks for comparable sequence lengths.

Limitations

- Local filters don't inherently provide a direct mechanism for modeling arbitrarily distant token relationships.
- Larger receptive fields or deeper layers are needed to combine distant information.

When to use CNN: Short text classification, sentiment analysis, spam detection, and tasks where local phrases are strong indicators of the class.

## 7. Approach 3: Recurrent Neural Networks (RNNs)

Unlike a CNN, which processes local windows, an RNN processes a text sequence one token at a time.

Consider this sentence:

> "The movie started slowly, but the ending was fantastic."

An RNN reads each token in order and updates its hidden state, which carries information from earlier tokens.

### Architecture

Sequence processed from left to right

The

movie

was

fantastic

\\(h\_{n}\\)

Hidden state

\\(h\_{n}\\)

Hidden state

\\(h\_{n}\\)

Hidden state

\\(h\_{n}\\)

Hidden state

Each state uses the current token and the previous hidden state.

Final hidden representation → classifier

### RNN mathematics

At each time step \\(t\\), the model calculates a new hidden state:

\\[ h_t=\tanh(W_xx_t+W_hh\_{t-1}+b_h) \\]

Where:

- \\(x_t\\): embedding of the current token.
- \\(h\_{t-1}\\): previous hidden state.
- \\(W_x,W_h\\): trainable weight matrices.
- \\(b_h\\): bias.
- \\(\tanh\\): activation function.

The classifier uses the hidden representation to produce a prediction.

### Why can a basic RNN struggle?

During training, the gradient must flow through many sequential steps. For long sequences, gradients may become extremely small or large. This is known as the vanishing gradient problem or exploding gradient problem.

A basic RNN may consequently struggle to retain information from far earlier in a long text.

For example, the word "not" in "I thought the movie would be good, but it was not good" can be important to the final sentiment. A model needs a way to retain relevant context while processing the sequence.

This motivates LSTM and GRU architectures.

## 8. Approach 4: LSTM and GRU

LSTM stands for Long Short-Term Memory. GRU stands for Gated Recurrent Unit.

Both are recurrent neural networks designed to control what information is retained, updated, and forgotten.

### LSTM: how the memory mechanism works

An LSTM maintains a hidden state \\(h_t\\) and a cell state \\(c_t\\). Its gates control the flow of information.

Forget gate

What should be removed?

Decides which information in the previous cell state is no longer useful.

Input gate

What new information should be stored?

Controls how much of the candidate information is added to memory.

Output gate

What information should be exposed?

Determines which parts of the cell state form the current hidden state.

The forget gate, for example, is computed as:

\\[ f_t=\sigma(W_f[h\_{t-1},x_t]+b_f) \\]

The cell state is updated approximately as:

\\[ c_t=f_t\odot c\_{t-1}+i_t\odot\tilde c_t \\]

Here, \\(i_t\\) is the input gate, \\(\tilde c_t\\) is candidate memory, and \\(\odot\\) denotes element-wise multiplication.

These gates help the network retain useful information across longer sequences.

### LSTM implementation

```
lstm_model = models.Sequential([    layers.Input(shape=(), dtype=tf.string),    vectorizer,    layers.Embedding(        input_dim=max_tokens,        output_dim=128,        mask_zero=True    ),    # Read the token sequence sequentially    layers.LSTM(64),    layers.Dropout(0.4),    layers.Dense(32, activation="relu"),    layers.Dense(1, activation="sigmoid")])lstm_model.compile(    optimizer="adam",    loss="binary_crossentropy",    metrics=["accuracy"])lstm_model.fit(    X_train,    y_train,    validation_split=0.2,    epochs=10,    batch_size=32,    callbacks=[early_stopping])lstm_model.evaluate(X_test, y_test)
```

### What is the difference between LSTM and GRU?

| Feature             | LSTM                                            | GRU                                         |
| ------------------- | ----------------------------------------------- | ------------------------------------------- |
| Memory components   | Hidden state + cell state                       | Hidden state                                |
| Gates               | Three main gates                                | Two main gates                              |
| Parameters          | Usually more                                    | Usually fewer                               |
| Training speed      | Can be slower                                   | Often faster                                |
| Long-range learning | Designed to help retain long-term information   | Also designed to handle longer dependencies |
| Practical choice    | Useful when its additional memory control helps | Useful as a simpler recurrent baseline      |

Neither is guaranteed to outperform the other. Performance depends on the dataset, sequence length, architecture, and training setup.

To use a GRU in the preceding model, replace:

```
layers.LSTM(64)
```

with:

```
layers.GRU(64)
```

When to use LSTM/GRU: When sequence order matters and you want to learn recurrent neural networks, especially for sequential or time-dependent text tasks. They are valuable learning steps, although pretrained Transformers are often preferred for many modern NLP applications.

## 9. Approach 5: Transformer-based text classification

Transformers are among the most important deep learning architectures in modern NLP.

Models such as BERT learn contextual representations: a word's representation depends on the surrounding words.

For example:

- "I deposited money at the bank."
- "I sat on the river bank."

A static embedding may give the word `bank` the same base vector in both sentences. A contextual model can generate different representations based on the meaning in each context.

### How does a Transformer work?

A Transformer encoder typically contains these major steps:

1\. Tokenization

Convert text into model-specific token IDs

2\. Token + position information

Represent token identity and sequence position

3\. Self-attention

Allow each token to use information from other tokens

4\. Feedforward layers

Transform the contextual token representations

Steps 3–4 are repeated across multiple Transformer layers.

5\. Classification head

Produce class logits and probabilities

### Understanding self-attention

Self-attention allows each token to determine which other tokens are useful for representing its meaning.

For each token representation, the model computes a query \\(Q\\), key \\(K\\), and value \\(V\\). Attention is calculated as:

\\[ \operatorname{Attention}(Q,K,V)= \operatorname{softmax} \left(\frac{QK^\top}{\sqrt{d_k}}\right)V \\]

Where:

- \\(QK^\top\\) measures relationships between tokens.
- \\(d_k\\) is the key-vector dimension.
- Softmax converts scores into normalized weights.
- \\(V\\) supplies the information combined using those weights.

For example, in "The movie was not good," the model can learn relationships between `not` and `good`, helping it represent the negation.

This is a learned mechanism—not a hard-coded rule that always interprets negation correctly.

### What is BERT?

BERT stands for Bidirectional Encoder Representations from Transformers.

A pretrained BERT model has already learned language patterns from a large corpus of text. For a classification task, we typically:

1. Load a pretrained tokenizer and model.
2. Add or use a classification head.
3. Fine-tune the model on labeled examples.
4. Evaluate it on held-out data.

Fine-tuning adapts pretrained weights to your specific task, such as sentiment analysis, spam detection, or topic classification.

### Fine-tune BERT using Hugging Face Transformers

This is a practical approach for real NLP projects. We'll use the `transformers` and `datasets` libraries.

Install the dependencies:

```
pip install transformers datasets accelerate scikit-learn torch
```

Assume `reviews.csv` has `review` and `sentiment` columns, with two classes: `negative` and `positive`.

The following code fine-tunes a pretrained DistilBERT model, a smaller Transformer in the BERT family.

```
import pandas as pdimport numpy as npimport torchfrom sklearn.model_selection import train_test_splitfrom sklearn.preprocessing import LabelEncoderfrom sklearn.metrics import classification_report, accuracy_scorefrom datasets import Datasetfrom transformers import (    AutoTokenizer,    AutoModelForSequenceClassification,    TrainingArguments,    Trainer,    DataCollatorWithPadding)# 1. Load and clean the datasetdf = pd.read_csv("reviews.csv")df = df.dropna(subset=["review", "sentiment"]).copy()df["review"] = df["review"].astype(str)df["sentiment"] = df["sentiment"].astype(str).str.strip()# 2. Encode class labelsencoder = LabelEncoder()df["label"] = encoder.fit_transform(df["sentiment"])# 3. Split the data before trainingtrain_df, test_df = train_test_split(    df[["review", "label"]],    test_size=0.2,    random_state=42,    stratify=df["label"])
```

A few practical notes:

- `max_length=256` truncates longer tokenized reviews. Increase it if longer context is essential and your hardware can handle it.
- Batch size and training speed depend on available memory and whether you have a GPU. If you run out of memory, reduce the batch size or sequence length.
- The first run downloads pretrained model files, so an internet connection is needed.
- The model checkpoint must support the tokenizer and sequence-classification architecture being used.
- The code uses `processing_class=tokenizer`, which is supported by current Transformers versions. If using an older version that doesn't accept this argument, update Transformers or use the version-appropriate tokenizer argument.

### Test the saved model on new text

```
from transformers import pipelineclassifier = pipeline(    "text-classification",    model="./saved-text-classifier",    tokenizer="./saved-text-classifier")print(classifier(    "The movie was beautifully written and performed."))
```

This produces a predicted class and a model score. The score is not automatically a perfectly calibrated probability.

### Why use a pretrained Transformer instead of training from scratch?

Training a Transformer from scratch requires large amounts of data and computing resources. Fine-tuning reuses language knowledge learned during pretraining, often providing better results with a much smaller labeled dataset.

When to use a Transformer: When classification accuracy matters, context is important, and you have sufficient computing resources for fine-tuning—or can use a suitable pretrained model.

## 10. Choosing the correct output layer and loss function

The correct output layer depends on your classification problem. This applies to all the architectures discussed above.

### A. Binary classification

Example: positive vs. negative sentiment.

Output layer:

```
layers.Dense(1, activation="sigmoid")
```

Loss:

```
loss="binary_crossentropy"
```

The output is one score between zero and one. A threshold such as `0.5` converts it into a predicted class.

### B. Multiclass classification

Example: classifying a news article as sports, politics, technology, or entertainment. Each article belongs to exactly one class.

Output layer:

```
num_classes = 4layers.Dense(num_classes, activation="softmax")
```

Loss:

```
loss="sparse_categorical_crossentropy"
```

Your labels should be integer class IDs such as `0`, `1`, `2`, and `3`.

Softmax converts logits into a probability distribution:

\\[ P(y=i\mid x)=\frac{e^{z_i}}{\sum\_{j=1}^{C}e^{z_j}} \\]

The probabilities sum to one, and the model usually predicts the class with the largest probability.

### C. Multilabel classification

Example: one article may simultaneously concern technology, politics, and business.

Each class is predicted independently, so several classes can be positive at once.

Output layer:

```
num_labels = 5layers.Dense(num_labels, activation="sigmoid")
```

Loss:

```
loss="binary_crossentropy"
```

Unlike multiclass classification, the outputs do not need to sum to one. Each label has its own target of zero or one.

| Task                       | Output activation | Loss                             |
| -------------------------- | ----------------- | -------------------------------- |
| Binary                     | Sigmoid           | Binary cross-entropy             |
| Multiclass, integer labels | Softmax           | Sparse categorical cross-entropy |
| Multiclass, one-hot labels | Softmax           | Categorical cross-entropy        |
| Multilabel                 | Sigmoid per label | Binary cross-entropy             |

Common mistake: Using softmax for a multilabel problem. Softmax assumes the classes compete with one another, whereas multilabel tasks allow multiple positive labels.

## 11. How to evaluate deep learning text classifiers

Accuracy alone is not enough, particularly if your dataset is imbalanced.

Imagine a spam dataset containing 950 legitimate messages and 50 spam messages. A model that predicts every message as legitimate achieves 95% accuracy but detects no spam.

Use several metrics:

- Precision: Of the examples predicted positive, how many were actually positive?
- Recall: Of all actual positive examples, how many did the model detect?
- F1-score: The harmonic mean of precision and recall.
- Confusion matrix: Shows which classes the model confuses.
- Macro F1: Calculates F1 independently for each class and averages them equally, making it useful when class sizes differ.

For the Keras models, you can evaluate predictions with scikit-learn:

```
from sklearn.metrics import (    classification_report,    confusion_matrix,    f1_score)probabilities = lstm_model.predict(X_test, verbose=0).ravel()y_pred = (probabilities >= 0.5).astype(int)print("Macro F1:", f1_score(y_test, y_pred, average="macro"))print(confusion_matrix(y_test, y_pred))print(classification_report(y_test, y_pred))
```

For a multiclass neural network, use `argmax` on the class probabilities instead of the binary threshold.

### Watch out for overfitting

A model may achieve excellent training accuracy but perform poorly on new text.

Healthy training

Training loss and validation loss both decrease, then stabilize.

Possible overfitting

Training loss continues decreasing while validation loss starts increasing.

Ways to reduce overfitting include dropout, weight decay, early stopping, more representative training data, and reducing model complexity.

Keep a separate test set for final evaluation. Use the validation set for selecting epochs, hyperparameters, and other modeling decisions.

## 12. Comparing the deep learning approaches

| Approach         | Learns sequence order?                          | Strengths                                     | Limitations                          |
| ---------------- | ----------------------------------------------- | --------------------------------------------- | ------------------------------------ |
| MLP + embeddings | Mostly loses order after pooling                | Simple baseline                               | Limited sequence modeling            |
| CNN              | Local patterns                                  | Fast, effective phrase detection              | Less direct long-range modeling      |
| Basic RNN        | Yes                                             | Sequential processing                         | Vanishing/exploding gradients        |
| LSTM             | Yes                                             | Gated memory for longer dependencies          | Sequential computation can be slow   |
| GRU              | Yes                                             | Simpler gated recurrent model                 | Still processes tokens sequentially  |
| Transformer/BERT | Yes, through attention and position information | Contextual representations, transfer learning | More compute and memory requirements |

## 13. Which approach should you learn and implement first?

Since you have already studied TF-IDF, Word2Vec, and traditional ML classification, I recommend this sequence:

1. Embedding + MLP: Understand token IDs, embeddings, output layers, cross-entropy, backpropagation, and training/validation curves.
2. Text CNN: Learn convolution filters, kernel size, feature maps, and pooling.
3. RNN, LSTM, and GRU: Understand hidden states, sequence processing, gating, and vanishing gradients.
4. Transformers and BERT: Learn attention, tokenization, pretrained models, fine-tuning, and inference.
5. Deployment: Save the selected model, create an inference function, and expose it through a FastAPI endpoint.

For your first experiment, train an MLP, CNN, and LSTM on the same dataset and train/test split. Compare their macro F1, training time, and validation loss. Then fine-tune DistilBERT and compare it against those baselines.

One important experimental rule: do not choose a model based only on training accuracy. The model that performs best on unseen data is the more useful classifier.