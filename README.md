![Alt](/monkeypox-small.jpg "monkeypox-image")

# Monkeypox misinformation detector project

## 1. Overview of repository contents

This repository contains source code and other materials developed as part of a machine learning project on **monkeypox misinformation detection**. The project involved the following main outputs:

1. The creation of an [annotated dataset of monkeypox](https://www.kaggle.com/datasets/stephencrone/monkeypox) related Twitter misinformation (using **tweet-collector.py**).
2. The development and evaluation of a [transformer model](https://huggingface.co/smcrone/monkeypox-misinformation) (using **model-experiments.ipynb**) that is trained to accurately classify monkeypox misinformation. (See sub-directories for experimental data.)
3. The proposal of a function designed to identify misinformation 'superspreaders' on Twitter and the optimisation of this function's hyperparameters (using **ss-hyperparam-calc.py**).
4. The development of a Streamlit web app in which the aforementioned transformer model and superspreader measure could be integrated (see **big.py**). This app can be found on [Streamlit Cloud](https://stephen-crone-monkeypox-big-bk7q0p.streamlitapp.com/) and on [HuggingFace Spaces](https://huggingface.co/spaces/smcrone/monkeypox).
5. The re-evaluation of the final model (using **model-retesting.ipynb**) based on a more [recently collected dataset](https://www.kaggle.com/datasets/stephencrone/monkeypox), in order to gauge the effects of temporal shift on model performance.

## 2. Guide to experimental data sub-directories

In order to pick a winning model, experiments were conducted in model-experiments.ipynb using three different paradigms: zero-shot learning, few-shot learning and the fine-tuning approach. Experimental data for each of these approaches is stored in the repository in a dedicated sub-directory.

### 2.1. Zero-shot learning data

Files names capture: (i) the model used in the experiment (either bart-large-mlni or nli-distilroberta-base); (ii) the labels used to guide the model (e.g. 'true' indicates 'true / false' label options); and (iii) the version of the monkeypox dataset that was used.

Each xlsx file itself contains three sheets. One of these is a 'metadata' sheet that stores much the same information conveyed by the file name itself. Another sheet ('class. report') contains an sklearn classification report based on the data in the 'data' sheet. The 'data' sheet, meanwhile, contains six columns:

| Column name | Description |
|-------------|-------------|
|text|the tweet text to be predicted|
|class|the ground truth class to which the example belongs|
|class zero label|the confidence that the example belongs to whatever the zero class label is|
|class one label|the confidence that the example belongs to whatever the one class label is|
|predictedLabel|the prediction of the model based on confidence scores generated|
|timeRequired|the number of seconds taken to produce the prediction|

### 2.2. Few-shot learning data

The few-shot learning data files are similar to the zero-shot learning files, but with some differences. In terms of the file names themselves, these reflect: (i) the number of examples (i.e. 'shots') that the model gets, ranging from six at the lowest to ten at the highest; and (ii) similar to the zero-shot files, the class labels that were used. (All experiments with this paradigm were conducted with GPT-J 6b -- hence why this detail is omitted from the file name.)

In terms of the file contents, the only differences compared to the zero-shot learning files are as follows:

1. The full 'prompt head' is provided as metadata, which contains the entire string that was presented to the model as context / examples.
2. The 'data' sheet does not include confidence scores for each example for the two classes -- only the ground truth and predicted labels.

In the 'Effects of minor changes' sub-directory, there are some additional experiments in which the ordering and balance of examples were perturbed to observe the effects on model performance.

### 2.3. Fine-tuning data

These experiments were conducted in three rounds (each focusing on optimising a particular aspect of the model design and training), and this is reflected in the sub-directory structure (which has three sub-folders). In terms of file naming conventions:

1. Round 1 file names capture: (i) the amount of training data used to train the model (either 100 per cent or 75 per cent); and (ii) the name of the model used.
2. Round 2 file names capture: (i) the name of the model used; and (ii) the combination of dataset features that were concatenated and fed into the transformer model.
3. Round 3 file names capture (i) the name of the model used; and (ii) the alteration made to default hyperparameters.

The content of files from all three rounds is fundamentally the same. One sheet contains the data collected from training; the other contains a classification report from evaluation of the model on test data. In terms of the features of the training data sheet, see the following explanatory notes:

| Column name | Description |
|-------------|-------------|
|loss| the training loss|
|sparse_categorical_accuracy| the training accuracy|
|val_loss| the validation loss|
|val_sparse_categorical_accuracy| the validation accuracy|
|lr| the learning rate|
|time_per_epoch| the amount of time (in seconds) taken per epoch of training|
|datasetChoice| whether the bigger or smaller dataset version was chosen (which corresponds with the choice of labelling scheme)|
|percentageKept| the percentage of the overall dataset that was used for training, validation and testing|
|samplesUsed| the number of samples that were used for training, validation and testing|
|trainSize| the percentage size of the training set relative to the size of the dataset overall|
|valSize| the percentage size of the validation set relative to the size of the dataset overall|
|testSize| the percentage size of the test set relative to the size of the dataset overall|
|model| the model used (e.g. BERT-base-cased, RoBERTa-base)|
|batchSize| the batch size used for training|
|LRstrategy| whether the learning rate was conditioned to decay or reduce on plateau|
|dropoutRate| the dropout rate for the dropout layer preceding the classification head (where applicable)|
|number| the internal id number of the tweet -- boolean denotes whether this feature was fed to model during training|
|created_at| the datetime that the tweet was published -- boolean denotes whether this feature was fed to model during training|
|text| the tweet text -- boolean denotes whether this feature was fed to model during training|
|source| the platform / tool used to publish the tweet -- boolean denotes whether this feature was fed to model during training|
|user is verified| whether the user is verified by Twitter -- boolean denotes whether this feature was fed to model during training|
|user has url| whether the user includes a URL in their profile -- boolean denotes whether this feature was fed to model during training|
|user description| the user description provided by the user -- boolean denotes whether this feature was fed to model during training|
|user created at| when the user profile was created -- boolean denotes whether this feature was fed to model during training|
|retweet_count| the number of retweets the tweet received -- boolean denotes whether this feature was fed to model during training|
|reply_count|	the number of replies the tweet received -- boolean denotes whether this feature was fed to model during training|
|like_count| the number of likes the tweet received -- boolean denotes whether this feature was fed to model during training|
|quote_count| the number of quote tweets the tweet received -- boolean denotes whether this feature was fed to model during training|
|followers count| the number of followers that the user has -- boolean denotes whether this feature was fed to model during training|
|following count| the number of accounts that the user follows -- boolean denotes whether this feature was fed to model during training|
|tweet count| the number of tweets published by the user -- boolean denotes whether this feature was fed to model during training|
|listed_count| the number of Twitter lists to which the user belongs -- boolean denotes whether this feature was fed to model during training|
|user location| user-supplied description of their location -- boolean denotes whether this feature was fed to model during training|
|years since account created| the number of years since user's account was created -- boolean denotes whether this feature was fed to model during training|
|tweets per day| the number of tweets that the user publishes per day on average -- boolean denotes whether this feature was fed to model during training|
|follower to following ratio| the user's follower-to-following ratio -- boolean denotes whether this feature was fed to model during training|
|isFastTokenizer| whether a fast tokenizer was used|
|optimizer| which optimizer was used|
|serialNumber| a number uniquely identifying the row using datetime information|

For any data files where one or more of these features is not present, it can be inferred that the default / baseline option was used. These were learning rate 5e-6 (reducing on plateau with patience of two epochs), using the Adam optimizer. Unless otherwise specified, the batch size used with the maximum achievable based on the resources available. For BERT-base as an example, this typically meant a batch size of eight; whereas for a BERT-large model, the maximum possible batch size was four. Dropout rates were only altered for the COVID-Twitter-BERT model that made it to Round 3. Here, the default dropout rate for the dropout layer was 0.1.  

