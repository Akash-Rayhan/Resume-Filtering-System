# Resume-Filtering-System

## Setup Process

This guide outlines the steps to set up the required Python environment and install the necessary packages.

To begin, create a Python virtual environment using Python 3.8. Make sure you have pip version 20.0.2 installed. Run the following commands:

```bash
python3.8 -m venv myenv        # Replace `myenv` with your preferred environment name
source myenv/bin/activate    # Activate the virtual environment
```
Git clone this repository
```bash
git clone <this repository>
```
**Download the [trained model](https://drive.google.com/file/d/1xvCXp8b-SWXJnpEDTwCD9MF_uIarcdm_/view?usp=drive_link) and save .keras file in "model" directory within repository files.**

To use this system you need to install some necessary libraries that are mentioned in requirements.txt.
```python

pip install -r requirements.txt

```
## Development Pipeline
This project was developed using [this](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) kaggle dataset. The dataset used for this project consists of resume text along with their corresponding categories. The aim is to automatically categorize resumes into different job-related categories based on the textual content. 
### Machine Learning Approach
The approach involves data preprocessing, feature vectorization, model selection, hyperparameter tuning, and evaluation. Before using the data for model training, a series of preprocessing (e.g. Lowercasing, Whitespace Removal, URL Removal, HTML Tag Removal, Punctuation Removal, Newline Removal, Numeric Words Removal, Stopword Removal, Lemmatization) steps were applied to clean and standardize the text. To represent the textual data in a format suitable for machine learning models, TF-IDF (Term Frequency-Inverse Document Frequency) vectorization was applied. This process converts the raw text into numerical features by considering the importance of words within the context of the entire corpus.The textual data was tokenized and converted into a matrix of TF-IDF features.TF values were scaled using sublinear scaling to mitigate the impact of highly frequent words. For model selection Random Forest Classifier was chosen as the initial machine learning model. A grid search with cross-validation was employed to find the best combination of hyperparameters. Parameters like the number of estimators, maximum features, maximum depth, and splitting criterion were tuned. After training the Random Forest Classifier with the best-tuned hyperparameters, the model achieved an accuracy of 72% on the training dataset and 57% on test set.
At a glance the classifier was performing poor. Hoping a type of complex model might perform better.

### Deep Learning Approach

Resumes contain sequential information, where the order of words can be important for context. LSTMs are designed to capture such sequential dependencies, making them well-suited for tasks involving text data. LSTMs are capable of capturing long-range dependencies in the data, which is essential for understanding the context of the entire resume and making accurate categorization decisions. Text preprocessing steps were done as earlier mentioned way. After tokenizing clean texts unique words were added to the corpus additionally calculated the maximum number of tokens in any single resume. The Tokenizer class from Keras tokenizes the text data, encoding words into integers based. Padding was implemented to ensure that shorter sequences are extended with zeros, and if necessary, longer sequences are truncated to the desired length(maximum number of tokens). 


The LSTM-based model is constructed using Keras API. The first embedding layer converts the integer-encoded words into dense vectors of fixed size. The next Bidirectional LSTM layer processes input vectors in both forward and backward directions. This allows the model to capture context from both directions, enhancing its understanding of the text's context. The model includes fully connected dense layers for non-linear feature extraction. Dropout layers are added to mitigate overfitting by randomly deactivating a fraction of neurons during training. The final layer uses a softmax activation function. The model training and evaluation are performed using K-fold cross-validation.
This time it got better accuracy, precision, recall than the previous one. Training accuracy 79% and test accuracy 71%. 

### key insights

If we take a look on confusion matrix it is clearly seen the model couldn't classify any single resume in BPO and automobile categories. As the data distribution in categories is imbalanced for this dataset. BPO and Automobile categories have the least amount of data observsations. Deep learning models, like LSTMs, often require a significant amount of data to generalize well. If we can increase the amount of data points for those classes the model may predict better.

![confusion_matrix](https://github.com/Akash-Rayhan/Resume-Filtering-System/assets/40039916/708a34bc-d726-4171-b462-b6c72c77bc79)


Also LSTMs require word embeddings. Techniques like word embeddings (Word2Vec, GloVe) or even more advanced contextual embeddings (BERT) may enhance model performance.

## Inference Pipeline
 Before running inference pipeline make sure you have placed the trained model file as guided in setup process.
 To categorize resumes, use the script.py script provided in the repository. Replace path/to/directory with the path to the directory containing the resumes you want to categorize.
 Run the script using the following command:
 ```powershell

(myenv)akash@akash:~$ python script.py path/to/directory

```

The script will categorize each resume based on its content using the trained model. It will then move each resume to the respective category folder within the input directory. The script will create a CSV file named categorized_resumes.csv in the repository directory. The CSV file will have two columns: filename and category.  Each row corresponds to a categorized resume, providing its filename and the assigned category.
