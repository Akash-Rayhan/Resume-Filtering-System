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
**Download the [trained model](https://drive.google.com/file/d/1xvCXp8b-SWXJnpEDTwCD9MF_uIarcdm_/view?usp=drive_link) and save .keras file in model directory.**

To use this system you need to install some necessary libraries that are mentioned in requirements.txt.
```python

pip install -r requirements.txt

```
## Development Pipeline
This project was developed using [this](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) kaggle dataset. The dataset used for this project consists of resume text along with their corresponding categories. The aim is to automatically categorize resumes into different job-related categories based on the textual content. 
### Machine Learning Approach
The approach involves data preprocessing, feature vectorization, model selection, hyperparameter tuning, and evaluation. Before using the data for model training, a series of preprocessing like (e.g. Lowercasing, Whitespace Removal, URL Removal, HTML Tag Removal, Punctuation Removal, Newline Removal, Numeric Words Removal, Stopword Removal, Lemmatization) steps were applied to clean and standardize the text. To represent the textual data in a format suitable for machine learning models, TF-IDF (Term Frequency-Inverse Document Frequency) vectorization was applied. This process converts the raw text into numerical features by considering the importance of words within the context of the entire corpus.
