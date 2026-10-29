# Hotel Review Sentiment Analysis

A machine learning project that classifies European hotel reviews as positive or negative using NLP, TF-IDF and Logistic Regression, with an interactive widget to test new reviews.

## Dataset
[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe) (Kaggle, CC0 licence).
This project uses a subset of **26,844 reviews**, which gives about 40,000 labelled review texts after cleaning.
The dataset is not included in this repo because of its size. Download `Hotel_Reviews.csv` from the link above.

## Approach
1. Split each review into its positive and negative text and label them (1 = positive, 0 = negative)
2. Remove placeholder reviews such as "No Negative" and "No Positive", and drop duplicates
3. Clean the text: lowercase, handle negations ("wasn't" becomes "was not"), remove punctuation
4. Extract TF-IDF features (unigrams and bigrams, up to 50,000 features), keeping negation words such as "not" and "no"
5. Train a Logistic Regression classifier with balanced class weights
6. Evaluate with precision, recall, F1 and a confusion matrix
7. Interpret the model by listing the words that push predictions towards each sentiment

## Results
| Metric | Score |
|---|---|
| Accuracy | 94% |
| Macro F1 | 0.936 |
| Precision (Negative / Positive) | 0.91 / 0.96 |
| Recall (Negative / Positive) | 0.95 / 0.93 |

**Top words for positive reviews:** location, comfortable, great, excellent, helpful, staff, clean, friendly
**Top words for negative reviews:** not, no, small, poor, noisy, expensive, slow, rude

## Interactive demo
The notebook ends with a text box and an "Analyze Review" button. Type any hotel review and the model returns the sentiment and its probability.

## Tech stack
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, ipywidgets

## How to run
1. Download `Hotel_Reviews.csv` from Kaggle and place it next to the notebook
2. Install the libraries: `pip install pandas numpy scikit-learn matplotlib seaborn joblib ipywidgets`
3. Open `Hotel_Review.ipynb` in Jupyter or Google Colab and run all cells

## Limitations
- Labels come from the Positive/Negative review columns, not from the overall reviewer rating
- Sarcasm and mixed reviews can be misclassified
- Only a subset of the full dataset was used

## Future improvements
- Compare with Naive Bayes, Linear SVM and transformer models
- Predict the numeric reviewer score
- Deploy as a Streamlit web app

## Author
Anusha Chaursiya | [LinkedIn](https://linkedin.com/in/anusha-chaursiya-440085257)
