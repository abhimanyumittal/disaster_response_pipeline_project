import sys
import nltk
import string
import pickle
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report

nltk.download('stopwords')
nltk.download('punkt')


def load_data(database_filepath):
    """
    Load data and split into X and Y
    Args:
        database_filepath: path of the db file
    Returns:
        X: features (messages)
        Y: target variable
        category_names: categories
    """
    # load data from db
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_response', engine)

    # split into X and Y
    X = df.loc[:, 'message']
    Y = df.iloc[:, 4:]
    category_names = list(Y)

    return X.values, Y.values, category_names


def tokenize(text):
    """
    tokenize text
    Args:
        text: text to be tokenized
    Returns:
        word_tokens: list of tokens
    """
    word_tokens = []
    sw = stopwords.words('english') + list(string.punctuation)
    for word in word_tokenize(text.lower()):
        if word not in sw:
            word_tokens.append(word)

    return word_tokens


def build_model():
    """
    build a multiclassification model
    Returns:
        cv: GridSearchCV object
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__learning_rate': [0.001, 0.01, 0.1, 0.9, 1],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    check model performance
    Args:
        model: multiclassification model
        X_test: test set features
        Y_test: test set target labels
        category_names: category names
    """
    Y_pred = model.predict(X_test)
    # show report
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    output model to pickle file
    Args:
        model: multiclassification model
        model_filepath: filepath of pickle file
    """
    # dump model to pickle file
    with open(model_filepath, 'wb') as outfile:
        pickle.dump(model, outfile)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
