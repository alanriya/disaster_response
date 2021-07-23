import sys
import pickle
import pandas as pd
from sqlalchemy import create_engine
import pdb

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# sklearn
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# pickle for storing the model.
import pickle

# download the corpus for the nltk library
nltk.download(['punkt', 'wordnet', 'stopwords'])
STOPWORDS = stopwords.words('english')

def load_data(database_filepath):
    """
    Summary:
        1) create connection to the sqlite database,
        2) read table data into dataframe        
    Input:
        database_filepath[str] : path in string to the database filepath
       
    Output:
        X : independent variable
        Y : dependent variable
        category_names: list of string with the dependent variable names.
    """
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table('message', engine)
    X = df['message'].copy(deep=True)
    Y = df[['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']].copy()
    category_names = list(Y.columns)
    return X, Y, category_names

def tokenize(text):
    """
    Summary:
        1) remove punctuations
        2) remove stop words
        3) lemmatize word to root form and return list of words
    Input:
        text[str] : message
       
    Output:
        clean_tokens : clean list of words
    """
    punctuation_tokenizer = RegexpTokenizer(r'\w+')
    removed_puntuctuations = punctuation_tokenizer.tokenize(text)
    # remove stopwords
    tokens = [i for i in removed_puntuctuations if i not in STOPWORDS]
    clean_tokens = []
    lemmatizer = WordNetLemmatizer()
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
    return clean_tokens


def build_model():
    """
    Summary:
        1) Create Machine Learning Pipeline
        2) Create GridSearch for the Pipeline
        3) return the model for prediction 
       
    Output:
        model [sklearn classifier] : machine learning model for prediction.
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(verbose=True)))
    ])
    
    # parameters for GridSearch
    parameters = { 
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'tfidf__use_idf': (True, False),
        'clf__estimator__min_samples_split': [2, 3, 4],
        'clf__estimator__max_depth': [3,6,8,10]
    }
    
    model = GridSearchCV(pipeline, param_grid = parameters)
    return model    

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Summary:
        1) Do prediction based on the test set
        2) return the classification report and the accuracy score   
    Input:
        model[sklearn model] : trained ML model
        X_test: testing set independent variable
        Y_test: testing set dependent variable
        category_names: list of string with the dependent variable names.        
    """
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(Y_test, columns = category_names)
    
    for name in category_names:
        print(f'Category : {name}')
        print(classification_report(list(Y_test[name]), list(y_pred_df[name])))
        print(f'Accuracy : {accuracy_score(list(Y_test[name]), list(y_pred_df[name]))}\n')

def save_model(model, model_filepath):
    """
    Summary:
        1) Saving model to pickle file
       
    Input:
        model[sklearn classifier]: trained machine learning model.
        model_filepath: filepath of the pickle file which the model is stored.
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()