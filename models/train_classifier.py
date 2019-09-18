import sys
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from nltk import word_tokenize, WordNetLemmatizer
import nltk
nltk.download()
from sklearn.metrics import precision_recall_fscore_support
import re
import pickle
import warnings
warnings.filterwarnings('ignore')


def load_data(database_filepath):
    '''
    This function will load in data from the database
    
    Input: database address
    Output: X, y and categories names
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponseData', engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original','index', 'genre'], axis=1)
    category_names = y.columns

    return X, y, category_names


def tokenize(text):
    
    '''
    This function will normalize the text and return lemmatized & normalized text for ML
    '''
    
    # keep only numbers and characters
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(max_depth=10)))
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    
    Y_pred = model.predict(X_test)
    
    results = pd.DataFrame(columns=['Category', 'precision', 'recall', 'f_score'])
    
    for category in range(0, len(category_names)):
        
        precision, recall, f_score, support = precision_recall_fscore_support(Y_test[category_names[category]], Y_pred[:, category], average='weighted')
            
        results.set_value(category+1, 'Category', category_names[category])
        results.set_value(category+1, 'precision', precision)
        results.set_value(category+1, 'recall', recall)
        results.set_value(category+1, 'f_score', f_score)

    print('Aggregated f_score:', results['f_score'].mean())
    print('Aggregated precision:', results['precision'].mean())
    print('Aggregated recall:', results['recall'].mean())
    return results  


def save_model(model, model_filepath):
    pickle.dump(model, model_filepath)


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
        save_model(model, open(model_filepath, 'wb'))

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()