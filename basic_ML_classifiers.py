# spot check machine learning algorithms on the adult imbalanced dataset
from numpy import mean
from numpy import std
import pdb, argparse
import pandas as pd
from pandas import read_csv
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier



def load_dataset( full_path):
    # load the dataset as a numpy array
    # dataframe = read_csv(full_path, header=None, na_values='?')
    dataframe = read_csv(full_path,na_values='?')
    # drop rows with missing
    dataframe = dataframe.dropna()
    # split into inputs and outputs
    last_ix = 'sex'
    X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix]

    # select categorical and numerical features
    cat_ix = X.select_dtypes(include=['object', 'bool']).columns
    num_ix = X.select_dtypes(include=['int64', 'float64']).columns
    c_ix = []
    n_ix = []
    for i, v in enumerate(dataframe.columns.tolist()):
        if v in cat_ix:
            c_ix.append(i)
        elif v in num_ix:
            n_ix.append(i)
    # label encode the target variable to have the classes 0 and 1
#     y = pd.to_numeric(y.values[:,0])
    # y = LabelEncoder().fit_transform(y)
    return X.values, y.values, c_ix, n_ix


# evaluate a model
def evaluate_model( X, y, model, kfold='false'):
    if kfold.lower() == 'false':
        perc=0.15
        print(f"Running Train-Test split with {perc}% test set.")
        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=perc, random_state=42)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = accuracy_score(y_test,preds)
    
    else:
        k = 10
        print(f"Running {k}k-fold cross validation training.")
        # define evaluation procedure
        cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=1, random_state=1)
        
        # evaluate model
        score = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=2)

    return score

# define models to test
def get_models():
    models, names = list(), list()
    # CART
    # models.append(DecisionTreeClassifier())
    # names.append('CART')
    # SVM
    models.append(SVC(gamma='scale'))
    names.append('SVM')
#     # Bagging
    models.append(BaggingClassifier(n_estimators=100))
    names.append('BAG')
    # RF
    models.append(MLPClassifier(max_iter=100))
    names.append('MLP')
# #     # GBM
    models.append(GradientBoostingClassifier(n_estimators=100))
    names.append('GBM')
    return models, names

filen = 'laftr_test_og-attr'
# full_path = f'../focus_data/recons_data/{filen}.csv'
full_path = f"../GeneralDatasets/sanitized_output/{filen}.csv"
X, y, cat_ix, num_ix = load_dataset(full_path)

# define models
models, names = get_models()
results = list()
total_texts = []
# evaluate each model

for i in range(len(models)):
    # define steps
    steps = [('c',OneHotEncoder(handle_unknown='ignore'),cat_ix), ('n',MinMaxScaler(),num_ix)]
    # one hot encode categorical, normalize numerical
    ct = ColumnTransformer(steps)
    # wrap the model i a pipeline
    pipeline = Pipeline(steps=[('t',ct),('m',models[i])])

    # evaluate the model and store results
    scores = evaluate_model(X,y, pipeline)
    results.append(scores)
    # summarize performance
    print_text = "{},{:.2f},{:.4f}".format(names[i], mean(scores), std(scores))
    
    total_texts.append(print_text)
    print(print_text)

total_score = 0
for result in total_texts:
    score = result.split(',')[1]
    total_score = total_score + float(score)
average = total_score/len(total_texts)

print(f"Average of {i+1} classifiers: {average}")

    # with open(f"{path}{file_name1}.txt", 'w+') as f:
    #             f.write(f"Input file: {file_name1}")
    #             f.write(f"Results: {results} \n \n")
    #             f.write(f"Test Output: {total_texts}\n")
    #             f.write(f"Classifiers average:{average}")
