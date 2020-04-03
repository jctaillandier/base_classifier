
# spot check machine learning algorithms on the adult imbalanced dataset
from numpy import mean
from numpy import std
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

# load the dataset
def load_dataset(full_path):
    # load the dataset as a numpy array
    dataframe = read_csv(full_path, header=None, na_values='?')
    # drop rows with missing
    dataframe = dataframe.dropna()
    # split into inputs and outputs
    last_ix = len(dataframe.columns) - 1
    X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix]
    # select categorical and numerical features
    cat_ix = X.select_dtypes(include=['object', 'bool']).columns
    num_ix = X.select_dtypes(include=['int64', 'float64']).columns
    # label encode the target variable to have the classes 0 and 1
    y = LabelEncoder().fit_transform(y)
    return X.values, y, cat_ix, num_ix


# evaluate a model
def evaluate_model(X, y, model):
    # Train test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
#     model.fit(X_train, y_train)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=1)
    
    return scores

# define models to test
def get_models():
    models, names = list(), list()
    # CART
    models.append(DecisionTreeClassifier())
    names.append('CART')
    # SVM
    models.append(SVC(gamma='scale'))
    names.append('SVM')
    # Bagging
    models.append(BaggingClassifier(n_estimators=100))
    names.append('BAG')
    # RF
    models.append(RandomForestClassifier(n_estimators=100))
    names.append('RF')
    # GBM
    models.append(GradientBoostingClassifier(n_estimators=100))
    names.append('GBM')
    return models, names


file = 'Adult_NotNA__sex'
# define the location of the dataset
full_path = f'../GeneralDatasets/sex_last/{file}.csv'
# load the dataset
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
    scores = evaluate_model(X, y, pipeline)
    results.append(scores)
    # summarize performance
    print_text = "{},{},{}".format(names[i], mean(scores), std(scores))
    total_texts.append(print_text)
    print(print_text)

# plot the results
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.savefig(f"./experiments/metadata_{file}.png")

# Write the results to file
total_score = 0
for result in total_texts:
    score = result.split(',')[1]
    total_score = total_score + float(score)
average = total_score/len(total_texts)



with open(f"./experiments/metadata_{file}.txt", 'w+') as f:
            f.write(f"Results: {results} \n \n")
            f.write(f"Test Output: {total_texts}\n")
            f.write(f"Classifiers average:{average}")

