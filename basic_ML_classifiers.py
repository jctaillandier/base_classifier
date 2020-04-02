{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      ">CART 0.824 (0.017)\n",
      ">SVM 0.846 (0.013)\n"
     ]
    }
   ],
   "source": [
    "# spot check machine learning algorithms on the adult imbalanced dataset\n",
    "from numpy import mean\n",
    "from numpy import std\n",
    "from pandas import read_csv\n",
    "from matplotlib import pyplot\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.model_selection import cross_val_score\n",
    "from sklearn.model_selection import RepeatedStratifiedKFold\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.ensemble import GradientBoostingClassifier\n",
    "from sklearn.ensemble import BaggingClassifier\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# load the dataset\n",
    "def load_dataset(full_path):\n",
    "    # load the dataset as a numpy array\n",
    "    dataframe = read_csv(full_path, header=None, na_values='?')\n",
    "    # drop rows with missing\n",
    "    dataframe = dataframe.dropna()\n",
    "    # split into inputs and outputs\n",
    "    last_ix = len(dataframe.columns) - 1\n",
    "    X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix]\n",
    "    # select categorical and numerical features\n",
    "    cat_ix = X.select_dtypes(include=['object', 'bool']).columns\n",
    "    num_ix = X.select_dtypes(include=['int64', 'float64']).columns\n",
    "    # label encode the target variable to have the classes 0 and 1\n",
    "    y = LabelEncoder().fit_transform(y)\n",
    "    return X.values, y, cat_ix, num_ix\n",
    "\n",
    "\n",
    "# evaluate a model\n",
    "def evaluate_model(X, y, model):\n",
    "    # Train test split\n",
    "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)\n",
    "    \n",
    "    # define evaluation procedure\n",
    "    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)\n",
    "    # evaluate model\n",
    "    model.fit(X_train, y_train)\n",
    "    scores = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)\n",
    "    \n",
    "    return scores\n",
    "\n",
    "# define models to test\n",
    "def get_models():\n",
    "    models, names = list(), list()\n",
    "    # CART\n",
    "    models.append(DecisionTreeClassifier())\n",
    "    names.append('CART')\n",
    "    # SVM\n",
    "    models.append(SVC(gamma='scale'))\n",
    "    names.append('SVM')\n",
    "    # Bagging\n",
    "    models.append(BaggingClassifier(n_estimators=100))\n",
    "    names.append('BAG')\n",
    "    # RF\n",
    "    models.append(RandomForestClassifier(n_estimators=100))\n",
    "    names.append('RF')\n",
    "    # GBM\n",
    "    models.append(GradientBoostingClassifier(n_estimators=100))\n",
    "    names.append('GBM')\n",
    "    return models, names\n",
    "\n",
    "\n",
    "file = 'Adult_NotNA__sex'\n",
    "# define the location of the dataset\n",
    "full_path = f'../GeneralDatasets/sex_last/{file}.csv'\n",
    "# load the dataset\n",
    "X, y, cat_ix, num_ix = load_dataset(full_path)\n",
    "# define models\n",
    "models, names = get_models()\n",
    "results = list()\n",
    "total_texts = []\n",
    "# evaluate each model\n",
    "for i in range(len(models)):\n",
    "    # define steps\n",
    "    steps = [('c',OneHotEncoder(handle_unknown='ignore'),cat_ix), ('n',MinMaxScaler(),num_ix)]\n",
    "    # one hot encode categorical, normalize numerical\n",
    "    ct = ColumnTransformer(steps)\n",
    "    # wrap the model i a pipeline\n",
    "    pipeline = Pipeline(steps=[('t',ct),('m',models[i])])\n",
    "\n",
    "    # evaluate the model and store results\n",
    "    scores = evaluate_model(X, y, pipeline)\n",
    "    results.append(scores)\n",
    "    # summarize performance\n",
    "    print_text = '>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores))\n",
    "    total_texts.append(print_text)\n",
    "    print(print_text)\n",
    "\n",
    "# plot the results\n",
    "pyplot.boxplot(results, labels=names, showmeans=True)\n",
    "pyplot.savefig(f\"./experiments/metadata_{file}.png\")\n",
    "# Write the results to file"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "with open(f\"./experiments/metadata_{file}.txt\", 'w+') as f:\n",
    "            f.write(f\"Results: {results} \\n \\n\")\n",
    "            f.write(f\"TEst Output: {total_texts} \\n \\n\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
