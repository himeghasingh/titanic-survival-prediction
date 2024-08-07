# Author: Megha Singh
import warnings
import pandas as pd
from pip._internal.utils.misc import tabulate
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
from tabulate import tabulate
import matplotlib.pyplot as plt
from pandas.plotting import table

# Define hyperparameters for Decision Tree
dt_hyperparameters = {
    "min_samples_split": range(10, 500, 20),
    "max_depth": range(1, 20, 2),
    "min_samples_leaf": range(1, 10),
    "criterion": ['gini', 'entropy']
}

# Define hyperparameters for Support Vector Machine
svm_hyperparameters = {"kernel": ['rbf'],
                       "gamma": [0.001, 0.01, 0.1, 1, 10],
                       "C": [1, 10, 50, 100, 250, 500, 1000, 2000], "probability": True}

# Define hyperparameters forMLP models
mlp_hyperparameters = {
    'hidden_layer_sizes': [(50,), (100,), (150,)],
    'activation': ['relu', 'tanh', 'logistic'],
    'alpha': [1e-3, 1e-4, 1e-5],
    'learning_rate': ['constant', 'invscaling', 'adaptive']
}

# Create dictionaries to store the best models and their corresponding scores
best_models = {}
all_scores = {}


# def analyse_data(df):
#     # Reference: https://towardsdatascience.com/a-beginners-guide-to-kaggle-s-titanic-problem-3193cb56f6ca
#     plt.figure(figsize=(8, 6))
#     ax = sns.countplot(data=df, x='Pclass')
#
#     for p in ax.patches:
#         ax.annotate(format(p.get_height(), '.0f'),
#                     (p.get_x() + p.get_width() / 2., p.get_height()),
#                     ha='center', va='center',
#                     xytext=(0, 10),
#                     textcoords='offset points')
#
#     plt.title('Distribution of Passenger Classes')
#     plt.xlabel('Passenger Class')
#     plt.ylabel('Count')
#     plt.show()
#
#     # %%
#     plt.figure(figsize=(8, 6))
#     ax = sns.countplot(data=df, x='Sex')
#
#     for p in ax.patches:
#         ax.annotate(format(p.get_height(), '.0f'),
#                     (p.get_x() + p.get_width() / 2., p.get_height()),
#                     ha='center', va='center',
#                     xytext=(0, 10),
#                     textcoords='offset points')
#
#     plt.title('Distribution of Sex')
#     plt.xlabel('Sex of Passengers')
#     plt.ylabel('Count')
#     plt.show()
#
#     survival_rates = df.groupby('Sex')['Survived'].mean().reset_index()
#
#     sns.barplot(x='Sex', y='Survived', data=survival_rates)
#
#     for index, value in enumerate(survival_rates['Survived']):
#         plt.text(index, value, f'{value:.4f}', ha='center', va='bottom')
#
#     plt.title('Survival Rates by Sex')
#     plt.ylabel('Average Survival Rate')
#     plt.xlabel('Sex')
#
#     plt.show()
#
#     sns.histplot(x="Age", hue="Survived", palette="mako", data=df[["Age", "Survived"]])
#     plt.title("Age Distribution", color='black', fontsize=14)
#     plt.yticks([])
#     plt.box(False)
#     plt.show()
#
#     plt.figure(figsize=(10, 6))
#     ax = sns.countplot(data=df, x='SibSp')
#
#     for p in ax.patches:
#         ax.annotate(format(p.get_height(), '.0f'),
#                     (p.get_x() + p.get_width() / 2., p.get_height()),
#                     ha='center', va='center',
#                     xytext=(0, 10),
#                     textcoords='offset points')
#
#     plt.title('Distribution of Siblings Amount')
#     plt.xlabel('Title of Passengers')
#     plt.ylabel('Count')
#     plt.show()
#
#     survival_by_sibsp = df.groupby('SibSp')['Survived'].mean()
#
#     plt.figure(figsize=(8, 6))
#     ax = sns.barplot(x=survival_by_sibsp.index, y=survival_by_sibsp.values)
#     plt.title('Survival Rate by SibSp')
#     plt.xlabel('Number of Siblings/Spouses')
#     plt.ylabel('Survival Rate')
#
#     total = float(len(df))
#     for p in ax.patches:
#         height = p.get_height()
#         ax.text(p.get_x() + p.get_width() / 2., height + 0.02, '{:.1%}'.format(height), ha="center")
#
#     plt.show()
#
#     df["Ticket"].value_counts()
#
#     newdf = df.copy()
#
#     newdf["Group_Size"] = newdf.groupby("Ticket")["PassengerId"].transform("count")
#     newdf["Group_Size"]
#
#     plt.figure(figsize=(10, 6))
#     ax = sns.countplot(data=newdf, x='Group_Size')
#
#     for p in ax.patches:
#         ax.annotate(format(p.get_height(), '.0f'),
#                     (p.get_x() + p.get_width() / 2., p.get_height()),
#                     ha='center', va='center',
#                     xytext=(0, 10),
#                     textcoords='offset points')
#
#     plt.title('Distribution of Group Size')
#     plt.xlabel('Group Size of Passengers')
#     plt.ylabel('Count')
#     plt.show()
#
#     newdf = df.copy()
#     newdf["Cabin"].fillna("Unknown", inplace=True)
#     newdf["Cabin"] = newdf["Cabin"].str[0]
#     sns.set(style="darkgrid")
#     sns.countplot(x='Survived', data=newdf, hue="Cabin", palette="Set1");
#
#     sns.set(style="darkgrid")
#     sns.countplot(x='Survived', data=df, hue="Embarked", palette="Set1");
#
#     sns.set(style="darkgrid")
#     sns.countplot(x='Survived', data=df, hue="Survived", palette="Set1");


def svm_model(X_train, X_val, Y_train, Y_val, test_df):
    # clf_svm = GridSearchCV(svm(), param_grid=svm_hyperparameters, cv=StratifiedKFold(n_splits=10),
    #                        scoring="accuracy", n_jobs=-1, verbose=1)
    # clf_svm.fit(X_train, Y_train)
    # print(clf_svm.best_params_)

    # Use the best hyperparameters obtained from tuning
    clf_svm_best_params = {'C': 2000, 'gamma': 0.01, 'kernel': 'rbf', 'probability': True}
    # Create model with optimal hyperparameters and fit training data on it
    clf_svm_best_model = SVC(**clf_svm_best_params)
    clf_svm_best_model.fit(X_train, Y_train)

    svm_y_train_pred = clf_svm_best_model.predict(X_train)
    svm_accuracy_train = accuracy_score(Y_train, svm_y_train_pred)
    print('Accuracy of SVM on training data: ', svm_accuracy_train)

    # Generate predictions for the validation set
    svm_y_val_pred = clf_svm_best_model.predict(X_val)
    svm_accuracy_val = accuracy_score(Y_val, svm_y_val_pred)
    print('Accuracy of SVM on validation data: ', svm_accuracy_val)

    # Generate predictions for the test set
    test_data = test_df.drop('PassengerId', axis=1)
    svm_y_test_pred = clf_svm_best_model.predict(test_data)
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': svm_y_test_pred
    })
    submission.to_csv('submission_svm.csv', index=False)

    # Store model information in the 'best_models' dictionary
    best_models["Support Vector Machine"] = {
        "model": clf_svm_best_model,
        "accuracy_train": svm_accuracy_train,
        "y_val_pred": svm_y_val_pred,
        "params": clf_svm_best_params
    }


def decisiontree_model(X_train, X_val, Y_train, Y_val, test_df):
    # Create and train Decision Tree Classifier
    # clf_dt = RandomizedSearchCV(DecisionTreeClassifier(), param_distributions=dt_hyperparameters,
    #                             cv=StratifiedKFold(n_splits=20),
    #                             scoring="accuracy", n_jobs=-1, verbose=1)
    #
    # clf_dt.fit(X_train, Y_train)
    # print(clf_dt.best_params_)

    # Use the best hyperparameters obtained from tuning
    clf_dt_best_params = {'min_samples_split': 130, 'min_samples_leaf': 6, 'max_depth': 19, 'criterion': 'gini'}
    # Create model with optimal hyperparameters and fit training data on it
    clf_dt_best_model = DecisionTreeClassifier(**clf_dt_best_params)
    clf_dt_best_model.fit(X_train, Y_train)

    dt_y_train_pred = clf_dt_best_model.predict(X_train)
    dt_accuracy_train = accuracy_score(Y_train, dt_y_train_pred)
    print('Accuracy of Decision Tree on training data: ', dt_accuracy_train)

    # Generate predictions for the validation set
    dt_y_val_pred = clf_dt_best_model.predict(X_val)
    dt_accuracy_val = accuracy_score(Y_val, dt_y_val_pred)
    print('Accuracy of Decision Tree on validation data: ', dt_accuracy_val)

    # Generate predictions for the test set
    test_data = test_df.drop('PassengerId', axis=1)
    dt_y_test_pred = clf_dt_best_model.predict(test_data)
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': dt_y_test_pred
    })
    submission.to_csv('submission_dt.csv', index=False)

    # Store model information in the 'best_models' dictionary
    best_models["Decision Tree"] = {
        "model": clf_dt_best_model,
        "accuracy_train": dt_accuracy_train,
        "y_val_pred": dt_y_val_pred,
        "params": clf_dt_best_params
    }


def mlp_model(X_train, X_val, Y_train, Y_val, test_df):
    # Create and train MLPClassifier
    # warnings.filterwarnings("ignore", category=ConvergenceWarning)
    #
    # clf_mlp = RandomizedSearchCV(MLPClassifier(), param_distributions=mlp_hyperparameters,
    #                              cv=StratifiedKFold(n_splits=10),
    #                              scoring="accuracy", n_jobs=-1, verbose=1)
    # clf_mlp.fit(X_train, Y_train)
    # print(clf_mlp.best_params_)

    # Use the best hyperparameters obtained from tuning
    clf_mlp_best_params = {'learning_rate': 'invscaling', 'hidden_layer_sizes': (150,), 'alpha': 0.001,
                           'activation': 'tanh'}
    # Create model with optimal hyperparameters and fit training data on it
    clf_mlp_best_model = MLPClassifier(**clf_mlp_best_params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        clf_mlp_best_model.fit(X_train, Y_train)

    mlp_y_train_pred = clf_mlp_best_model.predict(X_train)
    mlp_accuracy_train = accuracy_score(Y_train, mlp_y_train_pred)
    print('Accuracy of MLP on training data: ', mlp_accuracy_train)

    # Generate predictions for the validation set
    mlp_y_val_pred = clf_mlp_best_model.predict(X_val)
    mlp_accuracy_val = accuracy_score(Y_val, mlp_y_val_pred)
    print('Accuracy of MLP on validation data: ', mlp_accuracy_val)

    # Generate predictions for the test set
    test_data = test_df.drop('PassengerId', axis=1)
    mlp_y_test_pred = clf_mlp_best_model.predict(test_data)
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': mlp_y_test_pred
    })
    submission.to_csv('submission_mlp.csv', index=False)

    # Store model information in the 'best_models' dictionary
    best_models["Multilayer Perceptron"] = {
        "model": clf_mlp_best_model,
        "accuracy_train": mlp_accuracy_train,
        "y_val_pred": mlp_y_val_pred,
        "params": clf_mlp_best_params
    }


def print_missing_values(df):
    missing_values = df.isnull().sum()
    total_values = df.shape[0]
    missing_percentage = (missing_values / total_values) * 100

    missing_info = pd.DataFrame({
        'Column': df.columns,
        'Missing Values': missing_values,
        'Percentage': missing_percentage

    })
    print(missing_info)


def preprocess_data(df):
    """
        Preprocesses the input DataFrame by performing various transformations.

        :param df: Input DataFrame containing the raw data.
        :return: Preprocessed DataFrame.

        This function handles missing values, converts categorical features to numerical representations,
        creates new features, and drops unnecessary columns.
        """

    # Fill missing values for 'Embarked' with the mode and map to numerical values
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # Fill missing values for 'Fare', categorize into bins, and map to numerical values
    df['Fare'] = df['Fare'].fillna(0)
    df['Fare'] = pd.cut(df['Fare'], bins=[0, 7.91, 14.454, 31, float('inf')], labels=[0, 1, 2, 3],
                        include_lowest=True).astype(int)

    # Extract 'Salutation' from 'Name', map to numerical values, and create a new feature 'Salutation_Integer'
    df['Salutation'] = df['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())
    unique_salutations = df['Salutation'].unique()
    salutation_mapping = {salutation: idx for idx, salutation in enumerate(unique_salutations)}
    df['Salutation_Integer'] = df['Salutation'].map(salutation_mapping)

    # Fill missing values for 'Age' based on grouped median values and categorize into bins
    grouped_data = df.groupby(['Salutation', 'Sex', 'Pclass', 'Parch'])
    df['Age'] = grouped_data['Age'].transform(lambda x: x.fillna(x.median()))
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Age'] = pd.cut(df['Age'], bins=[0, 8, 16, 24, 32, 40, 48, 56, 64, float('inf')],
                       labels=[0, 1, 2, 3, 4, 5, 6, 7, 8], include_lowest=True).astype(int)

    # Map 'Sex' to numerical values
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # Create new features 'FamilySize' and 'isAlone'
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['isAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'isAlone'] = 1

    # Drop unnecessary columns
    columns_to_drop = ['Name', 'Ticket', 'FamilySize', 'Parch', 'SibSp', 'Cabin', 'Salutation']
    df = df.drop(columns=columns_to_drop)

    return df


def run_task():
    # Read training and testing data
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # print(train_df.head())
    # print(train_df.columns)
    #
    # analyse_data(train_df)
    # Print missing values in the training and testing datasets
    print("Missing values in Training data set:")
    print_missing_values(train_df)
    print("Missing values in Testing data set:")
    print_missing_values(test_df)
    # Preprocess the data
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)

    # Drop unnecessary columns and convert 'Survived' column to integer type

    train_df = train_df.drop('PassengerId', axis=1)
    train_df["Survived"] = train_df["Survived"].astype(int)

    # Create a correlation matrix
    correlation_matrix = train_df.corr()

    # Visualize the correlation matrix using a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix After Preprocessing')
    plt.show()

    # Split the data into training and validation sets

    Y = train_df["Survived"].values
    X = train_df.drop('Survived', axis=1)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=105)

    # norm_tech value can be set so as to use normalization/standardization
    # norm_tech = ''
    # if norm_tech == 'standardization':
    #     # Standardization
    #     scaler = StandardScaler()
    #     X_train = scaler.fit_transform(X_train)
    #     X_val = scaler.fit_transform(X_val)
    #     test_df = scaler.transform(test_df)
    #
    # if norm_tech == 'normalization':
    #     # Normalization
    #     scaler = MinMaxScaler()
    #     X_train = scaler.fit_transform(X_train)
    #     X_val = scaler.fit_transform(X_val)
    #     test_df = scaler.transform(test_df)

    # Use different models and store the best ones in the 'best_models' dictionary

    svm_model(X_train, X_val, Y_train, Y_val, test_df.copy())
    decisiontree_model(X_train, X_val, Y_train, Y_val, test_df.copy())
    mlp_model(X_train, X_val, Y_train, Y_val, test_df.copy())

    all_scores = {}
    # Kaggle accuracies for reference, hardcoded
    kaggle_accuracies = {
        'Support Vector Machine': 0.78708,
        'Decision Tree': 0.76794,
        'Multilayer Perceptron': 0.7799
    }
    table_data = []
    # Evaluate models and store scores in 'all_scores' dictionary
    for model_name, model_info in best_models.items():
        model = model_info["model"]
        y_val_pred = model_info["y_val_pred"]
        scores = evaluate_model(model, y_val_pred, Y_val, X_val, model_name)
        all_scores[model_name] = scores
        all_scores[model_name]['Kaggle_Accuracy'] = kaggle_accuracies.get(model_name, None)

    for metric in all_scores[list(best_models.keys())[0]].keys():
        plt.figure(figsize=(8, 4))  # Adjust the size as needed

        scores = [all_scores[model_name][metric] for model_name in best_models]
        colors = ['skyblue', 'lightgreen', 'lightcoral']  # Adjust colors as needed
        plt.bar(best_models.keys(), scores, color=colors, width=0.6)  # Adjust width as needed

        plt.xlabel('Models')
        plt.ylabel(metric.capitalize() + ' Score')
        plt.title(f'{metric.capitalize()} Score Comparison Among Models')

        for i, score in enumerate(scores):
            plt.text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom')

    # Plot ROC curves for model comparison
    plot_roc_curves_comparison(best_models, X_val, Y_val)

    # Print Kaggle accuracy and best parameters for each model
    for model_name, model_info in best_models.items():
        best_params = model_info.get("params", None)
        scores = all_scores.get(model_name, None)

        print(f"Model: {model_name}")
        print(f"Best Parameters: {best_params}")
        print("All Scores:")
        if scores:
            for metric, score in scores.items():
                print(f"  {metric.capitalize()}: {score:.3f}")
        else:
            print("  No scores available.")
        print("=" * 40)


# Function to evaluate a model and generate various metrics
def evaluate_model(model, y_val_pred, Y_val, X_val, model_name):
    """
        Evaluates a machine learning model using various metrics and visualizations.

        :param model: Trained machine learning model.
        :param y_val_pred: Predictions on the validation set.
        :param Y_val: True labels of the validation set.
        :param X_val: Features of the validation set.
        :param model_name: Name of the model for display purposes.
        :return: Dictionary of evaluation scores.

        This function calculates and displays metrics such as accuracy, precision, recall, F1 score,
        ROC curve, precision-recall curve, confusion matrix, and classification report.
        """
    scores = {}

    # Calculate and store accuracy
    accuracy = accuracy_score(Y_val, y_val_pred)
    scores['Validation_Accuracy'] = accuracy

    # Calculate and store precision
    precision = precision_score(Y_val, y_val_pred)
    scores['precision'] = precision

    # Calculate and store recall
    recall = recall_score(Y_val, y_val_pred)
    scores['recall'] = recall

    # Calculate and store F1 score
    f1 = f1_score(Y_val, y_val_pred)
    scores['f1'] = f1

    # Calculate and store ROC AUC if applicable
    if hasattr(model, 'predict_proba') and callable(getattr(model, 'predict_proba')):
        roc_auc = roc_auc_score(Y_val, model.predict_proba(X_val)[:, 1])
        scores['roc_auc'] = roc_auc
    else:
        scores['roc_auc'] = None

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(Y_val, model.predict_proba(X_val)[:, 1])
    plt.plot(recall, precision, label=f'{model_name} (AUC = {auc(recall, precision):.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(Y_val, model.predict_proba(X_val)[:, 1])
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(Y_val, y_val_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'],
                yticklabels=['Not Survived', 'Survived'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

    # Classification Report
    report = classification_report(Y_val, y_val_pred)
    print(f'Classification Report - {model_name}:\n{report}')

    return scores


# Function to plot ROC curves for model comparison
def plot_roc_curves_comparison(models_dict, X_val, Y_val):
    plt.figure(figsize=(10, 6))

    for model_name, model_info in models_dict.items():
        model = model_info["model"]
        fpr, tpr, _ = roc_curve(Y_val, model.predict_proba(X_val)[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison Among Models')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    run_task()
