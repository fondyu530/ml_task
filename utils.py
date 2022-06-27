import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate


def int_time_to_str(x: str):
    return x[:4] + '-' + x[4:6] + '-' + x[6:8] + ' ' + x[8:10] + ':' + x[10:12] + ':' + x[12:]


def add_sessions(main_data, sessions):
    main_data = main_data.merge(sessions, how="left", on="id")
    actions = main_data.drop(["age", "signup_flow"], axis=1)\
                       .groupby(["gender", "signup_method"])\
                       .mean()\
                       .round()\
                       .reset_index()
    cols_with_actions = sessions.columns[1:]
    for col in cols_with_actions:
        actions = actions.rename(columns={col: f"{col}_mean"})

    cols_with_actions_means = [col + "_mean" for col in cols_with_actions]

    main_data = main_data.merge(actions, on=["gender", "signup_method"])

    for i in range(len(cols_with_actions)):
        main_data[cols_with_actions[i]] = main_data[cols_with_actions[i]].fillna(main_data[cols_with_actions_means[i]])
        main_data = main_data.drop(cols_with_actions_means[i], axis=1)

    return main_data


def clear_data(main_data):
    # set appropriate data types
    main_data["date_account_created"] = main_data["date_account_created"].astype("datetime64[ns]")
    main_data["timestamp_first_active"] = main_data["timestamp_first_active"].astype(str) \
                                                                             .apply(int_time_to_str) \
                                                                             .astype("datetime64[ns]")
    main_data["date_first_booking"] = main_data["date_first_booking"].astype("datetime64[ns]")
    main_data["signup_flow"] = main_data["signup_flow"].astype(str)
    
    # Dealing with NaN values and clearing dataset
    main_data = main_data[(main_data["age"] < 100) | (main_data["age"].isnull())]
    main_data = main_data.dropna(subset=["first_affiliate_tracked"])
    main_data = main_data.drop("date_first_booking", axis=1)
    
    ages = main_data.groupby(["gender"]).mean().reset_index()
    ages = ages.rename(columns={"age": "mean_age"})
    ages["mean_age"] = ages["mean_age"].round()
    main_data = main_data.merge(ages, on=["gender"])
    main_data["age"] = main_data["age"].fillna(main_data["mean_age"])
    main_data = main_data.drop("mean_age", axis=1)
    
    return main_data


def encode_data(data):
    df = data.drop(["id",
                    "date_account_created",
                    "timestamp_first_active",
                    "language",
                    "first_browser",
                    "first_affiliate_tracked",
                    "signup_flow",
                    "affiliate_provider",
                    "signup_app",
                    "first_device_type",
                    "affiliate_channel"], axis=1)

    df = pd.get_dummies(df, columns=["gender",
                                     "signup_method"])
    
    return df
    

def conf_matrix(y_test, y_pred):
    
    cm = confusion_matrix(y_test, y_pred)

    class_names = ['NDF', 'US', 'other', 'CA', 'FR', 'ES', 'GB', 'IT', 'PT', 'NL', 'DE', 'AU']

    # Plot confusion matrix in a beautiful manner
    fig = plt.figure(figsize=(16, 14))
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt='g'); #annot=True to annotate cells
    # labels, title and ticks
    ax.set_xlabel('Predicted', fontsize=20)
    ax.xaxis.set_label_position('bottom')
    plt.xticks(rotation=90)
    ax.xaxis.set_ticklabels(class_names, fontsize=10)
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True', fontsize=20)
    ax.yaxis.set_ticklabels(class_names, fontsize=10)
    plt.yticks(rotation=0)

    plt.title('Refined Confusion Matrix', fontsize=20)

    plt.show()
    

def evaluate_metrics(estimator, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    scoring = ('accuracy', 'precision_macro', 'recall_macro', 'f1_macro')
    scores = cross_validate(estimator, X, y, scoring=scoring, cv=cv)

    accuracy = scores['test_accuracy'].mean()
    precision = scores['test_precision_macro'].mean()
    recall = scores['test_recall_macro'].mean()
    f1_score = scores["test_f1_macro"].mean()

    print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1: {f1_score}")
