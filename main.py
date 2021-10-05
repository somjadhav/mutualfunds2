import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

def main():
    now = time.time()

    col_types = {'net_assets':np.float64, 'fund_yield':np.float64,
        'net_annual_expense_ratio_fund':np.float64,'portfolio_stocks':np.float64,
        'price_earnings':np.float64, 'price_book':np.float64, 'price_sales':np.float64, 'price_cashflow':np.float64,
        'basic_materials':np.float64, 'consumer_cyclical':np.float64, 'financial_services':np.float64,
        'real_estate':np.float64, 'consumer_defensive':np.float64, 'healthcare':np.float64, 'utilities':np.float64,
        'communication_services':np.float64, 'energy':np.float64, 'industrials':np.float64, 'technology':np.float64, 
        'fund_beta_3years':np.float64, 'fund_beta_5years':np.float64, 'fund_beta_10years':np.float64,
        'fund_standard_deviation_3years':np.float64, 'fund_standard_deviation_5years':np.float64,
        'fund_standard_deviation_10years':np.float64, 'fund_return_10years':np.float64}

    df_mf = pd.read_csv('mfdata.csv',dtype=col_types)
    df_mf.shape[0]

    df_mf = df_mf[(df_mf['investment'] == 'Blend') | (df_mf['investment'] == 'Growth') | (df_mf['investment'] == 'Value')]
    df_mf = df_mf[(df_mf['portfolio_stocks'] + df_mf['portfolio_cash']) > 95]
    df_mf = df_mf[df_mf['portfolio_stocks'] > 80]
    df_mf = df_mf[df_mf['fund_return_10years'] != 0]

    selected_columns = [
        'net_assets', 'fund_yield', 'investment', 'size',
        'net_annual_expense_ratio_fund','portfolio_stocks',
        'price_earnings', 'price_book', 'price_sales', 'price_cashflow',
        'basic_materials', 'consumer_cyclical', 'financial_services',
        'real_estate', 'consumer_defensive', 'healthcare', 'utilities',
        'communication_services', 'energy', 'industrials', 'technology', 'fund_beta_3years',
        'fund_beta_5years', 'fund_beta_10years',
        'fund_standard_deviation_3years', 'fund_standard_deviation_5years',
        'fund_standard_deviation_10years', 'fund_return_10years']

    df_mf = df_mf[selected_columns]
    df_mf.shape[0]
    df_mf.shape[0] - df_mf.dropna().shape[0]
    df_mf.dropna(inplace=True)
    df_mf.shape[0] - df_mf.dropna().shape[0]
    s_rets = df_mf['fund_return_10years']
    df_mf = df_mf.drop('fund_return_10years',axis=1)

    s_rets = s_rets > 13.6
    s_rets = s_rets.astype(int)
    sum(s_rets)/len(s_rets)


    investment_dummies = pd.get_dummies(df_mf['investment'],prefix="investment")
    size_dummies = pd.get_dummies(df_mf['size'],prefix="size")

    df_mf = pd.concat([df_mf,investment_dummies,size_dummies], axis=1)
    df_mf = df_mf.drop(['investment','size'],axis=1)

    df_mf.head()

    X_train, X_test, y_train, y_test = train_test_split(df_mf, s_rets, test_size=0.20, random_state=42)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train.to_numpy())
    X_test_scaled = scaler.transform(X_test.to_numpy())

    pca = PCA(.95)

    pca.fit(X_train_scaled)

    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # print number of columns in old and new datasets
    print(X_train_scaled.shape[1])
    pca.n_components_



    knn_clf = KNeighborsClassifier()
    knn_params = {'n_neighbors' : list(range(3,52,2))}

    knn_gridsearch = GridSearchCV(knn_clf,knn_params,cv=5,scoring='accuracy', n_jobs=-1)
    knn_gridsearch.fit(X_train_pca,y_train.to_numpy())

    print("Best params:",knn_gridsearch.best_params_)
    best_knn = knn_gridsearch.best_estimator_

    knn_accuracy = cross_val_score(best_knn,X_train_pca, y_train.to_numpy(), cv=5, scoring='accuracy', n_jobs=-1)
    print('Average train accuracy:', round(sum(knn_accuracy)/len(knn_accuracy)*100,3))

    knn_preds = best_knn.predict(X_test_pca)

    print(confusion_matrix(y_test.to_numpy(), knn_preds))
    print(classification_report(y_test.to_numpy(), knn_preds))

    svm_clf = SVC()
    svm_params = {
        'C':[0.1,0.5,1,5,10,50,100],
        'gamma':['scale','auto'],
        'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
        'degree':[1,2,3,4,5]
    }

    svm_gridsearch = GridSearchCV(svm_clf, svm_params, cv=5, scoring='accuracy', n_jobs=-1)
    svm_gridsearch.fit(X_train_pca, y_train.to_numpy())

    print("Best params:",svm_gridsearch.best_params_)
    best_svm = svm_gridsearch.best_estimator_

    svm_accuracy = cross_val_score(best_svm, X_train_pca, y_train.to_numpy(), cv=5, scoring = 'accuracy', n_jobs=-1)
    print('Average train accuracy:',round(sum(svm_accuracy)/len(svm_accuracy)*100,3))

    svm_preds = best_svm.predict(X_test_pca)


    print(confusion_matrix(y_test.to_numpy(), svm_preds))
    print(classification_report(y_test.to_numpy(), svm_preds))

    lr_clf = LogisticRegression(random_state=42)
    lr_params = {
        'C':[0.1,0.5,1,5,10,50,100],
        'penalty':['l2'],
        'solver':['liblinear', 'sag', 'lbfgs', 'newton-cg']
    }

    lr_gridsearch = GridSearchCV(lr_clf, lr_params, cv=5, scoring='accuracy', n_jobs=-1)
    lr_gridsearch.fit(X_train_pca, y_train.to_numpy())

    print("Best params:",lr_gridsearch.best_params_)
    best_lr = lr_gridsearch.best_estimator_

    lr_accuracy = cross_val_score(best_lr, X_train_pca, y_train.to_numpy(), cv=5, scoring = 'accuracy', n_jobs=-1)
    print('Average train accuracy:',round(sum(lr_accuracy)/len(lr_accuracy)*100,3))

    lr_preds = best_lr.predict(X_test_pca)

    print(confusion_matrix(y_test.to_numpy(), lr_preds))
    print(classification_report(y_test.to_numpy(), lr_preds))

    dtc_clf = DecisionTreeClassifier(random_state=42)
    dtc_params = {
        'criterion':['gini','entropy'],
        'splitter':['best','random'],
        'max_depth':[5,10,15,20],
        'max_features':['log2', 'sqrt'],
        'min_samples_leaf':[5,10,15]
    }

    dtc_gridsearch = GridSearchCV(dtc_clf, dtc_params, cv=5, scoring='accuracy', n_jobs=-1)
    dtc_gridsearch.fit(X_train_pca, y_train.to_numpy())

    print("Best params:",dtc_gridsearch.best_params_)
    best_dtc = dtc_gridsearch.best_estimator_

    dtc_accuracy = cross_val_score(best_dtc, X_train_pca, y_train.to_numpy(), cv=5, scoring = 'accuracy', n_jobs=-1)
    print('Average train accuracy:',round(sum(dtc_accuracy)/len(dtc_accuracy)*100,3))

    dtc_preds = best_dtc.predict(X_test_pca)

    print(confusion_matrix(y_test.to_numpy(), dtc_preds))
    print(classification_report(y_test.to_numpy(), dtc_preds))

    adb_clf = AdaBoostClassifier(random_state=42)
    adb_params = {'n_estimators':[50,100,200,300,500]}

    adb_gridsearch = GridSearchCV(adb_clf, adb_params, cv=5, scoring='accuracy', n_jobs=-1)
    adb_gridsearch.fit(X_train_pca, y_train.to_numpy())

    print("Best params:",adb_gridsearch.best_params_)
    best_adb = adb_gridsearch.best_estimator_

    adb_accuracy = cross_val_score(best_adb, X_train_pca, y_train.to_numpy(), cv=5, scoring = 'accuracy', n_jobs=-1)
    print('Average train accuracy:',round(sum(adb_accuracy)/len(adb_accuracy)*100,3))

    adb_preds = best_adb.predict(X_test_pca)

    print(confusion_matrix(y_test.to_numpy(), adb_preds))
    print(classification_report(y_test.to_numpy(), adb_preds))

    rdf_clf = RandomForestClassifier(random_state=42)
    rdf_params = {
        'n_estimators':[50,100,200],
        'criterion':['gini','entropy'],
        'max_depth':[5,10,15,20],
        'max_features':['log2', 'sqrt'],
        'min_samples_leaf':[5,10,15]
    }


    rdf_gridsearch = GridSearchCV(rdf_clf, rdf_params, cv=5, scoring='accuracy', n_jobs=-1)
    rdf_gridsearch.fit(X_train_pca, y_train.to_numpy())

    print("Best params:",rdf_gridsearch.best_params_)
    best_rdf = rdf_gridsearch.best_estimator_

    rdf_accuracy = cross_val_score(best_rdf, X_train_pca, y_train.to_numpy(), cv=5, scoring = 'accuracy', n_jobs=-1)
    print('Average train accuracy:',round(sum(rdf_accuracy)/len(rdf_accuracy)*100,3))

    rdf_preds = best_rdf.predict(X_test_pca)

    print(confusion_matrix(y_test.to_numpy(), rdf_preds))
    print(classification_report(y_test.to_numpy(), rdf_preds))

    df_etfs = pd.read_csv('ETFs.csv',dtype=col_types)
    df_etfs.shape[0]

    df_etfs = df_etfs[(df_etfs['investment'] == 'Blend') | (df_etfs['investment'] == 'Growth') | (df_etfs['investment'] == 'Value')]
    df_etfs = df_etfs[df_etfs['portfolio_stocks'] > 90]
    df_etfs = df_etfs[df_etfs['fund_return_10years'] != 0]

    df_etfs = df_etfs[selected_columns]

    df_etfs.shape[0]

    df_etfs.shape[0] - df_etfs.dropna().shape[0]

    df_etfs.dropna(inplace=True)

    s_rets_etf = df_etfs['fund_return_10years']
    df_etfs = df_etfs.drop('fund_return_10years',axis=1)

    s_rets_etf = s_rets_etf > 13.6
    s_rets_etf = s_rets_etf.astype(int)
    sum(s_rets_etf)/len(s_rets_etf)

    investment_dummies = pd.get_dummies(df_etfs['investment'],prefix="investment")
    size_dummies = pd.get_dummies(df_etfs['size'],prefix="size")

    df_etfs = pd.concat([df_etfs,investment_dummies,size_dummies], axis=1)
    df_etfs = df_etfs.drop(['investment','size'],axis=1)
    df_etfs.reset_index(inplace=True,drop=True)

    df_etfs.head()

    X_etfs_scaled = scaler.transform(df_etfs.to_numpy())
    X_etfs_pca = pca.transform(X_etfs_scaled)

    etf_acc = best_rdf.score(X_etfs_pca,s_rets_etf.to_numpy())
    etf_acc

    later = time.time()

    print(int(later - now))

if __name__ == '__main__':
    main()