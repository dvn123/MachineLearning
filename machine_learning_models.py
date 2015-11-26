import numpy as np

from sklearn import svm, metrics
from sklearn import neighbors
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def naive_bayes(train_data_features, test_data_features, labels):
    gnb = GaussianNB()
    model = gnb.fit(train_data_features, labels)
    return model.predict_proba(test_data_features), model.predict(test_data_features), model

def perc(train_data_features, test_data_features, labels):
    prc = Perceptron()
    model = prc.fit(train_data_features, labels)
    return model.predict_proba(test_data_features), model.predict(test_data_features), model

def log_res(train_data_features, test_data_features, labels, using_cross_validation2, kf):
    if using_cross_validation2:
        #print("LOGRES")
        logres_C = 1
        logres_results = []
        for train, test in kf:
            C = logres_C
            p = 'l1'
            clf_l1_LR = LogisticRegression(C=C, penalty=p, tol=0.01)
            model = clf_l1_LR.fit(train_data_features[train], labels[train])
            predicted_classes = model.predict(train_data_features[test])
            class_probabilities = model.predict_proba(train_data_features[test])
            print(" n points:", len(predicted_classes), ", wrong: ", (labels[test] != predicted_classes).sum(), " percentage: ",(labels[test] != predicted_classes).sum()*100/len(predicted_classes),"%")
            logres_results.append((labels[test] != predicted_classes).sum())
            logres_C += 1
        logres_C = logres_results.index(min(logres_results)) + 1
        print("Log Res C: ", logres_C)
        clf_l1_LR = LogisticRegression(C=logres_C, penalty=p, tol=0.01)
        model = clf_l1_LR.fit(train_data_features, labels)
        return model.predict_proba(test_data_features), model.predict(test_data_features), model
    else:
        C = 1
        p = 'l1'
        clf_l1_LR = LogisticRegression(C=C, penalty=p, tol=0.01)
        model = clf_l1_LR.fit(train_data_features, labels)
        return model.predict_proba(test_data_features), model.predict(test_data_features), model

def cross_test(train_data_features, test_data_features, labels, using_cross_validation2, kf):
    if using_cross_validation2:
        C_base = 0.025
        C_step = 0.005
        C = C_base
        _results = []
        for train, test in kf:
            svc = SVC(kernel="linear", C=C, probability=True)
            model = svc.fit(train_data_features[train], labels[train])
            predicted_classes = model.predict(train_data_features[test])
            class_probabilities = model.predict_proba(train_data_features[test])
            print("n points:", len(predicted_classes), ", wrong: ", (labels[test] != predicted_classes).sum(), " percentage: ",(labels[test] != predicted_classes).sum()*100/len(predicted_classes),"%")
            _results.append((labels[test] != predicted_classes).sum())
            C += C_step
        C = C_base + C_step * _results.index(min(_results))
        print("C: ", C)
        svc = SVC(kernel="linear", C=C, probability=True)
        model = svc.fit(train_data_features, labels)
        return model.predict_proba(test_data_features), model.predict(test_data_features), model
    else:
        svc = SVC(kernel="linear", C=0.025, probability=True)
        model = svc.fit(train_data_features, labels)
        return model.predict_proba(test_data_features), model.predict(test_data_features), model

def linear_svm(train_data_features, test_data_features, labels, using_cross_validation2, kf):
    if using_cross_validation2:
        C_base = 0.025
        C_step = 0.005
        C = C_base
        _results = []
        for train, test in kf:
            svc = SVC(kernel="linear", C=C, probability=True)
            model = svc.fit(train_data_features[train], labels[train])
            predicted_classes = model.predict(train_data_features[test])
            class_probabilities = model.predict_proba(train_data_features[test])
            print("n points:", len(predicted_classes), ", wrong: ", (labels[test] != predicted_classes).sum(), " percentage: ",(labels[test] != predicted_classes).sum()*100/len(predicted_classes),"%")
            _results.append((labels[test] != predicted_classes).sum())
            C += C_step
        C = C_base + C_step * _results.index(min(_results))
        print("C: ", C)
        svc = SVC(kernel="linear", C=C, probability=True)
        model = svc.fit(train_data_features, labels)
        return model.predict_proba(test_data_features), model.predict(test_data_features), model
    else:
        svc = SVC(kernel="linear", C=0.025, probability=True)
        model = svc.fit(train_data_features, labels)
        return model.predict_proba(test_data_features), model.predict(test_data_features), model

def adaboost(train_data_features, test_data_features, labels, using_cross_validation2, kf):
    if using_cross_validation2:
        _results = np.zeros(10)
        base_n_estimators = 100 # week learners
        step_n_estimators = 100
        ada_results = []
        n_estimators = base_n_estimators
        lr = 1.48
        for train, test in kf:
            #dt = DecisionTreeClassifier(max_depth=26).fit(train_data_features, labels)
            rf = RandomForestClassifier(max_depth=395, n_estimators=80, max_features=7).fit(train_data_features, labels)
            #max_d += 2
            clf = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=rf, n_estimators = n_estimators, learning_rate = lr)
            #lr += 0.01
            model = clf.fit(train_data_features[train], labels[train])
            predicted_classes = model.predict(train_data_features[test])
            class_probabilities = model.predict_proba(train_data_features[test])
            print("ada week learners: ", n_estimators ,"learning rate ",lr," n points:", len(predicted_classes), ", wrong: ", (labels[test] != predicted_classes).sum(), " percentage: ",(labels[test] != predicted_classes).sum()*100/len(predicted_classes),"%"," sum of errors: ", _results[0])
            _results[0] += (labels[test] != predicted_classes).sum()
            ada_results.append((labels[test] != predicted_classes).sum())
            n_estimators += step_n_estimators
        n_estimators = base_n_estimators + step_n_estimators * ada_results.index(min(ada_results))
        print("optimized week learners ", n_estimators)
        #dt = DecisionTreeClassifier(max_depth=26).fit(train_data_features, labels)
        rf = RandomForestClassifier(max_depth=395, n_estimators=80, max_features=7).fit(train_data_features, labels)
        clf = AdaBoostClassifier(base_estimator=rf, n_estimators = n_estimators, learning_rate = lr)
        model = clf.fit(train_data_features, labels)
        return model.predict_proba(test_data_features), model.predict(test_data_features), model
    else:
        clf_l1_LR = LogisticRegression(C=1, penalty='l1', tol=0.01)
        lr = clf_l1_LR.fit(train_data_features, labels)
        dt = DecisionTreeClassifier()
        dt = dt.fit(train_data_features, labels)
        clf = AdaBoostClassifier(
            base_estimator=dt,
            learning_rate=1,
            n_estimators=250)
        model = clf.fit(train_data_features, labels)
        return model.predict_proba(test_data_features), model.predict(test_data_features), model

def des_tree(train_data_features, test_data_features, labels, using_cross_validation2, kf):
    if using_cross_validation2:
        _results = []
        base_max_depth = 6
        max_depth = base_max_depth
        step_max_depth = 100
        for train, test in kf:
            clf = DecisionTreeClassifier(max_depth=max_depth)
            model = clf.fit(train_data_features[train], labels[train])
            predicted_classes = model.predict(train_data_features[test])
            class_probabilities = model.predict_proba(train_data_features[test])
            print("maxd ",max_depth," |n points:", len(predicted_classes), ", wrong: ", (labels[test] != predicted_classes).sum(), " percentage: ",(labels[test] != predicted_classes).sum()*100/len(predicted_classes),"%")
            max_depth += step_max_depth
            _results.append((labels[test] != predicted_classes).sum())
        max_depth = 6 + step_max_depth * list(_results).index(min(_results))
        print("opt max depth ",max_depth)
        clf = DecisionTreeClassifier(max_depth = max_depth)
        model = clf.fit(train_data_features, labels)
        return model.predict_proba(test_data_features), model.predict(test_data_features), model
    else:
        clf = DecisionTreeClassifier()
        model = clf.fit(train_data_features, labels)
        return model.predict_proba(test_data_features), model.predict(test_data_features), model

def svm_model(train_data_features, test_data_features, labels, using_cross_validation2, kf):
    if using_cross_validation2:
        svm_results = np.zeros(10)
        #k_neighbors = 2
        #k_neighbors_results = []
        for train, test in kf:
            _svm = svm.SVC(probability=True, kernel='sigmoid')
            model = _svm.fit(train_data_features[train], labels[train])
            predicted_classes = model.predict(train_data_features[test])
            class_probabilities = model.predict_proba(train_data_features[test])
            print("n points:", len(predicted_classes), ", wrong: ", (labels[test] != predicted_classes).sum(), " sum of errors: ", svm_results[0])
            svm_results[0] += (labels[test] != predicted_classes).sum()
        k_neighbors = list(svm_results).index(min(svm_results)) + 2
        _svm = svm.SVC(probability=True)
        model = _svm.fit(train_data_features, labels)
        return model.predict_proba(test_data_features), model.predict(test_data_features), model

def k_nearest_neighbors(train_data_features, test_data_features, labels, using_cross_validation2, kf):
    if using_cross_validation2:
        k_neighbors_results = np.zeros(10)
        #k_neighbors = 1
        #k_neighbors_results = []
        for train, test in kf:
            for k_neighbors in range(2,10):
                clf = neighbors.KNeighborsClassifier(k_neighbors)
                model = clf.fit(train_data_features[train], labels[train])
                predicted_classes = model.predict(train_data_features[test])
                class_probabilities = model.predict_proba(train_data_features[test])
                print("K result, i - ", k_neighbors, ", n points:", len(predicted_classes), ", wrong: ", (labels[test] != predicted_classes).sum(), " sum of errors: ", k_neighbors_results[k_neighbors], " percentage: ",(labels[test] != predicted_classes).sum()*100/len(predicted_classes),"%")
                k_neighbors_results[k_neighbors] += (labels[test] != predicted_classes).sum()
        k_neighbors = list(k_neighbors_results).index(min(k_neighbors_results)) + 2
        print("k = ",k_neighbors)
        clf = neighbors.KNeighborsClassifier(k_neighbors)
        model = clf.fit(train_data_features, labels)
        predicted_classes = model.predict(test_data_features)
        return model.predict_proba(test_data_features), model.predict(test_data_features), model
    else:
        k_neighbors = 8
        clf = neighbors.KNeighborsClassifier(k_neighbors)
        model = clf.fit(train_data_features, labels)
        return model.predict_proba(test_data_features), model.predict(test_data_features), model