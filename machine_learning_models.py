import numpy as np

from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier

import aux_functions
from sklearn import svm, metrics
from sklearn import neighbors
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold

#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def naive_bayes(train_data_features, train_data_split_crossfold_features, test_data_features, labels, labels_cross_validation_classwise, kf, settings):
    gnb = GaussianNB()
    model = gnb.fit(train_data_features, labels)
    return model.predict_proba(test_data_features), model.predict(test_data_features), model

def perc(train_data_features, train_data_split_crossfold_features, test_data_features, labels, labels_cross_validation_classwise, kf, settings):
    prc = Perceptron()
    model = prc.fit(train_data_features, labels)
    return model.predict_proba(test_data_features), model.predict(test_data_features), model

def log_res(train_data_features, train_data_cross_validation_classwise_features, test_data_features, labels, labels_cross_validation_classwise, using_cross_validation2, kf, settings):
    if using_cross_validation2:
        logres_C = 1
        logres_results = []
        if(len(train_data_cross_validation_classwise_features) > 0):
            train_all = np.append(train_data_features, train_data_cross_validation_classwise_features, axis=0)
            labels_all = np.append(labels, labels_cross_validation_classwise)
            kf_all = KFold(len(train_all)-1, n_folds=int(settings['Data']['CrossValidation2']), shuffle=True)
            for train, test in kf_all:
                C = logres_C
                p = 'l1'
                clf_l1_LR = LogisticRegression(C=C, penalty=p, tol=0.01)
                model = clf_l1_LR.fit(train_all[train], labels_all[train])
                predicted_classes = model.predict(train_all[test])
                predicted_classes_train = model.predict(train_all[train])
                print("N points:", len(predicted_classes), " percentage: ",(labels_all[test] != predicted_classes).sum()*100/len(predicted_classes),"%, percentage_train: ", (labels_all[train] != predicted_classes_train).sum()*100/len(predicted_classes_train))
                logres_results.append((labels_all[test] != predicted_classes).sum())
                logres_C += 1
        else:
            for train, test in kf:
                C = logres_C
                p = 'l1'
                clf_l1_LR = LogisticRegression(C=C, penalty=p, tol=0.01)
                model = clf_l1_LR.fit(train_data_features[train], labels[train])
                predicted_classes = model.predict(train_data_features[test])
                predicted_classes_train = model.predict(train_data_features[train])
                print("N points:", len(predicted_classes), " percentage: ",(labels[test] != predicted_classes).sum()*100/len(predicted_classes),"%, percentage_train: ", (labels[train] != predicted_classes_train).sum()*100/len(predicted_classes_train))
                logres_results.append((labels[test] != predicted_classes).sum())
                logres_C += 1
        logres_C = logres_results.index(min(logres_results)) + 1
        print("Log Res C: ", logres_C)
        if(len(train_data_cross_validation_classwise_features) > 0):
            clf_l1_LR = LogisticRegression(C=logres_C, penalty=p, tol=0.01)
            model = clf_l1_LR.fit(train_data_features, labels)
            predicted_classes = model.predict(train_data_cross_validation_classwise_features)
            predicted_classes_train = model.predict(train_data_features)
            class_probabilities = model.predict_proba(train_data_cross_validation_classwise_features)
            print("N points:", len(predicted_classes), " percentage: ",(labels_cross_validation_classwise != predicted_classes).sum()*100/len(predicted_classes),"%, percentage_train: ", (labels != predicted_classes_train).sum()*100/len(predicted_classes_train))
            print("Log_loss: ", log_loss(labels_cross_validation_classwise, class_probabilities))
        clf_l1_LR = LogisticRegression(C=logres_C, penalty=p, tol=0.01)
        model = clf_l1_LR.fit(train_data_features, labels)
        return model.predict_proba(test_data_features), model.predict(test_data_features), model
    else:
        C = 1
        p = 'l1'
        clf_l1_LR = LogisticRegression(C=C, penalty=p, tol=0.01)
        model = clf_l1_LR.fit(train_data_features, labels)
        return model.predict_proba(test_data_features), model.predict(test_data_features), model

def cross_test(train_data_features, train_data_split_crossfold_features, test_data_features, labels, labels_cross_validation_classwise, using_cross_validation2, kf, settings):
    if using_cross_validation2:

        _results = []
        global_results = []

        names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
                 "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis",
                 "Quadratic Discriminant Analysis"]
        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025, probability=True),
            SVC(gamma=2, C=1, probability=True),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            AdaBoostClassifier(),
            GaussianNB(),
            LinearDiscriminantAnalysis(),
            QuadraticDiscriminantAnalysis()]

        for name, clf in zip(names, classifiers):
            for train, test in kf:
                model = clf.fit(train_data_features[train], labels[train])
                predicted_classes = model.predict(train_data_features[test])
                class_probabilities = model.predict_proba(train_data_features[test])
                print(name," n points:", len(predicted_classes), ", wrong: ", (labels[test] != predicted_classes).sum(), " percentage: ",(labels[test] != predicted_classes).sum()*100/len(predicted_classes),"%")
                _results.append((labels[test] != predicted_classes).sum())
            result = min(_results)
            global_results.append((name,result))
        print(global_results)

        clf = AdaBoostClassifier()
        model = clf.fit(train_data_features, labels)
        predicted_classes = model.predict(test_data_features)
        class_probabilities = model.predict_proba(test_data_features)
    else:
        clf = AdaBoostClassifier()
        model = clf.fit(train_data_features, labels)
        predicted_classes = model.predict(test_data_features)
        class_probabilities = model.predict_proba(test_data_features)

def linear_svm(train_data_features, train_data_cross_validation_classwise_features, test_data_features, labels, labels_cross_validation_classwise, using_cross_validation2, kf, settings):
    if using_cross_validation2:
        C_base = 4.5
        C_step = 0.5#0.005
        C = C_base
        _results = []
        if(len(train_data_cross_validation_classwise_features) > 0):
            train_all = np.append(train_data_features, train_data_cross_validation_classwise_features, axis=0)
            labels_all = np.append(labels, labels_cross_validation_classwise)
            kf_all = KFold(len(train_all)-1, n_folds=int(settings['Data']['CrossValidation2']), shuffle=True)
            for train, test in kf_all:
                svc = SVC(kernel="linear", C=C, probability=True)
                model = svc.fit(train_all[train], labels_all[train])
                predicted_classes = model.predict(train_all[test])
                predicted_classes_train = model.predict(train_all[train])
                class_probabilities = model.predict_proba(train_all[test])
                print("C: ",C," n points:", len(predicted_classes), " percentage: ",(labels_all[test] != predicted_classes).sum()*100/len(predicted_classes),"% percentage_train: ", (labels_all[train] != predicted_classes_train).sum()*100/len(predicted_classes_train),"%")
                _results.append((labels_all[test] != predicted_classes).sum())
                C += C_step
        else:
            for train, test in kf:
                svc = SVC(kernel="linear", C=C, probability=True)
                model = svc.fit(train_data_features[train], labels[train])
                predicted_classes = model.predict(train_data_features[test])
                predicted_classes_train = model.predict(train_data_features[train])
                class_probabilities = model.predict_proba(train_data_features[test])
                print("C: ",C," n points:", len(predicted_classes), " percentage: ",(labels[test] != predicted_classes).sum()*100/len(predicted_classes),"% percentage_train: ", (labels[train] != predicted_classes_train).sum()*100/len(predicted_classes_train),"%")
                _results.append((labels[test] != predicted_classes).sum())
                C += C_step
        C = C_base + C_step * _results.index(min(_results))
        print("C: ", C)
        if(len(train_data_cross_validation_classwise_features) > 0):
            svc = SVC(kernel="linear", C=C, probability=True)
            model = svc.fit(train_data_features, labels)
            predicted_classes = model.predict(train_data_cross_validation_classwise_features)
            class_probabilities = model.predict_proba(train_data_cross_validation_classwise_features)
            print("C: ",C," N points:", len(predicted_classes), " percentage: ",(labels_cross_validation_classwise != predicted_classes).sum()*100/len(predicted_classes),"%")
            print("Log_loss: ", log_loss(labels_cross_validation_classwise, class_probabilities))
        svc = SVC(kernel="linear", C=C, probability=True)
        model = svc.fit(train_data_features, labels)
        return model.predict_proba(test_data_features), model.predict(test_data_features), model
    else:
        svc = SVC(kernel="linear", C=8, probability=True)
        model = svc.fit(train_data_features, labels)
        return model.predict_proba(test_data_features), model.predict(test_data_features), model
def rbf_svm(train_data_features, train_data_cross_validation_classwise_features, test_data_features, labels, labels_cross_validation_classwise, using_cross_validation2, kf, settings):
    if using_cross_validation2:
        C_base = 4.5
        C_step = 0.5#0.005
        C = C_base
        gamma_base = 0.40
        gamma_step = 0.00
        gamma = gamma_base
        _results = []
        if(len(train_data_cross_validation_classwise_features) > 0):
            train_all = np.append(train_data_features, train_data_cross_validation_classwise_features, axis=0)
            labels_all = np.append(labels, labels_cross_validation_classwise)
            kf_all = KFold(len(train_all)-1, n_folds=int(settings['Data']['CrossValidation2']), shuffle=True)
            for train, test in kf_all:
                svc = SVC(kernel="rbf", C=C, gamma = gamma, probability=True)
                model = svc.fit(train_all[train], labels_all[train])
                predicted_classes = model.predict(train_all[test])
                predicted_classes_train = model.predict(train_all[train])
                class_probabilities = model.predict_proba(train_all[test])
                print("C: ",C," n points:", len(predicted_classes), " percentage: ",(labels_all[test] != predicted_classes).sum()*100/len(predicted_classes),"% percentage_train: ", (labels_all[train] != predicted_classes_train).sum()*100/len(predicted_classes_train),"%")
                _results.append((labels_all[test] != predicted_classes).sum())
                C += C_step
        else:
            for train, test in kf:
                svc = SVC(kernel="rbf", C=C, gamma = gamma)
                model = svc.fit(train_data_features[train], labels[train])
                predicted_classes = model.predict(train_data_features[test])
                predicted_classes_train = model.predict(train_data_features[train])
                print("C: ",C," n points:", len(predicted_classes), " percentage: ",(labels[test] != predicted_classes).sum()*100/len(predicted_classes),"% percentage_train: ", (labels[train] != predicted_classes_train).sum()*100/len(predicted_classes_train),"%")
                _results.append((labels[test] != predicted_classes).sum())
                C += C_step
        C = C_base + C_step * _results.index(min(_results))
        print("C: ", C)
        if(len(train_data_cross_validation_classwise_features) > 0):
            svc = SVC(kernel="rbf", C=C, gamma = gamma, probability=True)
            model = svc.fit(train_data_features, labels)
            predicted_classes = model.predict(train_data_cross_validation_classwise_features)
            class_probabilities = model.predict_proba(train_data_cross_validation_classwise_features)
            print("C: ",C," N points:", len(predicted_classes), " percentage: ",(labels_cross_validation_classwise != predicted_classes).sum()*100/len(predicted_classes),"%")
            print("Log_loss: ", log_loss(labels_cross_validation_classwise, class_probabilities))
        svc = SVC(kernel="rbf", C=C, gamma = gamma, probability=True)
        model = svc.fit(train_data_features, labels)
        return model.predict_proba(test_data_features), model.predict(test_data_features), model
    else:
        svc = SVC(kernel="rbf", C=8, gamma = 0.4, probability=True)
        model = svc.fit(train_data_features, labels)
        return model.predict_proba(test_data_features), model.predict(test_data_features), model

def adaboost(train_data_features, train_data_split_crossfold_features, test_data_features, labels, labels_cross_validation_classwise, using_cross_validation2, kf, settings):
    if using_cross_validation2:
        _results = np.zeros(10)
        base_n_estimators = 100 # week learners
        step_n_estimators = 100
        ada_results = []
        n_estimators = base_n_estimators
        lr = 1.48
        for train, test in kf:
            rf = RandomForestClassifier(max_depth=395, n_estimators=80, max_features=7).fit(train_data_features, labels)
            clf = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=rf, n_estimators = n_estimators, learning_rate = lr)
            model = clf.fit(train_data_features[train], labels[train])
            predicted_classes = model.predict(train_data_features[test])
            class_probabilities = model.predict_proba(train_data_features[test])
            print("ada week learners: ", n_estimators ,"learning rate ",lr," n points:", len(predicted_classes),
                  " percentage: ",(labels[test] != predicted_classes).sum()*100/len(predicted_classes),"%"," sum of errors: ", _results[0])
            _results[0] += (labels[test] != predicted_classes).sum()
            ada_results.append((labels[test] != predicted_classes).sum())
            n_estimators += step_n_estimators
        n_estimators = base_n_estimators + step_n_estimators * ada_results.index(min(ada_results))
        print("optimized week learners ", n_estimators)
        if(len(train_data_split_crossfold_features) > 0):
            rf = RandomForestClassifier(max_depth=395, n_estimators=80, max_features=7).fit(train_data_features, labels)
            clf = AdaBoostClassifier(base_estimator=rf, n_estimators = n_estimators, learning_rate = lr)
            model = clf.fit(train_data_features, labels)
            predicted_classes = model.predict(train_data_split_crossfold_features)
            class_probabilities = model.predict_proba(train_data_split_crossfold_features)
            print("N points:", len(predicted_classes), " percentage: ",(labels_cross_validation_classwise != predicted_classes).sum()*100/len(predicted_classes),"%")
            print("Log_loss: ", log_loss(labels_cross_validation_classwise, class_probabilities))
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

def des_tree(train_data_features, train_data_split_crossfold_features, test_data_features, labels, labels_cross_validation_classwise, using_cross_validation2, kf, settings):
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

def svm_model(train_data_features, train_data_split_crossfold_features, test_data_features, labels, labels_cross_validation_classwise, using_cross_validation2, kf, settings):
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

def k_nearest_neighbors(train_data_features, train_data_split_crossfold_features, test_data_features, labels, labels_cross_validation_classwise, using_cross_validation2, kf, settings):
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