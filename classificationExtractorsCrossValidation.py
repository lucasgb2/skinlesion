import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer

from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, recall_score, f1_score, precision_score, accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix, ConfusionMatrixDisplay
import scikitplot as skplt
import pandas as pd
import seaborn as sns
from datetime import datetime, timezone
import os
import constHam1000 as ham
import neptune
from neptunecontrib.monitoring.metrics import * #log_classification_report, log_binary_classification_metrics
from neptunecontrib.api import log_table, log_chart

project = ''
neptune.init(project,  api_token='')
neptune.set_project('lucasgb2/skin-lesion-2')

#DIR_FEATURES = os.path.join(ham.FEATURES, 'without-seg','without-agu','features')
#DIR_FEATURES = os.path.join(ham.FEATURES, 'without-seg','without-agu','deep')

#DIR_FEATURES = os.path.join(ham.FEATURES, 'without-seg','with-agu','features')
#DIR_FEATURES = os.path.join(ham.FEATURES, 'without-seg','with-agu','deep')

#DIR_FEATURES = os.path.join(ham.FEATURES, 'with-seg','with-agu','features')
DIR_FEATURES = os.path.join(ham.FEATURES, 'with-seg','with-agu-seg-kaggle','features')
#DIR_FEATURES = os.path.join(ham.FEATURES, 'with-seg','with-agu','deep')

#NAME_FILE_COMPLETE = 'without-seg-without-agu-'
#NAME_FILE_COMPLETE = 'without-seg-with-agu-'
NAME_FILE_COMPLETE = 'with-seg-with-agu-'
GROUP_EXPERIMENTE = NAME_FILE_COMPLETE




PARAMS = {
        'test_size':0.20,
        'shuffle':True,
        'stratify':True,        
        'split':5,
        'random_state':10,
        'imagens':'originais',        
        'segmentado':'não',
        'dir_features':DIR_FEATURES
        }


FEATURES = ['BIC.txt',
            'Moments.txt',         
            'AutoColorCorrelogram.txt',            
            'ReferenceColorSimilarity.txt',
            'CEDD.txt',
            'FCTH.txt',
            'Gabor.txt',
            'GCH.txt',
            'Haralick.txt',
            'HaralickColor.txt',
            'HaralickFull.txt',
            'JCD.txt',
            'LBP.txt',
            'LCH.txt',            
            'MPO.txt',
            'MPOC.txt',
            'PHOG.txt',                                    
            'Tamura.txt'
            ]
"""
FEATURES = [    
    'VGG16.txt',
    'MobileNet.txt',
    'VGG19.txt',
    'DenseNet121.txt',
    'ResNet50.txt',
    'InceptionV3.txt',          
    'Xception.txt'
]
"""

classifiersNames = [
                    "Nearest Neighbors", 
                    "Linear SVM",
                    #"RBF SVM", 
                    ##"Gaussian Process",
                    ##"Decision Tree", 
                    "Random Forest", 
                    "Neural Net", 
                    #"AdaBoost",
                    #"Naive Bayes", 
                    ##"QDA"
                    ]

classifiers = [
    KNeighborsClassifier(5, ),
    SVC(kernel="linear", ),
    #SVC(gamma=2, C=1),
    ##GaussianProcessClassifier(1.0 * RBF(1.0)),
    ##DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(),
    MLPClassifier(alpha=1, max_iter=1000, ),
    #AdaBoostClassifier(),
    #GaussianNB(),
    ##QuadraticDiscriminantAnalysis()
    ]

def autolabel(rects, axis):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        axis.annotate('{}'.format(round(height,3)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom')    

def getFeaturesForClassification():
    for featureName in FEATURES:
        pathFeature = os.path.join(DIR_FEATURES, NAME_FILE_COMPLETE + featureName)        
        
        if os.path.exists(pathFeature) == False:
            print('Arquivo não encontrado: '+pathFeature)
            continue

        df = pd.read_csv(pathFeature, sep=" ", header=None, skiprows=[0])  
        df = df.fillna(0)
        labels = df[1]  
        
        #removendo a primeira coluna que é um Index        
        df = df.drop(df.columns[0], axis=1)
        df = df.drop(df.columns[0], axis=1)
        yield df, labels


def run():
    dfResultMetricsAll = pd.DataFrame()
    for features in FEATURES:
        featureName = features
        dfResultMetrics = pd.DataFrame()
        
        features = os.path.join(DIR_FEATURES, NAME_FILE_COMPLETE+features)    
        
        if os.path.exists(features) == False:
            print('Arquivo não encontrado: '+features)
            continue
        
        print(features)
        df = pd.read_csv(features, delim_whitespace=True, header=None, skiprows=[0])
        #df = pd.read_csv(features, delim_whitespace=True, header=None)
        df = df.fillna(0)
        labels = df[1]  
        
        
        #removendo a primeira coluna que é um sequenciador        
        df = df.drop(df.columns[0], axis=1, )        
        df = df.drop(df.columns[0], axis=1, )    
        
                
        #df = df.reset_index(drop=True)               
        #random_state([0,10,20,30,40,50,60,70,80,90])
        #for seed in random_state:
        #x_train, x_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, shuffle=True, random_state=1, stratify=labels)           
        
        features = os.path.basename(features)

        neptune.create_experiment(name='extractor-features-'+featureName, 
                                  upload_source_files=__file__,
                                  description='Dataset HAM10000, Imagens segmentadas com com aumento.Conjunto do kaggle',
                                  params=PARAMS)
        neptune.set_property('group-experiment', GROUP_EXPERIMENTE)
        neptune.set_property('featureName', featureName)
        neptune.set_property('featureNameAll', features)
        neptune.log_text('log','características '+ features)        
        

        fig, axis = plt.subplots()
        axis.set_ylabel('Acurácia')
        axis.set_title(features)
        axis.set_xticks(np.arange(len(classifiersNames)))
        axis.set_xticklabels(classifiersNames)
        axis.legend()    
        index = 0    
        meanAcc = 0            
        
        for name, clf in zip(classifiersNames, classifiers):   
            neptune.log_text('log','-------------------------------------')                                         
            print(format('Executando classificação com algoritmo %s features: %s' % (name, features)))
            neptune.log_text('log', format('Executando classificação com algoritmo %s features: %s' % (name, features)))            
            neptune.log_text('log', str(clf))
            
            neptune.log_text('log', 'cross validation fold'+str(PARAMS['split']))
            print('Início da Predição: ', datetime.now())                
            
            #Está método instancia um ojeto responsável por fazer o k-fold estartificado por classe
            #o parâmetro n_split é a quantidade de split, no começo do código há a definição
            #o parâmetro random_state é a semente responsável por garantir que o split seja igual toda
            #a vez que for executado       
            splitCount = 1
            for styleSplit in [PARAMS['random_state']]:
                neptune.log_text('log-split', '5 vezes para testar todo o dataset, 5 vezes aleatório')
                skf = StratifiedKFold(n_splits=PARAMS['split'], 
                                    random_state=styleSplit, 
                                    shuffle=True)
                accs = []      
                precisions = []      
                recalls = []
                fscores = []
                balanceds = []
                
                        
                #cada interação o objeto skf irá retornar o index das amostras            
                for train_index, test_index in skf.split(df, labels):                
                    x_train, x_test = df.iloc[train_index], df.iloc[test_index]
                    y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]                                

                    x_train = Normalizer().fit_transform(x_train)
                    x_test = Normalizer().fit_transform(x_test)

                    neptune.log_text('log-split', name)                              
                    neptune.log_text('log-split', str(clf)) 
                    neptune.log_text('log-split', str(splitCount))                         
                    neptune.log_text('log-split', 'qtde-train %i qtde-test %i total %i' % ( len(x_train), len(x_test), len(x_train) + len(x_test)  ))
                    neptune.log_text('log-split', 'perc-train %0.4f perc-test %0.4f ' % ( len(x_train) / (len(x_train) + len(x_test)), len(x_test) /(len(x_train) + len(x_test))  ))
                    splitCount +=1

                    
                    clf.fit(x_train, y_train)
                    predict = clf.predict(x_test)
                    if hasattr(clf, "decision_function"):
                        predict_proba = clf.decision_function(x_test)
                    else:
                        predict_proba = clf.predict_proba(x_test)                                         
                    
                    acc = accuracy_score(y_test, predict)                                   
                    full = precision_recall_fscore_support(y_test, predict, average='macro')
                    balanced = balanced_accuracy_score(y_test, predict)                                   
                    
                    accs.append(acc)                            
                    precisions.append(full[0])
                    recalls.append(full[1])
                    fscores.append(full[2])
                    balanceds.append(balanced)
                    
                    score = "Accuracy: %0.4f  | Precision: %0.4f | Recall/Sensitivity: %0.4f | FScore: %0.4f | Balanced: %0.4f" % (acc, full[0], full[1], full[2], balanced)
                        
                    print(score)                
                    neptune.log_text('log', score)
                    figsize_ = (13,10)  
                    plot = skplt.metrics.plot_confusion_matrix(y_test, predict, normalize=True, figsize=figsize_, title_fontsize='large', text_fontsize='large', title='CM-Normalize '+name+'-'+features)
                    neptune.log_image(name, plot.figure, description='Matriz Confusão')        
                    del plot
                    
                    plot = skplt.metrics.plot_precision_recall(y_test, predict_proba, figsize=figsize_, title_fontsize='large', text_fontsize='large', title='Precision-Recall '+name+'-'+features)
                    neptune.log_image(name, plot.figure) 
                    del plot
                    
                    plot = skplt.metrics.plot_roc(y_test, predict_proba, figsize=figsize_, title_fontsize='large', text_fontsize='large', title='ROC '+name+'-'+features)
                    neptune.log_image(name, plot.figure) 
                    del plot
                    
                    del x_train
                    del x_test
                    del y_train
                    del y_test
                    del predict
               ############################################################
               #   MÉDIA DAS MÉTRICAS DA CLASSIFICAÇÃO POR CLASSIFICADOR  #
               ############################################################

                scores = "(MEAN) Accuracy: %0.4f (+/- %0.4f) | Precision: %0.4f | Recall/Sensitivity: %0.4f | FScore: %0.4f | Balanced: %0.4f | Estimator: %s " % \
                             (np.mean(accs), np.std(accs) * 2, np.mean(precisions), np.mean(recalls), np.mean(fscores), np.mean(balanceds), clf)
                
                print(scores)                
                neptune.log_text('log', scores)

                r = {
                'features':featureName,
                'estimator':name,             
                'acc':np.mean(accs).round(4),
                'std':np.std(accs).round(4) * 2,
                'precision':np.mean(precisions).round(4),
                'recall':np.mean(recalls).round(4),
                'fscore':np.mean(fscores).round(4),
                'params_estimator':[str(clf)]}
                
                metrics = pd.DataFrame(r)
                dfResultMetrics = dfResultMetrics.append(metrics)                             
                print('Predição concluída: ', datetime.now())            
                
                #média das acurácias das execuções do classificador
                meanAcc = np.mean(accs)                        
                b = axis.bar(index, meanAcc, width=0.2, align='center')                 
                autolabel(b, axis)
                index += 1
                
        dfResultMetrics.sort_values(by=['acc'], ascending=False, inplace=True)              
        log_table(features, dfResultMetrics)             
        neptune.log_image('ranking-acc-geral', fig, description='Média das ACCs de cada classificador')
        theBestAcc = '%0.4f %s' % (dfResultMetrics.iloc[0]['acc'], dfResultMetrics.iloc[0]['estimator'])
        neptune.log_text('best_ACC', theBestAcc)
        neptune.log_text('best-estimator', dfResultMetrics.iloc[0]['estimator'])
        neptune.log_metric('best-acc', dfResultMetrics.iloc[0]['acc'])
        neptune.log_metric('best-std', dfResultMetrics.iloc[0]['std'])
        neptune.log_metric('best-precision', dfResultMetrics.iloc[0]['precision'])
        neptune.log_metric('best-recall', dfResultMetrics.iloc[0]['recall'])
        neptune.log_metric('best-fscore', dfResultMetrics.iloc[0]['fscore'])
        
        neptune.stop()
        del df
        dfResultMetricsAll = dfResultMetricsAll.append(dfResultMetrics)                     
        dfResultMetricsAll.to_csv('all_features_metrics_'+NAME_FILE_COMPLETE+'.csv')
run()
