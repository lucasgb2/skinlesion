#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 20:35:05 2021

@author: lucas
"""

import numpy as np

import dhaaActiveLearning
from dhaaActiveLearning import AL_Strategy, AL_Parameters
from dhaaActiveLearning.classification import Classifier
from dhaaActiveLearning.dataset import Dataset
import json
import os
import neptune.new as neptune
from neptune.new.types import File
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    np.random.seed(1) #for reproducibility on some al strategies

    def configureNeptune():
 	project = ''
        run = neptune.init(project,                           
                       source_files=[__file__],
                       api_token=''             )
        return run
    
    strategies = ['MS','LC','EN','DBE','MST-BE','RDS']
    #estimators = ['SVM','k-NN','RF','NB']
    estimators = ['RF']    
  
    def runExperiments():
        superDataSetAllResults = []
        experiments = ['datasets_without-seg-with-agu-','datasets_with-seg-with-agu-','datasets_with-seg-kaggle-with-agu-']        
        dfAllResults = None 
        iniciado = False
        for groupExperiment in experiments:        
            
            datasets = Dataset.get_names(groupExperiment)
           
            for feature in datasets:    
                dataSetFeature = 'd1'
                run = configureNeptune()
                
                run['parameters/groupExperiment'] = groupExperiment            
                run['parameters/feature'] = feature            
                run['parameters/dataset'] = dataSetFeature
                run['parameters/commet'] = 'pegando mais dados do classificador'
                
                for strategy in strategies:     
                    run['parameters/strategy'] = strategy                    
                    for estimator in estimators:         
                        run['parameters/estimator'] = estimator                        
                        n_splits = 1            
                        al_params = AL_Parameters(dataset_path=groupExperiment,
                                                  dataset_name=feature, 
                                                  strategy_name=strategy, 
                                                  classifier_name=estimator,                                          
                                                  max_iterations=8000,
                                                  #n_instances=7,
                                                  #n_clusters=7
                                                  )                        
                        results = dhaaActiveLearning.run(al_params=al_params, 
                                                         n_splits=n_splits)
                        run['parameters/n_instances'] = al_params.n_instances     
                        run['parameters/n_clusters'] = al_params.n_clusters     
                        
                        i = 0       
                        listResults = []
                        
                        accs = np.array([])    
                        qtdeSampleByIteration = np.array([])
                        for r in results.results_dict:
                        #for r in results:
                            #results = results[r]
                            if (r != 'classifier') and (r != 'strategy'):
                                v = results.results_dict[r]
                                i+=1
                                accs = np.append(accs, v['accuracy'][0])                                
                                qtdeSampleByIteration = np.append(qtdeSampleByIteration, r)                                
                                dictResults = {'dataset': dataSetFeature,
                                                'feature':feature,
                                                #'split': v['split'][0], 
                                                'strategy': strategy,
                                                'estimator': estimator,
                                                'n_clusters': al_params.n_clusters,
                                                'n_instances': al_params.n_instances,                                                
                                                'samples': r,
                                                'interation':i,
                                                'acc':v['accuracy'][0],
                                                'acc_mean':np.mean(accs),
                                                'acc_std':np.std(accs),
                                                'classification_time':v['classification_time'][0],
                                                'fscore':v['fscore'][0],
                                                'precision':v['precision'][0],
                                                'querying_time':v['querying_time'][0],
                                                'recall':v['recall'][0]} 
                                
                                listResults.append(dictResults)
                                superDataSetAllResults.append(dictResults)
                                
                        df = pd.DataFrame(listResults)
                        df.to_csv('results_'+strategy+'_'+estimator+'_'+feature)                        
                        run['data/results_'+strategy+'_'+estimator+'_'+feature].upload(File.as_html(df))                    
                        
                        #-----------------------------------------------
                        # plotando taxa de acc x sample
                        #-----------------------------------------------                        
                        totalSample = np.max(qtdeSampleByIteration)
                        rateSample = np.array(qtdeSampleByIteration) / totalSample
                        rateSample = np.array(rateSample) * 100
                        rateAccs = np.array(accs) * 100    
                        plt.style.context('Solarize_Light2')
                        plt.rcParams["figure.figsize"] = (20, 10)
                        plt.title("%Amostras X ACC "+"{:10.4f}".format(np.max(rateAccs))+
                                  ': '+strategy+
                                  ', '+estimator+
                                  ', feature='+feature+
                                  ', clusters='+str(al_params.n_clusters)+
                                  ', instances='+str(al_params.n_instances))
                        plt.xlabel("ITERAÇÕES EXECUTADAS CONTEMPLANDO TODOS OS DADOS DE TREINAMENTO")
                        plt.ylabel("% ACC");                                                 
                        plt.yticks(range(0, 100, 2))                        
                        plt.xticks(range(0, int(totalSample), 100))
                        try:
                            plt.grid()
                        except Exception:
                            pass                        
                        plt.plot(rateSample, color='blue', label='% amostras')
                        plt.plot(rateAccs, color='green', label='% ACC')
                        plt.plot(np.argmax(rateAccs)+1, np.max(rateAccs), 'ro', color='red', label='ACC '+"{:10.4f}".format(np.max(rateAccs)))                    
                        plt.legend()
                        plt.show()                       
                        
                        try:
                            fig = plt.figure(fig)
                            run['plots/results_'+strategy+'_'+estimator].upload(fig)
                        except Exception:
                            pass  
                        
                        
          
                run.stop()
    
    runExperiments()
   



    
    
