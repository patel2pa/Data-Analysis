

import requests
import bs4 as bs
import pandas as pd
import pandas_datareader.data as web
import datetime as dt 
import os
from sklearn import svm, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np




def s_and_p_list():
   
    
    main_df = pd.DataFrame()
    
    
    
    
    for data in os.listdir('s_and_p_500/'):
        new_main_df = pd.DataFrame()
        df = pd.read_csv('s_and_p_500/{}'.format(data))
##        main_df['{}_volume'.format(data)] = df['Volume']
##        main_df['{}_prices'.format(data)] = df['Adj Close']
        new_main_df['{}_open'.format(data)] = df['High']
        new_main_df['{}_close'.format(data)] = df['Low']
        main_df['{}_avg'.format(data)] =  new_main_df.mean(axis = 1)
##        print(new_main_df)
        
    main_df.drop(main_df.index[[-1]], inplace = True)
    
    
    return main_df 
     
##s_and_p_list() 
    
def per_stock(stock):
    df = pd.DataFrame()
    new_df = pd.DataFrame()
    my_stock = pd.read_csv('s_and_p_500/{}.csv'.format(stock))
    df['open'] = my_stock['High']
    df['close'] = my_stock['Low']
    new_df['avg'] = df.mean(axis = 1)
      
    return (new_df)

##per_stock('AAPL')



def comparison(stock):
    stock_corr = []
    
    stocks = s_and_p_list()
    my_stock = per_stock(stock)
    stocks['avg'] = my_stock['avg']
    
    stocks['avg'] = stocks['avg'].shift(-1)
    stocks.drop(stocks.index[[-1]], inplace = True)
    stooo = pd.DataFrame()
##    stooo['stock'] = stocks['{}.csv_open'.format(stock)]
    stocks.drop(['{}.csv_avg'.format(stock)], axis=1, inplace = True)
    
  
    df_new = stocks[[ticker for ticker in stocks.columns.values.tolist()]].pct_change()
    

    df_new_new = pd.DataFrame()
    for st in df_new.columns:
        df_new_new[st] = df_new[st]
        for num in range(len(df_new[st])):
            try:
        
                if (df_new[st][num]) > 0.0 :
                    df_new_new.at[num, st] = 1
                else:
                    df_new_new.at[num, st] = -1
            except Exception:
                pass
            
            
            
            
    
    for stocksss in df_new_new.columns:
        stock_corr.append((df_new_new[stocksss].corr(df_new_new['avg'])))

    

    second_stock_corr = {}
    
    for stocksss in df_new_new.columns:
        corrr = df_new_new[stocksss].corr(df_new_new['avg'])
        second_stock_corr[stocksss] = corrr
    strong_stock_corr = {}

    for st in second_stock_corr:
        if   (second_stock_corr[st]) > .16:
            strong_stock_corr[st] = second_stock_corr[st]
    strong_stock_corr.pop('avg')

    strong_stock_table = pd.DataFrame()

    for sts in strong_stock_corr:
        strong_stock_table[sts] = df_new_new[sts]
    
    
    strong_stock_table['avg'] = df_new_new['avg']
    
    (strong_stock_table.drop(strong_stock_table.index[[0]],inplace = True))
    

##    print(second_stock_corr)
##    print(strong_stock_table)
    return (strong_stock_table)  
            
    
##    print(max_count)
##    print(min_count)
##    print(max_value)
##    print(min_value)

##(comparison('AAPL'))

def machine_stuff(stock):
    stock_list = comparison(stock)


    
##    df_new = stock_list[[ticker for ticker in stock_list.columns.values.tolist()]].pct_change()
    
    #main_df.drop(main_df.index[[-1]], inplace = True)
    df_new = stock_list
##    (df_new.drop(df_new.index[[0]],inplace = True))
    
##    df_new_new = pd.DataFrame()
##    
##    df_new_new['labels'] = df_new['avg']
##    
##    
##    for num in range(len(df_new['avg'])):
##        try:
##        
##            if (df_new['avg'][num]) > 0.0 :
##                df_new_new.at[num, 'labels'] = 1
##            else:
##                df_new_new.at[num, 'labels'] = -1
##        except Exception:
##            pass
##   
##    (df_new['labels']) = df_new_new['labels']
##    (df_new['labels'][df_new.index[-1]]) = 1
##    
##    print(df_new)
##    df_new.drop('avg', axis = 1, inplace = True)

    
        
   
    
    
##    print(df_new['{}.csv_open'.format(stock)], df_new['labels'])
    
##    x = np.array(df_new['{}'.format(col)])
##    X = x.reshape(-1,1)
        
        
    y = np.array(df_new['avg'])

    
    X = np.array(df_new.drop(['avg'], 1))
        
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = .75)
        
    clf = VotingClassifier([('lsvc',svm.LinearSVC()),
                                ('knn',neighbors.KNeighborsClassifier()),
                                ('rfor',RandomForestClassifier())])
        
    clf.fit(x_train, y_train)
        
    confidence = clf.score(x_test, y_test)
    
        
    print(confidence)
        
    return confidence

(machine_stuff('AAPL'))

##y = []
##    
##for r in range(30):
##    if r < 25:
##        y.append((machine_stuff('F')))
        
