

'''
for the first algo, which will be a simple algo, will compare the volume
with the movement of the price, do this by finding the percent change
of the volume and also the percent change of the price. find pattern,
the end goal is to "predict the future". 
find the corralation between two company, in which the company A moves 
and company B moves one day after company B, this is to predict the future
'''

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

pd.set_option('display.expand_frame_repr', False)

def get_stocks():
    list_of_stock = requests.get('https://en.wikipedia.org/wiki/S%26P_100')
    soup = bs.BeautifulSoup(list_of_stock.text)
    tab = (soup.find('table', {'class':'wikitable sortable'}))
    tab_1 = tab.find_all('td')
    contant = []
    for row in tab_1:
        if row.get_text().upper() and len(row.get_text())<=5:
            contant.append(row.get_text())
    
    new_contant = [x.replace('\n','') for x in contant]
    return new_contant


def get_stock_data():
    datas = get_stocks()
    if not os.path.exists('stock_mid_data'):
        os.makedirs('stock_mid_data')

    start = dt.datetime(2018, 1, 1)
    end = dt.datetime(2019,1, 22)

    for data in datas:
        if not os.path.exists('stock_mid_data/{}.csv'.format(data)):
            df = web.DataReader(data, 'yahoo', start, end)
            df.to_csv('stock_mid_data/{}.csv'.format(data))
        else:
            print('got it')
    
  
    
def calc():
    datas = get_stocks()
    
    main_df = pd.DataFrame()
    try: 
        for data in datas:
            df = pd.read_csv('stock_mid_data/{}.csv'.format(data))
            main_df['{}_volume'.format(data)] = df['Volume']
            main_df['{}_prices'.format(data)] = df['Adj Close']
    
    except Exception:
        pass
    return main_df 


def corr(stock_name):

    list_of_stock = calc()
    print(list_of_stock)
    main_df = pd.DataFrame()
    main_df['{}_prices'.format(stock_name)] = list_of_stock['{}_prices'.format(stock_name)]
    main_df['{}_volume'.format(stock_name)] = list_of_stock['{}_volume'.format(stock_name)]

    return main_df


def precent_change(stock_name):
    df = corr(stock_name)
##  df_new = pd.DataFrame(columns = ['prices_precent_change', 'volume_precent_change'])
    
    
    #AAPL_prices  AAPL_volume
   
##
##            precent_change = ((df.AAPL_prices[i+1]-df.AAPL_prices[i])/df.AAPL_prices[i])*100
##            df_new = df_new.append({'prices_precent_change': precent_change }, ignore_index=True)
##
##            precent_change_vol = ((df.AAPL_volume[i+1]-df.AAPL_volume[i])/df.AAPL_volume[i])*100
##            df_new = df_new.append({'volume_precent_change': precent_change_vol }, ignore_index=True)

    df_new = df[[ticker for ticker in df.columns.values.tolist()]].pct_change()
    df_new['prices_precent_change'] = df_new['{}_prices'.format(stock_name)]
    df_new['volume_precent_change'] = df_new['{}_volume'.format(stock_name)]
    df_new.drop(['{}_prices'.format(stock_name), '{}_volume'.format(stock_name)], axis = 1, inplace = True)
    df_new_new = pd.DataFrame()
    df_new_new['label'] = df_new['prices_precent_change']
    df_new_new['volume_label'] =  df_new['prices_precent_change']
    
    for tic in range(len(df_new.prices_precent_change)):
       
        
            
        if (df_new.prices_precent_change[tic]) > 0.001:
            df_new_new.at[tic, 'label'] = 1
            ##the .at() method replaces the row value with new value
        elif (df_new.prices_precent_change[tic]) < 0:
            df_new_new.at[tic, 'label'] = -1 
        else:
            df_new_new.at[tic, 'label'] = 0
    
            
    df_new['label'] = df_new_new.label
    df_new.prices_precent_change[0] = 0
    df_new.volume_precent_change[0] = 0
    list_of_label = (list(df_new['label']))
    count_1 = list_of_label.count(-1)
    count_2 = list_of_label.count(0)
    count_3 = list_of_label.count(1)
##    print(df_new)
##    print('the count of -1:', count_1)
##    print(10*'-')
##    print('the count of 0:', count_2)
##    print(10*'-')
##    print('the count of 1:', count_3)
##    print(df)
    
    stock_avg_colu = (df['{}_volume'.format(stock_name)].mean())
    for vol in range(len(df['{}_volume'.format(stock_name)])):


        if (df['{}_volume'.format(stock_name)][vol]) > stock_avg_colu:
            df_new_new.at[vol, 'volume_label'] = 1


        elif (df['{}_volume'.format(stock_name)][vol]) < stock_avg_colu:
            df_new_new.at[vol, 'volume_label'] = -1


        else:
            df_new_new.at[vol, 'volume_label'] = 0


    df_new['volume_label'] = df_new_new['volume_label']
    print(df_new)
   
         
    return df_new
##precent_change('AAPL')
    
def machining_the_data(stock):
    stock_data = precent_change(stock)
    X = np.array(stock_data.drop(['label'],1))
    y = np.array(stock_data['label'])
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .95)

    clf = VotingClassifier([('lsvc',svm.LinearSVC()),
                            ('knn',neighbors.KNeighborsClassifier()),
                            ('rfor',RandomForestClassifier())])

    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('accuracy:',confidence)
    print(X_test)
    
    predictions = clf.predict(np.array([[ 1.44231225,-9.70757936 ,-1.00000000],
 [ 2.96192498, -2.77678573 , 1.00000000],
 [ 2.50403844 , 6.29763054, -1.00000000]]))
    print(predictions)
     
    

machining_the_data('AAPL')
