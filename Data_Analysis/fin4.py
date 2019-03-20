'''
For this algo, the table will have the values of open, percent change in volume and
adj close will be use as the label. The theory is that the amount of volume will have an impact on the
adj close price
'''

#need to change the algo, get rid of the price depandence
# need to add a different parameter

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
    list_of_stock = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
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
    if not os.path.exists('s_and_p_500'):
        os.makedirs('s_and_p_500')

    start = dt.datetime(2018, 1, 1)
    end = dt.datetime(2019,1, 22)
    
    for data in datas:
        
        if not os.path.exists('s_and_p_500/{}.csv'.format(data)):
            try: 
            
                df = web.DataReader(data, 'yahoo', start, end)
                df.to_csv('s_and_p_500/{}.csv'.format(data))
                
            except Exception:
                pass
        else:
            print('got it')
    
get_stock_data()  
    
def calc():
    datas = get_stocks()
    
    main_df = pd.DataFrame()
    try: 
        for data in datas:
            df = pd.read_csv('stock_mid_data/{}.csv'.format(data))
            main_df['{}_volume'.format(data)] = df['Volume']
            main_df['{}_prices'.format(data)] = df['Adj Close']
            main_df['{}_open'.format(data)] = df['Open']
    
    except Exception:
        pass
    return main_df 


def corr(stock_name):

    list_of_stock = calc()
    
    main_df = pd.DataFrame()
    main_df['{}_prices'.format(stock_name)] = list_of_stock['{}_prices'.format(stock_name)]
    main_df['{}_volume'.format(stock_name)] = list_of_stock['{}_volume'.format(stock_name)]
    main_df['{}_open'.format(stock_name)] = list_of_stock['{}_open'.format(stock_name)]
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
    df_new['prices_precent_change'] = df_new['{}_open'.format(stock_name)]
    df_new['volume_precent_change'] = df_new['{}_volume'.format(stock_name)]
    df_new['close_price_precent_change'] = df_new['{}_prices'.format(stock_name)]
    
    
    df_new.drop(['{}_open'.format(stock_name), '{}_volume'.format(stock_name),'{}_prices'.format(stock_name) ], axis = 1, inplace = True)
    df_new_new = pd.DataFrame()##machining_the_data('AAPL')

    df_new_new['label'] = df_new['prices_precent_change']
    df_new_new['volume_label'] =  df_new['prices_precent_change']
   
    for tic in range(len(df['{}_prices'.format(stock_name)])):
       
        
            
        if (df['{}_prices'.format(stock_name)][tic]) > (df['{}_open'.format(stock_name)][tic]):
            df_new_new.at[tic, 'label'] = 1
            ##the .at() method replaces the row value with new value
        elif (df['{}_prices'.format(stock_name)][tic]) < (df['{}_open'.format(stock_name)][tic]):
            df_new_new.at[tic, 'label'] = -1 
        else:
            df_new_new.at[tic, 'label'] = 0
    
            
    df_new['label'] = df_new_new.label
    (df_new['open'])=df['{}_open'.format(stock_name)]
    
    
    df_new.drop([('prices_precent_change'),('close_price_precent_change'), ('volume_precent_change')], axis = 1, inplace = True )
    
##    df_new.volume_precent_change[0] = 0
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
#precent_change('AAPL')
    
def machining_the_data(stock):
    stock_data = precent_change(stock)
    print(stock_data)
    X = np.array(stock_data.drop(['label'],1))
    y = np.array(stock_data['label'])
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .50)

    clf = VotingClassifier([('lsvc',svm.LinearSVC()),
                            ('knn',neighbors.KNeighborsClassifier()),
                            ('rfor',RandomForestClassifier())])

    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('accuracy:',confidence)
##    print(X_test)
    ## the first column is the precent change in volume, second is the open and third is the volume labels
    
    predictions = clf.predict(np.array([[ 34, 1.00000000]]))
   
    print(predictions)
     
    

##machining_the_data('AAPL')
