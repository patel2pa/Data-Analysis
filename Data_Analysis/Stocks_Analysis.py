import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
#from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates
import pandas as pd
import pandas_datareader.data as web
import bs4 as bs
import pickle
import requests
import datetime as dt
import os
import numpy as np 



'''
Function uses web scraper to get tags of S&P 500 companies from wikipedia
 and saves it in the tickers variable and returns it. 
'''
def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text)
    table = soup.find('table', {'class':'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
    print(tickers)
    return tickers

'''
Function uses the
'''
def get_data_from_yahoo():
    #tickers = save_sp500_tickers
    tickers = ['GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GS', 'GT']

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2000,1,1)
    end = dt.datetime(2016,12,31)

    for ticker in tickers:
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, 'yahoo', start, end)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))

        else:
            print('Already got it')
get_data_from_yahoo()     

def compile_data():
    
    tickers = ['GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GS', 'GT']
    main_df = pd.DataFrame()

    for ticker in tickers:
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace = True)

        df.rename(columns = {'Adj Close':ticker}, inplace = True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace = True)


        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how = 'outer')

       

    print(main_df.tail())
    main_df.to_csv('joint table.csv')

compile_data()

'''
def visualuze_data():
    df = pd.read_csv('joint table.csv')
##    df['GIS'].plot()
##   plt.show()
    ##compute the corrlation between the columns 
    df_corr = df.corr()
    print(df_corr.head())

    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0])+ 0.5, minor = False )
    ax.set_yticks(np.arange(data.shape[1]) +0.5, minor = False )
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    columns_labels = df_corr.columns
    row_labels = df_corr.index

    ax.set_xticklabels(columns_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation = 90)
    heatmap.set_clim(-1,1)
    plt.tight_layout()
    plt.show()
visualuze_data()
'''






'''
start = dt.datetime(2000,1,1)
end = dt.datetime(2016,1,1)

df = web.DataReader('TSLA', 'yahoo', start, end)

print(df.head())

##we convert the data to a csv file with using the following code
##df.to_csv('tsla.csv')



##df['100ma'] = df['Adj Close'].rolling(window = 100).mean()
##the above line is will find the 100 day moving avarage


df.dropna(inplace = True)
print(df.tail())


df_ohlc = df['Adj Close'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum()


df_ohlc.reset_index(inplace = True)
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)


print(df_ohlc.head())
##print(df_volume.head())





ax1 = plt.subplot2grid((6,1),(0,0), rowspan=5, colspan = 1)
ax2 = plt.subplot2grid((6,1),(5,0), rowspan=1, colspan = 1,sharex = ax1 )
ax1.xaxis_date()
ax1.plot(df.index, df['Adj Close'])

ax2.bar(df.index, df['Volume'])

plt.show()
'''





