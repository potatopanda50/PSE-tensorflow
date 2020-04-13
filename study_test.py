from plotly.subplots import make_subplots
import plotly as py
import plotly.graph_objs as go

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output, State

#####################################################################
import tensorflow as tf
from collections import deque
import pandas as pd
import numpy as np
import os
import random
from sklearn import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session
from datetime import datetime,timedelta
from statistics import mean
#####################################################################

#reading CSV
def reading_csv():
    dfs = []

    for item in os.listdir('data'):
        df = pd.read_csv(f'data/{item}',
                        header=None,
                        names=['stock code','Date','open','high','low','close','volume','netforeign'])
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date',inplace=True)
        df.dropna(inplace=True)
        
        dfs.append(df)

    main_df = pd.concat(dfs)
    main_df.tail()
    # main_df=main_df.groupby('stock code')

    return main_df

#sorting and grouping
def sorting_and_grouping(df):
    #sorting base on dates
    df.sort_values('Date',inplace=True)
    #grouping by stock code
    df=df.groupby('stock code')
    return df

#list of all groups
def getting_allgroups(df):
    allgroups=[]
    for group,data in df:
        allgroups.append(group)
    return allgroups

#model list
def model_list():
    path = "model_saved"
    model_list = []
    for model in os.listdir(path):
        model_list.append(model)
    print(model_list)
    return model_list

#chart display
def chart_display(data):

    
    price = go.Candlestick(x=data.index,
                   open=data['open'],
                   high=data['high'],
                   low=data['low'],
                   close=data['close'],
                   name='prices')


    volume = go.Bar( 
        x = data.index,
        y = data['volume'],
        #fill= 'tozeroy',
        name= 'volume')

    layout = go.Layout(
        height = 700,
        xaxis = dict(
            rangeslider = dict(
                visible = False
            ),
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label='1m',
                         step='month',
                         stepmode='backward'),
                    dict(count=6,
                         label='6m',
                         step='month',
                         stepmode='backward'),
                    dict(count=1,
                        label='YTD',
                        step='year',
                        stepmode='todate'),
                    dict(count=1,
                        label='1y',
                        step='year',
                        stepmode='backward'),
                    dict(step='all')
                ])
            ),
        ),
        yaxis = dict(
            autorange=True
            )
        )
    fig = make_subplots(rows=4, cols=1,specs=[[{'rowspan': 3}],
                                            [None],
                                            [None],
                                            [{}]],
                      shared_xaxes=True
                     )

    fig.append_trace(price, 1,1)
    fig.append_trace(volume, 4,1)



    fig['layout'].update(layout)

    return fig

#CHART DISPLAY TEST
def chart_display_test(data,buy,sell):
    
    price = go.Candlestick(x=data.index,
                   open=data['open'],
                   high=data['high'],
                   low=data['low'],
                   close=data['close'],
                   name='prices')

    buys = go.Scatter(
                x = buy.index,
                y = buy['price'],
                mode = 'markers',
                marker = {'color': 'skyblue'},
                name = 'buy'        
                )

    sells = go.Scatter(
                    x = sell.index,
                    y = sell['price'],
                    mode = 'markers',
                    marker = {'color': 'red'},
                    name = 'sell'           
                )

    volume = go.Bar( 
        x = data.index,
        y = data['volume'],
        #fill= 'tozeroy',
        name= 'volume')

    layout = go.Layout(
        height = 700,
        xaxis = dict(
            rangeslider = dict(
                visible = False
            ),
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label='1m',
                         step='month',
                         stepmode='backward'),
                    dict(count=6,
                         label='6m',
                         step='month',
                         stepmode='backward'),
                    dict(count=1,
                        label='YTD',
                        step='year',
                        stepmode='todate'),
                    dict(count=1,
                        label='1y',
                        step='year',
                        stepmode='backward'),
                    dict(step='all')
                ])
            ),
        ),
        yaxis = dict(
            autorange=True
            )
        )

    fig = make_subplots(rows=4, cols=1,specs=[[{'rowspan': 3}],
                                        [None],
                                        [None],
                                        [{}]],
                  shared_xaxes=True
                 )

    fig.append_trace(price, 1,1)
    fig.append_trace(buys, 1,1)
    fig.append_trace(sells, 1,1)
    fig.append_trace(volume, 4,1)



    fig['layout'].update(layout)

    return fig
###############################################################
def buys_and_sells(data,model,data_copy):
    data.drop('stock code',1,inplace=True)
    data.drop('netforeign',1,inplace=True)
    data_copy.drop('stock code',1,inplace=True)
    data_copy.drop('netforeign',1,inplace=True)
    data['Date']=data.index

    sample,sequence_length,feature = model.input_shape 
    future_to_predict = 2

    for col in data.columns[:-1]:
            data[col] = data[col].pct_change()*100  # pct change "normalizes" the different currencies (each crypto coin has vastly diff values, we're really more interested in the other coin's movements)
            data.dropna(inplace=True)  # remove the nas created by pct_change
            data[col] = preprocessing.scale(data[col].values)  # scale between 0 and 1.
            
    data = data.interpolate()
    sequential_data = []
    prev_days = deque(maxlen=sequence_length)

    for i in data.values:  # iterate over the values
        #print(i)
        prev_days.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days) == sequence_length:  # make sure we have 60 sequences!
            sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!


    buys = []
    sells = []
    holds = []
    trades = []


    for seq, date in sequential_data[-100:]: #test on last 100 samples only
        seq = seq.reshape(-1,sequence_length,feature)
        x = model.predict([seq])
        x = np.argmax(x)
        
        price = float(data_copy[data_copy.index==date]['close'].values)
        print(x,date,price)
        if(x==1):
            trades.append([date,price,'Buy'])
            buys.append([date,price])
        elif(x==2):
            trades.append([date,price,'Sell'])
            sells.append([date,price])
        else:
            holds.append([date,price])

    trade_list = pd.DataFrame(trades,columns=['date','price','action'])

    buy = pd.DataFrame(buys,columns=['date','price'])
    sell = pd.DataFrame(sells,columns=['date','price'])
    hold = pd.DataFrame(holds,columns=['date','price'])

    buy.set_index('date',inplace=True)
    sell.set_index('date',inplace=True)
    hold.set_index('date',inplace=True)

    return buy,sell,hold,trade_list

#########################################################################
def Stats(buy,sell,trade_list):
    #holding period and profits

    buy_i = 0
    sell_i = 0
    count = 0
    sell_length = len(sell)
    buy_length = len(buy)

    #last 50 buy and sell
    #buy_i = buy_length-50
    # sell_i = sell_length-50


    holding_days_list = []
    total_holding_days = timedelta()

    percent_profit_list = []
    total_percent_profit = float()

    win = 0
    loss = 0


    while sell_i < sell_length:
        while buy.index[buy_i]<sell.index[sell_i]:
            
            ## holding days ... min,max,avg
            holding_days = sell.index[sell_i]-buy.index[buy_i]
            total_holding_days += holding_days
            holding_days_list.append(holding_days)
            
            ## %profit ... min,max,avg
            percent_profit = (sell['price'][sell_i]-buy['price'][buy_i])/buy['price'][buy_i]*100
            total_percent_profit += percent_profit
            percent_profit_list.append(percent_profit)
            
            ## winrate , win , loss
            if percent_profit > 2:
                win +=1
            else:
                loss +=1
            
            count += 1
            if buy_i==buy_length-1:
                break
            else:
                buy_i += 1
        sell_i +=1

    #RESULTS
    # holding
    min_days = str(min(holding_days_list))
    max_days = str(max(holding_days_list))
    avg_days = str(total_holding_days/count)

    # %profits
    win_p = []
    loss_p = []
    for p in percent_profit_list:
        if p >= 1.19:
            win_p.append(p)
        else:
            loss_p.append(p)
    avg_win = round(mean(win_p), 2)
    max_win = round(max(win_p), 2)
    min_win = round(min(win_p), 2)
    avg_loss = round(mean(loss_p), 2)
    max_loss = round(min(loss_p), 2)
    min_loss = round(max(loss_p), 2)
    total_profit = round(sum(percent_profit_list), 2)

    # trades , winrate , win ,loss
    num_trades = count
    winrate = win/count*100
    wins = win
    loss = loss

  

    print(total_profit)
    print(min_days)
    print(max_days)
    print(avg_days)
    print(min_win)
    print(max_win)
    print(avg_win)
    print(min_loss)
    print(max_loss)
    print(avg_loss)
    print(num_trades)
    print(winrate)
    print(wins)
    print(loss)

    #Stats
    trade_list.sort_values('date',ascending=False,inplace=True)
    # print(trade_list)
    ####################################################################################################
    ####################################################################################################
    #STATS
    stats = html.Div([
                html.H5('Holding:'),
                html.H6(f'min: {min_days}'),
                html.H6(f'max: {max_days}'),
                html.H6(f'avg: {avg_days}'),
                html.Hr(),
                html.H5('% Profit:'),
                html.H6(f'Total Profit: {total_profit}%'),
                html.H6(f'average win: {avg_win}%'),
                html.H6(f'max win: {max_win}%'),
                html.H6(f'min win: {min_win}%'),
                html.H6(f'average loss: {avg_loss}%'),
                html.H6(f'max loss: {max_loss}%'),
                html.H6(f'min loss: {min_loss}%'),
                html.Hr(),
                html.H6(f'Number of trades: {num_trades}'),
                html.H6(f'Winrate: {winrate}%'),
                html.H6(f'Winners: {wins}'),
                html.H6(f'Losers: {loss}'),
        ])

    return stats

def stock_info(data):
    last_data = data[-1:]
    last_data['Date'] = last_data.index 
    date = last_data.iloc[0]['Date']
    date = date.strftime("%m/%d/%Y")

    stock_code=str(last_data.iloc[0]['stock code'])
    open=float(last_data['open'])
    high=float(last_data['high'])
    low=float(last_data['low'])
    close=float(last_data['close'])
    volume=float(last_data['volume'])
    netforeign=float(last_data['netforeign'])
    contents = html.Div(children=[
        html.H3("Stock Info "),
        html.Div(children=[
            html.Div([                    
                html.H5(f"Stock Code: {stock_code}"), 
                html.H5(f"Open: {open}"),
                html.H5(f"High: {high}"),
                html.H5(f"Volume: {volume}"),
                ],className='six columns'),

            html.Div([
                html.H5(f"Date: {date}"),
                html.H5(f"Close: {close}"),
                html.H5(f"Low: {low}"),
                html.H5(f"Netforeign: {netforeign}"),
                ],className='six columns')

            ],className='row'),

        html.Hr(),
        ])
    return contents



#####################################################################
##RUNNING FUNCTIONS##
df = reading_csv()
df = sorting_and_grouping(df)
allgroups = getting_allgroups(df)
models = model_list()
#####################################################################

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(className='container-fluid',children=[
    html.H1(children='Stock App Predictior'),
    html.Hr(),
    html.Div(className='row',children=[
    #here row forcharts and info
        html.Div(id='',className='eight columns',children=[
            #column for chart and select stock
            #select stocks
            dcc.Dropdown(
                options=[
                    {'label':group,'value':group} for group in allgroups
                ],
                id='Stock_list',
                value='JFC',
                placeholder='Stock'
                ),
            #chart
            html.Hr(),
            dcc.Loading([
                html.Div(id='Chart'),
                html.Hr(),
                html.H4('Trades'),
                html.Div([
                        dash_table.DataTable(
                        id='Records',
                        fixed_rows={ 'headers': True, 'data': 0 },
                        style_cell={'width': '150px'},
                        ),
                    ])
                ],fullscreen=False,type='cube'),
            ]),

        html.Div(id='',className='four columns',children=[
            dcc.Dropdown(
                options=[
                    {'label':model,'value':model} for model in models
                ],
                id='test_model',
                value='',
                placeholder='model',
                clearable=True,
                ),
            html.Hr(),
            dcc.Loading([html.Div(id='Info')]),
            # dcc.Loading([html.Div(id='Trade_Stats')]),
            dcc.Loading([html.Div(id='Trade_Stats')]),
            
            
            ])
        ])
])
######################################################################
##CALLBACKS##
#stock chart
@app.callback(
    [Output(component_id='Chart', component_property='children'),
    Output(component_id='Info', component_property='children'),
    Output(component_id='Trade_Stats', component_property='children'),
    Output(component_id='Records', component_property='columns'),
    Output(component_id='Records', component_property='data'),],
    [Input(component_id='Stock_list', component_property='value'),
    Input(component_id='test_model', component_property='value')]
)

def charts(Stock,Model):
    if not Stock:
        content = ""
        Stock_info = ""
        stats = ""
        trade_record_col = []
        trade_record_data = []
    else:
        data = df.get_group(Stock)
        # data.sort_values('Date',inplace=True)
        data = data.interpolate()
        data.dropna(inplace=True)
        data_copy = df.get_group(Stock)
        data_copy = data.interpolate()
        data_copy.dropna(inplace=True)
        Stock_info = stock_info(data)
        stats = ""
        trade_record_col = []
        trade_record_data = []

        if not Model:
            figure = chart_display(data)
            content = dcc.Graph(figure=figure)
            # stats = ""
            # trade_record = ""
        else:
            model = load_model(f'model_saved/{Model}')
            figure = chart_display(data)
            content = dcc.Graph(figure=figure)
            trade_list = ""
            try:
                buy,sell,hold,trade_list = buys_and_sells(data,model,data_copy)
                figure = chart_display_test(data_copy,buy,sell)
                content = dcc.Graph(figure=figure)
                stats = Stats(buy,sell,trade_list)
                # trade_record = Trade_Record(trade_list)
                trade_record_col = [{"name": i, "id": i} for i in trade_list.columns]
                trade_record_data = trade_list.to_dict('records')
            except:
                pass    

    return content,Stock_info,stats,trade_record_col,trade_record_data




if __name__ == '__main__':
    app.run_server(debug=True)