import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

# Define file path
d_path = 'C:/Users/jack/Coding Shit/Stock Project/Data/'

# Load and preprocess data
prices = pd.read_csv(d_path + 'Training Data/Organized Mass Prices.csv').set_index('Date')
prices = prices.replace(np.nan, 0).sample(frac=1, axis=1)#.reset_index(drop=True)
print(prices)

# Process each ticker
for x in range(len(prices.columns[6:])):
    ticker = prices.columns[2:][x]
    progress = str(x / len(prices.columns[2:] * 100))
    print(progress + ' percent finished')
    window = 275
    window_increment = 75

    # Remove days with no data
    tick_prices = prices[ticker][prices[ticker] != 0]
    
    # Ensure non-zero index
    if tick_prices.empty:
        continue

    startIdx = 0
    endIdx = 0
    ticker_total = 0
    while endIdx + window < len(tick_prices):
        ticker_total += 1
        month_underperform = False
        week_underperform = False

        # 5 max per ticker to prevent overflow of data from a single ticker to save storage space
        if ticker_total > 5:
            break
        else:
            endIdx = startIdx + window
            
            # Plotting the prices
            plt.plot(tick_prices.index[startIdx:endIdx], tick_prices.values[startIdx:endIdx])
            plt.xticks([])
            # plt.show()

            start_date = tick_prices.index[startIdx]
            end_date = tick_prices.index[endIdx]

            current_price = tick_prices[end_date]
            week_ahead_date = (datetime.datetime.strptime(end_date, '%Y-%m-%d') + datetime.timedelta(8)).strftime('%Y-%m-%d')
            month_ahead_date = (datetime.datetime.strptime(end_date, '%Y-%m-%d') + datetime.timedelta(31)).strftime('%Y-%m-%d')

            week_ahead_prices = tick_prices[end_date:week_ahead_date]
            month_ahead_prices = tick_prices[end_date:month_ahead_date]

            week_ahead_max = week_ahead_prices.max()
            month_ahead_max = month_ahead_prices.max()

            month_indicator = (month_ahead_max - current_price) / current_price * 100
            week_indicator = (week_ahead_max - current_price) / current_price * 100

            month_inc_standards = [50, 70, 90]
            for x in range(len(month_inc_standards)): 
                standard = month_inc_standards[x]
                if x == 0:
                    if month_indicator < standard:
                        month_underperform = True
                    else:
                        plt.savefig(d_path + 'Graph Data/' +  'Month' + str(standard) + '/' + str(ticker) + '-' + str(end_date) + '.png')
                elif month_indicator > standard:
                    plt.savefig(d_path + 'Graph Data/' +  'Month' + str(standard) + '/' + str(ticker) + '-' + str(end_date) + '.png')
                else: 
                    break

            week_inc_standards = [20, 30, 40, 50, 60]
            for x in range(len(week_inc_standards)): 
                standard = week_inc_standards[x]
                if x == 0:
                    if week_indicator < standard:
                        week_underperform = True
                    else:
                        plt.savefig(d_path + 'Graph Data/' +  'Week' + str(standard) + '/' + str(ticker) + '-' + str(end_date) + '.png')
                elif week_indicator > standard:
                    plt.savefig(d_path + 'Graph Data/' +  'Week' + str(standard) + '/' + str(ticker) + '-' + str(end_date) + '.png')
                else: 
                    break

            if month_underperform:
                plt.savefig(d_path + 'Graph Data/Month-Underperform/' + str(ticker) + '-' + str(end_date) + '.png')
            if week_underperform: 
                plt.savefig(d_path + 'Graph Data/Week-Underperform/' + str(ticker) + '-' + str(end_date) + '.png')

            startIdx += window_increment
            plt.close()



'''Alternate Metrics Graphing'''

# graph_types = ['Price', 'SMA', 'EMA', 'MACD', 'Bollinger Bands', 'RSI']
#
# for ticker in unique_ticks[478:]:
#     print(ticker)
#     sma_vals = pd.DataFrame()
#     ema_vals = pd.DataFrame()
#     short_ema_vals = pd.DataFrame()
#     short_sma_vals = pd.DataFrame()
#     macd_vals = pd.DataFrame()
#     bb_low = pd.DataFrame()
#     bb_high = pd.DataFrame()
#     rsi_vals = pd.DataFrame()
#     first = True
#     short_first = True
#     nan = False
#     window = 26
#     short_window = 12
#     smoothing = 2
#     w_count = 0
#     window_list = []
#     short_window_list = []
#     up_move = []
#     down_move = []
#
#     for date in allprices.index:
#         value = allprices.loc[date, ticker]
#         if str(value) != 'nan' and str(value) != 'Nan' and str(value) != 'NaN':
#             try:
#                 change = value - short_window_list[-1]
#                 if change > 0:
#                     up_move.append(change)
#                     down_move.append(0)
#                 elif change < 0:
#                     down_move.append(abs(change))
#                     up_move.append(0)
#                 else:
#                     down_move.append(0)
#                     up_move.append(0)
#
#             except Exception as e:
#                 pass
#
#             window_list.append(value)
#             short_window_list.append(value)
#
#             if w_count < short_window:
#                 short_ema_vals.loc[date, ticker] = np.nan
#                 short_sma_vals.loc[date, ticker] = np.nan
#                 rsi_vals.loc[date, ticker] = np.nan
#             else:
#                 short_sma_vals.loc[date, ticker] = round(sum(short_window_list) / len(short_window_list), 2)
#                 if short_first:
#                     prev_val = short_sma_vals[ticker].values[-1]
#                     if str(prev_val) != 'nan' and str(prev_val) != 'Nan':
#                         short_ema_vals.loc[date, ticker] = round((value * (smoothing/(short_window+1))) + (prev_val * (1-(smoothing/(short_window+1)))), 2)
#                     else:
#                         stop = False
#                         x = -2
#                         while not stop:
#                             x -= 1
#                             prev_val = short_sma_vals[ticker].values[x]
#                             print(prev_val)
#                             if str(prev_val) != 'nan' and str(prev_val) != 'Nan':
#                                 short_ema_vals.loc[date, ticker] = round((value * (smoothing / (short_window + 1))) + (prev_val * (1 - (smoothing / (short_window + 1)))), 2)
#                                 stop = True
#                             else:
#                                 continue
#                     short_first = False
#                 else:
#                     prev_val = short_ema_vals[ticker].values[-1]
#                     if str(prev_val) != 'nan' and str(prev_val) != 'Nan':
#                         short_ema_vals.loc[date, ticker] = round((value * (smoothing/(short_window+1))) + (prev_val * (1-(smoothing/(short_window+1)))), 2)
#                     else:
#                         stop = False
#                         x = -2
#                         try:
#                             while not stop:
#                                 x -= 1
#                                 prev_val = short_ema_vals[ticker].values[x]
#                                 if str(prev_val) != 'nan' and str(prev_val) != 'Nan':
#                                     short_ema_vals.loc[date, ticker] = round((value * (smoothing / (short_window + 1))) + (prev_val * (1 - (smoothing / (short_window + 1)))), 2)
#                                     stop = True
#                                 else:
#                                     continue
#                         except Exception as e:
#                             print(e, 'short ema')
#                             short_ema_vals.loc[date, ticker] = np.nan
#
#
#                 avgU = sum(up_move) / short_window
#                 avgD = sum(down_move) / short_window
#                 try:
#                     RS = avgU / avgD
#                     rsi_vals.loc[date, ticker] = round(100 - 100/(1 + RS), 2)
#                 except:
#                     rsi_vals.loc[date, ticker] = np.nan
#
#                 up_move.pop(0)
#                 down_move.pop(0)
#                 short_window_list.pop(0)
#
#             if w_count < window:
#                 w_count += 1
#                 sma_vals.loc[date, ticker] = np.nan
#                 ema_vals.loc[date, ticker] = np.nan
#                 bb_low.loc[date, ticker] = np.nan
#                 bb_high.loc[date, ticker] = np.nan
#                 macd_vals.loc[date, ticker] = np.nan
#             else:
#                 sma_vals.loc[date, ticker] = round(sum(window_list) / len(window_list), 2)
#                 if first:
#                     prev_val = sma_vals[ticker].values[-1]
#                     if str(prev_val) != 'nan':
#                         ema_vals.loc[date, ticker] = round((value * (smoothing/(window+1))) + (prev_val * (1-(smoothing/(window+1)))), 2)
#                     else:
#                         stop = False
#                         x = -2
#                         try:
#                             while not stop:
#                                 x -= 1
#                                 prev_val = sma_vals[ticker].values[x]
#                                 if str(prev_val) != 'nan' and str(prev_val) != 'Nan':
#                                     ema_vals.loc[date, ticker] = round((value * (smoothing / (window + 1))) + (prev_val * (1 - (smoothing / (window + 1)))), 2)
#                                     stop = True
#                                 else:
#                                     continue
#                         except Exception as e:
#                             print(e, 'ema')
#                             ema_vals.loc[date, ticker] = np.nan
#                     first = False
#
#                 else:
#                     prev_val = ema_vals[ticker].values[-1]
#                     if str(prev_val) != 'nan':
#                         ema_vals.loc[date, ticker] = round((value * (smoothing / (window + 1))) + (prev_val * (1 - (smoothing / (window + 1)))), 2)
#                     else:
#                         stop = False
#                         x = -2
#                         try:
#                             while not stop:
#                                 x -= 1
#                                 prev_val = ema_vals.loc[:, ticker].values[x]
#                                 if str(prev_val) != 'nan' and str(prev_val) != 'Nan':
#                                     ema_vals.loc[date, ticker] = round((value * (smoothing / (window + 1))) + (prev_val * (1 - (smoothing / (window + 1)))), 2)
#                                     stop = True
#                                 else:
#                                     continue
#                         except Exception as e:
#                             print(e, 'ema')
#                             ema_vals.loc[date, ticker] = np.nan
#
#                 macd_vals.loc[date, ticker] = round(ema_vals.loc[date, ticker] - short_ema_vals.loc[date, ticker], 2)
#                 bb_low.loc[date, ticker] = round(sma_vals.loc[date, ticker] + (np.std(window_list) * 2), 2)
#                 bb_high.loc[date, ticker] = round(sma_vals.loc[date, ticker] - (np.std(window_list) * 2), 2)
#
#                 window_list.pop(0)
#         else:
#             short_ema_vals.loc[date, ticker] = np.nan
#             short_sma_vals.loc[date, ticker] = np.nan
#             sma_vals.loc[date, ticker] = np.nan
#             ema_vals.loc[date, ticker] = np.nan
#             bb_low.loc[date, ticker] = np.nan
#             bb_high.loc[date, ticker] = np.nan
#             macd_vals.loc[date, ticker] = np.nan
#             rsi_vals.loc[date, ticker] = np.nan
#             pass
#
#     for gtype in graph_types:
#         try:
#             os.mkdir(d_path + 'Graphs/' + gtype + ' Graph Data/' + ticker)
#         except:
#             try:
#                 os.mkdir(d_path + 'Graphs/' + gtype + ' Graph Data')
#                 os.mkdir(d_path + 'Graphs/' + gtype + ' Graph Data/' + ticker)
#             except:
#                 pass
#
#     sma_vals.to_csv(d_path + 'Tables/SMA Table Data/SMA ' + ticker + '.csv')
#     ema_vals.to_csv(d_path + 'Tables/EMA Table Data/EMA ' + ticker + '.csv')
#     short_ema_vals.to_csv(d_path + 'Tables/Short EMA Table Data/Short EMA ' + ticker + '.csv')
#     short_sma_vals.to_csv(d_path + 'Tables/Short SMA Table Data/Short SMA ' + ticker + '.csv')
#     macd_vals.to_csv(d_path + 'Tables/MACD Table Data/MACD ' + ticker + '.csv')
#     bb_low.to_csv(d_path + 'Tables/BB Low Table Data/BB Low ' + ticker + '.csv')
#     bb_high.to_csv(d_path + 'Tables/BB High Table Data/BB High ' + ticker + '.csv')
#     rsi_vals.to_csv(d_path + 'Tables/RSI Table Data/RSI ' + ticker + '.csv')

    # for date in prices.index:
    #     date = str(date).replace('/', '-')
    #     price_vals = allprices.loc[:date, ticker]
    #     if price_vals.isna().sum() != len(price_vals.index):
    #         date_vals = list(allprices.loc[:date].index)
    #         time_frame = 1825
    #
    #         # find start date
    #         for idx in range(len(allprices.index)):
    #             recent = datetime.datetime.strptime(date_vals[-1], '%Y-%m-%d').date()
    #             old = datetime.datetime.strptime(date_vals[idx], '%Y-%m-%d').date()
    #             if (recent - old).days > time_frame:
    #                 pass
    #
    #             else:
    #                 start_date = str(old)
    #                 print(start_date)
    #                 price_vals = price_vals.loc[start_date:].values
    #                 date_vals = allprices.loc[start_date:date].index
    #                 break
    #
    #         date_vals = [datetime.datetime.strptime(d, '%Y-%m-%d').date() for d in date_vals]
    #
    #         plt.figure(figsize = (20, 5))
    #         plt.plot(date_vals, price_vals)
    #         plt.savefig(d_path + 'Graphs/Price Graph Data/' + ticker + '/' + str(date).replace('/', '-'))
    #         plt.clf()
    #
    #         sma = sma_vals.loc[start_date:date].values
    #         plt.plot(date_vals, sma)
    #         plt.savefig(d_path + 'Graphs/SMA Graph Data/' + ticker + '/' + str(date).replace('/', '-'))
    #         plt.clf()
    #
    #         ema = ema_vals.loc[start_date:date].values
    #         plt.plot(date_vals, ema)
    #         plt.savefig(d_path + 'Graphs/EMA Graph Data/' + ticker + '/' + str(date).replace('/', '-'))
    #         plt.clf()
    #
    #         macd = macd_vals.loc[start_date:date].values
    #         plt.plot(date_vals, macd)
    #         plt.savefig(d_path + 'Graphs/MACD Graph Data/' + ticker + '/' + str(date).replace('/', '-'))
    #         plt.clf()
    #
    #         bb_h = bb_high.loc[start_date:date].values
    #         bb_l = bb_low.loc[start_date:date].values
    #         plt.plot(date_vals, bb_h)
    #         plt.plot(date_vals, bb_l)
    #         plt.plot(date_vals, sma)
    #         plt.savefig(d_path + 'Graphs/Bollinger Bands Graph Data/' + ticker + '/' + str(date).replace('/', '-'))
    #         plt.clf()
    #         plt.close()
    #
    #         plt.figure(figsize = (30, 5))
    #         rsi = rsi_vals.loc[start_date:date].values
    #         plt.plot(date_vals, rsi)
    #         plt.savefig(d_path + 'Graphs/RSI Graph Data/' + ticker + '/' + str(date).replace('/', '-'))
    #         plt.clf()
    #         plt.close()
    #
    #     else:
    #         pass

'''
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

d_path = 'C:/Users/jack/Coding Shit/Stock Project/Data/'
prices = pd.read_csv(d_path + 'Training Data/Organized Mass Prices.csv').set_index('Date')
prices = prices.replace(np.nan, 0)

for ticker in prices.columns[2:]:
    window = 275  # number of days to be processed
    window_increment = 75

    # removing days with no data
    print(prices[ticker].index)
    tick_prices = pd.Series(prices[ticker], index=prices[ticker].index)
    nonz_indices = tick_prices.nonzero()
    shaved_prices = pd.Series(tick_prices.iloc[nonz_indices], index=tick_prices.iloc[nonz_indices].index)
    print(shaved_prices)

    startIdx = 0
    for idx in prices[ticker].index:
        if prices.loc[idx, ticker] != 0:
            startIdx = idx
            break
    endIdx = startIdx + window

    # print(shaved_prices)
    # print(shaved_prices.index, shaved_prices.values)

    for day in shaved_prices.index:
        plot = plt.plot(shaved_prices.index[startIdx:endIdx], shaved_prices.values[startIdx:endIdx])
        plt.show()
        time.sleep(5)
        plt.close()

        start_date = shaved_prices.index[startIdx]
        end_date = shaved_prices.index[endIdx]

        current_price = shaved_prices[end_date]
        week_ahead_date = datetime.datetime.strptime(end_date, '%Y-%m-%d') + datetime.timedelta(8)
        month_ahead_date = datetime.datetime.strptime(end_date, '%Y-%m-%d') + datetime.timedelta(31)

        week_ahead_prices = shaved_prices[end_date:week_ahead_date]
        month_ahead_prices = shaved_prices[end_date:month_ahead_date]

        week_ahead_max = np.max(np.array(week_ahead_prices))
        month_ahead_max = np.max(np.array(month_ahead_prices))

        month_indicator = (month_ahead_max - current_price) / current_price * 100
        week_indicator = (week_ahead_max - current_price) / current_price * 100

        if month_indicator < 50:
            plt.savefig(d_path + str(end_date) + ' ' + str(ticker) + '.png')
        elif month_indicator < 70:
            plt.savefig(d_path + str(end_date) + ' ' + str(ticker) + '.png')
        elif month_indicator < 90:
            plt.savefig(d_path + str(end_date) + ' ' + str(ticker) + '.png')
        else:
            month_underperform = True

        if week_indicator < 20:
            plt.savefig(d_path + str(end_date) + ' ' + str(ticker) + '.png')
        elif week_indicator < 30:
            plt.savefig(d_path + str(end_date) + ' ' + str(ticker) + '.png')
        elif week_indicator < 40:
            plt.savefig(d_path + str(end_date) + ' ' + str(ticker) + '.png')
        elif week_indicator < 50:
            plt.savefig(d_path + str(end_date) + ' ' + str(ticker) + '.png')
        elif week_indicator < 60:
            plt.savefig(d_path + str(end_date) + ' ' + str(ticker) + '.png')
        else:
            week_underperform = True

        if month_underperform and week_underperform:
            plt.savefig(d_path + 'Graph Data/Underperform/' + str(end_date) + ' ' + str(ticker) + '.png')

        startIdx += window_increment
        endIdx += window_increment'''

''''''