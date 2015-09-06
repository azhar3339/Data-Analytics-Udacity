import pandas as pd
from ggplot import *

df = pd.read_csv('/home/azhar/Dropbox/Udacity/Data-Analytics-Udacity/Project2/Data/turnstile_data_master_with_weather.csv')
df1 = df[['ENTRIESn_hourly','rain']]
# print df.shape
dataText=pd.DataFrame.from_items([('x',[3000,3000]),('y',[5000,4500]),('text',['Black: Non rainy days','Blue:Rainy days'])])
#
# plot =  ggplot(aes(x = 'ENTRIESn_hourly'),data = df) + \
#             geom_histogram()
#
# print plot
#
# #########First plot
plot =  ggplot(df1,aes('ENTRIESn_hourly')) + \
        geom_histogram(data=df[df['rain'] == 0], fill = "black", alpha = 0.5,binwidth = 50) + \
        geom_histogram(data=df[df['rain'] == 1], fill = "blue", alpha = 0.5,binwidth = 50) +\
        scale_x_discrete(limits = [0, 5000]) +\
        scale_y_discrete(limits = [0, 6000]) +\
        theme_bw() +\
        xlab("Entries per Hour") +\
        ylab("Frequency") +\
        ggtitle("Distribution of entries per hour during raniny and normal days") +\
        geom_text(aes(x='x', y='y', label='text'), data=dataText)

print plot

#############Second plot
# unique_dates = df['DATEn'].unique()
#
# df1 = df[['ENTRIESn_hourly','DATEn']]
# df1_groupMean = df1.groupby(['DATEn'],as_index = False).agg(['mean'])
#
# df2 = pd.DataFrame({'Dates': unique_dates,
#                     'Mean': list(df1_groupMean.ix[:,0])})
#
# df2['Dates'] = pd.to_datetime(pd.Series(df2['Dates']))
#
# df2['weekdays'] = df2['Dates'].dt.dayofweek
#
# df2['weekdays'] = df2['weekdays'].replace(5,"Weekend")
# df2['weekdays'] = df2['weekdays'].replace(6,"Weekend")
# df2['weekdays'] = df2['weekdays'].replace(0,"Weekday")
# df2['weekdays'] = df2['weekdays'].replace(1,"Weekday")
# df2['weekdays'] = df2['weekdays'].replace(2,"Weekday")
# df2['weekdays'] = df2['weekdays'].replace(3,"Weekday")
# df2['weekdays'] = df2['weekdays'].replace(4,"Weekday")
#
#
# plot2 = ggplot(df2,aes(y = 'Mean', color = 'weekdays')) +\
#         geom_point(aes(x='Dates')) +\
#         xlab("Date") +\
#         ylab("Mean entries per hour") +\
#         ggtitle("Weekday vs Weekend Ridership") +\
#         theme_bw()
#
# print plot2
#
#
# #############Third plot
# # unique_dates = df['Hour'].unique()
# #
# # df1 = df[['ENTRIESn_hourly','Hour']]
# # df1_groupMean = df1.groupby(['Hour'],as_index = False).agg(['mean'])
# #
# # df2 = pd.DataFrame({'Hour': unique_dates,
# #                     'Mean': list(df1_groupMean.ix[:,0])})
# #
# #
# # # print df2
# #
# # plot3 = ggplot(df2,aes(y = 'Mean', x='Hour')) +\
# #         geom_point() +\
# #         geom_line() +\
# #         scale_x_discrete(limits = [0, 23]) +\
# #         xlab("Time of the Day") +\
# #         ylab("Mean entries per hour") +\
# #         ggtitle("Ridership throughout the day")
# #
# # print plot3
#
#
# def get_plots (turnstile_data):
#
#     df =  turnstile_data
#     df1 = df[['ENTRIESn_hourly','rain']]
#     dataText=pd.DataFrame.from_items([('x',[3000,3000]),('y',[5000,4500]),('text',['Black: Non rainy days','Blue:Rainy days'])])
#
#     # #########First plot
#     plot =  ggplot(df1,aes('ENTRIESn_hourly')) + \
#             geom_histogram(data=df[df['rain'] == 0], fill = "black", alpha = 0.5,binwidth = 50) + \
#             geom_histogram(data=df[df['rain'] == 1], fill = "blue", alpha = 0.5,binwidth = 50) +\
#             scale_x_discrete(limits = [0, 5000]) +\
#             scale_y_discrete(limits = [0, 6000]) +\
#             theme_bw() +\
#             xlab("Entries per Hour") +\
#             ylab("Frequency") +\
#             ggtitle("Distribution of entries per hour during raniny and normal days") +\
#             geom_text(aes(x='x', y='y', label='text'), data=dataText)
#
#     print plot
#
#     #############Second plot
#     unique_dates = df['DATEn'].unique()
#
#     df1 = df[['ENTRIESn_hourly','DATEn']]
#     df1_groupMean = df1.groupby(['DATEn'],as_index = False).agg(['mean'])
#
#     df2 = pd.DataFrame({'Dates': unique_dates,
#                         'Mean': list(df1_groupMean.ix[:,0])})
#
#     df2['Dates'] = pd.to_datetime(pd.Series(df2['Dates']))
#
#     df2['weekdays'] = df2['Dates'].dt.dayofweek
#
#     df2['weekdays'] = df2['weekdays'].replace(5,"Weekend")
#     df2['weekdays'] = df2['weekdays'].replace(6,"Weekend")
#     df2['weekdays'] = df2['weekdays'].replace(0,"Weekday")
#     df2['weekdays'] = df2['weekdays'].replace(1,"Weekday")
#     df2['weekdays'] = df2['weekdays'].replace(2,"Weekday")
#     df2['weekdays'] = df2['weekdays'].replace(3,"Weekday")
#     df2['weekdays'] = df2['weekdays'].replace(4,"Weekday")
#
#
#     plot2 = ggplot(df2,aes(y = 'Mean', color = 'weekdays')) +\
#             geom_point(aes(x='Dates')) +\
#             xlab("Date") +\
#             ylab("Mean entries per hour") +\
#             ggtitle("Weekday vs Weekend Ridership") +\
#             theme_bw()
#
#     print plot2

