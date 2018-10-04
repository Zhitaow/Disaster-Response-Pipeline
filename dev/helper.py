import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

# sanity check on the values in each column
def sanity_check(df):
    '''
    Usage: If there is any column containing values outside 0 or 1, impute with the mode in this column.
    '''
    flag = False
    for col in df.columns:
        value_set = set(df[col].unique())
        if (value_set - set([0, 1])):
            flag = True
            print("The column '{}' contains additional values: {}".format(col,value_set))
            idx = (~df[col].isin([0, 1]))
            print("There are {} out of {} ({}%) in the 'related' column" \
              .format(df[idx].shape[0], df.shape[0], 
                      df[idx].shape[0]/df.shape[0]*100))
            #print("Impute the abnormal value with the mode {} in this column.".format(df[col].mode()[0]))
            #df.loc[idx, col] = df[col].mode()[0]
    if flag is False:
        print("All data entries are either 0 or 1 in the dataset")
    return

def bar_chart(df0, df1, is_sort = True, fig_size = (18.5, 10.5)):
    '''
        Usage: plot stacked bar chart
        Input: 
            df0, df1 - 1-column dataframes with two categories
            fig_size: size of the figure
        Return:
            None
    '''
    header = ['value: 0', 'value: 1']
    
    if is_sort:
        idx = np.argsort(df0.values)
        data0 = df0.values[idx].tolist()
        data1 = df1.values[idx].tolist()
        xaxis = df0.index[idx].tolist()
    else:
        data0 = df0.values.tolist()
        data1 = df1.values.tolist()
        xaxis = df0.index.tolist()

    dataset= [data0, data1]

    matplotlib.rc('font', serif='Helvetica Neue')
    matplotlib.rc('text', usetex='false')
    matplotlib.rcParams.update({'font.size': 40})

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(fig_size[0], fig_size[1])

    configs = dataset[0]
    N = len(configs)
    ind = np.arange(N)
    width = 0.4

    p1 = plt.bar(ind, dataset[0], width, color='r')
    p2 = plt.bar(ind, dataset[1], width, bottom=dataset[0], color='b')

    plt.ylim([0,1.1])
    plt.yticks(fontsize=12)
    #plt.ylabel(output, fontsize=12)
    plt.xticks(ind, xaxis, fontsize=12, rotation=90)
    plt.xlabel('test', fontsize=12)
    plt.legend((p1[0], p2[0]), (header[0], header[1]), fontsize=12, ncol=4, framealpha=0, fancybox=True)
    plt.show()