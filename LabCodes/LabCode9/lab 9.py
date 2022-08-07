import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def read_and_filter(filename,filter_limit):
    df=pd.read_csv(filename)
    df = df[df["Y"] > float(filter_limit)]
    return df
def fix_deformation(dataframe):
    array = np.array(dataframe)
    for row in array:
        row[1]=float(row[1])*0.5+5
    rotation_radian = -45 * np.pi / 180
    for row in array:
        row[0] = float(row[0])
        row[0] = row[0] * np.cos(rotation_radian) + row[1] * np.sin(rotation_radian)
        row[1] = row[1] * np.cos(rotation_radian) - row[0] * np.sin(rotation_radian)
    for row in array:
        row[1] = (row[1]-1500)*0.01
    for row in array:
        row[0] = int(row[0])
    return array
def fit_and_predict(dataset,day):
    avg_x = np.average(dataset[0])
    avg_y = np.average(dataset[1])
    x=day
    y=dataset[day][1]
    min_beta_num=0
    min_beta_denum = 0
    for d in dataset:
        min_beta_num+=(x-avg_x)*(y-avg_y)
    for d in dataset:
        min_beta_denum+=(x-avg_x)**2
    beta_minimized=min_beta_num/min_beta_denum
    alpha_minimized=avg_y-beta_minimized*avg_x
    prediction=alpha_minimized+beta_minimized*x
    return (alpha_minimized,beta_minimized,prediction)
def plot(dataset, alpha, beta, prediction, day):
    array = np.array(dataset)
    plt.title("Exchange Rate")
    plt.xlabel("DAY")
    plt.ylabel("USD")
    plt.plot(array)
    plt.show()
def test():
    my_array=fix_deformation(read_and_filter("data.csv",9.0))
    alpha=fit_and_predict(my_array,-14)[0]
    beta=fit_and_predict(my_array,-14)[1]
    prediction=fit_and_predict(my_array,-14)[2]
    plot(my_array,alpha,beta,prediction,-14)
test()

