# -*- codeing = utf-8 -*-
# @Time : 2024/6/5 0:04
# @Author : Zhang Jingwei
# @Name : C_means.py
# @Software : PyCharm
import math
import matplotlib.pyplot as plt
import numpy as np

def Distance(a,b): #欧氏距离
    return (a[0]-b[0])**2 + (a[1]-b[1])**2


def ReadTrains():                   #读取训练集
    trains = []
    try:
        f = open("trains.txt","r")
        print("-----读取训练集成功-----")
        try:
            for line in f.readlines():
                train = line.split()
                temp = [float(train[0]), float(train[1]),0] #保持数据到嵌套数组，身高、体重、类别
                trains.append(temp)
            return trains
        finally:
            f.close()
            print("-----文件关闭-----")
    except Exception as ex:
        print("-----出现异常",ex,"-----")

def C_means(trains,c): #C均值聚类 输入数据和聚类个数
    all_center=[0,0]
    for i in range(len(trains)):
        all_center[0] += trains[i][0]  # 计算总体的类心 先算总的x,y
        all_center[1] += trains[i][1]
    all_center[0] /= len(trains)
    all_center[1] /= len(trains)
    centers = trains[0:c]                       #[切片]，选择C个点作为初始类心，这里将前c个作为初始类心
    # 生成 c 个 0 到 99 之间的随机整数
    nums = np.random.randint(0, 99, c)
    for i in range(c):  # 遍历样本点
        centers[i] = trains[nums[i]]
    new_centers =[]                             #记录新的类心
    numbers = []                                 #桶，记录一个类有多少个模式
    counts = 0                                  #记录未变的类的数量
    while counts < c:
        numbers = [0 for i in range(c)]
        new_centers = [[0,0] for i in range(c)]  #深复制，不会指向同一个目标
        for i in range(len(trains)):            #遍历样本点
            mindistance = 1e7                   #最小距离
            minindex = 0                        #记录离哪个类心距离近
            for j in range(len(centers)):       #遍历当前点，找到距离最小的类心
                if Distance(trains[i],centers[j]) < mindistance:
                    mindistance = Distance(trains[i],centers[j])
                    minindex = j
            trains[i][2] = minindex         #归属minindex类
            new_centers[minindex][0] += trains[i][0]         #计算新的类心 先算总的x,y
            new_centers[minindex][1] += trains[i][1]
            numbers[minindex] += 1              #该类中 模式的数量+1

        for i,center in enumerate(centers):     #遍历类心，比较新类心和旧类心是否发生变化
            new_centers[i][0] /= float(numbers[i])
            new_centers[i][1] /= float(numbers[i])
            if ((new_centers[i][0]- center[0] < 1e-6) and (new_centers[i][1] - center[1] < 1e-6)):
                counts += 1                     #未变的类数量+1
            centers[i] = new_centers[i]         #更新类心
    print("-----处理完毕，展示结果-----")
    colors = ["red","blue","green","coral","tan","yellow","brown","gold","orange","peru"]
    marks = ["+","x","o","v","^","<",">","1","2","3"]
    WCSS=0
    BCSS=0
    sum = np.zeros(c)
    sum2 = np.zeros(c)
    for i,center in enumerate(centers):
        print("当前第%d类，类心为：(%d,%d) 共有%d个模式，它们分别是："%(i + 1,center[0],center[1],numbers[i]))
        sum2[i]=Distance(all_center,center)*numbers[i]
        for j,train in enumerate(trains):
            if train[2] == i:
                sum[i]+=Distance(train, center)
                print("\t%d:(%d,%d)"%(j+1,train[0],train[1]))
                plt.scatter(train[0],train[1],marker = marks[i],c = colors[i])
        WCSS+=sum[i]
        BCSS+=sum2[i]
    chi[c]=(BCSS/(c-1))/(WCSS/(len(trains)-c))

    plt.title('C-means clustering')
    plt.xlabel('height/cm')
    plt.ylabel('weight/kg')
    plt.show()



def E(G1,G2): #计算分裂聚类中的指标E
    N1 = len(G1)
    N2 = len(G2)
    G1_center = [0,0]
    G2_center = [0,0]
    for i in range(N1):
        G1_center[0] += G1[i][0]
        G1_center[1] += G1[i][1]
    G1_center[0] /= N1
    G1_center[1] /= N1
    for i in range(N2):
        G2_center[0] += G2[i][0]
        G2_center[1] += G2[i][1]
    G2_center[0] /= N2
    G2_center[1] /= N2
    E=N1*N2/(N1+N2)*(pow((G1_center[0]-G2_center[0]),2)+pow((G1_center[1]-G2_center[1]),2))
    return E

def split_clustering(trains): #分裂聚类 只针对两类问题 输入数据
    G1=trains #将全部数据归为一类
    G1_copy=trains
    G2=[] #第二类开始为空集
    e=[]
    for i in range(len(trains)):  #第一次
        G2.append(G1_copy[i]) #将G1中每个元素依次加一个到G2
        G1.pop(i)             #将G1中的该元素删去
        e.append(E(G1,G2))    #计算此时E值，存入e数组
        G1.insert(i,G2[0])    #恢复原来G1、G2
        G2.pop()
    Emax=max(e)               #取e数组的最大值
    split_index = e.index(max(e))  #e最大值对应的下标为第一次要分裂到G2的元素
    G2.append(G1_copy[split_index])
    G1.pop(split_index)
    G1_copy=G1

    cnt=1                                       #记录循环轮数
    Emax_last=Emax
    while(Emax>=Emax_last):                     #循环终止条件：本轮循环e的最大值比上次循环e的最大值小
        e=[]                                    #清空e数组
        for i in range(len(G1)):
            G2.append(G1_copy[i])               #将G1中每个元素依次加一个到G2
            G1.pop(i)                           #将G1中的该元素删去
            e.append(E(G1, G2))                 #计算此时E值，存入e数组
            G1.insert(i, G2[cnt])               #恢复原来G1、G2
            G2.pop()
        Emax = max(e)                           #取e数组的最大值
        if(Emax>Emax_last):                     #循环终止条件：本轮循环e的最大值比上次循环e的最大值小
            Emax_last=Emax                      #更新上一轮最大值
            split_index = e.index(Emax)         #e最大值对应的下标为第一次要分裂到G2的元素
            G2.append(G1_copy[split_index])
            G1.pop(split_index)
            G1_copy=G1
            cnt+=1                              #轮数+1

    print("-----处理完毕，展示结果-----")
    colors = ["red","blue"]
    marks = ["+","x"]
    print("当前第1类，共有%d个模式，它们分别是：" % (len(G1)))
    for i,value in enumerate(G1):
        print("\t%d:(%d,%d)"%(i + 1,value[0],value[1]))
        plt.scatter(value[0],value[1],marker = marks[0],c = colors[0],label="Category A")
    print("当前第2类，共有%d个模式，它们分别是：" % (len(G2)))
    for i,value in enumerate(G2):
        print("\t%d:(%d,%d)"%(i + 1,value[0],value[1]))
        plt.scatter(value[0],value[1],marker = marks[1],c = colors[1],label="Category B")
    plt.title('split clustering')
    plt.xlabel('height/cm')
    plt.ylabel('weight/kg')
    plt.show()
    return 0


chi=np.zeros(10)
print("-----准备读取训练集-----")
trains = ReadTrains()
print(len(trains))
c = int(input("请输入需要分成多少类"))
print("-----C均值聚类开始-----")
C_means(trains,c)
'''
# 聚类指标CHI和聚类数的变化关系
for i in range(2,10):
    C_means(trains,i)
plt.scatter(range(2,10), chi[range(2,10)])
plt.title('clustering evaluation')
plt.xlabel('numbers of categories')
plt.ylabel('CHI')
plt.show()
'''
print("-----C均值聚类结束-----")

print("-----分裂均值聚类开始-----")
split_clustering(trains)
print("-----分裂均值聚类结束-----")






