#!/usr/bin/env python
# coding: utf-8

# In[133]:


import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier,plot_tree 
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns    


# In[134]:


# 데이터 불러오기 
data_folder = "dataset/"
fileExt = r".txt"
not_header = ['description.txt','documentation.txt','profile.txt']
filenames= [file for file in os.listdir(data_folder) if file.endswith(fileExt) and file not in not_header]
program_1 = pd.read_csv(data_folder+filenames[0],sep=",", index_col=0)

header = [x[:-4] for x in filenames]
dataFrame = pd.DataFrame(columns=header)
dataFrame

# 데이터 불러오기

for i in range(len(filenames)):
    
    with open(data_folder + filenames[i]) as f:
        lines_after_header = f.readlines()
        data_list = []

        for line in range(len(lines_after_header)):
            data_list.append(float(lines_after_header[line].split()[0]))
    f.close()
    dataFrame[header[i]]=data_list
    
# profile 불러오기
with open(data_folder + 'profile.txt') as f:
    lines_after_header = f.readlines()
    data_list = []

    for line in range(len(lines_after_header)):
        data_list.append(lines_after_header[line].split())
f.close()
profile = pd.DataFrame(columns=['Cooler condition','Valve condition','Internal pump leakage','Hydraulic accumulator',
                                 'stable flag'],data= data_list)
profile=profile.astype('int')

dataFrame = pd.concat([dataFrame,profile],axis=1)


# In[135]:


dataFrame.head()


# 2. 각 label들의 histogram

# In[136]:




dataFrame.iloc[:,-1].plot.hist()
plt.xlabel(dataFrame.columns[-1])
plt.show()


dataFrame.columns[-1]

dataFrame.iloc[:,-2].plot.hist()
plt.xlabel(dataFrame.columns[-2])
plt.show()

dataFrame.iloc[:,-3].plot.hist()
plt.xlabel(dataFrame.columns[-3])
plt.show()

dataFrame.iloc[:,-4].plot.hist(bins=20)
plt.xlabel(dataFrame.columns[-4])
plt.show()

dataFrame.iloc[:,-5].plot.hist(bins=20)
plt.xlabel(dataFrame.columns[-5])
plt.show()


# In[137]:


corr = dataFrame.corr()

# 컬럼 별로 관계가 0.3이상인 것들
co=0.3
for i in corr.columns[-5:]:
    print(i)
    print(corr[(corr[i]>co) | (corr[i]<-co)].index)
    print()


# In[138]:


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.scatter (dataFrame.index,dataFrame['SE'],s=0.2 ,label = 'SE')
ax1.legend(loc=(1.0,1.0))
ax1.set_xlabel('Index')
ax1.set_ylabel('SE')


# In[139]:


co=0.3
dataFrame2= dataFrame[dataFrame['SE']>20]

corr2=dataFrame2.corr()
print(corr2[(corr2['Cooler condition']>co) | (corr2['Cooler condition']<-co)].index)
for i in corr2[(corr2['Cooler condition']>co) | (corr2['Cooler condition']<-co)].index[:-1]:
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.scatter (dataFrame2.index,dataFrame2[i],s=0.2 ,label = i)
    ax2.plot(dataFrame['Cooler condition'],label = 'Cooler',color='orange')
    ax1.legend(loc=(1.0,1.0))
    ax1.set_xlabel('Index')
    ax1.set_ylabel(i)
    ax2.set_ylabel('Cooler')
    
    ax2.legend(loc=(0.8,1.0))


# ## 각 센서들의 상관관계

# In[140]:


dataFrame2.iloc[:,:-5].corr()


# In[141]:


plt.figure(figsize=(15,15))
sns.heatmap(data = dataFrame2.iloc[:,:-5].corr(), annot=True, 
fmt = '.2f', linewidths=.5, cmap='Blues')


# In[142]:


print(corr2[(corr2['Cooler condition']>co) | (corr2['Cooler condition']<-co)]['Cooler condition'])


# In[143]:


X_train, X_test, y_train, y_test = train_test_split(dataFrame2.iloc[:,:-5], dataFrame2['Cooler condition'], test_size=0.3, random_state=42)
dtc  = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Cooler condition  정확도:", accuracy)
print("")


# In[144]:


# 중요도 값 확인
importances = dtc.feature_importances_

# 중요도 값을 출력
for feature, importance in zip(X_train.columns, importances):
    print(f"Feature: {feature}, Importance: {importance}")

# 중요도 값을 그래프로 시각화
plt.bar(X_train.columns, importances)
plt.xlabel('Sensor') 
plt.ylabel('Importance')
plt.title('Feature Importance_Cooler condition')
plt.xticks(rotation=90)
plt.show()



# In[145]:


co=0.3
dataFrame2= dataFrame[dataFrame['SE']>20]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.scatter (dataFrame2.index,dataFrame2['CE'],s=0.2 ,label = "CE")
ax2.scatter (dataFrame2.index,dataFrame2['CP'],s=0.2 ,label = 'CP',color='orange')
ax1.legend(loc=(1.0,1.0))
ax1.set_xlabel('Index')
ax1.set_ylabel('CE')
ax2.set_ylabel('CP')

ax2.legend(loc=(0.8,1.0))


# In[146]:


x= pd.DataFrame(dataFrame2['CE'],columns=['CE'])
X_train, X_test, y_train, y_test = train_test_split(x, dataFrame2['Cooler condition'], test_size=0.3, random_state=42)
dtc  = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Cooler condition, CE  정확도:", accuracy)
print("")


# In[147]:


x= pd.DataFrame(dataFrame2['CP'],columns=['CP'])
X_train, X_test, y_train, y_test = train_test_split(x, dataFrame2['Cooler condition'], test_size=0.3, random_state=42)
dtc  = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Cooler condition, CP  정확도:", accuracy)
print("")


# 데이터 0~1 scaling

# In[148]:


# Min-Max 스케일링을 위한 scaler 객체 생성
scaler = MinMaxScaler()

# 데이터 프레임의 값에 Min-Max 스케일링 적용
scaled_df = pd.DataFrame(scaler.fit_transform(dataFrame2.iloc[:,:-5]), columns=dataFrame.columns[:-5])
scaled_df.head()


# In[ ]:





# In[149]:


col = 'CE'
input_col= [x for x in scaled_df.columns if x not in ['CE','CP']]
# print(input_col)
X_train, X_test, y_train, y_test = train_test_split(scaled_df[input_col], scaled_df[col], test_size=0.3, random_state=42)
lr  = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print("CE  정확도:", accuracy)
print("")


# In[150]:


# 가중치 확인
# 변수의 가중치 확인
coefficients = lr.coef_
intercept = lr.intercept_

# 각 변수의 가중치 출력
for feature, coef in zip(X_train.columns, coefficients):
    print(f"{feature}: {coef}")

# 절편(intercept) 출력
print("Intercept:", intercept)

# 가중치 값을 그래프로 시각화
plt.bar(X_train.columns, coefficients)
plt.xlabel('Sensor') 
plt.ylabel('Weight')
plt.title('Feature Weight_CE')
plt.xticks(rotation=90)
plt.show()


# In[151]:


col = 'CP'
input_col= [x for x in scaled_df.columns if x not in ['CE','CP']]
# print(input_col)
X_train, X_test, y_train, y_test = train_test_split(scaled_df[input_col], scaled_df[col], test_size=0.3, random_state=42)
lr  = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print("CP  정확도:", accuracy)
print("")


# In[152]:


# 가중치 확인
# 변수의 가중치 확인
coefficients = lr.coef_
intercept = lr.intercept_

# 각 변수의 가중치 출력
for feature, coef in zip(X_train.columns, coefficients):
    print(f"{feature}: {coef}")

# 절편(intercept) 출력
print("Intercept:", intercept)

# 가중치 값을 그래프로 시각화
plt.bar(X_train.columns, coefficients)
plt.xlabel('Sensor') 
plt.ylabel('Weight')
plt.title('Feature Weight_CP')
plt.xticks(rotation=90)
plt.show()


# CP 값을 구하기 위한 가중치 중 큰 값만을 선정하여 예측이 가능한지 확인

# In[153]:


li=[]

for feature, coef in zip(X_train.columns, coefficients):
    if (coef>0.05) | (coef <-0.05):
        li.append(feature)
print(li)
col = 'CP'
# print(input_col)
X_train, X_test, y_train, y_test = train_test_split(scaled_df[li], scaled_df[col], test_size=0.3, random_state=42)
lr  = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print("CP  정확도:", accuracy)
print("")


# In[154]:


# 가중치 확인
# 변수의 가중치 확인
coefficients = lr.coef_
intercept = lr.intercept_

# 각 변수의 가중치 출력
for feature, coef in zip(X_train.columns, coefficients):
    print(f"{feature}: {coef}")

# 절편(intercept) 출력
print("Intercept:", intercept)

# 가중치 값을 그래프로 시각화
plt.bar(X_train.columns, coefficients)
plt.xlabel('Sensor') 
plt.ylabel('Weight')
plt.title('Feature Weight_CP')
plt.xticks(rotation=90)
plt.show()


# In[155]:


li=['TS3','TS4']
col = 'CP'
# print(input_col)
X_train, X_test, y_train, y_test = train_test_split(scaled_df[li], scaled_df[col], test_size=0.3, random_state=42)
lr  = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print("CP  정확도:", accuracy)
print("")


# In[156]:


# 가중치 확인
# 변수의 가중치 확인
coefficients = lr.coef_
intercept = lr.intercept_

# 각 변수의 가중치 출력
for feature, coef in zip(X_train.columns, coefficients):
    print(f"{feature}: {coef}")

# 절편(intercept) 출력
print("Intercept:", intercept)

# 가중치 값을 그래프로 시각화
plt.bar(X_train.columns, coefficients)
plt.xlabel('Sensor') 
plt.ylabel('Weight')
plt.title('Feature Weight_CP')
plt.xticks(rotation=90)
plt.show()


# In[157]:


dataFrame2= dataFrame[dataFrame['SE']>20]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.scatter (dataFrame2.index,dataFrame2['TS3'],s=0.2 ,label = "CE")
ax2.scatter (dataFrame2.index,dataFrame2['TS4'],s=0.2 ,label = 'CP',color='orange')
ax1.legend(loc=(1.0,1.0))
ax1.set_xlabel('Index')
ax1.set_ylabel('TS3')
ax2.set_ylabel('TS4')

ax2.legend(loc=(0.8,1.0))


# In[158]:


fig, ax1 = plt.subplots()  # 왼쪽 축

ax1.scatter(dataFrame.index,dataFrame['TS4'], label='TS4',s=0.5,color='black')
ax1.scatter(dataFrame.index,dataFrame['TS3'], label='TS3',s=0.5,color='orange')

ax1.set_ylabel('TS3, TS4')  # 왼쪽 축 레이블 설정

fig.tight_layout()  # 그래프 간격 조정
fig.legend()  # 범례 위치 설정
plt.show()


# TS4나 TS3중 하나만 사용하여 CP를 예측한 경우

# In[159]:


x= pd.DataFrame(scaled_df['TS3'],columns=['TS3'])
X_train, X_test, y_train, y_test = train_test_split(x, scaled_df['CE'], test_size=0.3, random_state=42)
lr  = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print("TS3 - CP  정확도:", accuracy)
print("")


# In[160]:


x= pd.DataFrame(scaled_df['TS4'],columns=['TS4'])
X_train, X_test, y_train, y_test = train_test_split(x, scaled_df['CE'], test_size=0.3, random_state=42)
lr  = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print("TS4 - CP  정확도:", accuracy)
print("")


# In[161]:


a='TS1'
x= pd.DataFrame(scaled_df[a],columns=[a])
X_train, X_test, y_train, y_test = train_test_split(x, scaled_df['CE'], test_size=0.3, random_state=42)
lr  = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print(a,"- CP  정확도:", accuracy)
print("")


# In[162]:


a='TS2'
x= pd.DataFrame(scaled_df[a],columns=[a])
X_train, X_test, y_train, y_test = train_test_split(x, scaled_df['CE'], test_size=0.3, random_state=42)
lr  = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print(a,"- CP  정확도:", accuracy)
print("")


# In[163]:


col = 'TS3'
input_col= [x for x in scaled_df.columns if x not in ['CE','CP','TS3','TS4']]
# print(input_col)
X_train, X_test, y_train, y_test = train_test_split(scaled_df[input_col], scaled_df[col], test_size=0.3, random_state=42)
lr  = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print("TS3  정확도:", accuracy)
print("")


# In[164]:


# 가중치 확인
# 변수의 가중치 확인
coefficients = lr.coef_
intercept = lr.intercept_

# 각 변수의 가중치 출력
for feature, coef in zip(X_train.columns, coefficients):
    print(f"{feature}: {coef}")

# 절편(intercept) 출력
print("Intercept:", intercept)

# 가중치 값을 그래프로 시각화
plt.bar(X_train.columns, coefficients)
plt.xlabel('Sensor') 
plt.ylabel('Weight')
plt.title('Feature Weight_CP')
plt.xticks(rotation=90)
plt.show()


# In[165]:


a='TS3'
b='TS4'
dataFrame2= dataFrame[dataFrame['SE']>20]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.scatter (dataFrame2.index,dataFrame2[a],s=0.2 ,label = a)
ax2.scatter (dataFrame2.index,dataFrame2[b],s=0.2 ,label = b,color='orange')
ax1.legend(loc=(1.0,1.0))
ax1.set_xlabel('Index')
ax1.set_ylabel(a)
ax2.set_ylabel(b)

ax2.legend(loc=(0.8,1.0))


# In[166]:


a='TS1'
b='TS2'
dataFrame2= dataFrame[dataFrame['SE']>20]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.scatter (dataFrame2.index,dataFrame2[a],s=0.2 ,label = a)
ax2.scatter (dataFrame2.index,dataFrame2[b],s=0.2 ,label = b,color='orange')
ax1.legend(loc=(1.0,1.0))
ax1.set_xlabel('Index')
ax1.set_ylabel(a)
ax2.set_ylabel(b)

ax2.legend(loc=(0.8,1.0))


# In[167]:


a='TS1'
b='TS3'
dataFrame2= dataFrame[dataFrame['SE']>20]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.scatter (dataFrame2.index,dataFrame2[a],s=0.2 ,label = a)
ax2.scatter (dataFrame2.index,dataFrame2[b],s=0.2 ,label = b,color='orange')
ax1.legend(loc=(1.0,1.0))
ax1.set_xlabel('Index')
ax1.set_ylabel(a)
ax2.set_ylabel(b)

ax2.legend(loc=(0.8,1.0))


# In[168]:


a='TS2'
b='TS4'
dataFrame2= dataFrame[dataFrame['SE']>20]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.scatter (dataFrame2.index,dataFrame2[a],s=0.2 ,label = a)
ax2.scatter (dataFrame2.index,dataFrame2[b],s=0.2 ,label = b,color='orange')
ax1.legend(loc=(1.0,1.0))
ax1.set_xlabel('Index')
ax1.set_ylabel(a)
ax2.set_ylabel(b)

ax2.legend(loc=(0.8,1.0))


# TS4는 다른 어떤 센서랑 연관되어 있는 지 확인 (TS1~4, CE, CP 제외)

# In[169]:


col = 'TS4'
input_col= [x for x in scaled_df.columns if x not in ['TS1','TS2','TS3','TS4','CE','CP']]
# print(input_col)
X_train, X_test, y_train, y_test = train_test_split(scaled_df[input_col], scaled_df[col], test_size=0.3, random_state=42)
lr  = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print(col,"  정확도:", accuracy)
print("")


# In[170]:


# 가중치 확인
# 변수의 가중치 확인
coefficients = lr.coef_
intercept = lr.intercept_

# 각 변수의 가중치 출력
for feature, coef in zip(X_train.columns, coefficients):
    print(f"{feature}: {coef}")

# 절편(intercept) 출력
print("Intercept:", intercept)

# 가중치 값을 그래프로 시각화
plt.bar(X_train.columns, coefficients)
plt.xlabel('Sensor') 
plt.ylabel('Weight')
plt.title('Feature Weight_TS4')
plt.xticks(rotation=90)
plt.show()


# In[171]:


li=[]
a=0.4
for feature, coef in zip(X_train.columns, coefficients):
    if (coef>a) | (coef <-a):
        li.append(feature)
print(li)


# In[172]:


a='PS5'
b='TS4'
dataFrame2= dataFrame[dataFrame['SE']>20]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.scatter (dataFrame2.index,dataFrame2[a],s=0.2 ,label = a)
ax2.scatter (dataFrame2.index,dataFrame2[b],s=0.2 ,label = b,color='orange')
ax1.legend(loc=(1.0,1.0))
ax1.set_xlabel('Index')
ax1.set_ylabel(a)
ax2.set_ylabel(b)

ax2.legend(loc=(0.8,1.0))


# In[173]:


a='PS'
b='TS4'
for i in range(1,7):
    x= a+str(i)
    dataFrame2= dataFrame[dataFrame['SE']>20]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.scatter (dataFrame2.index,dataFrame2[x],s=0.2 ,label = x)
    ax2.scatter (dataFrame2.index,dataFrame2[b],s=0.2 ,label = b,color='orange')
    ax1.legend(loc=(1.0,1.0))
    ax1.set_xlabel('Index')
    ax1.set_ylabel(x)
    ax2.set_ylabel(b)

    ax2.legend(loc=(0.8,1.0))


# In[174]:


col = 'TS4'
# input_col= [x for x in scaled_df.columns if x not in ['TS1','TS2','TS3','TS4','CE','CP','PS5','PS6']]
input_col= ['FS1', 'FS2','SE', 'VS1']

print(input_col)
X_train, X_test, y_train, y_test = train_test_split(scaled_df[input_col], scaled_df[col], test_size=0.3, random_state=42)
lr  = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print(col,"  정확도:", accuracy)
print("")


# In[175]:


# 가중치 확인
# 변수의 가중치 확인
coefficients = lr.coef_
intercept = lr.intercept_

# 각 변수의 가중치 출력
for feature, coef in zip(X_train.columns, coefficients):
    print(f"{feature}: {coef}")

# 절편(intercept) 출력
print("Intercept:", intercept)

# 가중치 값을 그래프로 시각화
plt.bar(X_train.columns, coefficients)
plt.xlabel('Sensor') 
plt.ylabel('Weight')
plt.title('Feature Weight_TS4')
plt.xticks(rotation=90)
plt.show()


# In[176]:


a='EPS1'
b='PS1'
dataFrame2= dataFrame[dataFrame['SE']>20]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.scatter (dataFrame2.index,dataFrame2[a],s=0.2 ,label = a)
ax2.scatter (dataFrame2.index,dataFrame2[b],s=0.2 ,label = b,color='orange')
ax1.legend(loc=(1.0,1.0))
ax1.set_xlabel('Index')
ax1.set_ylabel(a)
ax2.set_ylabel(b)

ax2.legend(loc=(0.8,1.0))


# In[177]:


col = 'TS4'
# input_col= [x for x in scaled_df.columns if x not in ['TS1','TS2','TS3','TS4','CE','CP','PS5','PS6']]
input_col= ['FS1','SE', 'VS1']

print(input_col)
X_train, X_test, y_train, y_test = train_test_split(scaled_df[input_col], scaled_df[col], test_size=0.3, random_state=42)
lr  = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print(col,"  정확도:", accuracy)
print("")


# In[178]:


# 가중치 확인
# 변수의 가중치 확인
coefficients = lr.coef_
intercept = lr.intercept_

# 각 변수의 가중치 출력
for feature, coef in zip(X_train.columns, coefficients):
    print(f"{feature}: {coef}")

# 절편(intercept) 출력
print("Intercept:", intercept)

# 가중치 값을 그래프로 시각화
plt.bar(X_train.columns, coefficients)
plt.xlabel('Sensor') 
plt.ylabel('Weight')
plt.title('Feature Weight_TS4')
plt.xticks(rotation=90)
plt.show()


# In[179]:


a='FS2'
b='TS4'
dataFrame2= dataFrame[dataFrame['SE']>20]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.scatter (dataFrame2.index,dataFrame2[a],s=0.2 ,label = a)
ax2.scatter (dataFrame2.index,dataFrame2[b],s=0.2 ,label = b,color='orange')
ax1.legend(loc=(1.0,1.0))
ax1.set_xlabel('Index')
ax1.set_ylabel(a)
ax2.set_ylabel(b)

ax2.legend(loc=(0.8,1.0))


# In[180]:


a='FS1'
b='TS4'
dataFrame2= dataFrame[dataFrame['SE']>20]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.scatter (dataFrame2.index,dataFrame2[a],s=0.2 ,label = a)
ax2.scatter (dataFrame2.index,dataFrame2[b],s=0.2 ,label = b,color='orange')
ax1.legend(loc=(1.0,1.0))
ax1.set_xlabel('Index')
ax1.set_ylabel(a)
ax2.set_ylabel(b)

ax2.legend(loc=(0.8,1.0))


# In[181]:


col = 'TS4'
# input_col= [x for x in scaled_df.columns if x not in ['TS1','TS2','TS3','TS4','CE','CP','PS5','PS6']]
input_col= ['SE', 'VS1']

print(input_col)
X_train, X_test, y_train, y_test = train_test_split(scaled_df[input_col], scaled_df[col], test_size=0.3, random_state=42)
lr  = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print(col,"  정확도:", accuracy)
print("")

# 가중치 확인
# 변수의 가중치 확인
coefficients = lr.coef_
intercept = lr.intercept_

# 각 변수의 가중치 출력
for feature, coef in zip(X_train.columns, coefficients):
    print(f"{feature}: {coef}")

# 절편(intercept) 출력
print("Intercept:", intercept)

# 가중치 값을 그래프로 시각화
plt.bar(X_train.columns, coefficients)
plt.xlabel('Sensor') 
plt.ylabel('Weight')
plt.title('Feature Weight_TS4')
plt.xticks(rotation=90)
plt.show()


# In[182]:


a='VS1'
b='TS4'
dataFrame2= dataFrame[dataFrame['SE']>20]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.scatter (dataFrame2.index,dataFrame2[a],s=0.2 ,label = a)
ax2.scatter (dataFrame2.index,dataFrame2[b],s=0.2 ,label = b,color='orange')
ax1.legend(loc=(1.0,1.0))
ax1.set_xlabel('Index')
ax1.set_ylabel(a)
ax2.set_ylabel(b)

ax2.legend(loc=(0.8,1.0))


# In[183]:


a='TS3'
b='TS4'
c='TS1'
d='TS2'
dataFrame2= dataFrame[dataFrame['SE']>20]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.scatter (scaled_df.index,scaled_df[a],s=0.2 ,label = a,color='black')
ax2.scatter (scaled_df.index,scaled_df[b],s=0.2 ,label = b,color='orange')
ax2.scatter (scaled_df.index,scaled_df[c],s=0.2 ,label = c,color='red')
ax2.scatter (scaled_df.index,scaled_df[d],s=0.2 ,label = d,color='green')


ax1.legend(loc=(1.0,1.0))
ax1.set_xlabel('Index')
ax1.set_ylabel(a)
ax2.set_ylabel(b)

ax2.legend(loc=(0.8,1.0))


# In[184]:


li= ['CE', 'CP', 'EPS1', 'FS2', 'PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6', 'TS1', 'TS2', 'TS3', 'TS4', 'VS1']
for i in li:
    a= pd.DataFrame(dataFrame2[i],columns=[i])
    X_train, X_test, y_train, y_test = train_test_split(a, dataFrame2['Cooler condition'], test_size=0.3, random_state=42)
    dtc  = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(i,"로 계산한  정확도:", accuracy)
    print("")
    # 가중치 확인
    # 변수의 가중치 확인
    coefficients = lr.coef_
    intercept = lr.intercept_


# In[205]:


for i in ['VS1','PS3','PS4']:
    a=i
    b='PS1'
    dataFrame2= dataFrame[dataFrame['SE']>20]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.scatter (dataFrame2.index,dataFrame2[a],s=0.2 ,label = a)
    ax2.scatter (dataFrame2.index,dataFrame2[b],s=0.2 ,label = b,color='orange')
    ax1.legend(loc=(1.0,1.0))
    ax1.set_xlabel('Index')
    ax1.set_ylabel(a)
    ax2.set_ylabel(b)

    ax2.legend(loc=(0.8,1.0))


# ## Internal pump leakage

# In[185]:


label_columns=['Cooler condition', 'Internal pump leakage', 'Hydraulic accumulator','Valve condition']
label_sen=['SE','FS1','PS5']
for i in label_columns:
    X_train, X_test, y_train, y_test = train_test_split(dataFrame2[label_sen], dataFrame2[i], test_size=0.3, random_state=42)
    dtc  = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(i,"  정확도:", accuracy)
    print("")
    # 중요도 값 확인
    importances = dtc.feature_importances_

    # 중요도 값을 출력
    for feature, importance in zip(dataFrame.columns[:-5], importances):
        print(f"Feature: {feature}, Importance: {importance}")

    # 중요도 값을 그래프로 시각화
    plt.bar(X_train.columns, importances)
    plt.xlabel('Feature') 
    plt.ylabel('Importance')
    plt.title('Feature Importance_'+i)
    plt.xticks(rotation=90)
    plt.show()



# In[186]:


X_train, X_test, y_train, y_test = train_test_split(dataFrame2.iloc[:,:-5], dataFrame2['Internal pump leakage'], test_size=0.3, random_state=42)
dtc  = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Internal pump leakage  정확도:", accuracy)
print("")
# 중요도 값 확인
importances = dtc.feature_importances_

# 중요도 값을 출력
for feature, importance in zip(X_train, importances):
    print(f"Feature: {feature}, Importance: {importance}")

# 중요도 값을 그래프로 시각화
plt.bar(X_train.columns, importances)
plt.xlabel('Feature') 
plt.ylabel('Importance')
plt.title('Feature Importance_Internal pump leakage')
plt.xticks(rotation=90)
plt.show()



# In[187]:


x= [x for x in dataFrame2.columns[:-5] if x != 'SE']
X_train, X_test, y_train, y_test = train_test_split(scaled_df[x], scaled_df['SE'], test_size=0.3, random_state=42)
lr  = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print("SE  정확도:", accuracy)
print("")


# In[188]:


# 가중치 확인
# 변수의 가중치 확인
coefficients = lr.coef_
intercept = lr.intercept_

# 각 변수의 가중치 출력
for feature, coef in zip(X_train.columns, coefficients):
    print(f"{feature}: {coef}")

# 절편(intercept) 출력
print("Intercept:", intercept)

# 가중치 값을 그래프로 시각화
plt.bar(X_train.columns, coefficients)
plt.xlabel('Sensor') 
plt.ylabel('Weight')
plt.title('Feature Weight_SE')
plt.xticks(rotation=90)
plt.show()


# In[189]:


x= ['EPS1','PS1','FS2','TS3']
X_train, X_test, y_train, y_test = train_test_split(scaled_df[x], scaled_df['SE'], test_size=0.3, random_state=42)
lr  = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print("SE  정확도:", accuracy)
print("")


# In[190]:


# 가중치 확인
# 변수의 가중치 확인
coefficients = lr.coef_
intercept = lr.intercept_

# 각 변수의 가중치 출력
for feature, coef in zip(X_train.columns, coefficients):
    print(f"{feature}: {coef}")

# 절편(intercept) 출력
print("Intercept:", intercept)

# 가중치 값을 그래프로 시각화
plt.bar(X_train.columns, coefficients)
plt.xlabel('Sensor') 
plt.ylabel('Weight')
plt.title('Feature Weight_SE')
plt.xticks(rotation=90)
plt.show()


# In[191]:



x= ['EPS1','PS1','TS3']
X_train, X_test, y_train, y_test = train_test_split(scaled_df[x], scaled_df['SE'], test_size=0.3, random_state=42)
lr  = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print("SE  정확도:", accuracy)
print("")
# 가중치 확인
# 변수의 가중치 확인
coefficients = lr.coef_
intercept = lr.intercept_

# 각 변수의 가중치 출력
for feature, coef in zip(X_train.columns, coefficients):
    print(f"{feature}: {coef}")

# 절편(intercept) 출력
print("Intercept:", intercept)

# 가중치 값을 그래프로 시각화
plt.bar(X_train.columns, coefficients)
plt.xlabel('Sensor') 
plt.ylabel('Weight')
plt.title('Feature Weight_SE')
plt.xticks(rotation=90)
plt.show()


# In[192]:



x= ['EPS1','PS1','TS3']
for i in x:
    y= [a for a in x if a != i]
    X_train, X_test, y_train, y_test = train_test_split(scaled_df[y], scaled_df['SE'], test_size=0.3, random_state=42)
    lr  = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    accuracy = r2_score(y_test, y_pred)
    print("SE  정확도:", accuracy)
    print("")
    # 가중치 확인
    # 변수의 가중치 확인
    coefficients = lr.coef_
    intercept = lr.intercept_

    # 각 변수의 가중치 출력
    for feature, coef in zip(X_train.columns, coefficients):
        print(f"{feature}: {coef}")

    # 절편(intercept) 출력
    print("Intercept:", intercept)

    # 가중치 값을 그래프로 시각화
    plt.bar(X_train.columns, coefficients)
    plt.xlabel('Sensor') 
    plt.ylabel('Weight')
    plt.title('Feature Weight_SE')
    plt.xticks(rotation=90)
    plt.show()


# In[193]:


x= ['EPS1','PS1','TS3']
for i in x:
    a= pd.DataFrame(scaled_df[i],columns=[i])
    X_train, X_test, y_train, y_test = train_test_split(a, scaled_df['SE'], test_size=0.3, random_state=42)
    lr  = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    accuracy = r2_score(y_test, y_pred)
    print("SE  정확도:", accuracy)
    print("")
    # 가중치 확인
    # 변수의 가중치 확인
    coefficients = lr.coef_
    intercept = lr.intercept_

    # 각 변수의 가중치 출력
    for feature, coef in zip(X_train.columns, coefficients):
        print(f"{feature}: {coef}")

    # 절편(intercept) 출력
    print("Intercept:", intercept)

    # 가중치 값을 그래프로 시각화
    plt.bar(X_train.columns, coefficients)
    plt.xlabel('Sensor') 
    plt.ylabel('Weight')
    plt.title('Feature Weight_SE')
    plt.xticks(rotation=90)
    plt.show()


# In[ ]:





# In[ ]:





# SE, FS1을 제외한모든 값들이 비슷한형태의그래프를 보여줌

# In[ ]:





# 번외) 각 label들(stable flag제외)이 stable flag에 미치는 영향

# In[195]:


X_train, X_test, y_train, y_test = train_test_split(dataFrame2.iloc[:,-5:-1], dataFrame2.iloc[:,-1], test_size=0.3, random_state=42)
dtc  = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("정확도:", accuracy)


# In[198]:


#시각화 
fig = plt.figure(figsize=(15, 8))
_ = tree.plot_tree(dtc, 
                  feature_names=X_train.columns,
                  class_names=['0', '1'],
                  filled=True)


# In[203]:


dt_clf_model_text = tree.export_text(dtc, feature_names=list(X_train.columns))
print(dt_clf_model_text)


# In[194]:


labels=dataFrame.iloc[:,-5:]
pd.options.display.max_columns = None
pd.options.display.max_rows = None
label_columns=[ 'Valve condition', 'Cooler condition','Internal pump leakage', 'Hydraulic accumulator','stable flag']

# pure_labels = labels.groupby(label_columns).size()
# print(pure_labels)
labels_a= labels
pure_labels = labels_a.groupby(label_columns).size()
print(pure_labels)
pd.options.display.max_columns = 18
pd.options.display.max_rows = 20


# 1. Valve condition이 73 80 90일경우 무조건 0이 나옴 
# 

# 2. Internal pump leakage가 1이나 2이면 0이 10개, 1이 1개 나온다

# 3. valve = 100 ,internal pump = 0 일 때, (cooler, Hydraulic)이 (3, 130)일때 1이 나올 확률이 압도적, (20,90)일때 1이 나올 확률이 압도적, (100, 90)일때 1이 나올 확률이 압도적

# In[ ]:




