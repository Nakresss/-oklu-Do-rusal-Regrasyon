import pandas as pd 
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error
#Çift tırnak içine data dosyasının adı yazılacak
ad = pd.read_csv("Advertising.csv", usecols = [1,2,3,4])
df = ad.copy()
df.head()


X = df.drop("sales", axis = 1)
Y = df["sales"]
X.columns = ["feature1","feature2","feature3"]
X_train, X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.20, random_state=42)
print("X'in Train gözlemi",X_train.shape)
print("Y'nin train gözlemi",Y_train.shape)
print("X'in test sayısı",X_test.shape)
print("Y'nin test sayısı",Y_test.shape)
training = df.copy()
print("Veri setimizin ilk hali",training.shape)

#Statsmodels işlemleri

lm = sm.OLS(Y_train,X_train)
model = lm.fit()
model.summary()
a = input("Katsayıya göre tabloya bakmak ister misiniz? Y/N")
if a == "Y":
    b = input("kaç katsayı olduğunu girin")
    print(model.summary().tables(b))
else:
    pass

#Scikit learn model
lm = LinearRegression()
model = lm.fit(X_train,Y_train)
print("Sabit katsayı", model.intercept_)
print("Diğer Katsayılar",model.coef_)

#Tahmin işlemleri

c = input("Yeni Değer girin 1. için")
d = input("Yeni Değer girin 2. için")
e= input("Yeni Değer girin 3. için")
yeni_veri = [[c],[d],[e]]
yeni_veri = pd.DataFrame(yeni_veri).T

print("Değer Tahmini",model.predict(yeni_veri))
rmse = np.sqrt(mean_squared_error(Y_train,model.predict(X_train)))
print("eğitim hatanız",rmse)
rmse2 = np.sqrt(mean_squared_error(Y_test,model.predict(X_test)))
print("test seti hatanız",rmse2)
