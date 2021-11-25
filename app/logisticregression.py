# import numpy as np 
# from numpy.random import rand
# from numpy import log, dot, e
# import pandas as pd 

# path = '/content/gdrive/MyDrive/covid-symptoms-dataset.csv'
# df = pd.read_csv(path)

# pd.DataFrame(df.dtypes).rename(columns = {0:'dtype'})
# df_OHE = pd.get_dummies(df, drop_first=True) 
# df_OHE = df_OHE.drop(['Running Nose_Yes', 'Asthma_Yes', 'Chronic Lung Disease_Yes', 'Headache_Yes', 
#                       'Heart Disease_Yes', 'Diabetes_Yes', 'Hyper Tension_Yes', 'Fatigue _Yes', 
#                       'Gastrointestinal _Yes', 'Visited Public Exposed Places_Yes', 
#                       'Family working in Public Exposed Places_Yes'], axis=1)
# df_OHE.rename(columns = {'COVID-19_Yes':'COVID-19'}, inplace=True)

# X = df_OHE.drop('COVID-19', axis=1)
# y = df_OHE['COVID-19']

# class MinMaxScaler(object):
#     def __init__(self, feature_range=(0, 1)):
#         self._low, self._high = feature_range

#     def fit(self, X):
#         self._min = X.min(axis=0)
#         self._max = X.max(axis=0)
#         return self

#     def transform(self, X):
#         X_std = (X - self._min) / (self._max - self._min)
#         return X_std * (self._high - self._low) + self._low

#     def fit_transform(self, X):
#         return self.fit(X).transform(X)

# def train_test_split_(*arrays, test_size=None, train_size=None, random_state=None):
#     length = len(arrays[0])
#     if random_state:
#         np.random.seed(random_state)
        
#     p = np.random.permutation(length)

#     if type(test_size) == int:
#         index = length - test_size
#     elif type(test_size) == float:
#         index = length - np.ceil(length * test_size)
#     else:
#         if type(train_size) == int:
#             index = train_size
#         elif type(train_size) == float:
#             index = int(length * train_size)
#         else:
#             index = length - np.ceil(length * 0.25)

#     return [b for a in arrays for b in (a[p][:index], a[p][index:])]

# class LogisticRegression:
    
#     def sigmoid(self, z): 
#       return 1 / (1 + e**(-z))
    
#     def cost_function(self, X, y, weights):                 
#         z = dot(X, weights)
#         predict_1 = y * log(self.sigmoid(z))
#         predict_0 = (1 - y) * log(1 - self.sigmoid(z))
#         return -sum(predict_1 + predict_0) / len(X)
    
#     def fit(self, X, y, epochs=25, lr=0.05):        
#         loss = []
#         weights = rand(X.shape[1])
#         N = len(X)
                 
#         for _ in range(epochs):        
#             # Gradient Descent
#             y_hat = self.sigmoid(dot(X, weights))
#             weights -= lr * dot(X.T,  y_hat - y) / N            
#             # Saving Progress
#             loss.append(self.cost_function(X, y, weights)) 
            
#         self.weights = weights
#         self.loss = loss
    
#     def predict(self, X):        
#         # Predicting with sigmoid function
#         z = dot(X, self.weights)
#         # Returning binary result
#         return [1 if i > 0.5 else 0 for i in self.sigmoid(z)]

# scaler = MinMaxScaler(feature_range=(-1, 1))
# X_array = X.to_numpy()
# X_scaled = scaler.fit_transform(X_array)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y)

# logreg = LogisticRegression()
# logreg.fit(X_train, y_train, epochs=500, lr=0.5)

# user_given_features = [[-1.,  1., -1., -1., -1., -1., -1.]]

# x_example_test = [user_given_features]
# example = logreg.predict(x_example_test)
# print(example)

