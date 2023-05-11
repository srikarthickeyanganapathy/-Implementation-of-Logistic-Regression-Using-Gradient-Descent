# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the data file and import numpy, matplotlib and scipy.
2. Visulaize the data and define the sigmoid function, cost function and gradient descent.
3. Plot the decision boundary . 
4. Calculate the y-prediction.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Sri Karthickeyan Ganapathy
RegisterNumber:  212222240102
*/ 
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize    #to remove unwanted data and memory storage

data=np.loadtxt("/content/ex2data1 (1).txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

Visualizing the data
plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

Sigmoid fuction
def sigmoid(z):
  return 1/(1+np.exp(-z))
  
plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()

def costFuction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  J= -(np.dot(y, np.log(h)) + np.dot(1-y,np.log(1-h))) / X.shape[0]
  grad = np.dot(X.T, h-y) / X.shape[0]
  return J,grad
  
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J, grad=costFuction(theta, X_train, y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J, grad=costFuction(theta, X_train, y)
print(J)
print(grad)

def cost(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  J= -(np.dot(y, np.log(h)) + np.dot(1-y, np.log(1-h))) / X.shape[0]
  return J
  
def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad= np.dot(X.T, h-y) / X.shape[0]
  return grad
  
X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta= np.array([0,0,0])
res = optimize.minimize(fun=cost, x0=theta, args=(X_train,y),method="Newton-CG",jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min, x_max = X[:,0].min() - 1, X[:,0].max()+1
  y_min, y_max = X[:,1].min() - 1, X[:,1].max()+1
  xx, yy = np.meshgrid(np.arange(x_min,x_max,0.1),
                       np.arange(y_min,y_max,0.1))
  X_plot = np.c_[xx.ravel(), yy.ravel()]
  X_plot = np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot = np.dot(X_plot, theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
  plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not admitted")
  plt.contour(xx,yy,y_plot, levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
  
  plotDecisionBoundary(res.x,X,y)
  
prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
  X_train = np.hstack((np.ones((X.shape[0],1)),X))
  prob = sigmoid(np.dot(X_train,theta))
  return(prob >= 0.5).astype(int)
  
np.mean(predict(res.x,X)==y)
```

## Output:
![logistic regression using gradient descent](sam.png)
### Array Value of x :
![237655268-033e44c7-01d8-4694-af58-e47e586bc326](https://github.com/srikarthickeyanganapathy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393842/396431c8-6cf5-457b-be06-52999fdd6c64)

### Array Value of y :
![237655319-3d9afb2e-3520-4ab7-a959-1685a95c48cb](https://github.com/srikarthickeyanganapathy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393842/7fd3ea3e-466e-476d-a03d-0fe70502054c)

### Exam 1 - Score Graph :
![237659957-6c725e84-829f-459b-9dab-ec3c9c4316fb](https://github.com/srikarthickeyanganapathy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393842/9846913a-088d-4097-a74a-18ef6fa23575)

### Sigmoid Function Graph :
![237655512-b114a544-ffeb-42c5-a36d-e6bdb52a65ea](https://github.com/srikarthickeyanganapathy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393842/9a61afb5-cf4e-4380-a053-e43cd8e279ee)

### X_train_grad Value :
![237655762-6d006193-6d73-44d9-8290-4d82eb608d7e](https://github.com/srikarthickeyanganapathy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393842/1e539f83-56ab-4728-8efb-b08951ed70d1)

### Y_train_grad Value :
![237655834-71194e2e-7353-4bf9-93d7-027575624256](https://github.com/srikarthickeyanganapathy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393842/68c1ffb5-bc17-4932-bf0b-a1e374a694e7)

### Print Res.x :
![237656681-16c592fc-3657-4346-bdb5-c27acad5a7f8](https://github.com/srikarthickeyanganapathy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393842/74c46118-6c02-4f3c-9a61-c12907b14f9b)

### Decision Boundary - Graph For Exam Score :
![237660022-beecf6cd-c08f-47df-bb02-00eadb43d41f](https://github.com/srikarthickeyanganapathy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393842/06080c15-ad3e-4e1f-a7bf-04aea62926f1)

### Proability Value :
![237656790-b92b63f4-f532-462a-8593-d86f4cc83efd](https://github.com/srikarthickeyanganapathy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393842/6978e563-0ab5-437b-ab3d-64a33884fdc4)

### Prediction Value of Mean :
![237656853-f8ee18ed-c208-47d9-ac94-7704d6852df1](https://github.com/srikarthickeyanganapathy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393842/8ff09384-04a2-4e5c-8211-9b963a24570c)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

