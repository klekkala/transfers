
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

reg = 0.01

def get_gradient(w, b, x):  
  # evaluate class scores, [N x K]

  num_examples = x.shape[0]
  features = x[:,:-1]
  labels = x[:,-1].astype(int)
  scores = np.dot(features, w) + b 
  
  # compute the class probabilities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
  
  # compute the loss: average cross-entropy loss and regularization
  corect_logprobs = -np.log(probs[range(num_examples),labels])
  data_loss = np.sum(corect_logprobs)/num_examples
  reg_loss = 0.5*reg*np.sum(w*w)
  loss = data_loss + reg_loss
  
  # compute the gradient on scores
  dscores = probs
  dscores[range(num_examples),labels] -= 1
  dscores /= num_examples
  
  # backpropate the gradient to the parameters (W,b)
  dw = np.dot(features.T, dscores)
  db = np.sum(dscores, axis=0, keepdims=True)
  
  dw += reg*w # regularization gradient
  
  # perform a parameter update
  #W += -step_size * dW
  #b += -step_size * db

  return dw, db, loss



#Number of samples:
num = 10

# Range:
sa = 0
ea = 10
sb = 30
eb = 40



ax = np.random.random((num,2))*(ea-sa) + sa
data_a = np.concatenate((ax, np.random.randint(1, size = num)[:, np.newaxis]), axis = 1)

bx = np.random.random((num,2))*(eb-sb) + sb
data_b = np.concatenate((bx, 1+np.random.randint(1, size = num)[:, np.newaxis]), axis = 1)


data_x = np.random.random((10*num,2))*(45-sa) + sa

# *Add intercept data and normalize*

# In[4]:



ordera = np.random.permutation(len(data_a))
orderb = np.random.permutation(len(data_b))

portion = 6

train_a = data_a[ordera[:portion]]

train_b = data_b[orderb[:portion]]

val_a = data_a[ordera[portion:]]

val_b = data_b[orderb[portion:]]



train_ab = np.concatenate([train_a, train_b])
val_ab = np.concatenate([val_a, val_b])
np.random.shuffle(train_ab)



w = np.random.randn(2,2)
b = np.random.randn(1,2)

wv = np.random.randn(2,2)
bv = np.random.randn(1,2)

alpha = 0.5
tolerance = 1e-5

# Perform Gradient Descent
iterations = 1



# *Perform gradient descent on the training set
while True:
    dw, db, error = get_gradient(w, b, train_ab)
    new_w = w - alpha * dw
    new_b = b - alpha * db
    
    # Stopping Condition
    if np.sum(abs(new_w - w)) < tolerance:
        print "Converged."
        break
    
    # Print error every 50 iterations
    if iterations % 100 == 0:
        print "Iteration: %d - Error: %.4f" %(iterations, error)
    
    iterations += 1
    w = new_w
    b = new_b

print "w =",w
print "Test Cost =", get_gradient(w, b, val_ab)[1]



# *Perform gradient descent on the validations set
iterations = 1
while True:
    dw, db, error = get_gradient(wv, bv, val_ab)
    new_w = wv - alpha * dw
    new_b = bv - alpha * db
    
    # Stopping Condition
    if np.sum(abs(new_w - wv)) < tolerance:
        print "Converged."
        break
    
    # Print error every 50 iterations
    if iterations % 100 == 0:
        print "Iteration: %d - Error: %.4f" %(iterations, error)
    
    iterations += 1
    wv = new_w
    bv = new_b

print "w =",wv
print "Test Cost =", get_gradient(wv, bv, val_ab)[1]


# In[9]:
classa = []
classb = []
y = np.dot(data_x,w)+b
for i in range(len(y)):
  if y[i,0] > y[i,1]:
    classa.append(i)

  else:
    classb.append(i)

'''
plt.scatter(np.take(data_x, classa, axis=0)[:,0], np.take(data_x, classa, axis=0)[:,1], c='m', label='Model')
plt.scatter(np.take(data_x, classb, axis=0)[:,0], np.take(data_x, classb, axis=0)[:,1], c='k', label='Model')
plt.scatter(train_a[:,0], train_a[:,1], c='y', label='Train class A Set')
plt.scatter(train_b[:,0], train_b[:,1], c='r', label='Train class B Set')
plt.scatter(val_a[:,0], val_a[:,1], c='g', label='Val class A Set')
plt.scatter(val_b[:,0], val_b[:,1], c='b', label='Val class B Set')
plt.grid()
plt.legend(loc='best')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
'''
# *Plot the model obtained*

# In[10]:


w1 = np.linspace(-w[1,1]*3, w[1,1]*3, 300)
w0 = np.linspace(-w[0,0]*3, w[0,0]*3, 300)
J_vals = np.zeros(shape=(w1.size, w0.size))

for t1, element in enumerate(w1):
    for t2, element2 in enumerate(w0):
        wT = np.zeros((2,2))
        bT = np.zeros((1,2))
        wT[1,1] = element
        wT[0,0] = element2
        J_vals[t1, t2] = get_gradient(wT, bT, train_ab)[2]

wv1 = np.linspace(-wv[1,1]*3, wv[1,1]*3, 300)
wv0 = np.linspace(-wv[0,0]*3, wv[0,0]*3, 300)
Jv_vals = np.zeros(shape=(w1.size, w0.size))

for t1, element in enumerate(wv1):
    for t2, element2 in enumerate(wv0):
        wT = np.zeros((2,2))
        bT = np.zeros((1,2))
        wT[1,1] = element
        wT[0,0] = element2
        Jv_vals[t1, t2] = get_gradient(wT, bT, val_ab)[2]

plt.scatter(w[0,0], w[1,1], marker='*', color='r', s=40, label='Solution Found')
plt.scatter(wv[0,0], wv[1,1], marker='*', color='k', s=40, label='Solution Found')
CS = plt.contour(w0, w1, J_vals, np.logspace(-10,10,50), label='Cost Function')
CS = plt.contour(wv0, wv1, Jv_vals, np.logspace(-10,10,50), label='Cost Function')
plt.clabel(CS, inline=1, fontsize=10)
plt.title("Contour Plot of Cost Function")
plt.xlabel("w0")
plt.ylabel("w1")
plt.legend(loc='best')
plt.show()


# *Generate contour plot of the cost function*
