# Create your first MLP in Keras
# importing libs
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
# In[3]:
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# In[3]:
# load pima indians dataset

dataset = np.loadtxt("data/pima-indians-diabetes.csv", delimiter=",")

# In[3]:
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# In[3]:
# create model

model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
#model.summary()

# In[3]:
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# In[3]:
# Fit the model
model.fit(X,Y, nb_epoch=150, batch_size=10)

# In[3]:
# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
