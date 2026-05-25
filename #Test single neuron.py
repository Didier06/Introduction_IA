#Test single neuron
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam, SGD
#Input data
X_train = [.1,.2,.3, .4]
Y_train= [.05,.1, .15,.2]
print("X_train ", X_train)
print("Y_train ", Y_train)
#Model architecture
modelSimple = Sequential()
modelSimple.add(Dense(1, kernel_initializer= 'random_normal', input_shape=(1,)))
#modelSimple.add(Dense(3, kernel_initializer= 'uniform', input_shape=(1,)))
#modelSimple.add(Dense(1, kernel_initializer= 'uniform', input_shape=(1,)))

#Compile and fit model
LEARNING_RATE =0.05
modelSimple.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse")

# si on fixe manuellment les coefficients et biais couche 0
#coeff = np.array([[0.5]])
#biais = np.array([0])
#poids =[coeff,biais]
#modelSimple.layers[0].set_weights(poids)

modelSimple.fit(X_train, Y_train, batch_size=1, epochs=20, verbose=1)
#Print weights
print("")
print("Weights: \n")
print(modelSimple.get_weights())
#Print prediction
print("")
print("Prediction: \n")
print(modelSimple.predict(X_train, batch_size=1))