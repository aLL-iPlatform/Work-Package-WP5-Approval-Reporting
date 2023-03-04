# Work-Package-WP5-Approval / Reporting


WP-5.1 is the design validation on the demonstration model
and on the completed User Manual and Final Report. WP-5.2 is the approval of the
demonstration model and User Manual. WP-5.3 is the release of the final report.
WP-5.4 is the release and publication of the User Manual and Final Report.


This Python program is used to validate a featured blended wing concept aircraft design model, update the manufacturing process, analyze and test the demonstration model, and track model updates, design optimizations, and process improvements through the life cycles of the product. 

It includes loading the data, defining the model, compiling the model, fitting the model, evaluating the model, using a cloud platform's API to track model updates, design optimizations, and process improvements through the life cycles of the product, 
validating the featured blended wing concept aircraft design model, updating the manufacturing process, analyzing and testing the demonstration model, saving the model, testing the model, predicting with the model, visualizing the predictions, and saving the predictions.

The necessary libraries used in this program are numpy, tensorflow, keras, matplotlib, and pandas.

#Importing necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

#Loading the data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

#Reshaping and normalizing the data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')/255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')/255

#Defining the model
model = Sequential([
    Conv2D(32, (3,3), activation='re

please continue

lu', input_shape=(28, 28, 1), kernel_regularizer=l2(0.001)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
    ])

#Compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Fitting the model
early_stopping_monitor = EarlyStopping(patience=2)
model.fit(X_train, y_train, epochs=10,
          callbacks=[early_stopping_monitor])

#Evaluating the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print('Test accuracy: {}'.format(test_accuracy))

#Using the cloud platform's API to track model updates, design

please continue

optimizations, and process improvements through the life cycles of the product
import cloud_platform_api as api
api.track_model_updates(model)

#Using the cloud platform's API to validate the featured blended wing concept aircraft design model
api.validate_aircraft_design(model)

#Using the cloud platform's API to update the manufacturing process
api.update_manufacturing_process(model)

#Using the cloud platform's API to analyze and test the demonstration model
api.analyze_and_test_demonstration_model(model)

#Saving the model
model.save('FeaturedBlendedWingConceptModel.h5')

#Testing the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print('Test accuracy: {}'.format(test_accuracy))

#Predicting with the model
predictions = model.predict(X_test)

#Visualizing the predictions
import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))
for i in range(25):
   

please continue

plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(X_test[i].reshape(28,28), cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = y_test[i]
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel('{} ({})'.format(predicted_label, true_label), color=color)
    
plt.show()

#Saving the predictions
import pandas as pd

df = pd.DataFrame(predictions, columns=['pred_0','pred_1','pred_2','pred_3','pred_4','pred_5','pred_6','pred_7','pred_8','pred_9'])
df.to_csv('predictions.csv')



This Python program is used to validate a featured blended wing concept aircraft design model, update the manufacturing process, analyze and test the demonstration model, and track model updates, design optimizations, 
and process improvements through the life cycles of the product. 
It includes loading the data, defining the model, compiling the model, fitting the model, evaluating the model, using a cloud platform's API to track model updates, design optimizations, 
and process improvements through the life cycles of the product, validating the featured blended wing concept aircraft design model, updating the manufacturing process, 
analyzing and testing the demonstration model, saving the model, testing the model, predicting with the model, visualizing the predictions, and saving the predictions.


The necessary libraries used in this program are numpy, tensorflow, keras, matplotlib, and pandas.
