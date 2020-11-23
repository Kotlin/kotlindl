#!/usr/bin/env python
# coding: utf-8

# In[14]:


import tensorflow as tf

# In[15]:


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# In[16]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# In[17]:


model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test, verbose=2)

# In[18]:


tf.saved_model.save(model, "C:\\zaleslaw\\home\\models\\mnist1")

# In[ ]:
