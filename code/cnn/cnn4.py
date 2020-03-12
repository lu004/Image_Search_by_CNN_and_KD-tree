# A CNN model has much more parameters, however, which is not fit for the case study problem

import os
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from utl import d_dir

class Cnn:

	def __init__(self):
		self.p = os.path.join(d_dir, "cnn.h5")
		self.md()
		if os.path.exists(self.p):
			self.m.load_weights(self.p)
			print("load cnn:" + self.p)
		self.m.summary()

	def md(self):
		input_size = (28, 28, 3)
		cl_n = 6
		ft_layer = "dense_1"

		self.m = Sequential([
			Conv2D(32, (3, 3), padding='same', input_shape=input_size, activation='relu'),
			Conv2D(32, (3, 3), activation='relu'),
			MaxPooling2D(pool_size=(2, 2)),
			Dropout(0.25),

			Conv2D(64, (3, 3), padding='same', activation='relu'),
			Conv2D(64, (3, 3), activation='relu'),
			MaxPooling2D(pool_size=(2, 2)),
			Dropout(0.25),

			Flatten(),
			#Dense(512, activation='relu'),
			#Dropout(0.5),
			Dense(128, activation='relu'),
			Dense(cl_n, activation='softmax')
		])
		self.m.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
		self.m_f = Model(inputs=self.m.input, outputs=self.m.get_layer(ft_layer).output)

	def get_f(self, x):
		return self.m_f.predict(x)

	def sv(self):
		self.m.save_weights(self.p)
