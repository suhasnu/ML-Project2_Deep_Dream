import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image 
from keras.applications import MobileNetV2
from keras.models import Model
from keras.applications.mobilenet_v2 import preprocess_input


def preprocess_image(image_path):
  
  #Loadimg image and converting it into RGB image.
  original_image = Image.open(image_path).convert('RGB')

  #resize image to 224x224x3, PIL library implicitly handles the 3 channels
  original_image = original_image.resize((224, 224))

  #Converting the image into numpy array and adding the batch dimension.
  image_array = np.array(original_image)
  image_array = np.expand_dims(image_array, axis=0)

  #scale the values between -1 to 1
  image_array = preprocess_input(image_array)
  
  
  #convert to tensorflow variable to allow geadient updates
  return tf.Variable(image_array)

def deprocess_image(processed_image):
  x = processed_image.copy()
  #remove the batch dimension from (1, 224, 224, 3) to (224, 224, 3)
  if len(x.shape) == 4:
    x = np.squeeze(x, axis = 0)
  #Normalize from range(-1, 1) to range(0, 1)
  x = (x + 1.0) / 2.0
  #scale from (0, 1) to (0, 255)
  x = x * 255.0
  #Clip values to ensure they are in valid range of colors(0, 255)
  x = np.clip(x, 0, 255).astype('uint8')
  return x

#calculating loss
def calculate_loss(activations):
  loss = -tf.reduce_mean(tf.square(activations))
  return loss

#passing arguments
if len(sys.argv) < 4:
  print("Enter valid number of arguments")
  sys.exit(1)
  
image_path = sys.argv[1]
layer = sys.argv[2]
steps = sys.argv[3]
N = int(steps)

#Loading the model "turning the off the top classification layer"
base_model = MobileNetV2(weights = 'imagenet', include_top = False)

#selecting the specific layer
layer_id = int(layer)
if layer_id < 0 or layer_id >= len(base_model.layers):
  print("Layer out of bound")
  sys.exit(1)
target_layer = base_model.layers[layer_id]
#Creating a new model that outputs only activations of that layer
dream_model = Model(inputs=base_model.input, outputs=target_layer.output)

step_size = 0.01

original_image = preprocess_image(image_path)

@tf.function
def dream_step(image):
  #calculating gradients and loss
  with tf.GradientTape() as tape:
    tape.watch(image)
    activations = dream_model(image)
    loss = calculate_loss(activations)
    
  gradients = tape.gradient(loss, image)
  
  #normalizing gardients to avoid exploding/vanishing updates
  gradients /= (tf.math.reduce_std(gradients) + 1e-8)
  
  #update image
  image.assign_sub(gradients * step_size)
  #Clip pixel values to stay within the valid range (-1, 1)
  
  image.assign(tf.clip_by_value(image, -1, 1))
  
  return loss

for step in range(N):
  loss_value = dream_step(original_image)
  
  #Displaying progress every 10 steps
  if step % 10 == 0:
    print(f"Step {step}/{steps}, loss: {loss_value:.2f}")

#save the final result
dream_image = deprocess_image(original_image.numpy())
output_filename = f"dream_{os.path.basename(image_path)}"
Image.fromarray(dream_image).save(output_filename)

print("Dreaming complete")