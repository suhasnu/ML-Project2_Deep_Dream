import numpy as np
import tensorflow as tf
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras import Model
from PIL import Image

#Configurations
image_path = 'image_1.jpg'
layer_name = 'block_9_add'
steps = 500
step_size = 0.01

#Loading the Model
base_model = MobileNetV2(weights = 'imagenet', include_top = False) #Turning off the top classification layer

#selecting the specific layer
"""
#Searching for the layer
for layer in base_model.layers:
  print(layer.name)
"""
target_layer = base_model.get_layer(layer_name)

#Creating a new model that outputs only activations of that layer
dream_model = Model(inputs = base_model.input, outputs = target_layer.output)

def preprocess_image(image_path):
  image = Image.open(image_path)
  #resize to 224x224
  image = image.resize((224, 224))
  #Convert to numpy array
  image_array = np.array(image)
  #expanding dimension from (224, 224, 3) to (1, 224, 224, 3)
  image_array = np.expand_dims(image_array, axis = 0)
  #scale values between -1 and 1
  image_array = preprocess_input(image_array)
  
  #print(image_array.shape)
  return image_array

#preprocess_image(image_path)

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

#Loss function
def calculate_loss(layer_activations):
  #calculating the sum of aquared activation
  #square the activations to make negative values exciting too
  losses = tf.square(layer_activations)
  #sum up all the excitement in the layer
  total_loss = tf.reduce_sum(losses)
  return total_loss

#dreaming loop
@tf.function #to speed up execution
def dream_step(image, model, step_size):
  #calculating Gradient
  with tf.GradientTape() as tape:
    tape.watch(image)
    activations = model(image)
    loss = calculate_loss(activations)
  
  gradients = tape.gradient(loss, image)
  
  #normalizing gardients to avoid exploding/vanishing updates
  gradients /= (tf.math.reduce_std(gradients) + 1e-8)
  
  #update image
  image = image + (gradients * step_size)
  
  #Clip pixel values to stay within the valid range (-1, 1)
  image = tf.clip_by_value(image, -1, 1)
  
  return loss, image

#exceution part
image = preprocess_image(image_path)
image = tf.convert_to_tensor(image) #convert to Tensor for GradientTape

print("Starting to dream")

for step in range(steps):
  loss, image = dream_step(image, dream_model, step_size)
  
  #Displaying progress every 10 steps
  if step % 10 == 0:
    print(f"Step {step}/{steps}, loss: {loss:.2f}")
    
#save the final result
dream_image = deprocess_image(image.numpy())
output_filename = f"dream_{image_path}"
Image.fromarray(dream_image).save(output_filename)

print(f"Dreaming complete")
  
