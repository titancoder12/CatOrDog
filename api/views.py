from django.shortcuts import render
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework import status
import tensorflow as tf
import os

# Load the tflite model
tflite_model_file = os.path.abspath('./api/model1.tflite')
interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

# Create your views here.
@api_view(['POST'])
def predict(request):

    # Check if 'image' is in the request data
    if 'image' not in request.FILES:
        return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)
    
    # Get/Preprocess the image
    try:
        # Get the image from the request
        image = request.FILES['image']
        image = image.read()
        image = tf.io.decode_jpeg(image)
    except ValueError as e:
        print(str(e))
        return Response({'error': 'Invalid image'}, status=status.HTTP_400_BAD_REQUEST)
    
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.expand_dims(image, 0)

    # Make a prediction
    interpreter.set_tensor(input_index, image)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_index)
    
    # Get the class names
    class_names = ['cat', 'dog']

    # Get the predicted class index
    predicted_class_index = tf.argmax(prediction, axis=1).numpy()[0]

    # Get the predicted class name and prediction value
    predicted_class_name = class_names[predicted_class_index]
    prediction_value = prediction[0][predicted_class_index]

    # Return the class name and prediction value
    return Response({'class_name': predicted_class_name, 'prediction_value': prediction_value}, status=status.HTTP_200_OK)