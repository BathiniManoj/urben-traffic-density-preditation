import os
import cv2
from django.shortcuts import render, redirect
from django.http import HttpResponse
import numpy as np
import tensorflow as tf
import pandas as pd
from django.http import JsonResponse
from django.conf import settings
from .models import TrafficData
from django.core.files.storage import default_storage


# Get the absolute path to the model file
model_path = os.path.join(os.path.dirname(__file__), 'traffic', 'predtic', 'traffic_density_model.h5')

# Load TensorFlow model
model = tf.keras.models.load_model("C:/Users/manoj/vdt/traffic/predtic/traffic_density_model.h5")

IMG_WIDTH, IMG_HEIGHT = 224, 224  # Image size

# Load historical traffic data
traffic_data = pd.read_csv("C:/Users/manoj/vdt/traffic/predtic/futuristic_city_traffic.csv").drop(columns=["City"])

def home(request):
    return render(request, 'home.html')

def estimate_traffic_delay(vehicle_count, csv_data):
    """Estimate traffic delay based on the closest match in the CSV file."""
    closest_match = csv_data.iloc[(csv_data["Traffic Density"] - vehicle_count).abs().argsort()[:1]]
    estimated_speed = closest_match["Speed"].values[0] if not closest_match.empty else 30
    return round(60 / estimated_speed, 2) if estimated_speed > 0 else 10


# 
from django.shortcuts import render
from django.http import HttpResponse
import os
import cv2
import numpy as np
from django.conf import settings
# Load the trained model once when the server starts
#model = tf.keras.models.load_model("traffic_density_model.h5")
IMG_WIDTH, IMG_HEIGHT = 224, 224  # Image size
def estimate_traffic_delay(vehicle_count):
    """Determine traffic density and estimated delay based on vehicle count with better granularity."""
    
    # Very Low traffic: (0 to 5 vehicles)
    if vehicle_count <= 5:
        return "No traffic", 0.5  # Minimal delay, ~0.5 min/km
    
    # Low traffic: (6 to 15 vehicles)
    elif 6 <= vehicle_count <= 15:
        return "Low", 1  # Low traffic, ~1 min/km
    
    # Moderate traffic: (16 to 30 vehicles)
    elif 16 <= vehicle_count <= 30:
        return "Moderate", 2  # Moderate traffic, ~2 min/km
    
    # Busy traffic: (31 to 50 vehicles)
    elif 31 <= vehicle_count <= 50:
        return "Busy", 4  # Busy traffic, ~4 min/km
    
    # Heavy traffic: (51 to 75 vehicles)
    elif 51 <= vehicle_count <= 75:
        return "Heavy", 6  # Heavy traffic, ~6 min/km
    
    # Very Heavy traffic: (76 to 100 vehicles)
    elif 76 <= vehicle_count <= 100:
        return "Very Heavy", 8  # Very Heavy traffic, ~8 min/km
    
    # Extreme congestion: (101+ vehicles)
    else:
        return "Extreme", 10  # Extreme congestion, ~10+ min/km

    
def predict_traffic_from_image(request):
    """Predict traffic density and estimate delay from an uploaded image."""
    if request.method == 'POST' and request.FILES['image']:
        # Handle image upload
        image = request.FILES['image']

        # Generate a unique name for the image and save it
        image_name = f"{str(request.user.id)}_{image.name}"
        image_path = os.path.join(settings.MEDIA_ROOT, image_name)

        # Save the image to the media folder using Django's default storage
        with default_storage.open(image_path, 'wb') as f:
            for chunk in image.chunks():
                f.write(chunk)

        # Read the saved image using OpenCV (cv2)
        img = cv2.imread(image_path)

        if img is None:
            return HttpResponse("Error: Unable to read the uploaded image", status=400)

        # Proceed with image processing and prediction
        img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img_resized = img_resized / 255.0  # Normalize
        img_input = np.expand_dims(img_resized, axis=0)

        # Predict vehicle count using the model
        vehicle_count = model.predict(img_input)[0][0]
        vehicle_count = int(vehicle_count)  # Convert to integer

        # Get traffic density and estimated delay
        density_level, estimated_delay = estimate_traffic_delay(vehicle_count)

        # Save the prediction data into the session
        request.session['vehicle_count'] = vehicle_count
        request.session['estimated_delay'] = estimated_delay
        request.session['density_level'] = density_level
        request.session['image_url'] = f"/media/{image_name}"  # Save image URL for displaying

        # Redirect to success page or render the result
        return redirect('success_page')  # Redirect to the success page that will display the results

    return render(request, 'upload_image.html')  # Form for uploading the image
# Success page view to render the results after redirection
def success_page(request):
    """Render the success page with prediction results."""
    # Get data from session
    vehicle_count = request.session.get('vehicle_count', None)
    estimated_delay = request.session.get('estimated_delay', None)
    density_level = request.session.get('density_level', None)
    image_url = request.session.get('image_url', None)

    # If session data is not available, redirect to the upload page
    if not vehicle_count or not estimated_delay or not density_level or not image_url:
        return redirect('predict_image')

    context = {
        'vehicle_count': vehicle_count,
        'estimated_delay': estimated_delay,
        'density_level': density_level,
        'image_url': image_url,
    }

    # Optionally, clear session data after displaying the results (if needed)
    request.session.flush()

    return render(request, 'success_page.html', context)


def predict_traffic_from_camera(request):
    """Capture image from camera, predict traffic density, and return response."""
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return render(request, "error_template.html", {"error": "Camera not accessible"})
    
    img_resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
    img_input = np.expand_dims(img_resized, axis=0)

    vehicle_count = int(model.predict(img_input)[0][0])
    estimated_delay = estimate_traffic_delay(vehicle_count, traffic_data)

    return render(request, "prediction.html", {
        "vehicle_count": vehicle_count,
        "estimated_delay": estimated_delay
    })

def traffic_data_list(request):
    """Fetch all traffic data and render it in a list."""
    traffic_data = TrafficData.objects.all()  # Fetch all traffic data from the database
    context = {
        'traffic_data': traffic_data,
    }
    return render(request, 'traffic_data_list.html', context)

def base(request):
    """Render the home page."""
    return render(request, 'base.html')

# import os
# import tensorflow as tf
# import numpy as np
# from sklearn.metrics import mean_absolute_error, mean_squared_error

# # Load your model
# model_path = os.path.join(os.path.dirname(__file__), 'traffic', 'predtic', 'traffic_density_model.h5')
# model = tf.keras.models.load_model(model_path)

# # Example: Assume you have your test data (X_test and y_test) ready.
# # X_test: Input features (images or data points)
# # y_test: True target values (ground truth)

# # Example of loading or generating test data
# # X_test = np.array([...])  # Your test features
# # y_test = np.array([...])  # Your true labels/target values

# # Model prediction (replace with actual test data input)
# predictions = model.predict(X_test)

# # Ensure that y_test and predictions have the same shape
# y_test = y_test.reshape(-1)  # Flatten y_test if needed
# predictions = predictions.reshape(-1)  # Flatten predictions if needed

# # Calculate Mean Absolute Error (MAE)
# mae = mean_absolute_error(y_test, predictions)
# print(f'Mean Absolute Error (MAE): {mae}')

# # Calculate Root Mean Squared Error (RMSE)
# rmse = np.sqrt(mean_squared_error(y_test, predictions))
# print(f'Root Mean Squared Error (RMSE): {rmse}')
