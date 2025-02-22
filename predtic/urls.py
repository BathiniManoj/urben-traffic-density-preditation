from django.urls import path
from .views import *
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("", home, name="home"),
    path("traffic/predict/image/", predict_traffic_from_image, name="predict_image"),
    path("traffic/predict/camera/", predict_traffic_from_camera, name="predict_camera"),
    path('success/', views.success_page, name='success_page'),  # Success page after prediction
    path('traffic_data/', views.traffic_data_list, name='traffic_data_list'),


] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)