from django import forms
from .models import TrafficImage

class TrafficImageForm(forms.ModelForm):
    class Meta:
        model = TrafficImage
        fields = ['image', 'location', 'latitude', 'longitude']  # Include latitude and longitude
