from django.db import models



class TrafficData(models.Model):
    vehicle_type = models.CharField(max_length=100)
    weather = models.CharField(max_length=50)
    economic_condition = models.CharField(max_length=50)
    day_of_week = models.CharField(max_length=20)
    hour_of_day = models.IntegerField()
    speed = models.FloatField()
    is_peak_hour = models.BooleanField()
    event_occurred = models.BooleanField()
    energy_consumption = models.FloatField()
    traffic_density = models.FloatField()

    def _str_(self):
        return f"Density: {self.traffic_density}, Speed: {self.speed}"