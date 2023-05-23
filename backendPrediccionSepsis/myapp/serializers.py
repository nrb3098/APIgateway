from rest_framework.serializers import ModelSerializer
from .models import myApp

class myAppSerializer(ModelSerializer):
    class Meta:
        model = myApp
        fields = '__all__'
