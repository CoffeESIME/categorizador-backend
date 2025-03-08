# api/serializers.py
from rest_framework import serializers
from .models import UploadedFile

class UploadedFileSerializer(serializers.ModelSerializer):
    file_url = serializers.SerializerMethodField()

    class Meta:
        model = UploadedFile
        fields = ['id', 'original_name', 'file_type', 'size', 'upload_date', 'status', 'file_url']

    def get_file_url(self, obj):
        return obj.file.url
