# api/serializers.py
from rest_framework import serializers
from .models import UploadedFile
from .utils import generate_presigned_url

class UploadedFileSerializer(serializers.ModelSerializer):
    file_url = serializers.SerializerMethodField()

    class Meta:
        model = UploadedFile
        fields = ['id', 'original_name', 'file_type', 'size', 'upload_date', 'status', 'file_url', 'file_location']

    def get_file_url(self, obj):
        # Use MinIO presigned URL if file_location is set
        if obj.file_location:
            return generate_presigned_url(obj.file_location)
        # Fallback to Django file URL for legacy files
        if obj.file:
            return obj.file.url
        return None
