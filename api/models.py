
import os
import time
from django.db import models
import json

def custom_upload_path(instance, filename):
    base, ext = os.path.splitext(filename)
    timestamp = str(int(time.time() * 1000000))  # Timestamp en microsegundos
    new_filename = f"{base}_{timestamp}{ext}"
    return  new_filename

class UploadedFile(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pendiente'),
        ('categorized', 'Procesado'),
        ('deleted', 'Eliminado'),
    ]
    
    file = models.FileField(upload_to=custom_upload_path)
    original_name = models.CharField(max_length=255, blank=True)
    file_type = models.CharField(max_length=50)
    size = models.PositiveIntegerField()
    upload_date = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pendiente')
    
    def save(self, *args, **kwargs):
        # Guardamos el nombre original solo al crear el objeto
        if not self.pk:
            self.original_name = self.file.name
        super().save(*args, **kwargs)

    def __str__(self):
        return self.original_name
# api/models.py

class FileMetadataModel(models.Model):
    """
    Modelo que guarda, para cada archivo, metadatos procesados.
    """
    uploaded_file = models.ForeignKey('api.UploadedFile', on_delete=models.CASCADE, null=True, blank=True)
    author = models.CharField(max_length=255, blank=True, null=True)
    title = models.CharField(max_length=255, blank=True, null=True)
    content = models.TextField(blank=True, null=True)
    analysis = models.TextField(blank=True, null=True)
    description = models.TextField(blank=True, null=True)

    # Si deseas guardar todo el JSON original:
    metadata_json = models.TextField(blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def set_metadata_json(self, data: dict):
        self.metadata_json = json.dumps(data)

    def get_metadata_json(self) -> dict:
        return json.loads(self.metadata_json) if self.metadata_json else {}

    def __str__(self):
        return f"Metadata for FileID={self.uploaded_file_id if self.uploaded_file else 'None'}"
