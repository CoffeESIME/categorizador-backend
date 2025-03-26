import os
import time
from django.db import models
import json

def custom_upload_path(instance, filename):
    base, ext = os.path.splitext(filename)
    timestamp = str(int(time.time() * 1000000))  # Timestamp en microsegundos
    new_filename = f"{base}_{timestamp}{ext}"
    
    # Determinar la carpeta según el tipo de archivo
    file_type = instance.file_type if hasattr(instance, 'file_type') else ""
    
    if file_type.startswith('image/'):
        folder = 'images'
    elif file_type.startswith('video/'):
        folder = 'videos'
    elif file_type.startswith('audio/'):
        folder = 'audio'
    elif file_type in ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                      'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                      'application/vnd.ms-powerpoint', 'application/vnd.openxmlformats-officedocument.presentationml.presentation']:
        folder = 'documents'
    elif file_type.startswith('text/'):
        folder = 'texts'
    else:
        folder = 'others'
    
    # Crear la ruta completa
    upload_path = os.path.join( folder, new_filename)
    
    # Guardar para usarlo posteriormente en el campo file_location
    if hasattr(instance, '_file_location'):
        instance._file_location = upload_path
    
    return upload_path

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
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    file_location = models.CharField(max_length=255, blank=True)
    
    def save(self, *args, **kwargs):
        if not self.pk:
            self.original_name = self.file.name
            self._file_location = ''
        
        super().save(*args, **kwargs)
        
        # Si tenemos la ubicación del archivo y el campo file_location está vacío
        if hasattr(self, '_file_location') and self._file_location and not self.file_location:
            self.file_location = self._file_location
            type(self).objects.filter(pk=self.pk).update(file_location=self._file_location)

    def __str__(self):
        return self.original_name

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