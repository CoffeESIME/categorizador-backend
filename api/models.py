
import os
import time
from django.db import models

def custom_upload_path(instance, filename):
    base, ext = os.path.splitext(filename)
    timestamp = str(int(time.time() * 1000000))  # Timestamp en microsegundos
    new_filename = f"{base}_{timestamp}{ext}"
    return os.path.join("uploads/", new_filename)

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
