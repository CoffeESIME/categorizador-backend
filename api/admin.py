from django.contrib import admin
from .models import UploadedFile  # si tienes otro modelo, inclúyelo

admin.site.register(UploadedFile)
