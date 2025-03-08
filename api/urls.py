# api/urls.py
from django.urls import path
from .views import MultiFileUploadView, PendingFilesView

urlpatterns = [
path('files/upload', MultiFileUploadView.as_view(), name='file_upload'),
    path('files/', PendingFilesView.as_view(), name='pending_files'),
]
