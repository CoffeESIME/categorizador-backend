from django.urls import path
from .views import MultiFileUploadView, PendingFilesView, MetadataProcessingView, UnconnectedNodesView
from .llm_views import LLMProcessView

urlpatterns = [
    path('files/upload', MultiFileUploadView.as_view(), name='file_upload'),
    path('files/', PendingFilesView.as_view(), name='pending_files'),
    path('llm/process', LLMProcessView.as_view(), name='llm_process'),
    path('metadata/save', MetadataProcessingView.as_view(), name ='metadata_process'),
    path('metadata/unconnected-nodes', UnconnectedNodesView.as_view(), name='metadata/unconnected-nodes')
]
