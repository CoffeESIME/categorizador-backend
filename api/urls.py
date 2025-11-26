from django.urls import path

from .views.files import (
    MultiFileUploadView,
    PendingFilesView,
    MetadataProcessingView,
    TextProcessView,
    TextMetadataProcessingView,
)
from .views.graph import GraphView, AdvancedGraphSearchView
from .views.nodes import (
    UnconnectedNodesView,
    NodeConnectionsView,
    NodeTypesView,
    NodeCreationView,
    NodesByTypeView,
    UpdateNodeView,
    DeleteNodeConnectionView,
    ConnectNodesView,
    ConnectUnconnectedNodeView,
)
from .llm_views import LLMProcessView, AudioMetadataCuratorView
from .rag_views import MultiModalSearchView

urlpatterns = [
    path('files/upload', MultiFileUploadView.as_view(), name='file_upload'),
    path('files/', PendingFilesView.as_view(), name='pending_files'),
    path('llm/process', LLMProcessView.as_view(), name='llm_process'),
    path('llm/audio-metadata', AudioMetadataCuratorView.as_view(), name='llm_audio_metadata'),
    path('metadata/save', MetadataProcessingView.as_view(), name='metadata_process'),
    path('metadata/unconnected-nodes', UnconnectedNodesView.as_view(), name='metadata/unconnected-nodes'),
    path('nodes/create-node', NodeCreationView.as_view(), name='node_creation'),
    path('graph', GraphView.as_view(), name='graph_view'),
    path('graph/search/', AdvancedGraphSearchView.as_view(), name='advanced_graph_search'),
    path('nodes/node-types/<str:nodeType>/', NodesByTypeView.as_view(), name='nodes_by_type'),
    path('nodes/<str:nodeId>/connections', NodeConnectionsView.as_view(), name='node_connections'),
    path('nodes/<str:nodeId>/connections-actions', UpdateNodeView.as_view(), name='update_node_connection'),
    path('nodes/create-relationship', ConnectNodesView.as_view(), name='connect_nodes'),
    path('nodes/delete-relationship', DeleteNodeConnectionView.as_view(), name='delete_node_connection'),
    path('node-types', NodeTypesView.as_view(), name='node_types'),
    path('nodes/connect-unconnected', ConnectUnconnectedNodeView.as_view(), name='connect_unconnected_node'),
    path('content/process', TextProcessView.as_view(), name='content_process'),
    path('content/metadata/save', TextMetadataProcessingView.as_view(), name='content_metadata_save'),
    path("search", MultiModalSearchView.as_view(), name="multimodal-search"),
]
