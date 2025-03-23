from django.urls import path
from .views import DeleteNodeConnectionView, GraphView, MultiFileUploadView, NodeConnectionsView, NodeCreationView, NodeTypesView, PendingFilesView, MetadataProcessingView, UnconnectedNodesView, UpdateNodeView, NodesByTypeView
from .llm_views import LLMProcessView

urlpatterns = [
    path('files/upload', MultiFileUploadView.as_view(), name='file_upload'),
    path('files/', PendingFilesView.as_view(), name='pending_files'),
    path('llm/process', LLMProcessView.as_view(), name='llm_process'),
    path('metadata/save', MetadataProcessingView.as_view(), name ='metadata_process'),
    path('metadata/unconnected-nodes', UnconnectedNodesView.as_view(), name='metadata/unconnected-nodes'),
    path('nodes/create-node', NodeCreationView.as_view(), name= 'node_creation'),
    path('graph', GraphView.as_view(), name='graph_view'),
path('nodes/node-types/<str:nodeType>/', NodesByTypeView.as_view(), name='nodes_by_type'),
    
    
    
    
    
    
    
    
    path('nodes/<str:nodeId>/connections', NodeConnectionsView.as_view(), name='node_connections'),
    path('nodes/<str:nodeId>/connections-actions', UpdateNodeView.as_view(), name='update_node_connection'),
    path('nodes/<str:nodeId>/connections-actions/<str:connectionNodeId>', DeleteNodeConnectionView.as_view(), name='delete_node_connection'),
    path('node-types', NodeTypesView.as_view(), name="node_types")
]
