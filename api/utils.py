def serialize_neo4j_node(node_data):
    if not node_data:
        return {}
    serialized_node = {}
    for key, value in node_data.items():
        if hasattr(value, 'isoformat'):
            serialized_node[key] = value.isoformat()
        elif isinstance(value, (list, dict)): # Podrías necesitar manejo recursivo para listas/dicts anidados
            serialized_node[key] = str(value) # o un manejo más específico
        else:
            serialized_node[key] = value
    return serialized_node