import json

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from ..neo4j_client import driver
class GraphView(APIView):
    """
    Endpoint para obtener un subgrafo filtrado.
    Parámetros opcionales (query parameters):
      - startingNodeId: id del nodo de inicio.
      - maxDepth: profundidad de la búsqueda (por defecto 2).
      - relationshipType: tipo de relación para filtrar.
      - nodeLabels: etiquetas (separadas por coma) para filtrar nodos.
      - limit: número máximo de nodos a retornar (por defecto 100).
    
    Retorna un objeto JSON con:
      { "nodes": [...], "edges": [...] }
    """
    def get(self, request, *args, **kwargs):
        starting_node_id = request.query_params.get("startingNodeId", None)
        try:
            max_depth = int(request.query_params.get("maxDepth", 2))
        except ValueError:
            max_depth = 2
        relationship_type = request.query_params.get("relationshipType", None)
        node_labels_str = request.query_params.get("nodeLabels", "")
        node_labels = (
            [label.strip() for label in node_labels_str.split(",") if label.strip()]
            if node_labels_str
            else []
        )
        try:
            limit = int(request.query_params.get("limit", 100))
        except ValueError:
            limit = 100

        nodes = []
        edges = []

        with driver.session() as session:
            if starting_node_id:
                # Buscar subgrafo a partir de un nodo de inicio, excluyendo nodos de tipo NodeType
                query_nodes = f"""
                    MATCH (start)
                    WHERE start.id = $startingNodeId
                      AND NOT start:NodeType
                    WITH start
                    MATCH (start)-[r*0..$maxDepth]-(n)
                    WHERE NOT n:NodeType
                """
                if node_labels:
                    query_nodes += " AND ANY(label IN labels(n) WHERE label IN $nodeLabels) "
                query_nodes += " RETURN DISTINCT n, labels(n) as labels"
                result = session.run(
                    query_nodes,
                    startingNodeId=starting_node_id,
                    maxDepth=max_depth,
                    nodeLabels=node_labels,
                )
            else:
                # Sin nodo de inicio: buscar nodos, excluyendo NodeType, con límite
                query_nodes = "MATCH (n) WHERE NOT n:NodeType "
                if node_labels:
                    query_nodes += " AND ANY(label IN labels(n) WHERE label IN $nodeLabels) "
                query_nodes += " RETURN n, labels(n) as labels LIMIT $limit"
                result = session.run(query_nodes, nodeLabels=node_labels, limit=limit)

# ... (dentro de GraphView) ...
            for record in result:
                node_data = record.get("n")
                node = {}
                if node_data:
                    for key, value in node_data.items():
                        # Convertir tipos de fecha/hora de Neo4j a string
                        if hasattr(value, 'isoformat'): # Verifica si es un objeto de fecha/hora compatible
                            node[key] = value.isoformat()
                        else:
                            node[key] = value
                
                # Usamos 'doc_id' si existe, o 'id' como respaldo para identificar el nodo
                node_id = node.get("doc_id") or node.get("id")
                node["nodeId"] = node_id # Asegurarse de que nodeId se asigna después de procesar las propiedades
                node["labels"] = record.get("labels")
                nodes.append(node)

            # Obtener las relaciones (edges) entre los nodos recuperados
            node_ids = [node["nodeId"] for node in nodes if node.get("nodeId")]
            if node_ids:
                query_edges = """
                    MATCH (a)-[r]->(b)
                    WHERE (a.doc_id IN $nodeIds OR a.id IN $nodeIds)
                      AND (b.doc_id IN $nodeIds OR b.id IN $nodeIds)
                """
                if relationship_type:
                    query_edges += " AND type(r) = $relationshipType "
                query_edges += " RETURN a, b, type(r) as relation"
                
                params = {"nodeIds": node_ids}
                if relationship_type:
                    params["relationshipType"] = relationship_type
                
                edge_result = session.run(query_edges, **params)
                for record in edge_result:
                    a_data = record.get("a")
                    b_data = record.get("b")
                    
                    a_node = {}
                    if a_data:
                        for key, value in a_data.items():
                            if hasattr(value, 'isoformat'):
                                a_node[key] = value.isoformat()
                            else:
                                a_node[key] = value
                                
                    b_node = {}
                    if b_data:
                        for key, value in b_data.items():
                            if hasattr(value, 'isoformat'):
                                b_node[key] = value.isoformat()
                            else:
                                b_node[key] = value
                    
                    source = a_node.get("doc_id") or a_node.get("id")
                    target = b_node.get("doc_id") or b_node.get("id")
                    edges.append({
                        "source": source,
                        "target": target,
                        "relation": record.get("relation")
                    })

        return Response({"nodes": nodes, "edges": edges}, status=status.HTTP_200_OK)


def serialize_neo4j_value(value):
    """Serializa valores de Neo4j, especialmente tipos temporales."""
    if hasattr(value, 'isoformat'): # Para DateTime, Date, Time, etc.
        return value.isoformat()
    # Puedes añadir más conversiones aquí para tipos espaciales de Neo4j, etc.
    elif isinstance(value, list):
        return [serialize_neo4j_value(item) for item in value]
    elif isinstance(value, dict):
        return {k: serialize_neo4j_value(v) for k, v in value.items()}
    return value

def serialize_neo4j_node_properties(node_data):
    if not node_data:
        return {}
    serialized_node = {}
    for key, value in node_data.items():
        serialized_node[key] = serialize_neo4j_value(value)
    return serialized_node


class AdvancedGraphSearchView(APIView):
    def post(self, request, *args, **kwargs):
        try:
            params = request.data
            
            # Validar parámetros básicos (puedes usar un Serializer de DRF para esto)
            # Por ahora, una validación simple
            if not isinstance(params, dict):
                return Response({"error": "El cuerpo de la solicitud debe ser un objeto JSON."},
                                status=status.HTTP_400_BAD_REQUEST)

            # --- Construcción de la Consulta Cypher ---
            # Esta parte será compleja y requerirá construir la consulta dinámicamente
            # basándose en los parámetros.
            
            cypher_query_parts = []
            cypher_params = {}
            
            # 1. Nodos de Inicio (MATCH inicial)
            # Esta es la parte más compleja de generalizar. Podrías empezar con un MATCH (n) general
            # y luego filtrar, o construir cláusulas MATCH más específicas si hay start_nodes.
            # Por simplicidad, aquí asumiremos un inicio más general o un filtrado posterior.
            
            # Ejemplo de manejo de start_nodes (simplificado):
            # Si se proveen start_nodes, podrías hacer un UNWIND y MATCH
            # O construir un WHERE con ORs para los IDs/propiedades de los nodos de inicio.

            # 2. Criterios de Coincidencia para Nodos
            node_alias = "n" # Alias para los nodos principales en el path
            start_node_alias = "start_node"
            end_node_alias = "end_node" # Alias para los nodos finales del path
            rel_alias = "r" # Alias para las relaciones

            # MATCH clause
            # Determinar la profundidad del path
            min_depth = params.get("traversal_options", {}).get("min_depth", 0)
            max_depth = params.get("traversal_options", {}).get("max_depth", 3)
            
            path_pattern = f"({start_node_alias})-[{rel_alias}*"""
            if min_depth is not None and max_depth is not None:
                 path_pattern += f"{min_depth}..{max_depth}"
            elif max_depth is not None:
                 path_pattern += f"..{max_depth}"
            else: # Solo min_depth o ninguno (usa un default razonable)
                 path_pattern += f"{min_depth}.." # O simplemente * si min_depth es 0
            path_pattern += f"]-({end_node_alias})"

            cypher_query_parts.append(f"MATCH path = {path_pattern}")

            # WHERE clause
            where_clauses = []

            # Filtrado de Nodos de Inicio (start_node_alias)
            start_nodes_criteria = params.get("start_nodes", [])
            if start_nodes_criteria:
                start_node_conditions = []
                for i, criteria in enumerate(start_nodes_criteria):
                    alias_param = f"start_crit_{i}"
                    condition_parts = []
                    if "id" in criteria:
                        condition_parts.append(f"{start_node_alias}.id = ${alias_param}_id")
                        cypher_params[f"{alias_param}_id"] = criteria["id"]
                    if "label" in criteria:
                        condition_parts.append(f"'{criteria['label']}' IN labels({start_node_alias})")
                    if "properties" in criteria:
                        for p_key, p_value in criteria["properties"].items():
                            prop_param_name = f"{alias_param}_prop_{p_key}"
                            condition_parts.append(f"{start_node_alias}.`{p_key}` = ${prop_param_name}")
                            cypher_params[prop_param_name] = p_value
                    if condition_parts:
                         start_node_conditions.append(f"({' AND '.join(condition_parts)})")
                if start_node_conditions:
                    where_clauses.append(f"({' OR '.join(start_node_conditions)})")


            # Filtrado de Nodos en el Path (n) - se aplica a start_node y end_node
            match_criteria = params.get("match_criteria", {})
            node_labels = match_criteria.get("node_labels", [])
            if node_labels:
                # Aplicar a ambos nodos del path (o a todos los nodos si UNWIND)
                for alias in [start_node_alias, end_node_alias]:
                    label_conditions = [f"'{label}' IN labels({alias})" for label in node_labels]
                    if label_conditions: # Si solo algunas etiquetas aplican a un tipo de nodo
                        where_clauses.append(f"({' OR '.join(label_conditions)})")


            node_prop_filters = match_criteria.get("node_properties_filter", [])
            if node_prop_filters:
                for alias in [start_node_alias, end_node_alias]: # Aplicar a ambos extremos del path
                    for i, prop_filter in enumerate(node_prop_filters):
                        filter_param_name = f"{alias}_node_prop_filter_{i}"
                        # Cuidado con la inyección de Cypher si 'key' u 'operator' vienen del usuario sin sanear
                        # Es mejor tener un mapeo de operadores seguros
                        operator = prop_filter.get("operator", "=")
                        # Validar/mapear operadores aquí para seguridad
                        safe_operators = {
                            "=": "=", ">": ">", "<": "<", ">=": ">=", "<=": "<=",
                            "CONTAINS": "CONTAINS", "STARTS WITH": "STARTS WITH", "ENDS WITH": "ENDS WITH",
                            "IN": "IN"
                        }
                        if operator not in safe_operators:
                            continue # O lanzar error
                        
                        where_clauses.append(f"{alias}.`{prop_filter['key']}` {safe_operators[operator]} ${filter_param_name}")
                        cypher_params[filter_param_name] = prop_filter['value']
            
            # Filtrado de Relaciones (r)
            # Esto requiere iterar sobre las relaciones en el path si quieres filtrar cada una.
            # UNWIND relationships(path) as r_in_path
            # WITH start_node, end_node, r_in_path, path
            # WHERE ...
            # Esto añade complejidad. Por ahora, un filtro general para los tipos de relación:
            rel_types = match_criteria.get("relationship_types", [])
            if rel_types:
                 # Esta condición se aplicaría a CUALQUIER relación en el path si es *
                 # Si el path es de longitud fija, puedes ser más específico.
                 # Para paths de longitud variable, se necesita UNWIND o ALL/ANY en predicados.
                rel_type_conditions = [f"type({rel_alias_in_path}) = '{rt}'" for rt in rel_types]
                # Esto necesitaría un `UNWIND relationships(path) AS rel_alias_in_path` y aplicar el WHERE
                # O usar un predicado:
                if rel_types:
                    rel_type_check = " AND ".join([f"ANY(r_in_path IN relationships(path) WHERE type(r_in_path) = '{rt}')" for rt in rel_types])
                    # Esto es más complejo de lo que parece para múltiples tipos opcionales.
                    # Una forma más simple si quieres que *todas* las relaciones sean de ciertos tipos:
                    # where_clauses.append(f"ALL(r_in_path IN relationships(path) WHERE type(r_in_path) IN $rel_types_param)")
                    # cypher_params["rel_types_param"] = rel_types
                    # O si *alguna* relación debe ser de un tipo (lo cual es raro para un path entero):
                    # where_clauses.append(f"ANY(r_in_path IN relationships(path) WHERE type(r_in_path) IN $rel_types_param)")
                    # Por ahora, simplifiquemos: si se especifica un tipo, asumimos que se aplica a *alguna* relacion del path
                    # (esto puede no ser lo que se quiere siempre)
                    # Si la longitud del path es 1, entonces es type(r)
                    if min_depth == 1 and max_depth == 1:
                        rel_type_conditions_single = [f"type({rel_alias}) = '{rt}'" for rt in rel_types]
                        if rel_type_conditions_single:
                            where_clauses.append(f"({' OR '.join(rel_type_conditions_single)})")
                    # Para paths variables, filtrar tipos de relación es más complejo y usualmente se hace
                    # con UNWIND o con funciones de lista de predicados.

            # Dirección de la relación (ya implícita en el MATCH start-[...]-end)
            # Si quieres cambiar la direccionalidad dinámicamente, la construcción del MATCH cambia.

            if where_clauses:
                cypher_query_parts.append("WHERE " + " AND ".join(where_clauses))

            # RETURN clause
            # Desestructurar el path para obtener nodos y relaciones individualmente
            cypher_query_parts.append("WITH nodes(path) AS path_nodes, relationships(path) AS path_rels")
            cypher_query_parts.append("UNWIND path_nodes AS n_in_path")
            cypher_query_parts.append("UNWIND path_rels AS r_in_path")
            
            # Aplicar filtros de propiedades a r_in_path si es necesario aquí
            # ...
            
            # Decidir qué retornar
            result_options = params.get("result_options", {})
            return_nodes_str = "RETURN DISTINCT n_in_path" # Por defecto, todos los nodos del path
            return_rels_str = ", COLLECT(DISTINCT r_in_path) AS relationships" if result_options.get("return_edges", True) else ""
            
            # Proyección de propiedades de nodo
            # node_props_to_return = result_options.get("node_properties_to_return")
            # if node_props_to_return:
            #     prop_map = ", ".join([f"{p}: n_in_path.`{p}`" for p in node_props_to_return])
            #     return_nodes_str = f"RETURN DISTINCT n_in_path {{ .id, labels: labels(n_in_path), {prop_map} }}"
            # else: # Retornar todas las propiedades (o las serializadas)
            #     return_nodes_str = f"RETURN DISTINCT n_in_path"


            # cypher_query_parts.append(f"{return_nodes_str} {return_rels_str}")
            # Esta estructura de RETURN es un poco compleja para combinar nodos y relaciones de forma distintiva.
            # Usualmente se retornan nodos y relaciones por separado o como un path.
            # Para la visualización:
            cypher_query_parts.append("RETURN DISTINCT n_in_path, r_in_path")


            # Límites
            # El límite se aplica al final. Puede ser complejo aplicar límites separados a nodos y relaciones
            # de esta manera sin subconsultas.
            # limit_nodes = result_options.get("limit_nodes", 100)
            # limit_edges = result_options.get("limit_edges", 200) # No es trivial aplicarlo directamente aquí
            # cypher_query_parts.append(f"LIMIT {limit_nodes}") # Esto limitaría el número total de filas n_in_path, r_in_path

            final_query = "\n".join(cypher_query_parts)
            print(f"Generated Cypher: {final_query}") # Reemplazo de self.stdout.write
            print(f"Parameters: {cypher_params}")

            nodes = {} # Usar un diccionario para evitar duplicados por ID
            edges = []
            
            with driver.session() as session:
                result = session.run(final_query, **cypher_params)
                for record in result:
                    node_data = record.get("n_in_path")
                    if node_data:
                        processed_node_props = serialize_neo4j_node_properties(node_data.get('_properties', {})) # _properties es donde neo4j.Result anida las props
                        node_id = node_data.get('id') # o node_data.element_id si usas IDs internos
                        
                        # Si 'id' no está en las propiedades, intenta obtenerlo del elemento mismo
                        if not node_id and hasattr(node_data, 'id'): # id del objeto nodo (entero)
                             node_id = str(node_data.id) 
                        elif not node_id and 'id' in processed_node_props: # id de tus propiedades
                             node_id = processed_node_props['id']


                        if node_id and node_id not in nodes:
                            nodes[node_id] = {
                                "id": node_id, # Asegúrate que este es el ID que usas para identificar unívocamente
                                "labels": list(node_data.labels),
                                "properties": processed_node_props
                            }

                    rel_data = record.get("r_in_path")
                    if rel_data:
                        # Para relaciones, los IDs de start y end son internos de Neo4j
                        # Necesitas mapearlos a tus IDs de propiedad si los usas para la visualización.
                        # Esto es más fácil si buscas los nodos conectados por sus propiedades `id`
                        start_node_element_id = str(rel_data.start_node.id) # ID interno del nodo Neo4j
                        end_node_element_id = str(rel_data.end_node.id) # ID interno del nodo Neo4j
                        
                        # Necesitarías una forma de mapear estos IDs internos a tus IDs de propiedad ('id' o 'doc_id')
                        # si no los tienes ya en `nodes`.
                        # O, mejor, la consulta debería devolver los IDs de propiedad de los nodos de la relación.
                        # Esto se complica. Una forma más simple es que el front-end reconstruya desde la lista de nodos.

                        # Por ahora, guardamos la relación con los IDs de los nodos que ya deberíamos tener en `nodes`
                        # Esto asume que los nodos de la relación ya fueron procesados.
                        # Se necesitará obtener el 'id' de propiedad de start_node y end_node.
                        # La consulta actual no facilita esto directamente en el `r_in_path` solo.

                        # Una mejor forma de retornar para visualización:
                        # RETURN n, r, m (nodo_inicio, relacion, nodo_fin)
                        # Luego procesas n, r, m en cada record.

                        # Replantear el RETURN y el procesamiento para que sea más fácil construir el grafo:
                        # MATCH (n1)-[r]->(n2) WHERE ... RETURN n1, r, n2
                        # Esto es mucho más simple si los filtros se pueden aplicar así.
                        
                        # Por ahora, vamos a omitir el procesamiento de relaciones detallado
                        # ya que la consulta actual con UNWIND lo hace complicado de reconstruir sin
                        # información adicional o una estructura de query diferente.
                        # La `GraphView` original tiene un enfoque más directo para esto.
                        pass


            # La consulta actual devuelve n_in_path y r_in_path, que no es ideal para reconstruir un grafo
            # de nodos y aristas únicos.
            # Vamos a simplificar la consulta para que se parezca más a tu GraphView original
            # y luego añadimos filtros.

            # --- REVISIÓN DE LA CONSTRUCCIÓN DE LA CONSULTA (Enfoque más simple) ---
            
            nodes_map = {}
            edges_list = []
            
            match_parts = []
            where_conditions = []
            query_parameters = {}

            start_nodes_config = params.get("start_nodes", [])
            path_min_depth = params.get("traversal_options", {}).get("min_depth", 0)
            path_max_depth = params.get("traversal_options", {}).get("max_depth", 2) 
            result_options = params.get("result_options", {}) # <--- Asegúrate de que result_options se defina aquí
            match_criteria = params.get("match_criteria", {}) # <--- Asegúrate de que match_criteria se defina aquí

            
            # Construir el patrón de path
            # (start_node_alias)-[rel_alias*min..max]-(end_node_alias)
            # Si no hay start_nodes_config, el match es más general: MATCH (n)
            if start_nodes_config:
                # Unir los nodos de inicio con OR y hacer el path desde ellos
                start_node_matches = []
                for i, sn_config in enumerate(start_nodes_config):
                    sn_alias = f"sn_{i}"
                    sn_match_parts = []
                    sn_where_parts = []
                    label_part = f":`{sn_config['label']}`" if "label" in sn_config else ""
                    sn_match_parts.append(f"({sn_alias}{label_part})")
                    
                    if "id" in sn_config:
                        param_name = f"sn_{i}_id"
                        sn_where_parts.append(f"{sn_alias}.id = ${param_name}")
                        query_parameters[param_name] = sn_config["id"]
                    if "properties" in sn_config:
                        for p_key, p_value in sn_config["properties"].items():
                            param_name = f"sn_{i}_prop_{p_key}"
                            sn_where_parts.append(f"{sn_alias}.`{p_key}` = ${param_name}")
                            query_parameters[param_name] = p_value
                    
                    path_query = f"MATCH path = ({sn_alias})-[r_path*{path_min_depth}..{path_max_depth}]-(m_path) "
                    if sn_where_parts:
                        path_query += "WHERE " + " AND ".join(sn_where_parts) + " "
                    
                    # Aplicar filtros de match_criteria a m_path y r_path aquí
                    sub_where_clauses_for_path = self._build_path_filters(
                        "m_path", "r_path", match_criteria, query_parameters, f"path_{i}_"
                    )
                    if sub_where_clauses_for_path:
                         path_query += ("AND " if sn_where_parts else "WHERE ") + " AND ".join(sub_where_clauses_for_path)


                    path_query += "RETURN nodes(path) AS path_nodes, relationships(path) AS path_rels"
                    match_parts.append(path_query)

                final_query = " UNION ".join(match_parts) # Si hay múltiples start_nodes, UNION los paths
            else:
                # Búsqueda general, no anclada a nodos de inicio específicos
                # Esto es más como tu GraphView actual, pero con más filtros.
                # MATCH (n)-[r*min..max]-(m)
                # O simplemente MATCH (n), luego MATCH (n)-[r]->(m) para obtener relaciones si es necesario
                # Para simplificar, si no hay start_nodes, buscamos nodos que cumplan criterios
                # y luego sus relaciones.

                node_filters_where = self._build_path_filters(
                    "n", None, match_criteria, query_parameters, "node_"
                ) # Solo filtros de nodo por ahora
                
                query_str = "MATCH (n) "
                if node_filters_where:
                    query_str += "WHERE " + " AND ".join(node_filters_where) + " "
                
                # Decidimos qué retornar: por ahora nodos y luego sus relaciones
                # Esto es más fácil de manejar que paths complejos para la salida de grafo JSON.
                # Primero obtenemos los nodos
                query_str_nodes = query_str + "RETURN DISTINCT n "
                limit_nodes = result_options.get("limit_nodes", 100)
                query_str_nodes += f"LIMIT {limit_nodes}"
                
                final_query = query_str_nodes # Ejecutaremos esto primero.

            if not final_query: # Si no se pudo construir una consulta (ej. start_nodes vacío y no se implementó alternativa)
                 return Response({"nodes": [], "edges": []}, status=status.HTTP_200_OK)


            print(f"Generated Cypher: {final_query}") # O logger.debug(f"Generated Cypher: {final_query}")
            # self.stdout.write(f"Parameters: {cypher_params}") se convierte en:
            print(f"Parameters: {cypher_params}") 

            collected_nodes_from_paths = {} # {neo4j_element_id: processed_node_dict}
            collected_rels_from_paths = {} # {neo4j_element_id: processed_rel_dict}

            with driver.session() as session:
                if start_nodes_config: # Lógica de paths
                    result = session.run(final_query, **query_parameters)
                    for record in result:
                        path_nodes_data = record.get("path_nodes", [])
                        path_rels_data = record.get("path_rels", [])

                        for node_data in path_nodes_data:
                            if node_data.element_id not in collected_nodes_from_paths:
                                props = serialize_neo4j_node_properties(node_data.get('_properties', {}))
                                node_render_id = props.get('id', props.get('doc_id', node_data.element_id))
                                collected_nodes_from_paths[node_data.element_id] = {
                                    "id": node_render_id,
                                    "labels": list(node_data.labels),
                                    "properties": props,
                                    "_neo4j_id": node_data.element_id # Guardar para mapeo de relaciones
                                }
                        for rel_data in path_rels_data:
                            if rel_data.element_id not in collected_rels_from_paths:
                                props = serialize_neo4j_node_properties(rel_data.get('_properties', {}))
                                collected_rels_from_paths[rel_data.element_id] = {
                                    "id": rel_data.element_id,
                                    "type": rel_data.type,
                                    "properties": props,
                                    "start_node_neo4j_id": rel_data.start_node.element_id,
                                    "end_node_neo4j_id": rel_data.end_node.element_id
                                }
                    
                    # Convertir a formato final
                    final_nodes_list = list(collected_nodes_from_paths.values())
                    final_edges_list = []
                    for rel_dict in collected_rels_from_paths.values():
                        start_node_internal_id = rel_dict["start_node_neo4j_id"]
                        end_node_internal_id = rel_dict["end_node_neo4j_id"]
                        
                        # Encontrar el ID de renderizado (el que usa el frontend)
                        source_render_id = collected_nodes_from_paths.get(start_node_internal_id, {}).get("id")
                        target_render_id = collected_nodes_from_paths.get(end_node_internal_id, {}).get("id")

                        if source_render_id and target_render_id:
                            final_edges_list.append({
                                "source": source_render_id,
                                "target": target_render_id,
                                "relation": rel_dict["type"],
                                "properties": rel_dict["properties"],
                                "_neo4j_id": rel_dict["id"]
                            })
                    
                    return Response({"nodes": final_nodes_list, "edges": final_edges_list}, status=status.HTTP_200_OK)

                else: # Lógica de búsqueda general de nodos primero, luego relaciones
                    node_results = session.run(final_query, **query_parameters)
                    temp_nodes_map = {} # usa el 'id' de propiedad como clave
                    node_neo4j_ids_for_rels = []

                    for record in node_results:
                        node_data = record.get("n")
                        if node_data:
                            props = serialize_neo4j_node_properties(node_data.get('_properties', {}))
                            node_render_id = props.get('id', props.get('doc_id', node_data.element_id))
                            
                            if node_render_id not in temp_nodes_map:
                                temp_nodes_map[node_render_id] = {
                                    "id": node_render_id,
                                    "labels": list(node_data.labels),
                                    "properties": props
                                }
                                node_neo4j_ids_for_rels.append(props.get('id', node_data.element_id)) # Usa el ID que usas en las relaciones

                    # Ahora obtener relaciones para estos nodos
                    final_edges_list = []
                    if temp_nodes_map and result_options.get("return_edges", True):
                        # Usar los IDs de propiedad para el MATCH de relaciones
                        rel_query_parameters = {"nodeDbIds": node_neo4j_ids_for_rels} 
                        rel_match_str = "MATCH (n1)-[r]-(n2) WHERE n1.id IN $nodeDbIds AND n2.id IN $nodeDbIds "
                        
                        # Aplicar filtros de relación aquí
                        rel_filters_where = self._build_path_filters(
                            None, "r", match_criteria, rel_query_parameters, "rel_"
                        )
                        if rel_filters_where:
                             rel_match_str += "AND " + " AND ".join(rel_filters_where)
                        
                        rel_match_str += "RETURN DISTINCT n1, r, n2 "
                        limit_edges = result_options.get("limit_edges", 200)
                        rel_match_str += f"LIMIT {limit_edges}"
                        print(f"Generated Cypher: {final_query}") 
                        print(f"Final Query (Stage 2 - Edges): {rel_match_str}")
                        print(f"Parameters (Edges): {rel_query_parameters}")

                        edge_results = session.run(rel_match_str, **rel_query_parameters)
                        for record in edge_results:
                            n1_data = record.get("n1")
                            r_data = record.get("r")
                            n2_data = record.get("n2")

                            if n1_data and r_data and n2_data:
                                n1_props = serialize_neo4j_node_properties(n1_data.get('_properties', {}))
                                n2_props = serialize_neo4j_node_properties(n2_data.get('_properties', {}))
                                r_props = serialize_neo4j_node_properties(r_data.get('_properties', {}))

                                source_id = n1_props.get('id', n1_props.get('doc_id', n1_data.element_id))
                                target_id = n2_props.get('id', n2_props.get('doc_id', n2_data.element_id))
                                
                                # Asegurarse de que los nodos de la relación estén en nuestro conjunto de nodos
                                if source_id in temp_nodes_map and target_id in temp_nodes_map:
                                    final_edges_list.append({
                                        "source": source_id,
                                        "target": target_id,
                                        "relation": r_data.type,
                                        "properties": r_props
                                    })
                    
                    return Response({"nodes": list(temp_nodes_map.values()), "edges": final_edges_list}, status=status.HTTP_200_OK)


        except Exception as e:
            # self.stderr.write(traceback.format_exc()) se convierte en:
            print(f"ERROR: {traceback.format_exc()}") # O logger.error(traceback.format_exc())
            return Response({"error": str(e), "trace": traceback.format_exc()},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def _build_path_filters(self, node_alias, rel_alias, match_criteria, query_params, param_prefix=""):
        """Helper para construir cláusulas WHERE para nodos y relaciones en un path."""
        where_clauses = []
        
        # Filtros de etiqueta de nodo (si node_alias está presente)
        if node_alias:
            node_labels = match_criteria.get("node_labels", [])
            if node_labels:
                # Si un nodo DEBE tener TODAS las etiquetas especificadas:
                # label_conditions = [f"'{label}' IN labels({node_alias})" for label in node_labels]
                # where_clauses.append("(" + " AND ".join(label_conditions) + ")")
                # Si un nodo DEBE tener ALGUNA de las etiquetas especificadas (más común para filtro):
                label_conditions = [f"{node_alias}:`{label}`" for label in node_labels] # Usa sintaxis de match de etiqueta
                if label_conditions:
                    where_clauses.append("(" + " OR ".join(label_conditions) + ")")


        # Filtros de propiedades de nodo (si node_alias está presente)
        if node_alias:
            node_prop_filters = match_criteria.get("node_properties_filter", [])
            for i, prop_filter in enumerate(node_prop_filters):
                filter_param_name = f"{param_prefix}{node_alias}_prop_{i}"
                operator = prop_filter.get("operator", "=")
                key = prop_filter['key']
                value = prop_filter['value']
                
                # Mapeo de operadores seguros (expandir según sea necesario)
                safe_operators = {
                    "=": "=", "!=": "<>", ">": ">", "<": "<", ">=": ">=", "<=": "<=",
                    "CONTAINS": "CONTAINS", "STARTS WITH": "STARTS WITH", 
                    "ENDS WITH": "ENDS WITH", "IN": "IN"
                }
                if operator not in safe_operators:
                    self.stderr.write(f"Operador no seguro o desconocido: {operator}")
                    continue
                
                # Para el operador IN, el valor debe ser una lista
                if safe_operators[operator] == "IN" and not isinstance(value, list):
                    self.stderr.write(f"Valor para operador IN debe ser una lista para la clave {key}")
                    continue

                where_clauses.append(f"{node_alias}.`{key}` {safe_operators[operator]} ${filter_param_name}")
                query_params[filter_param_name] = value

        # Filtros de tipo de relación (si rel_alias está presente)
        if rel_alias:
            rel_types = match_criteria.get("relationship_types", [])
            if rel_types:
                # Si la relación DEBE ser DE ALGUNO de estos tipos
                type_conditions = [f"type({rel_alias}) = '{rt}'" for rt in rel_types]
                if type_conditions:
                    where_clauses.append("(" + " OR ".join(type_conditions) + ")")
        
        # Filtros de propiedades de relación (si rel_alias está presente)
        if rel_alias:
            rel_prop_filters = match_criteria.get("relationship_properties_filter", [])
            for i, prop_filter in enumerate(rel_prop_filters):
                filter_param_name = f"{param_prefix}{rel_alias}_prop_{i}"
                operator = prop_filter.get("operator", "=")
                key = prop_filter['key']
                value = prop_filter['value']

                safe_operators = {
                    "=": "=", "!=": "<>", ">": ">", "<": "<", ">=": ">=", "<=": "<=",
                    "CONTAINS": "CONTAINS", "STARTS WITH": "STARTS WITH", 
                    "ENDS WITH": "ENDS WITH", "IN": "IN"
                } # Reutilizar el mismo mapeo
                if operator not in safe_operators:
                    self.stderr.write(f"Operador no seguro o desconocido para relación: {operator}")
                    continue
                
                if safe_operators[operator] == "IN" and not isinstance(value, list):
                     self.stderr.write(f"Valor para operador IN (relación) debe ser una lista para la clave {key}")
                     continue

                where_clauses.append(f"{rel_alias}.`{key}` {safe_operators[operator]} ${filter_param_name}")
                query_params[filter_param_name] = value
        
        return where_clauses


