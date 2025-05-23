import os
import shutil
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings

# Importar cliente Weaviate y la clase Filter
from api.weaviate_client import CLIENT as weaviate_client
from weaviate.classes.query import Filter # <--- Importar Filter

# Importar modelos Django
from api.models import UploadedFile, FileMetadataModel

# Importar driver Neo4j y función para cerrar
from api.neo4j_client import driver as neo4j_driver, close_driver as close_neo4j_connection

class Command(BaseCommand):
    help = 'Resetea la aplicación: vacía Neo4j, SQLite, Weaviate y elimina los archivos subidos.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--confirm',
            action='store_true',
            help='Confirma la ejecución de la acción destructiva.',
        )

    def handle(self, *args, **options):
        if not options['confirm']:
            self.stdout.write(self.style.WARNING(
                "Este comando es destructivo y borrará todos los datos de Neo4j, SQLite, Weaviate "
                "y los archivos en la carpeta 'uploads'."
            ))
            self.stdout.write(self.style.WARNING(
                "Ejecuta de nuevo con la opción --confirm para proceder."
            ))
            return

        self.stdout.write(self.style.WARNING("Iniciando el reseteo de la aplicación..."))

        try:
            # 1. Vaciar Neo4j
            try:
                self.stdout.write("Vaciando base de datos Neo4j...")
                with neo4j_driver.session() as session:
                    session.run("MATCH (n) DETACH DELETE n")
                self.stdout.write(self.style.SUCCESS("Neo4j vaciada correctamente."))
            except Exception as e:
                raise CommandError(f"Error al vaciar Neo4j: {e}")

            # 2. Vaciar tablas de SQLite
            try:
                self.stdout.write("Vaciando tablas de SQLite (UploadedFile, FileMetadataModel)...")
                UploadedFile.objects.all().delete()
                FileMetadataModel.objects.all().delete()
                self.stdout.write(self.style.SUCCESS("Tablas de SQLite vaciadas correctamente."))
            except Exception as e:
                raise CommandError(f"Error al vaciar SQLite: {e}")

            # 3. Vaciar Weaviate
            try:
                self.stdout.write("Vaciando colecciones de Weaviate...")
                collection_names = ["Imagenes", "Textos", "Audio", "Video", "PdfChunks"]
                
                # Definir un filtro que probablemente coincida con todos los objetos
                # Asumimos que la propiedad 'doc_id' es de tipo texto y existe en la mayoría de los objetos.
                # Si 'doc_id' no siempre existe o no es texto, podrías necesitar un filtro diferente,
                # o filtrar por una propiedad que sí sea universal en tus datos.
                # Filter.by_property("doc_id").is_not_none() también es una buena opción.
                match_all_filter = Filter.by_property("doc_id").like("*")

                for name in collection_names:
                    try:
                        if weaviate_client.collections.exists(name):
                            collection = weaviate_client.collections.get(name)
                            # Usar el filtro definido en lugar de where=None
                            collection.data.delete_many(where=match_all_filter)
                            self.stdout.write(self.style.SUCCESS(f"Objetos de la colección '{name}' en Weaviate eliminados."))
                        else:
                            self.stdout.write(self.style.NOTICE(f"La colección '{name}' no existe en Weaviate, omitiendo."))
                    except Exception as e_coll:
                        self.stdout.write(self.style.WARNING(f"Advertencia al procesar la colección '{name}' en Weaviate: {e_coll}"))
                self.stdout.write(self.style.SUCCESS("Colecciones de Weaviate procesadas."))
            except Exception as e:
                raise CommandError(f"Error al procesar Weaviate: {e}")

            # 4. Eliminar archivos de la carpeta 'uploads'
            try:
                uploads_path = settings.MEDIA_ROOT
                self.stdout.write(f"Eliminando contenido de la carpeta de uploads: {uploads_path}...")
                if os.path.exists(uploads_path):
                    subdirs_to_clear = ['images', 'videos', 'audio', 'documents', 'texts', 'others']
                    
                    for subdir_name in subdirs_to_clear:
                        dir_path = os.path.join(uploads_path, subdir_name)
                        if os.path.isdir(dir_path):
                            shutil.rmtree(dir_path)
                            os.makedirs(dir_path)
                            self.stdout.write(self.style.SUCCESS(f"Directorio '{dir_path}' limpiado y recreado."))
                    
                    for item in os.listdir(uploads_path):
                        item_path = os.path.join(uploads_path, item)
                        if os.path.isfile(item_path) and item.endswith(".json"): # Elimina todos los JSON en la raíz de uploads
                            os.remove(item_path)
                            self.stdout.write(self.style.SUCCESS(f"Archivo JSON '{item_path}' eliminado."))
                else:
                    self.stdout.write(self.style.WARNING(f"La carpeta de uploads '{uploads_path}' no existe."))
                self.stdout.write(self.style.SUCCESS("Archivos de uploads eliminados y directorios base recreados."))
            except Exception as e:
                raise CommandError(f"Error al eliminar archivos de uploads: {e}")

        except CommandError as e: # Re-lanzar CommandError para que Django lo maneje.
            raise e
        except Exception as e: # Capturar otros errores inesperados.
            raise CommandError(f"Un error inesperado ocurrió durante el reseteo: {e}")
        finally:
            self.stdout.write("Cerrando conexiones...")
            try:
                if weaviate_client and hasattr(weaviate_client, 'is_connected') and weaviate_client.is_connected():
                    weaviate_client.close()
                    self.stdout.write(self.style.SUCCESS("Conexión Weaviate cerrada."))
                elif weaviate_client and hasattr(weaviate_client, 'close'): # Intento genérico de cierre
                     weaviate_client.close()
                     self.stdout.write(self.style.SUCCESS("Conexión Weaviate cerrada (intento genérico)."))
            except Exception as e:
                self.stdout.write(self.style.WARNING(f"Error al cerrar conexión Weaviate: {e}"))
            
            try:
                # Usar la función close_driver importada de tu neo4j_client.py
                close_neo4j_connection()
                self.stdout.write(self.style.SUCCESS("Conexión Neo4j cerrada."))
            except Exception as e:
                self.stdout.write(self.style.WARNING(f"Error al cerrar conexión Neo4j: {e}"))

        self.stdout.write(self.style.SUCCESS("Reseteo de la aplicación completado."))