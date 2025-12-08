import os
import shutil
import boto3  # <--- Importar boto3
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings

# Importar cliente Weaviate y la clase Filter
from api.weaviate_client import CLIENT as weaviate_client
from weaviate.classes.query import Filter

# Importar modelos Django
from api.models import UploadedFile, FileMetadataModel

# Importar driver Neo4j y función para cerrar
from api.neo4j_client import driver as neo4j_driver, close_driver as close_neo4j_connection

class Command(BaseCommand):
    help = 'Resetea la aplicación: vacía Neo4j, SQLite, Weaviate y MinIO (S3).'

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
                "y los archivos en MinIO."
            ))
            self.stdout.write(self.style.WARNING(
                "Ejecuta de nuevo con la opción --confirm para proceder."
            ))
            return

        self.stdout.write(self.style.WARNING("Iniciando el reseteo TOTAL de la aplicación..."))

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
                self.stdout.write("Vaciando tablas de SQLite...")
                UploadedFile.objects.all().delete()
                FileMetadataModel.objects.all().delete()
                self.stdout.write(self.style.SUCCESS("Tablas de SQLite vaciadas correctamente."))
            except Exception as e:
                raise CommandError(f"Error al vaciar SQLite: {e}")

            # 3. Vaciar Weaviate
            try:
                self.stdout.write("Vaciando colecciones de Weaviate...")
                # Lista actualizada con tus colecciones reales (incluyendo Image, Video, etc.)
                collection_names = ["Image", "Video", "Audio", "Document", "Imagenes", "Textos", "PdfChunks"]
                
                # Intentamos borrar todo lo que tenga un doc_id (o sea, todo)
                match_all_filter = Filter.by_property("doc_id").like("*")

                for name in collection_names:
                    try:
                        if weaviate_client.collections.exists(name):
                            collection = weaviate_client.collections.get(name)
                            # Borrado masivo
                            collection.data.delete_many(where=match_all_filter)
                            self.stdout.write(self.style.SUCCESS(f"Objetos de '{name}' eliminados."))
                        else:
                            pass # Silencioso si no existe
                    except Exception as e_coll:
                        self.stdout.write(self.style.WARNING(f"Warn en Weaviate '{name}': {e_coll}"))
                self.stdout.write(self.style.SUCCESS("Weaviate procesado."))
            except Exception as e:
                raise CommandError(f"Error al procesar Weaviate: {e}")

            # 4. Eliminar archivos de 'uploads' (Local - Legacy)
            # Mantenemos esto por si acaso quedan archivos locales viejos
            try:
                uploads_path = settings.MEDIA_ROOT
                if os.path.exists(uploads_path):
                    self.stdout.write(f"Limpiando carpeta local {uploads_path}...")
                    for item in os.listdir(uploads_path):
                        item_path = os.path.join(uploads_path, item)
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
                    self.stdout.write(self.style.SUCCESS("Archivos locales eliminados."))
            except Exception as e:
                self.stdout.write(self.style.WARNING(f"Error limpiando uploads locales: {e}"))

            # 5. Vaciar MinIO (S3) - ¡NUEVO!
            try:
                bucket_name = settings.AWS_STORAGE_BUCKET_NAME
                self.stdout.write(f"Vaciando Bucket S3: {bucket_name}...")
                
                s3 = boto3.resource(
                    's3',
                    endpoint_url=settings.AWS_S3_ENDPOINT_URL,
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                )
                
                bucket = s3.Bucket(bucket_name)
                
                # Verificar si el bucket existe antes de intentar borrar
                if bucket.creation_date:
                    # Esta instrucción borra TODOS los objetos y versiones en el bucket
                    bucket.objects.all().delete()
                    self.stdout.write(self.style.SUCCESS(f"Bucket '{bucket_name}' vaciado completamente."))
                else:
                    self.stdout.write(self.style.WARNING(f"El bucket '{bucket_name}' no existe o no es accesible."))

            except Exception as e:
                # No lanzamos CommandError fatal, solo advertencia, por si MinIO está apagado
                self.stdout.write(self.style.ERROR(f"Error al limpiar MinIO: {e}"))

        except CommandError as e:
            raise e
        except Exception as e:
            raise CommandError(f"Error inesperado: {e}")
        finally:
            self.stdout.write("Cerrando conexiones...")
            # Cerrar Weaviate
            try:
                if weaviate_client:
                    weaviate_client.close()
            except:
                pass
            
            # Cerrar Neo4j
            try:
                close_neo4j_connection()
            except:
                pass

        self.stdout.write(self.style.SUCCESS("✅ RESETEO COMPLETADO EXITOSAMENTE."))