from django.core.management.base import BaseCommand

from api.seeds import (
    seed_node_types,
    ensure_upload_directories,
    seed_weaviate_schema,
)


class Command(BaseCommand):
    help = (
        "Inicializa la aplicación creando directorios de uploads, "
        "sembrando los tipos de nodo en Neo4j y el esquema de Weaviate."
    )

    def handle(self, *args, **options):
        ensure_upload_directories()
        seed_node_types()
        seed_weaviate_schema()
        self.stdout.write(self.style.SUCCESS("Datos iniciales cargados."))

