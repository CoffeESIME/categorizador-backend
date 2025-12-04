import os
import django
from django.conf import settings

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'categorizador.settings')
django.setup()

try:
    from api.tasks import process_pdf_task
    print("Import api.tasks successful")
except Exception as e:
    print(f"Import api.tasks failed: {e}")

try:
    from api.services.processing import _encode_image
    print("Import api.services.processing successful")
except Exception as e:
    print(f"Import api.services.processing failed: {e}")
