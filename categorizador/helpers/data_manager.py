import os
import time

def custom_upload_path(instance, filename):
    base, ext = os.path.splitext(filename)
    # Obtener el timestamp en microsegundos
    timestamp = str(int(time.time() * 1000000))
    # Combinar el nombre original con el timestamp para asegurar la unicidad
    new_filename = f"{base}_{timestamp}{ext}"
    return os.path.join("uploads/", new_filename)
