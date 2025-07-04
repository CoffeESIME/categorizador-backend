from unittest import mock
from django.test import Client
from django.urls import reverse

from api.models import UploadedFile


def test_metadata_processing_view_delegates(db):
    UploadedFile.objects.create(
        original_name="sample.txt",
        file="/tmp/sample.txt",
        file_type="text/plain",
        status="pending",
        file_location="texts/sample.txt",
    )

    data = [{"original_name": "sample.txt", "embedding_type": "pdf", "file_location": "texts/sample.txt"}]
    client = Client()
    with mock.patch("api.services.processing.process_pdf") as m_pdf:
        m_pdf.return_value = None
        response = client.post(reverse("metadata_process"), data, content_type="application/json")
        assert response.status_code == 200
        m_pdf.assert_called_once()
