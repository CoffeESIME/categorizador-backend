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
    with mock.patch("api.tasks.process_pdf_task.delay") as m_delay:
        m_delay.return_value = mock.Mock(id="123")
        response = client.post(reverse("metadata_process"), data, content_type="application/json")
        assert response.status_code == 200
        m_delay.assert_called_once()
