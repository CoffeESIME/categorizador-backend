"""
Unit tests for Celery tasks with MinIO integration.
"""
from unittest.mock import patch, MagicMock, mock_open
from django.test import TestCase
from celery.exceptions import Retry

from api.tasks import (
    process_pdf_task,
    process_image_with_description_task,
    process_audio_file_task,
    process_video_file_task,
    process_text_embeddings_task,
)


class CeleryTasksTestCase(TestCase):
    """Tests for Celery processing tasks."""
    
    @patch('api.tasks.download_file_from_minio')
    @patch('api.tasks.embed_pdf_and_store')
    @patch('api.tasks.store_embedding')
    @patch('api.tasks.CLIENT')
    def test_process_pdf_task_success(self, mock_client, mock_store, mock_embed, mock_download):
        """Test PDF processing task downloads from MinIO and processes."""
        # Mock the context manager
        mock_download.return_value.__enter__ = MagicMock(return_value='/tmp/test.pdf')
        mock_download.return_value.__exit__ = MagicMock(return_value=False)
        
        meta = {
            'file_location': 'documents/test.pdf',
            'title': 'Test PDF'
        }
        
        # Call the task synchronously
        result = process_pdf_task.apply(args=['test_file_id', meta])
        
        mock_download.assert_called_once()
    
    @patch('api.tasks.download_file_from_minio')
    @patch('api.tasks._encode_image')
    @patch('api.tasks.guardar_imagen_en_weaviate')
    @patch('api.tasks.store_embedding')
    @patch('api.tasks.OllamaEmbeddings')
    @patch('api.tasks.limpiar_meta')
    @patch('api.tasks.CLIENT')
    def test_process_image_with_description_task_success(
        self, mock_client, mock_limpiar, mock_ollama, mock_store, mock_guardar, mock_encode, mock_download
    ):
        """Test image processing task with description."""
        mock_download.return_value.__enter__ = MagicMock(return_value='/tmp/test.jpg')
        mock_download.return_value.__exit__ = MagicMock(return_value=False)
        mock_encode.return_value = [0.1, 0.2, 0.3]
        mock_limpiar.return_value = {}
        
        mock_embed_instance = MagicMock()
        mock_embed_instance.embed_documents.return_value = [[0.1, 0.2]]
        mock_ollama.return_value = mock_embed_instance
        
        meta = {
            'file_location': 'images/test.jpg',
            'description': 'A test image',
            'style': 'modern'
        }
        
        result = process_image_with_description_task.apply(
            args=['test_file_id', meta, 'images/test.jpg']
        )
        
        mock_download.assert_called_once()
        mock_encode.assert_called_once()
    
    @patch('api.tasks.download_file_from_minio')
    @patch('api.tasks._encode_audio')
    @patch('api.tasks._transcribe_audio')
    @patch('api.tasks.process_text_embeddings')
    @patch('api.tasks.guardar_audio_en_weaviate')
    @patch('api.tasks.store_embedding')
    @patch('api.tasks.limpiar_meta')
    @patch('api.tasks.CLIENT')
    def test_process_audio_task_with_transcription(
        self, mock_client, mock_limpiar, mock_store, mock_guardar, 
        mock_text_embed, mock_transcribe, mock_encode, mock_download
    ):
        """Test audio processing task with transcription."""
        mock_download.return_value.__enter__ = MagicMock(return_value='/tmp/test.mp3')
        mock_download.return_value.__exit__ = MagicMock(return_value=False)
        mock_encode.return_value = [0.1, 0.2]
        mock_transcribe.return_value = "Transcribed text"
        mock_text_embed.return_value = [0.3, 0.4]
        mock_limpiar.return_value = {}
        
        meta = {'file_location': 'audio/test.mp3'}
        
        result = process_audio_file_task.apply(
            args=['test_file_id', meta, 'audio/test.mp3'],
            kwargs={'transcribe': True}
        )
        
        mock_transcribe.assert_called_once()
    
    @patch('api.tasks.download_file_from_minio')
    @patch('api.tasks._encode_video')
    @patch('api.tasks._encode_audio')
    @patch('api.tasks.process_text_embeddings')
    @patch('api.tasks.guardar_video_en_weaviate')
    @patch('api.tasks.store_embedding')
    @patch('api.tasks.limpiar_meta')
    @patch('api.tasks.CLIENT')
    def test_process_video_task_with_audio(
        self, mock_client, mock_limpiar, mock_store, mock_guardar,
        mock_text_embed, mock_encode_audio, mock_encode_video, mock_download
    ):
        """Test video processing task with audio extraction."""
        mock_download.return_value.__enter__ = MagicMock(return_value='/tmp/test.mp4')
        mock_download.return_value.__exit__ = MagicMock(return_value=False)
        mock_encode_video.return_value = [0.1, 0.2]
        mock_encode_audio.return_value = [0.3, 0.4]
        mock_text_embed.return_value = [0.5, 0.6]
        mock_limpiar.return_value = {}
        
        meta = {'file_location': 'videos/test.mp4'}
        
        result = process_video_file_task.apply(
            args=['test_file_id', meta, 'videos/test.mp4'],
            kwargs={'include_audio': True}
        )
        
        mock_encode_video.assert_called_once()
        mock_encode_audio.assert_called_once()
    
    @patch('api.tasks.process_text_embeddings')
    def test_process_text_embeddings_task_success(self, mock_process):
        """Test text embeddings task."""
        mock_process.return_value = [0.1, 0.2, 0.3]
        
        meta = {
            'content': 'Test content',
            'title': 'Test Title'
        }
        
        result = process_text_embeddings_task.apply(args=[meta])
        
        mock_process.assert_called_once_with(meta)


class TaskErrorHandlingTestCase(TestCase):
    """Tests for task error handling."""
    
    @patch('api.tasks.download_file_from_minio')
    def test_pdf_task_handles_download_error(self, mock_download):
        """Test PDF task handles MinIO download errors."""
        mock_download.side_effect = Exception("MinIO connection failed")
        
        meta = {'file_location': 'documents/test.pdf'}
        
        with self.assertRaises(Exception):
            process_pdf_task.apply(args=['test_file_id', meta])
    
    @patch('api.tasks.download_file_from_minio')
    @patch('api.tasks._encode_image')
    def test_image_task_handles_encoding_error(self, mock_encode, mock_download):
        """Test image task handles encoding errors."""
        mock_download.return_value.__enter__ = MagicMock(return_value='/tmp/test.jpg')
        mock_download.return_value.__exit__ = MagicMock(return_value=False)
        mock_encode.side_effect = Exception("Encoding failed")
        
        meta = {'file_location': 'images/test.jpg'}
        
        with self.assertRaises(Exception):
            process_image_with_description_task.apply(
                args=['test_file_id', meta, 'images/test.jpg']
            )
