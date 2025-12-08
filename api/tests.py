"""
Unit tests for file upload and MinIO integration.
"""
from unittest.mock import patch, MagicMock
from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APITestCase, APIClient
from rest_framework import status
from io import BytesIO
from django.core.files.uploadedfile import SimpleUploadedFile

from api.models import UploadedFile
from api.utils import generate_presigned_url, get_minio_client


class MinIOUtilsTestCase(TestCase):
    """Tests for MinIO utility functions."""
    
    @patch('api.utils.boto3.client')
    def test_get_minio_client_creates_client(self, mock_boto_client):
        """Test that get_minio_client creates a boto3 client with correct params."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        
        client = get_minio_client()
        
        mock_boto_client.assert_called_once()
        call_kwargs = mock_boto_client.call_args
        self.assertEqual(call_kwargs[0][0], 's3')
    
    @patch('api.utils.get_minio_client')
    def test_generate_presigned_url_success(self, mock_get_client):
        """Test presigned URL generation."""
        mock_client = MagicMock()
        mock_client.generate_presigned_url.return_value = 'http://minio:9005/bucket/key?signature=xxx'
        mock_get_client.return_value = mock_client
        
        url = generate_presigned_url('images/test.jpg')
        
        self.assertIsNotNone(url)
        self.assertIn('http', url)
        mock_client.generate_presigned_url.assert_called_once()
    
    @patch('api.utils.get_minio_client')
    def test_generate_presigned_url_failure(self, mock_get_client):
        """Test presigned URL handles exceptions."""
        mock_client = MagicMock()
        mock_client.generate_presigned_url.side_effect = Exception("Connection error")
        mock_get_client.return_value = mock_client
        
        url = generate_presigned_url('images/test.jpg')
        
        self.assertIsNone(url)


class MultiFileUploadViewTestCase(APITestCase):
    """Tests for file upload endpoint."""
    
    def setUp(self):
        self.client = APIClient()
        self.upload_url = reverse('file_upload')
    
    @patch('api.views.files.upload_file_to_minio')
    @patch('api.views.files.generate_presigned_url')
    def test_upload_single_file_success(self, mock_presigned, mock_upload):
        """Test uploading a single file."""
        mock_upload.return_value = 'images/test.jpg'
        mock_presigned.return_value = 'http://minio:9005/bucket/images/test.jpg?sig=xxx'
        
        file = SimpleUploadedFile(
            "test.jpg",
            b"file_content",
            content_type="image/jpeg"
        )
        
        response = self.client.post(
            self.upload_url,
            {'files': file},
            format='multipart'
        )
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('files', response.data)
        self.assertEqual(len(response.data['files']), 1)
        mock_upload.assert_called_once()
    
    @patch('api.views.files.upload_file_to_minio')
    @patch('api.views.files.generate_presigned_url')
    def test_upload_multiple_files_success(self, mock_presigned, mock_upload):
        """Test uploading multiple files."""
        mock_upload.return_value = 'images/test.jpg'
        mock_presigned.return_value = 'http://minio:9005/bucket/images/test.jpg?sig=xxx'
        
        file1 = SimpleUploadedFile("test1.jpg", b"content1", content_type="image/jpeg")
        file2 = SimpleUploadedFile("test2.png", b"content2", content_type="image/png")
        
        response = self.client.post(
            self.upload_url,
            {'files': [file1, file2]},
            format='multipart'
        )
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['files']), 2)
    
    def test_upload_no_files_returns_error(self):
        """Test uploading with no files returns error."""
        response = self.client.post(self.upload_url, {}, format='multipart')
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.data)
    
    @patch('api.views.files.upload_file_to_minio')
    def test_upload_minio_failure_returns_error(self, mock_upload):
        """Test MinIO upload failure is handled."""
        mock_upload.side_effect = Exception("MinIO connection failed")
        
        file = SimpleUploadedFile("test.jpg", b"content", content_type="image/jpeg")
        
        response = self.client.post(
            self.upload_url,
            {'files': file},
            format='multipart'
        )
        
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)


class PendingFilesViewTestCase(APITestCase):
    """Tests for pending files endpoint."""
    
    def setUp(self):
        self.client = APIClient()
        self.pending_url = reverse('pending_files')
    
    @patch('api.serializers.generate_presigned_url')
    def test_get_pending_files_empty(self, mock_presigned):
        """Test getting pending files when none exist."""
        response = self.client.get(self.pending_url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['files'], [])
    
    @patch('api.serializers.generate_presigned_url')
    def test_get_pending_files_returns_pending_only(self, mock_presigned):
        """Test that only pending files are returned."""
        mock_presigned.return_value = 'http://minio/test?sig=xxx'
        
        # Create pending file
        UploadedFile.objects.create(
            original_name='pending.jpg',
            file_type='image/jpeg',
            size=1024,
            status='pending',
            file_location='images/pending.jpg'
        )
        
        # Create categorized file (should not be returned)
        UploadedFile.objects.create(
            original_name='done.jpg',
            file_type='image/jpeg',
            size=1024,
            status='categorized',
            file_location='images/done.jpg'
        )
        
        response = self.client.get(self.pending_url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['files']), 1)
        self.assertEqual(response.data['files'][0]['original_name'], 'pending.jpg')
    
    @patch('api.serializers.generate_presigned_url')
    def test_pending_files_include_presigned_url(self, mock_presigned):
        """Test that pending files include presigned URLs."""
        mock_presigned.return_value = 'http://minio/bucket/key?sig=xxx'
        
        UploadedFile.objects.create(
            original_name='test.jpg',
            file_type='image/jpeg',
            size=1024,
            status='pending',
            file_location='images/test.jpg'
        )
        
        response = self.client.get(self.pending_url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('file_url', response.data['files'][0])
        mock_presigned.assert_called()


class UploadedFileModelTestCase(TestCase):
    """Tests for UploadedFile model."""
    
    def test_create_file_with_minio_location(self):
        """Test creating a file record with MinIO location."""
        file = UploadedFile.objects.create(
            original_name='test.jpg',
            file_type='image/jpeg',
            size=1024,
            status='pending',
            file_location='images/test.jpg'
        )
        
        self.assertEqual(file.original_name, 'test.jpg')
        self.assertEqual(file.file_location, 'images/test.jpg')
        self.assertIsNone(file.file.name if file.file else None)
    
    def test_str_representation(self):
        """Test string representation of UploadedFile."""
        file = UploadedFile.objects.create(
            original_name='test.jpg',
            file_type='image/jpeg',
            size=1024,
            status='pending'
        )
        
        self.assertEqual(str(file), 'test.jpg')
