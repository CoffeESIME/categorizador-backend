from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from celery.result import AsyncResult

class TaskStatusView(APIView):
    """
    Endpoint to check the status of multiple Celery tasks.
    GET /api/tasks/status/?ids=task1,task2,task3
    """
    def get(self, request, *args, **kwargs):
        task_ids = request.query_params.get('ids', '')
        if not task_ids:
             return Response({"error": "No task IDs provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        ids_list = [tid.strip() for tid in task_ids.split(',') if tid.strip()]
        results = {}
        
        for task_id in ids_list:
            res = AsyncResult(task_id)
            task_data = {
                "status": res.status,
            }
            if res.status == 'SUCCESS':
                # Assuming the result is JSON serializable
                task_data["result"] = res.result
            elif res.status == 'FAILURE':
                task_data["error"] = str(res.result)
            
            results[task_id] = task_data
            
        return Response(results, status=status.HTTP_200_OK)
