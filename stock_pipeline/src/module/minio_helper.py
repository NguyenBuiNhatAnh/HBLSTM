# modules/minio_helper.py
from minio import Minio
import os

class MinIOHelper:
    def __init__(self, endpoint="localhost:9005", access_key="admin", secret_key="password123"):
        # Nhớ dùng port 9005 mà chúng ta đã đổi để né Hadoop nhé
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False # Chạy ở localhost không có https nên đặt là False
        )
        print("Đã kết nối thành công tới MinIO!")

    def ensure_bucket_exists(self, bucket_name):
        """Kiểm tra bucket đã tồn tại chưa, chưa có thì tạo mới."""
        found = self.client.bucket_exists(bucket_name)
        if not found:
            self.client.make_bucket(bucket_name)
            print(f" Đã tạo bucket mới: {bucket_name}")

    def upload_file(self, bucket_name, file_path, object_name=None):
        """Upload 1 file từ máy tính lên MinIO."""
        self.ensure_bucket_exists(bucket_name)
        
        # Nếu không truyền tên object, lấy luôn tên file gốc
        if object_name is None:
            object_name = os.path.basename(file_path)
            
        try:
            self.client.fput_object(bucket_name, object_name, file_path)
            print(f" Đã upload thành công '{file_path}' lên '{bucket_name}/{object_name}'")
        except Exception as err:
            print(f" Lỗi khi upload {file_path}: {err}")