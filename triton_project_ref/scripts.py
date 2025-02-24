import numpy as np
from trism.model import TritonModel

# Khởi tạo Triton model
model = TritonModel(
    model="viencoder",   # Tên model theo Triton
    version=1,           # Phiên bản model
    url="localhost:8001",
    grpc=True
)

# Kiểm tra metadata
for inp in model.inputs:
    print(f"name: {inp.name}, shape: {inp.shape}, datatype: {inp.dtype}\n")
for out in model.outputs:
    print(f"name: {out.name}, shape: {out.shape}, datatype: {out.dtype}\n")

# Chuyển dữ liệu thành np.bytes_
input_data = np.array([b"Hello, world!"], dtype=np.bytes_)

# Chạy inference
outputs = model.run(data=[input_data])

# Hiển thị kết quả
print(outputs)
