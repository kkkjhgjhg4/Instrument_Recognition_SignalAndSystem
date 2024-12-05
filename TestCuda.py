import torch
import time

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    device = torch.device("cuda")
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Number of CUDA Devices: {torch.cuda.device_count()}")

    # Create large tensors on GPU
    a = torch.randn(10000, 10000, device=device)
    b = torch.randn(10000, 10000, device=device)

    # Warm-up GPU
    for _ in range(10):
        torch.matmul(a, b)

    # Time the matrix multiplication
    start_time = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()  # Ensure all CUDA operations are complete
    end_time = time.time()

    print(f"Time taken for matrix multiplication on {device}: {end_time - start_time:.4f} seconds")
else:
    print("CUDA is not available. Please check your PyTorch installation and CUDA setup.")