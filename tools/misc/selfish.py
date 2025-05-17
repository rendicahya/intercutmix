import time

import torch


def occupy_gpu_memory(percentage=0.95):
    """
    Allocates a tensor on the GPU to occupy a specified percentage of its memory.

    Args:
        percentage (float): The percentage of GPU memory to occupy (between 0 and 1).
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. This code requires a CUDA-enabled GPU.")
        return

    device = torch.device("cuda")
    total_memory = torch.cuda.get_device_properties(device).total_memory
    target_memory = int(percentage * total_memory)

    try:
        allocated_tensor = torch.empty((target_memory // 4,), dtype=torch.float32, device=device)
        print(f"Successfully allocated approximately {percentage*100:.2f}% of GPU memory.")
        return allocated_tensor
    except torch.cuda.OutOfMemoryError:
        print(f"Could not allocate {percentage*100:.2f}% of GPU memory. You may have other processes using the GPU.")
        return None

if __name__ == "__main__":
    occupied_tensor = occupy_gpu_memory(percentage=0.9885)

    if occupied_tensor is not None:
        print("Holding the allocated memory. Press Ctrl+C to release.")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            del occupied_tensor
            print("GPU memory released.")