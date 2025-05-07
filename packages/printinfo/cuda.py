import torch

def printCUDAinfo():
    print("CUDA Information:")
    if torch.cuda.is_available():
        # PyTorch가 사용하는 CUDA 버전
        print(f"CUDA version: {torch.version.cuda}")
        # cuDNN 사용 가능 여부
        print(f"cuDNN: {torch.backends.cudnn.enabled}")
        print(f"CUDA is available! (Number of GPUs: {torch.cuda.device_count()})")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB 단위로 변환
            allocated_memory = torch.cuda.memory_allocated(i) / (1024**3)  # GB 단위로 변환
            reserved_memory = torch.cuda.memory_reserved(i) / (1024**3) # GB 단위로 변환
            available_memory = total_memory - reserved_memory

            print(f"Total Memory: {total_memory:.2f} GB")
            print(f"Allocated Memory: {allocated_memory:.2f} GB")
            print(f"Reserved Memory: {reserved_memory:.2f} GB")
            print(f"Available Memory: {available_memory:.2f} GB")
    else:
        print("CUDA is NOT available. Please check your CUDA installation and drivers.")

if __name__ == "__main__":
    printCUDAinfo()