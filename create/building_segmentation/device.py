import torch

def cudaFreeBytes(index: int) -> int:
    with torch.cuda.device(index):
        return torch.cuda.mem_get_info()[0]

def pickCudaDevice(index: int | None = None) -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("config requested CUDA, but torch.cuda.is_available() is False")

    count = torch.cuda.device_count()
    if index is not None:
        if index >= count:
            raise SystemExit(f"cuda:{index} requested, but only {count} CUDA device(s) are visible")
    
        chosen = index
    
    elif count == 1:
        chosen = 0
    
    else:
        chosen = max(range(count), key=lambda i: (cudaFreeBytes(i), i))
        free = [round(cudaFreeBytes(i) / 1024**2) for i in range(count)]
        print(f"CUDA devices: {count}  free MiB={free}  using cuda:{chosen} ({torch.cuda.get_device_name(chosen)})")

    torch.cuda.set_device(chosen)
    return torch.device("cuda", chosen)

def getDevice(preference: str = "auto") -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return pickCudaDevice()
        
        if torch.backends.mps.is_available():
            return torch.device("mps")
        
        return torch.device("cpu")

    elif preference == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        
        print("MPS is not available; falling back to CPU")
        return torch.device("cpu")

    device = torch.device(preference)
    if device.type != "cuda":
        return device

    return pickCudaDevice(device.index)