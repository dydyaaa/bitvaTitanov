import os, psutil, torch
process = psutil.Process(os.getpid())
ram_mb = process.memory_info().rss / 1024 ** 2

print(f"[MEM][{tag}] RAM used: {ram_mb:.2f} MB")

    # видеопамять (если есть CUDA)
if torch.cuda.is_available():
    vram_alloc = torch.cuda.memory_allocated() / 1024 ** 2
    vram_res = torch.cuda.memory_reserved() / 1024 ** 2
    print(f"[MEM][{tag}] VRAM allocated: {vram_alloc:.2f} MB, reserved: {vram_res:.2f} MB")
