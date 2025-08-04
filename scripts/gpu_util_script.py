import GPUtil

# GPU情報の取得
gpus = GPUtil.getGPUs()

for gpu in gpus:
    print(f"GPU ID: {gpu.id}")
    print(f"Name: {gpu.name}")
    print(f"Load: {gpu.load * 100:.1f}%")
    print(f"Free memory: {gpu.memoryFree:.1f} MB")
    print(f"Used memory: {gpu.memoryUsed:.1f} MB")
    print(f"Total memory: {gpu.memoryTotal:.1f} MB")
    print(f"Temperature: {gpu.temperature} °C\n")
