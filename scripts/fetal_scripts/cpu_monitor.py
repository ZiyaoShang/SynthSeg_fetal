import psutil
import time
import pynvml

# Function to monitor CPU and GPU resources
def monitor_resources(interval_seconds, gpu_index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    while True:
        # Get CPU usage
        cpu_usage = psutil.cpu_percent(interval=interval_seconds)

        # Get GPU usage
        gpu_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_usage = gpu_info.gpu

        # Print or log the resource usage
        print(f"{cpu_usage}%")
        # print(f"GPU Usage: {gpu_usage}%")

        # Sleep for the specified interval
        time.sleep(interval_seconds)

# Call the function with the desired monitoring interval and GPU index
monitor_resources(1, 1) 
