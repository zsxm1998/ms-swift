import subprocess
import time
import argparse

def is_all_gpus_idle():
    try:
        # 使用 nvidia-smi 命令获取 GPU 的运行状态
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            print(f"Error running nvidia-smi: {result.stderr}")
            return False

        # 检查输出是否为空，表示所有 GPU 上都没有进程
        return not result.stdout.strip()
    except FileNotFoundError:
        print("Error: nvidia-smi command not found. Ensure NVIDIA drivers are installed.")
        return False

def run_command_on_idle(command):
    while True:
        if is_all_gpus_idle():
            print("All GPUs are idle. Running command...")
            subprocess.run(command, shell=True)
            break
        else:
            print("GPUs are not idle. Checking again in 10 seconds...")
        time.sleep(10)  # 每10s检查一次

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Monitor GPU usage and run a command when all GPUs are idle.")
    parser.add_argument("command", default='echo "All GPUs are idle."', type=str, help="The bash command to run when all GPUs are idle.")
    
    # 解析命令行参数
    args = parser.parse_args()

    # 监控 GPU 并在空闲时运行命令
    run_command_on_idle(args.command)
