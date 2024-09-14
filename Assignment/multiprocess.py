import multiprocessing
import subprocess

def run_script(script_args):
    script_name, *args = script_args
    # print(f"Executing script: {script_name} with args: {args}")
    subprocess.run(['python3', script_name, *args])

if __name__ == "__main__":
    # 假设你有三个脚本文件：script1.py、script2.py、script3.py
    # 并且每个脚本接受不同的参数
    script_args_list = []
    for i in range(200, 600, 50):
        script_args_list.append(('./Assignment/a2_visualize.py', f'{i}'))

    # 创建一个进程池
    with multiprocessing.Pool() as pool:
        # 使用 map 函数将工作分配给每个进程
        pool.map(run_script, script_args_list)
