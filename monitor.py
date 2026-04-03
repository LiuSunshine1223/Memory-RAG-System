import threading
import time
import torch
import matplotlib.pyplot as plt


class VRAMMonitor:
    def __init__(self):
        self.is_running = False # 控制线程是否继续运行
        self.vram_data = [] # 每次采样的显存大小(MB)
        self.timestamps = [] # 每次采样的时间戳，单位秒，从监控开始算起
        self.start_time = 0 # 监控开始的时间点

    def _monitor_loop(self):
        """后台采样线程，每 0.5 秒记录一次 PyTorch 真实显存"""
        while self.is_running:
            # torch.cuda.memory_allocated()算你代码用的显存，单位bytes，除以1024*1024转换成MB
            used_vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            self.vram_data.append(used_vram_mb)
            self.timestamps.append(time.time() - self.start_time)
            time.sleep(0.5)

    def start(self):
        if not torch.cuda.is_available():
            print("警告：未检测到 GPU，监控脚本将只记录 0 MB。")
            return

        print("[VRAM Monitor] 显存监控已在后台启动...")
        self.is_running = True
        self.start_time = time.time()
        # 创建一个执行流（线程），目标函数是_monitor_loop
        self.thread = threading.Thread(target=self._monitor_loop)
        # 执行流开始运行，进入_monitor_loop循环，直到is_running变为False
        self.thread.start()

    def stop_and_plot(self, save_path="vram_test_result.png"):
        if not self.is_running: return # 如果监控还没开始，直接返回，不要报错

        self.is_running = False
        self.thread.join() # 主线程等待，直到监控线程完全结束，确保数据采集完整
        print("[VRAM Monitor] 监控结束，正在生成压测报告...")

        # ======= 开始绘制工业级图表 =======
        plt.figure(figsize=(10, 5), dpi=300)

        # 画主体折线图
        plt.plot(self.timestamps, self.vram_data, label='Allocated VRAM (MB)', color='#1f77b4', linewidth=2)
        # 下方填充淡淡的蓝色，看起来更高级
        plt.fill_between(self.timestamps, self.vram_data, color='#1f77b4', alpha=0.1)

        # 坐标轴和标题设置
        plt.title("RTX 5070 Ti VRAM Usage During Multi-turn RAG Chat", fontsize=14, fontweight='bold')
        plt.xlabel("Interaction Time (Seconds)", fontsize=12)
        plt.ylabel("VRAM Usage (MB)", fontsize=12)

        # 加上网格线，充满学术感
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc="upper left")
        plt.tight_layout()

        # 保存并显示
        plt.savefig(save_path)
        print(f"图表已保存至项目根目录: {save_path}")