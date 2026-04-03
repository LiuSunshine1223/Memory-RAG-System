# OLLama模型接口 LLma3:8B 的简单封装，提供 ask 方法进行文本生成请求。
# 使用 Ollama 托管跑（4-bit 量化，80亿参数 × 0.5 Bytes = 4GB）比 放在项目结构里裸跑（FP16量化，80亿参数 × 2 Bytes = 16GB）
# 更省资源、更稳定，且无需担心显存溢出问题
import requests

class LLMModel:
    def __init__(self, url, model_name):
        self.url = url
        self.model = model_name

    def ask(self, prompt):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False, # 非流式输出，等待完整回复后再返回结果，适合一次性生成较长文本的场景
            "options": {
                "temperature": 0.0,  # 绝对理智，消除乱码和随机发散
                "top_p": 0.1  # 极度收拢词汇表，只选最确定的词
            }
        }
        try:
            # timeout 设为 120 秒，给 8B 模型留足生成时间
            # requests.post() 发送 HTTP POST 请求到指定 URL，携带 JSON 格式的 payload 作为请求体，并设置超时时间为 120 秒。
            # Response 对象包含服务器返回的响应数据。
            # 通过 response.json() 方法将响应内容解析为 JSON 格式，并使用 .get("response", "LLM 无返回内容") 提取其中的 "response" 字段，如果该字段不存在，则返回默认消息 "LLM 无返回内容"。
            response = requests.post(self.url, json=payload, timeout=120)
            return response.json().get("response", "LLM 无返回内容")
        except Exception as e:
            return f"[错误] 无法连接到 Ollama: {e}\n请检查宿主机 IP 和 OLLAMA_HOST 设置。"