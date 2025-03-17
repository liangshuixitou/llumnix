#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import aiohttp
import time


async def send_prompt():
    """向服务器发送硬编码的提示并获取响应（调试用）"""

    # 硬编码的参数
    server_address = "127.0.0.1:37037"  # 服务器地址
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # 采样参数
    sampling_params = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 2048}

    # 请求超时时间(秒)
    timeout_seconds = 120

    # 设置超时
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    try:
        print(f"正在向服务器 {server_address} 发送请求...")

        # 保存每个提示的响应
        prompt_responses = {}

        # 创建一个会话用于所有请求
        async with aiohttp.ClientSession(timeout=timeout) as client:
            # 循环发送每个提示
            for prompt in prompts:
                print(f"发送提示: {prompt}")

                # 构建单个请求体
                request_data = {"prompt": prompt, "stream": False, **sampling_params}

                # 发送请求
                start_time = time.time()
                response = await client.post(
                    f"http://{server_address}/generate", json=request_data
                )

                # 检查响应状态
                if response.status != 200:
                    error_text = await response.text()
                    print(f"错误: 服务器返回状态码 {response.status}")
                    print(f"错误信息: {error_text}")
                    continue

                # 解析响应
                response_json = await response.json()
                prompt_responses[prompt] = (
                    response_json["text"][0]
                    if "text" in response_json and response_json["text"]
                    else ""
                )

                # 计算耗时
                elapsed = time.time() - start_time
                print(f"完成提示处理，耗时: {elapsed:.2f}秒")

        # 打印所有结果
        print("\n===== 生成的文本 =====")
        for i, prompt in enumerate(prompts):
            if prompt in prompt_responses:
                print(f"\n[提示 {i+1}]: {prompt}")
                print(f"[响应 {i+1}]: {prompt_responses[prompt]}")
            else:
                print(f"\n[提示 {i+1}]: {prompt}")
                print("[响应 {i+1}]: <未收到有效响应>")
        print("======================\n")

    except asyncio.TimeoutError:
        print(f"错误: 请求超时，已等待{timeout_seconds}秒")
    except Exception as e:
        print(f"错误: {str(e)}")


if __name__ == "__main__":
    print("开始调试请求...")
    asyncio.run(send_prompt())
    print("请求完成")
