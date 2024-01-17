### 无缝兼容OpenAI客户端，只需要重写customer_gpt_api
# 本模版做了以下兼容工作
## 1. 必须是'/chat/completions' 
# 2. 能够接收与 OpenAI 相同结构的请求。这意味着你的端点必须能够处理 model 和 messages 这样的字段，并且能够正确地从请求中解析它们。
# 3. 你的 Flask 应用程序必须能够返回与 OpenAI API 相兼容的响应。包括 id、object、created、model、choices 以及 usage 字段。

from flask import Flask, request, jsonify
import time
import requests

app = Flask(__name__)
# 入参示例
# messages：[
#     {
#         "role": "user",
#         "content": "Hello, who are you?"
#     },
#     {
#         "role": "assistant",
#         "content": "I am an AI created by OpenAI."
#     },
#     {
#         "role": "user",
#         "content": "What can you do?"
#     }
# ]
# model: gpt-4
@app.route('/chat/completions', methods=['POST'])
def chat_completions():
    print(f'chat_completions##start, request={request.json}')
    # 这里可以解析 OpenAI 客户端库发送的请求格式
    data = request.json
    messages = data['messages']
    model = data.get('model', 'gpt-4')
    model='WENXIN'
    
    # 假设 messages 只包含一个用户消息
    query = messages[0]['content']
    print(f'query={query}')
    # 这里是你的自定义逻辑来生成回复
    answer, taskId= customer_gpt_api(query, model)
    # print(f'answer={answer},taskId={taskId}')
    # 伪造一个 OpenAI 兼容的响应
    # 格式化响应以匹配 OpenAI 的响应格式
    formatted_response = {
        "id": taskId,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                # 假设 answer 包含完整的回复文本
                "message": {"role": "user", "content": answer},
                "index": 0,
                "finish_reason": "length"
            }
        ],
        "usage": {
            "prompt_tokens": len(query.split()),  # 这是一个简化的 token 计数
            "completion_tokens": len(answer.split()),  # 同上
            "total_tokens": len(query.split()) + len(answer.split())  # 同上
        }
    }
    print(f'formatted_response={formatted_response}')
    return jsonify(formatted_response)




## TODO: 自定义的llm调用接口
def customer_gpt_api(query, model='GPT4'):
    answer = None
    task_id = None
    ## 自定义llm接口调用实现 ，根据实际情况编写
    raise Exception("""
            must overwrite customer_gpt_api!
    """)
    return answer, task_id


def deploy(debug=True):
    app.run(debug, port=5000)



if __name__ == '__main__':
    deploy()
    




