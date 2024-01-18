### 无缝兼容OpenaI客户端，需要重写customer_gpt_api
### 兼容OpenaiEmbedding
# 本模版做了以下兼容工作
# 1. 必须是'/chat/completions' 
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

### 兼容OpenaiEmbedding
# 函数描述：获取嵌入
# 入参：
#curl https://api.openai.com/v1/embeddings \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer $OPENAI_API_KEY" \
#   -d '{
#     "input": "Your text string goes here",
#     "model": "text-embedding-ada-002"
#   }'
# 响应：
# {
#   "data": [
#     {
#       "embedding": [
#         -0.006929283495992422,
#         -0.005336422007530928,
#         ...
#         -4.547132266452536e-05,
#         -0.024047505110502243
#       ],
#       "index": 0,
#       "object": "embedding"
#     }
#   ],
#   "model": "text-embedding-ada-002",
#   "object": "list",
#   "usage": {
#     "prompt_tokens": 5,
#     "total_tokens": 5
#   }
# }
# 


@app.route('/embeddings', methods=['POST'])
def embed_query():
    # 加权逻辑
    from langchain_community.embeddings import ModelScopeEmbeddings
    # 解析请求数据
    request_data = request.get_json()
    print(f'embed_query##request={request_data}')
    
    # 获取文本输入和模型参数
    input_text = request_data.get('input')
    model = request_data.get('model', 'text-embedding-ada-002')  # 如果模型未指定，使用默认值
    # 初始化嵌入模型（根据你的实际情况来初始化适当的模型）
    query_result = customer_embedding_api(model=model, input_text=input_text)
    data_list = [
        {
            "embedding": query_result,
            "index": 0,  # 如果有多个结果，这里应该是结果的索引
            "object": "embedding"
        }  

    ]
    # data_list = [
    #     {
    #         "embedding": result,
    #         "index": index,  # 如果有多个结果，这里应该是结果的索引
    #         "object": "embedding"
    #     }  for index, result in enumerate(query_result)

    # ]
        
    # 构建响应
    response = {
        "data": data_list,
        "model": model,  # 这应该是实际使用的模型标识
        "object": "list",
        # 这里你可能需要根据你的系统提供相关的用量信息
        "usage": {
            "prompt_tokens": len(input_text.split()),  # 假设每个词是一个token
            "total_tokens": len(input_text.split())
        }
    }
    
    # 返回JSON响应
    return jsonify(response)


## TODO: 自定义的embedding调用接口
def customer_embedding_api(input_text, model:str=None):
    query_result = list()
    ## 自定义llm接口调用实现 ，根据实际情况编写
    # raise Exception("""
    #         must overwrite customer_embedding_api!
    # """)
    from langchain.embeddings import HuggingFaceBgeEmbeddings
    model_name = "BAAI/bge-small-zh-v1.5"
    model_kwargs = {}#{'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
    model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
        # ,query_instruction="为这个句子生成表示以用于检索相关文章："
    )
    query_result= model.embed_query(input_text)
    return query_result
    

## TODO: 自定义的llm调用接口
def customer_gpt_api(query, model='GPT4'):
    answer = None
    task_id = None
    ## 自定义llm接口调用实现 ，根据实际情况编写
    raise Exception("""
            must overwrite customer_gpt_api!
    """)
    return answer, task_id



# 部署
def deploy(debug=True):
    app.run(debug=True, port=5000)
    print(f'已启动自定义llm服务，端口: 5000')



if __name__ == '__main__':
    # inputs = '你好'
    # query_result = customer_embedding_api(input_text=inputs)
    # print(f'finish! {query_result}')
    
    deploy()