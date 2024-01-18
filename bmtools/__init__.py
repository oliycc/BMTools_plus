from .tools.serve import ToolServer
from .utils.logging import get_logger

import os
import util.construct_customer_openai_template as customer_openai
# 由于国内openAI接口无法使用，使用自定义的url无缝兼容openAI库
use_custemer_openai_api = os.environ.get('USE_CUSTOMER_OPENAI_API')
if use_custemer_openai_api is not None and use_custemer_openai_api==True:
    # 启动自定义openai服务
    customer_openai.deploy(debug=False)
    