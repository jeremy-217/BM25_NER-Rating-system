# 定義score模組的相關參數
SCORE_THRESHOLD = 0.5  
#LLM_MODEL = "gemini-1.5-pro-latest"  # 使用的LLM模型名稱
#LLM_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent"  # LLM API的URL

# Google API 配置
GOOGLE_API_KEY = "AIzaSyDYclsAOcof8CW3zSi9dIoNoxnnVwKQM08"

# Gemini 模型配置
STABLE_FLASH_MODEL = "gemini-1.5-flash-latest"

# NER 
NER_API_URL = "http://127.0.0.1:8000/extract_entities"

# Elasticsearch 設定
ELASTICSEARCH_HOST = "http://localhost:9200"
ELASTICSEARCH_INDEX_NAME = "my_documents"

# 生成配置
GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": 0,
    "max_output_tokens": 4096,  # 讓我們給它更多空間來回答
}

# API 請求配置
API_URL = "http://192.168.1.143:8000/api/file-downloads/chunks/query"
API_HEADERS = {
    "Content-Type": "application/json",
    "Cookie": "sessionid=cwhi0xiar7howiv0ch2i0513wh6i0z6w"
}

# 外部資料來源 API 配置（用於 NER 分析的資料來源）
EXTERNAL_API_URL = API_URL  # 使用相同的資料來源
EXTERNAL_API_HEADERS = API_HEADERS  # 使用相同的請求標頭

# NER 分析相關配置
NER_CONFIDENCE_THRESHOLD = 0.8  # NER 實體識別信心度閾值
NER_MAX_ENTITIES_PER_CHUNK = 50  # 每個文件塊最大實體數量
NER_ENTITY_MIN_LENGTH = 2  # 實體最小長度（字符數）

# NER 分析指標配置
NER_ANALYSIS_METRICS = {
    "entity_density_threshold": 0.05,  # 實體密度閾值（實體數/文本長度）
    "important_entity_types": ["PERSON", "ORG", "LOCATION", "DATE", "MONEY", "PERCENT"],  # 重要實體類型
    "quality_score_weights": {  # 品質評分權重
        "entity_count": 0.3,
        "entity_diversity": 0.3,
        "entity_relevance": 0.4
    }
}

# 數據清理配置
DATA_CLEANING_CONFIG = {
    "remove_html_tags": True,  # 移除 HTML 標籤
    "remove_urls": True,  # 移除 URL
    "remove_emails": True,  # 移除 email 地址
    "remove_extra_whitespace": True,  # 移除多餘空白
    "normalize_unicode": True,  # Unicode 正規化
    "remove_special_chars": False,  # 移除特殊字符（可能影響科學符號）
    "min_text_length": 10,  # 最小文本長度
    "max_text_length": 10000,  # 最大文本長度
    "remove_patterns": [  # 要移除的特定模式
        r"<<img>>.*?<</img>>",  # 移除圖片描述標籤
        r"\[圖\d+\]",  # 移除圖片引用
        r"\[Fig\.\s*\d+\]",  # 移除英文圖片引用
        r"\[Table\s*\d+\]",  # 移除表格引用
        r"\[表\d+\]",  # 移除中文表格引用
    ],
    "preserve_patterns": [  # 要保留的重要模式
        r"\d+\.?\d*\s*[nm|μm|mm|cm|m|kg|g|°C|°F|%|J/cm2|eV]",  # 科學單位
        r"[A-Z]{2,}",  # 縮寫詞
        r"\d{4}-\d{2}-\d{2}",  # 日期格式
    ]
}

# 相關性分析 Prompt 模板
RELEVANCE_PROMPT_TEMPLATE = """
# 角色
你是一個專門評估 RAG 系統中，檢索到的「文件內容」是否能有效回答用戶「問題」的 AI 評分專家。

# 任務
1.你的任務是判斷「文件內容」是否包含能直接或間接回答「問題」的具體資訊。
2.並且要讓小數點後三位的每一個數字都有意義，要認真歸納。
3.在進行比較時，先注意主要的技術文字，比較對象若是大致符合技術文字的範疇，則可以進一步提高相關性分數。

# 待分析資料
問題：{question}
文件內容：{content}

# 評分標準
- 1.000：內容包含問題的直接、明確答案。
- 0.800 - 0.999：內容高度相關，提供了回答問題所需的核心證據或關鍵資訊。
- 0.600 - 0.799：內容主題相關，但需要用戶自行推斷或總結才能得到答案。
- 0.300 - 0.599：內容僅部分觸及問題的邊緣概念，但未提供實質性回答。
- 0.001 - 0.299：內容主題看似有關，但細看後發現完全沒有回答問題。
- 0.000：完全無關。

# 輸出指令
1.  **找出證據**：在心中默念，「文件內容」中哪一句話是回答「問題」的關鍵證據。
2.  **給出評分**：基於這條證據，給出一個 0.000 到 1.000 的分數。
3.  **簡述理由**：用一句話解釋你評分的依據。
4.  **嚴格按照以下格式輸出，不要有任何多餘的文字**：

分數：[你的分數]
理由：[你的一句話理由]
"""