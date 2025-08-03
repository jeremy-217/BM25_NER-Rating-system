# test_final_and_working.py

import google.generativeai as genai
import requests
import json
from config import (
    SCORE_THRESHOLD, 
    GOOGLE_API_KEY, 
    STABLE_FLASH_MODEL, 
    GENERATION_CONFIG,
    API_URL,
    API_HEADERS,
    RELEVANCE_PROMPT_TEMPLATE
)

# output the response as json and read it into dict to use
'''
response = {
    "id": "unique_id",
    "content_length": 1234,
    "analysis": "This is a sample analysis result.",
    "score": 0.85
    }
or list of dicts
response = [
    {
        "id": "unique_id_1",
        "content_length": 1234,
        "analysis": "This is a sample analysis result.",
        "score": 0.85
    },
    {
        "id": "unique_id_2",
        "content_length": 5678,
        "analysis": "This is another sample analysis result.",
        "score": 0.90
    }
]
'''
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("✅ Gemini API 金鑰配置成功！")

except Exception as e:
    print(f"❌ 配置 API 金鑰時發生錯誤: {e}")
    exit()


# --- 2. 初始化模型 ---
model = genai.GenerativeModel(
    model_name=STABLE_FLASH_MODEL,
    generation_config=GENERATION_CONFIG
    #...
)

print(f"模型 '{model.model_name}' 已準備就緒。")


# --- 3. 開始互動式聊天 ---

# --- 問題與 content 相關性分析區塊 ---

# 讓使用者輸入查詢內容
user_query = input("請輸入您的查詢問題: ")

# API 請求函數
def fetch_chunks_from_api(query):
    data = {"query": query}
    
    try:
        response = requests.post(API_URL, headers=API_HEADERS, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ API 請求失敗: {e}")
        return None

# 獲取 API 資料
print(f"正在查詢: {user_query}")
api_response = fetch_chunks_from_api(user_query)

if api_response is None:
    print("無法獲取資料，使用預設資料進行測試...")
    # 使用原本的測試資料
    question = "LPP-EUV 燒蝕二氧化矽的效能如何?"
    data_list = [{
        "id": "06fe1d94-df74-431a-b4b9-2b87051dd7ad",
        "content": "Irradiation fluence at 13.5 nm was about 0.2 J/cm2. The silica glass was successfully ablated. The maximum ablation rate was about 25 nm per shot, which is about half that of 11-nm light.\n\nThe most likely cause of these results is the difference in the absorption cross section of silica glass around 12 nm; the absorption cross section increases at wavelengths shorter than 12 nm, and the absorption coefficient around 11 nm is about 1.5 times larger than that at 13.5 nm(11). The other cause may be the irradiation fluence. The EUV energy of the 13.5-nm light was lower than that of the 11-nm light in these experiments.\n\nThe EUV spectra of Nd:YAG LPP, as shown in Figs. 2 and 3, have a large component on the short-wavelength side around 7 nm.\n\nWe attempted to ablate using EUV light of Nd:YAG laser-produced Al plasma (LPP-Al), shown in Fig. 4, to investigate the effect of the short-wavelength side. As a result, the silica glass was not ablated, and it was confirmed that emission at wavelengths shorter than 8 nm (150 eV) did not contribute to ablation in the present experiments. The incident grazing angle of the ellipsoidal mirror (200 mrad) gave a low focusing efficiency for wavelengths shorter than 8 nm(12), thus we obtain this result. On the other hand, we have to consider the effect of longer wavelength. Makimura et al. reported that ablation is not caused by irradiation with VUV light longer than 80 nm, and the EUV light in the range of 30-80 nm is not always necessary for the\n\n1500\n\nCCD SIGNAL\n1000\n<<img>>這是一張線圖，X軸為距離（μm），範圍0到250；Y軸為深度（nm），範圍0到-600。圖表顯示一條不平整的表面輪廓線，其中有數個深度超過-300奈米的尖銳凹陷，最深處約-450奈米。<</img>>\n500\n0\n0\n5\n10\n15\n20\nWAVELENGTH [nm]\n\nFig. 4. Light emission spectrum of a Nd:YAG laser produced Al plasma.\n\n(a)\n<<img>>這是一張線圖，X軸為距離（μm），範圍0到300；Y軸為深度（μm），範圍0到-6。圖表顯示一條具有密集且深度不一的凹槽輪廓線，最深處約為-4微米。<</img>>\n<<img>>這是一張顯微影像，呈現了一個佈滿圓形微孔的陣列結構。影像中央有一條紅色虛線橫貫，標示出其中一排微孔的位置，可能用於量測或對位分析。<</img>>\n2000\nCO2 LPP\nXe\nSn\n1500\nCCD SIGNAL\n1000\n500\n0\n(b)\n6\n8\n10\n12\n14\n16\n18\n20\nWAVELENGTH [nm]\n\nFig. 6. A microscope image and the cross-sectional profile of silica surface irradiated by emission of CO2 laser produced (a) Xe plasma, and (b) Sn plasma.\nFig. 5. Light emission spectrum of a CO2 laser produced Xe plasma and Sn plasma.",
        "metadata": {
            "file_name": "130_1779.pdf",
            "page_number": 4,
            "file_path": "/tmp/tmpx3doi9_k/c38a485c-4fc3-4682-9b25-8f65b7a74b77_130_1779.pdf",
            "file_type": "application/pdf",
            "file_size": 723339,
            "creation_date": "2025-07-16",
            "last_modified_date": "2025-07-16",
            "file_id": "c38a485c-4fc3-4682-9b25-8f65b7a74b77"
        },
        "score": 1.0
    }]
else:
    print("✅ 成功獲取 API 資料")
    question = user_query
    
    # 處理多個資料的情況
    if isinstance(api_response, list):
        print(f"共獲取到 {len(api_response)} 筆資料")
        data_list = api_response
    else:
        print("獲取到單筆資料")
        data_list = [api_response]

# 後台調整 prompt 的地方
# prompt 模板已移動到 config.py

# --- 處理資料並分析多個 content ---
results = []

for i, data in enumerate(data_list):
    extracted = {"id": data.get("id", f"unknown_{i}"), "content": data.get("content", "")}
    print(f"\n=== 分析第 {i+1} 筆資料 ===")
    print(f"ID: {extracted['id']}")
    print(f"Content 長度: {len(extracted['content'])} 字符")
    print("-" * 30)

    # --- 執行相關性分析 ---
    analysis_prompt = RELEVANCE_PROMPT_TEMPLATE.format(
        question=question, 
        content=extracted["content"]
    )

    print(f"分析問題: {question}")
    print("正在分析相關性...")

    try:
        response = model.generate_content(analysis_prompt)
        analysis_result = response.text.strip()
        print(f"\nGemini 分析結果:")
        print(analysis_result)
        
        # 儲存結果
        results.append({
            "id": extracted["id"],
            "content_length": len(extracted["content"]),
            "analysis": analysis_result,
            "original_data": data
        })
        
    except Exception as e:
        print(f"❌ 分析時發生錯誤: {e}")
        results.append({
            "id": extracted["id"],
            "content_length": len(extracted["content"]),
            "analysis": f"錯誤: {e}",
            "original_data": data
        })
    
    print("-" * 50)

# --- 顯示所有結果總覽 ---
print("\n=== 所有結果總覽 ===")
for i, result in enumerate(results):
    print(f"{i+1}. ID: {result['id']}")
    print(f"   Content 長度: {result['content_length']} 字符")
    print(f"   分析結果: {result['analysis']}")
    print("-" * 30)