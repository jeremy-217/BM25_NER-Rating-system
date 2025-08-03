# ner.py
"""
簡化版 NER 分析工具 - 整合 Gemini API

功能特色：
- 內建 spaCy NER 服務進行實體識別
- 使用 Gemini API 進行智能分析和處理
- 整合 RAG 檢索結果
- 移除評分機制，專注於實體識別和智能分析

使用方法：
    uv run python ner.py

要求：
- spaCy 和 en_core_web_trf 模型已安裝
- 配置正確的 Gemini API 金鑰
- 配置正確的外部資料 API 設定
"""

import requests
import json
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Tuple, Optional
from html import unescape
import google.generativeai as genai

# 整合 NER 服務相關導入
try:
    import spacy
    SPACY_AVAILABLE = True
    print("✅ spaCy 可用，將使用內建 NER 服務")
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠️ spaCy 不可用，將使用外部 NER API")

# --- 從 config.py 匯入設定 ---
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from config import (
        NER_API_URL, 
        EXTERNAL_API_URL, 
        EXTERNAL_API_HEADERS,
        GOOGLE_API_KEY,
        STABLE_FLASH_MODEL,
        GENERATION_CONFIG,
        DATA_CLEANING_CONFIG
    )
    print("✅ 成功載入 NER 和 Gemini API 相關配置")
except ImportError as e:
    print(f"❌ 錯誤: 找不到 config.py 或必要的設定: {e}")
    print("請確保 config.py 中包含所有必要的配置。")
    exit()

# 配置 Gemini API
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("✅ Gemini API 金鑰配置成功！")
except Exception as e:
    print(f"❌ 配置 Gemini API 金鑰時發生錯誤: {e}")
    exit()

# 初始化 Gemini 模型
model = genai.GenerativeModel(
    model_name=STABLE_FLASH_MODEL,
    generation_config=GENERATION_CONFIG
)
print(f"✅ Gemini 模型 '{model.model_name}' 已準備就緒")

# --- 函式定義 ---

class LocalNERService:
    """內建的 NER 服務類"""
    
    def __init__(self):
        self.nlp = None
        self.model_loaded = False
        self.loaded_model_name = None
        
        if SPACY_AVAILABLE:
            # 嘗試載入 Transformer 模型
            model_options = ["en_core_web_trf", "en_core_web_sm"]
            
            print(f"🔍 嘗試載入 spaCy 模型，候選項目: {model_options}")
            
            for model_name in model_options:
                try:
                    print(f"   正在嘗試載入: {model_name}")
                    self.nlp = spacy.load(model_name)
                    self.model_loaded = True
                    self.loaded_model_name = model_name
                    print(f"✅ spaCy NER 模型 '{model_name}' 成功載入！")
                    
                    # 測試模型功能
                    test_doc = self.nlp("Apple Inc. is based in Cupertino.")
                    test_entities = [ent.text for ent in test_doc.ents]
                    print(f"   🧪 模型測試: 識別出 {len(test_entities)} 個實體: {test_entities}")
                    break
                except OSError as e:
                    print(f"   ❌ 模型 '{model_name}' 載入失敗: {e}")
                    continue
                except Exception as e:
                    print(f"   💥 模型 '{model_name}' 測試失敗: {e}")
                    continue
            
            if not self.model_loaded:
                print("❌ 錯誤：找不到任何可用的 spaCy 模型。")
                print("💡 建議執行以下命令安裝模型：")
                print("   uv run python -m spacy download en_core_web_sm")
                print("   uv run python -m spacy download en_core_web_trf")
                print("🔄 將回退到外部 NER API。")
        else:
            print("❌ spaCy 不可用，將只使用外部 NER API")
    
    def get_model_info(self):
        """返回當前載入的模型信息"""
        if self.model_loaded and self.nlp:
            return {
                "model_name": self.loaded_model_name,
                "model_loaded": True,
                "spacy_version": spacy.__version__,
                "model_lang": self.nlp.lang,
                "pipeline_components": list(self.nlp.pipe_names)
            }
        else:
            return {
                "model_name": None,
                "model_loaded": False,
                "spacy_version": spacy.__version__ if SPACY_AVAILABLE else "N/A",
                "error": "No model loaded"
            }
    
    def extract_entities(self, text: str) -> List[Dict]:
        """使用內建 spaCy 模型提取實體"""
        if not self.model_loaded or not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            raw_entities = [
                {
                    "text": ent.text, 
                    "label": ent.label_,
                    "confidence": 1.0,  # spaCy 不提供信心度，設為 1.0
                    "start": ent.start_char,
                    "end": ent.end_char
                } 
                for ent in doc.ents
            ]
            
            # 後處理：過濾和修正實體
            filtered_entities = self._post_process_entities(raw_entities, text)
            return filtered_entities
            
        except Exception as e:
            print(f"⚠️ 內建 NER 服務錯誤: {e}")
            return []
    
    def _post_process_entities(self, entities: List[Dict], text: str) -> List[Dict]:
        """後處理實體，過濾誤識別和修正邊界"""
        filtered_entities = []
        
        # 定義過濾規則
        exclude_patterns = {
            'PERSON': [r'^Fig$', r'^mA$', r'^Table$', r'^\d+$'],  # 排除圖表標籤被誤識為人名
            'ORG': [r'^Table\s+\d+$', r'^Fig\s+\d+$'],  # 排除表格/圖片標籤被誤識為組織
            'GPE': [r'^\d+$', r'^[A-Z]{1,3}$']  # 排除單個字母或數字被誤識為地名
        }
        
        # 定義最小長度要求
        min_lengths = {
            'PERSON': 2,
            'ORG': 2,
            'GPE': 2
        }
        
        for entity in entities:
            entity_text = entity['text'].strip()
            entity_label = entity['label']
            
            # 跳過空實體
            if not entity_text:
                continue
            
            # 檢查最小長度
            if entity_label in min_lengths and len(entity_text) < min_lengths[entity_label]:
                continue
            
            # 檢查排除模式
            should_exclude = False
            if entity_label in exclude_patterns:
                for pattern in exclude_patterns[entity_label]:
                    if re.match(pattern, entity_text):
                        should_exclude = True
                        break
            
            if should_exclude:
                continue
            
            # 清理實體文本
            cleaned_text = self._clean_entity_text(entity_text)
            if cleaned_text:
                entity['text'] = cleaned_text
                filtered_entities.append(entity)
        
        return filtered_entities
    
    def _clean_entity_text(self, text: str) -> str:
        """清理實體文本"""
        # 移除前後標點符號
        text = re.sub(r'^[^\w]+|[^\w]+$', '', text)
        
        # 移除多餘空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text if len(text) >= 2 else ""
    
    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """去除重複實體，保留最長版本"""
        if not entities:
            return entities
        
        # 按類型分組
        entities_by_type = {}
        for entity in entities:
            label = entity['label']
            if label not in entities_by_type:
                entities_by_type[label] = []
            entities_by_type[label].append(entity)
        
        deduplicated = []
        for label, type_entities in entities_by_type.items():
            # 按文本內容去重，保留最長的版本
            unique_entities = {}
            for entity in type_entities:
                text = entity['text'].lower()
                if text not in unique_entities or len(entity['text']) > len(unique_entities[text]['text']):
                    unique_entities[text] = entity
            
            deduplicated.extend(unique_entities.values())
        
        return deduplicated

# 全域 NER 服務實例
local_ner_service = LocalNERService()

def fetch_chunks_from_api(query: str) -> list:
    """
    向外部 API 發送請求以獲取與查詢相關的文件 chunks。
    """
    print(f"\n正在向外部資料 API 查詢: '{query}'")
    print(f"API 端點: {EXTERNAL_API_URL}")
    
    data_to_send = {"query": query}
    
    try:
        print(f"📤 發送請求數據: {data_to_send}")
        print(f"📤 請求標頭: {EXTERNAL_API_HEADERS}")
        
        response = requests.post(EXTERNAL_API_URL, headers=EXTERNAL_API_HEADERS, json=data_to_send, timeout=30)
        
        print(f"📥 回應狀態碼: {response.status_code}")
        print(f"📥 回應標頭: {dict(response.headers)}")
        
        if response.status_code != 200:
            print(f"❌ API 錯誤詳情: {response.text}")
            print(f"❌ 請求 URL: {response.url}")
            
            # 提供測試數據作為回退
            print("🔄 使用測試數據作為回退...")
            return [
                {
                    "id": "test_fallback_001",
                    "content": f"This is a test document about {query}. Apple Inc. is a major technology company founded by Steve Jobs in Cupertino, California. The company is known for innovative products like iPhone, iPad, and Mac computers."
                },
                {
                    "id": "test_fallback_002", 
                    "content": f"Another document discussing {query}. TSMC (Taiwan Semiconductor Manufacturing Company) is the world's largest contract chip manufacturer, producing semiconductors for companies like Apple, Nvidia, AMD, and Qualcomm."
                }
            ]
        
        response.raise_for_status()
        print("✅ 成功從資料 API 獲取 chunk！")
        
        # 假設 API 回應的格式是 JSON，且 chunks 就在其中
        api_response = response.json()
        print(f"📋 API 回應類型: {type(api_response)}")
        
        # 處理 API 可能返回單一物件或列表的情況
        if isinstance(api_response, list):
            print(f"📋 返回列表，包含 {len(api_response)} 個項目")
            return api_response
        elif isinstance(api_response, dict):
            print(f"📋 返回字典，鍵: {list(api_response.keys())}")
            # 如果 API 返回的是一個字典，我們假設 chunks 在某個鍵下，或它本身就是一個 chunk
            # 這裡我們簡單地將其放入列表中以統一處理，您可以根據實際情況調整
            return [api_response]
        else:
            print(f"⚠️ 警告: 從 API 收到了未知的資料格式: {type(api_response)}")
            return []
            
    except requests.exceptions.Timeout:
        print(f"⏰ 錯誤: API 請求超時")
        return None
    except requests.exceptions.ConnectionError:
        print(f"🔌 錯誤: 無法連接到 API 伺服器")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ 錯誤: 外部資料 API 請求失敗: {e}")
        print(f"   請求 URL: {EXTERNAL_API_URL}")
        print(f"   請求數據: {data_to_send}")
        print(f"   請求標頭: {EXTERNAL_API_HEADERS}")
        return None
    except Exception as e:
        print(f"💥 未預期的錯誤: {e}")
        return None


def clean_text_for_ner(text: str) -> str:
    """
    為 NER 分析清理文本數據
    """
    if not text or not text.strip():
        return ""
    
    original_text = text
    cleaning_stats = {
        "original_length": len(text),
        "steps_applied": []
    }
    
    # 1. 移除 HTML 標籤
    if DATA_CLEANING_CONFIG.get("remove_html_tags", True):
        text = re.sub(r'<[^>]+>', '', text)
        text = unescape(text)  # 解碼 HTML 實體
        cleaning_stats["steps_applied"].append("HTML標籤移除")
    
    # 2. 移除特定模式（如圖片標籤）
    for pattern in DATA_CLEANING_CONFIG.get("remove_patterns", []):
        old_len = len(text)
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE | re.DOTALL)
        if len(text) < old_len:
            cleaning_stats["steps_applied"].append(f"模式移除: {pattern[:20]}...")
    
    # 3. 移除 URL
    if DATA_CLEANING_CONFIG.get("remove_urls", True):
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        cleaning_stats["steps_applied"].append("URL移除")
    
    # 4. 移除 email 地址
    if DATA_CLEANING_CONFIG.get("remove_emails", True):
        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', ' ', text)
        cleaning_stats["steps_applied"].append("Email移除")
    
    # 5. Unicode 正規化
    if DATA_CLEANING_CONFIG.get("normalize_unicode", True):
        text = unicodedata.normalize('NFKC', text)
        cleaning_stats["steps_applied"].append("Unicode正規化")
    
    # 6. 移除多餘的空白字符
    if DATA_CLEANING_CONFIG.get("remove_extra_whitespace", True):
        # 保留單個空格，移除多餘空白
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        cleaning_stats["steps_applied"].append("空白字符清理")
    
    # 7. 移除特殊字符（可選，可能影響科學符號）
    if DATA_CLEANING_CONFIG.get("remove_special_chars", False):
        # 保留基本標點和字母數字
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'°%\/]', ' ', text)
        cleaning_stats["steps_applied"].append("特殊字符移除")
    
    # 8. 檢查文本長度限制
    min_length = DATA_CLEANING_CONFIG.get("min_text_length", 10)
    max_length = DATA_CLEANING_CONFIG.get("max_text_length", 10000)
    
    if len(text) < min_length:
        cleaning_stats["steps_applied"].append(f"文本過短 ({len(text)} < {min_length})")
        return ""
    
    if len(text) > max_length:
        text = text[:max_length]
        cleaning_stats["steps_applied"].append(f"文本截斷 (>{max_length})")
    
    # 最終清理
    text = text.strip()
    
    cleaning_stats["final_length"] = len(text)
    cleaning_stats["reduction_ratio"] = 1 - (len(text) / len(original_text)) if len(original_text) > 0 else 0
    
    return text


def get_entities_from_ner_service(text: str, enable_cleaning: bool = True, prefer_local: bool = True) -> Dict:
    """
    向 NER 服務發送請求，優先使用內建服務，包含數據清理功能
    
    Args:
        text: 要分析的文本
        enable_cleaning: 是否啟用數據清理
        prefer_local: 是否優先使用內建 NER 服務
    
    Returns:
        包含實體和清理統計的字典
    """
    if not text or not text.strip():
        return {
            "entities": [],
            "cleaning_applied": False,
            "service_used": "none"
        }

    original_text = text
    cleaning_applied = False
    
    # 數據清理
    if enable_cleaning:
        cleaned_text = clean_text_for_ner(text)
        if cleaned_text != text:
            cleaning_applied = True
            text = cleaned_text
        
        if not text:  # 清理後文本為空
            return {
                "entities": [],
                "cleaning_applied": cleaning_applied,
                "service_used": "none"
            }
    
    # 嘗試使用內建 NER 服務
    entities = []
    service_used = "external_api"
    
    if prefer_local and local_ner_service.model_loaded:
        try:
            entities = local_ner_service.extract_entities(text)
            service_used = "local_spacy"
            print(f"🧠 使用內建 spaCy NER 服務")
        except Exception as e:
            print(f"⚠️ 內建 NER 服務失敗: {e}")
            entities = []
    
    # 如果內建服務失敗或不可用，使用外部 API
    if not entities and not prefer_local or (prefer_local and not entities):
        try:
            print(f"🌐 使用外部 NER API: {NER_API_URL}")
            response = requests.post(NER_API_URL, json={"text": text}, timeout=10)
            response.raise_for_status()
            api_entities = response.json().get("entities", [])
            
            # 轉換外部 API 格式為統一格式
            entities = []
            for entity in api_entities:
                if isinstance(entity, dict):
                    entities.append({
                        "text": entity.get("text", ""),
                        "label": entity.get("label", "UNKNOWN"),
                        "confidence": entity.get("confidence", 1.0)
                    })
            
            service_used = "external_api"
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 外部 NER API 請求失敗: {e}")
            entities = [{"text": f"NER Service Error: {str(e)}", "label": "ERROR", "confidence": 0.0}]
            service_used = "error"
    
    return {
        "entities": entities,
        "cleaning_applied": cleaning_applied,
        "original_length": len(original_text),
        "cleaned_length": len(text),
        "service_used": service_used
    }


def analyze_entities_with_gemini(entities: List[Dict], text: str, query: str) -> Dict:
    """
    使用 Gemini API 分析實體識別結果
    """
    if not entities:
        return {
            "entity_count": 0,
            "gemini_analysis": "未識別到任何實體",
            "entity_summary": {}
        }
    
    # 準備實體資訊
    entity_info = []
    entity_types = {}
    
    for entity in entities:
        entity_text = entity.get('text', '')
        entity_label = entity.get('label', 'UNKNOWN')
        confidence = entity.get('confidence', 1.0)
        
        entity_info.append(f"- {entity_text} ({entity_label}, 信心度: {confidence:.2f})")
        
        if entity_label not in entity_types:
            entity_types[entity_label] = 0
        entity_types[entity_label] += 1
    
    # 構建 Gemini 分析提示
    prompt = f"""
請分析以下 NER (命名實體識別) 結果：

原始查詢: {query}
文本長度: {len(text)} 字符
識別到的實體數量: {len(entities)}

實體列表:
{chr(10).join(entity_info)}

實體類型統計:
{chr(10).join([f"- {label}: {count}個" for label, count in entity_types.items()])}

請提供以下分析：
1. 實體識別的整體品質評估
2. 實體與查詢的相關性
3. 實體的實用性和重要性
4. 是否有遺漏的重要實體
5. 對實體結果的總結和建議

請用繁體中文回答，提供詳細但簡潔的分析。
"""
    
    try:
        response = model.generate_content(prompt)
        gemini_analysis = response.text
    except Exception as e:
        gemini_analysis = f"Gemini 分析失敗: {str(e)}"
    
    return {
        "entity_count": len(entities),
        "entity_types": entity_types,
        "gemini_analysis": gemini_analysis,
        "entity_summary": {
            "total_entities": len(entities),
            "unique_types": len(entity_types),
            "most_common_type": max(entity_types, key=entity_types.get) if entity_types else "無"
        }
    }


def display_ner_analysis(chunk_id: str, content: str, ner_result: Dict, analysis: Dict, query: str):
    """
    顯示 NER 分析結果，包含 Gemini 智能分析
    """
    entities = ner_result.get("entities", [])
    
    print(f"\n{'='*60}")
    print(f"📄 文件 ID: {chunk_id}")
    print(f"📝 內容長度: {len(content)} 字符")
    print(f"🎯 內容預覽: \"{content[:100]}...\"")
    print(f"🔧 NER 服務: {ner_result.get('service_used', 'unknown')}")
    print(f"📋 查詢問題: {query}")
    print(f"{'='*60}")
    
    # 顯示數據清理信息
    if ner_result.get("cleaning_applied", False):
        print(f"🧹 數據清理:")
        print(f"   📏 原始長度: {ner_result.get('original_length', 0)} 字符")
        print(f"   ✂️  清理後長度: {ner_result.get('cleaned_length', 0)} 字符")
        reduction = ner_result.get('original_length', 0) - ner_result.get('cleaned_length', 0)
        if reduction > 0:
            reduction_pct = (reduction / ner_result.get('original_length', 1)) * 100
            print(f"   📉 移除內容: {reduction} 字符 ({reduction_pct:.1f}%)")
    else:
        print(f"🧹 數據清理: 未啟用")
    
    # 顯示實體統計
    print(f"\n📊 NER 識別結果:")
    print(f"   🔢 實體總數: {analysis['entity_count']}")
    
    if analysis['entity_types']:
        print(f"    實體類型分布:")
        sorted_types = sorted(analysis['entity_types'].items(), key=lambda x: x[1], reverse=True)
        for entity_type, count in sorted_types:
            percentage = (count / analysis['entity_count']) * 100
            print(f"      {entity_type}: {count} ({percentage:.1f}%)")
    
    # 詳細實體列表
    print(f"\n🔍 識別出的實體:")
    if not entities:
        print("   (無)")
    else:
        # 按類型分組顯示
        entities_by_type = {}
        for entity in entities:
            entity_type = entity.get('label', 'UNKNOWN')
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)
        
        for entity_type, type_entities in entities_by_type.items():
            print(f"\n   📌 {entity_type}:")
            for entity in type_entities:
                entity_text = entity.get('text', '')
                confidence = entity.get('confidence', 'N/A')
                if isinstance(confidence, (int, float)):
                    confidence_str = f"{confidence:.2f}"
                else:
                    confidence_str = str(confidence)
                print(f"      • {entity_text} (信心度: {confidence_str})")
    
    # 顯示 Gemini 智能分析
    print(f"\n🤖 Gemini 智能分析:")
    print(f"{analysis['gemini_analysis']}")
    
    print(f"{'='*60}")


def analyze_rag_results_with_gemini(query: str, all_results: List[Dict]) -> str:
    """
    使用 Gemini API 分析整體 RAG 結果
    """
    # 準備 RAG 結果摘要
    total_chunks = len(all_results)
    total_entities = sum(result['analysis']['entity_count'] for result in all_results)
    
    # 收集所有實體類型
    all_entity_types = {}
    chunk_summaries = []
    
    for result in all_results:
        chunk_id = result['id']
        entity_count = result['analysis']['entity_count']
        entity_types = result['analysis']['entity_types']
        
        chunk_summaries.append(f"文件 {chunk_id}: {entity_count}個實體")
        
        for entity_type, count in entity_types.items():
            if entity_type not in all_entity_types:
                all_entity_types[entity_type] = 0
            all_entity_types[entity_type] += count
    
    # 構建 Gemini 分析提示
    prompt = f"""
請分析以下 RAG (檢索增強生成) 系統的整體表現：

使用者查詢: {query}

檢索結果統計:
- 檢索到的文件數量: {total_chunks}
- 總識別實體數量: {total_entities}
- 平均每文件實體數: {total_entities/max(total_chunks, 1):.1f}

各文件實體統計:
{chr(10).join(chunk_summaries)}

整體實體類型分布:
{chr(10).join([f"- {label}: {count}個" for label, count in sorted(all_entity_types.items(), key=lambda x: x[1], reverse=True)])}

請提供以下分析：
1. RAG 檢索結果的品質評估
2. 實體識別對回答使用者查詢的幫助程度
3. 檢索到的文件是否能夠充分回答使用者問題
4. 實體分布是否符合查詢主題
5. 整體系統表現的改進建議

請用繁體中文回答，提供詳細的分析和建議。
"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini 整體分析失敗: {str(e)}"


def generate_summary_report(all_results: List[Dict], query: str):
    """
    生成整體的 NER 和 RAG 分析摘要報告
    """
    print(f"\n{'='*80}")
    print(f"整體 NER + RAG 分析摘要報告")
    print(f"{'='*80}")
    
    total_chunks = len(all_results)
    total_entities = sum(result['analysis']['entity_count'] for result in all_results)
    
    print(f"基本統計:")
    print(f" 查詢問題: {query}")
    print(f" 文件塊總數: {total_chunks}")
    print(f" 實體總數: {total_entities}")
    print(f" 平均每文件實體數: {total_entities/max(total_chunks, 1):.1f}")
    
    # 獲取 Gemini 整體分析
    print(f"\n🤖 Gemini 整體 RAG 分析:")
    rag_analysis = analyze_rag_results_with_gemini(query, all_results)
    print(rag_analysis)
    
    print(f"{'='*80}")


# --- 主程式執行區塊 ---

if __name__ == "__main__":
    # --- 步驟 1: 接收使用者輸入 ---
    user_query = input("請輸入您的查詢問題以獲取相關文件 chunks: ")
    if not user_query:
        print("查詢不能為空。")
        exit()

    print("\n" + "="*80)
    print("從 API 檢索文件並進行 NER + Gemini 智能分析")
    print("="*80)

    # --- 步驟 2: 從您的 API 獲取文件 chunks ---
    retrieved_chunks = fetch_chunks_from_api(user_query)

    if retrieved_chunks is None:
        print("❌ 無法獲取資料，程式終止。")
        print("💡 建議執行測試: uv run python ner_test_suite.py")
        exit()
    
    if not retrieved_chunks:
        print("⚠️ API 未返回任何文件 chunk。")
        exit()

    print(f"\n✅ 共檢索到 {len(retrieved_chunks)} 份文件 chunk，現逐一進行 NER + Gemini 分析...")

    # --- 步驟 3: 對每個 chunk 進行 NER 並使用 Gemini 分析 ---
    all_results = []
    for i, chunk in enumerate(retrieved_chunks):
        # 假設 chunk 是字典，且內容在 'content' 鍵下
        content_to_analyze = chunk.get("content", "")
        chunk_id = chunk.get("id", f"unknown_{i+1}")

        # 呼叫 NER 服務
        extracted_entities = get_entities_from_ner_service(content_to_analyze)
        
        # 使用 Gemini 分析實體結果
        analysis = analyze_entities_with_gemini(
            extracted_entities.get('entities', []), 
            content_to_analyze, 
            user_query
        )
        
        # 儲存結果
        result = {
            "id": chunk_id,
            "entities": extracted_entities,
            "analysis": analysis,
            "content_length": len(content_to_analyze)
        }
        all_results.append(result)

        # 顯示詳細分析結果
        display_ner_analysis(chunk_id, content_to_analyze, extracted_entities, analysis, user_query)

    # --- 步驟 4: 生成整體摘要報告（包含 Gemini RAG 分析） ---
    generate_summary_report(all_results, user_query)
    
    # --- 步驟 5: 可選擇性地將結果存儲到檔案中 ---
    save_results = input("\n是否要將分析結果保存到 JSON 檔案？ (y/N): ").lower().strip()
    if save_results in ['y', 'yes', '是']:
        filename = "ner_gemini_analysis_results.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"✅ 分析結果已儲存到 {filename}")
    
    print("\n🎉 NER + Gemini 智能分析完成！")