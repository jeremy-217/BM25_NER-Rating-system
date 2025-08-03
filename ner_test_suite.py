#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NER 系統測試套件
獨立的測試程式，用於驗證 NER 系統的各項功能

功能包括：
- 基本 NER 功能測試
- API 連接測試
- 模型載入測試
- 系統診斷
- 錯誤處理測試
- 回退機制測試

使用方法：
    uv run python ner_test_suite.py
"""

import sys
import os
import json
import time
from typing import Dict, List

# 添加路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 導入 NER 系統組件
try:
    from ner import (
        LocalNERService, 
        get_entities_from_ner_service,
        analyze_entities_with_gemini,
        fetch_chunks_from_api,
        local_ner_service
    )
    from config import (
        EXTERNAL_API_URL, 
        EXTERNAL_API_HEADERS,
        GOOGLE_API_KEY,
        STABLE_FLASH_MODEL
    )
    print("✅ 成功導入 NER 系統組件")
except ImportError as e:
    print(f"❌ 導入失敗: {e}")
    sys.exit(1)

class NERTestSuite:
    """NER 系統測試套件"""
    
    def __init__(self):
        self.test_results = []
        self.start_time = time.time()
        
        # 測試用例
        self.test_cases = [
            {
                "name": "基本實體識別",
                "text": "Apple Inc. is a technology company founded by Steve Jobs in Cupertino, California.",
                "expected_entities": ["Apple Inc", "Steve Jobs", "Cupertino", "California"]
            },
            {
                "name": "半導體產業",
                "text": "TSMC is manufacturing 3nm chips for Apple and Nvidia. Intel is also developing 18A process technology.",
                "expected_entities": ["TSMC", "Apple", "Nvidia", "Intel"]
            },
            {
                "name": "日期和數字",
                "text": "In 2024, the company achieved 95% success rate with $2.5 billion revenue.",
                "expected_entities": ["2024", "95%", "$2.5 billion"]
            },
            {
                "name": "複雜科技文本",
                "text": "Samsung's 3nm GAA technology competes with TSMC's N3 process. Both companies target Apple's A17 chip production.",
                "expected_entities": ["Samsung", "TSMC", "Apple", "A17"]
            },
            {
                "name": "地理和組織",
                "text": "Microsoft headquarters in Redmond, Washington, while Google operates from Mountain View, California.",
                "expected_entities": ["Microsoft", "Redmond", "Washington", "Google", "Mountain View", "California"]
            }
        ]
    
    def run_all_tests(self):
        """執行所有測試"""
        print("🧪 開始 NER 系統完整測試套件")
        print("=" * 80)
        
        # 1. 系統診斷測試
        self.test_system_diagnostics()
        
        # 2. 模型載入測試
        self.test_model_loading()
        
        # 3. 基本 NER 功能測試
        self.test_basic_ner_functionality()
        
        # 4. API 連接測試
        self.test_api_connectivity()
        
        # 5. 回退機制測試
        self.test_fallback_mechanism()
        
        # 6. Gemini 分析測試
        self.test_gemini_analysis()
        
        # 7. 壓力測試
        self.test_performance()
        
        # 生成測試報告
        self.generate_test_report()
    
    def test_system_diagnostics(self):
        """系統診斷測試"""
        print("\n🔧 測試 1: 系統診斷")
        print("-" * 50)
        
        try:
            model_info = local_ner_service.get_model_info()
            
            print(f"📊 spaCy 模型狀態:")
            for key, value in model_info.items():
                print(f"   {key}: {value}")
            
            # 檢查 Gemini API 配置
            api_configured = bool(GOOGLE_API_KEY and GOOGLE_API_KEY.strip())
            print(f"\n🤖 Gemini API 配置: {'✅ 已配置' if api_configured else '❌ 未配置'}")
            
            # 檢查外部 API 配置
            external_api_configured = bool(EXTERNAL_API_URL and EXTERNAL_API_HEADERS)
            print(f"🌐 外部 API 配置: {'✅ 已配置' if external_api_configured else '❌ 未配置'}")
            
            self.test_results.append({
                "test": "系統診斷",
                "status": "PASS",
                "details": {
                    "spacy_model": model_info.get("model_loaded", False),
                    "gemini_api": api_configured,
                    "external_api": external_api_configured
                }
            })
            
        except Exception as e:
            print(f"❌ 系統診斷失敗: {e}")
            self.test_results.append({
                "test": "系統診斷",
                "status": "FAIL",
                "error": str(e)
            })
    
    def test_model_loading(self):
        """模型載入測試"""
        print("\n🧠 測試 2: 模型載入")
        print("-" * 50)
        
        try:
            # 測試模型是否正確載入
            if local_ner_service.model_loaded:
                print("✅ spaCy 模型成功載入")
                
                # 快速功能測試
                test_text = "Test entity recognition with Apple Inc."
                entities = local_ner_service.extract_entities(test_text)
                
                print(f"🧪 快速測試: 識別到 {len(entities)} 個實體")
                
                self.test_results.append({
                    "test": "模型載入",
                    "status": "PASS",
                    "details": {
                        "model_name": local_ner_service.loaded_model_name,
                        "entities_found": len(entities)
                    }
                })
            else:
                print("⚠️ spaCy 模型未載入，將使用外部 API")
                self.test_results.append({
                    "test": "模型載入",
                    "status": "WARNING",
                    "details": "使用外部 API 回退"
                })
                
        except Exception as e:
            print(f"❌ 模型載入測試失敗: {e}")
            self.test_results.append({
                "test": "模型載入",
                "status": "FAIL",
                "error": str(e)
            })
    
    def test_basic_ner_functionality(self):
        """基本 NER 功能測試"""
        print("\n🔍 測試 3: 基本 NER 功能")
        print("-" * 50)
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n   測試案例 {i}: {test_case['name']}")
            print(f"   文本: {test_case['text'][:60]}...")
            
            try:
                # 執行 NER
                result = get_entities_from_ner_service(test_case['text'], prefer_local=True)
                entities = result.get('entities', [])
                
                # 統計結果
                entity_texts = [entity.get('text', '') for entity in entities]
                found_expected = sum(1 for expected in test_case['expected_entities'] 
                                   if any(expected.lower() in entity.lower() for entity in entity_texts))
                
                print(f"   🔢 識別實體: {len(entities)} 個")
                print(f"   ✅ 期望實體: {found_expected}/{len(test_case['expected_entities'])} 個")
                print(f"   🔧 使用服務: {result.get('service_used', 'unknown')}")
                
                # 顯示識別的實體
                if entities:
                    print(f"   📋 實體列表: {', '.join([e.get('text', '') for e in entities[:5]])}")
                
                self.test_results.append({
                    "test": f"NER功能-{test_case['name']}",
                    "status": "PASS",
                    "details": {
                        "entities_found": len(entities),
                        "expected_found": found_expected,
                        "service_used": result.get('service_used')
                    }
                })
                
            except Exception as e:
                print(f"   ❌ 測試失敗: {e}")
                self.test_results.append({
                    "test": f"NER功能-{test_case['name']}",
                    "status": "FAIL",
                    "error": str(e)
                })
    
    def test_api_connectivity(self):
        """API 連接測試"""
        print("\n🌐 測試 4: API 連接")
        print("-" * 50)
        
        try:
            # 使用簡單查詢測試 API
            test_query = "apple"
            print(f"測試查詢: '{test_query}'")
            
            result = fetch_chunks_from_api(test_query)
            
            if result is not None:
                print(f"✅ API 連接成功，返回 {len(result)} 個項目")
                self.test_results.append({
                    "test": "API連接",
                    "status": "PASS",
                    "details": {"chunks_returned": len(result)}
                })
            else:
                print("❌ API 連接失敗")
                self.test_results.append({
                    "test": "API連接",
                    "status": "FAIL",
                    "error": "API 返回 None"
                })
                
        except Exception as e:
            print(f"❌ API 連接測試失敗: {e}")
            self.test_results.append({
                "test": "API連接",
                "status": "FAIL",
                "error": str(e)
            })
    
    def test_fallback_mechanism(self):
        """回退機制測試"""
        print("\n🔄 測試 5: 回退機制")
        print("-" * 50)
        
        try:
            # 測試本地服務回退到外部 API
            test_text = "Testing fallback with Microsoft and Amazon."
            
            print("測試本地優先模式...")
            result_local = get_entities_from_ner_service(test_text, prefer_local=True)
            
            print("測試外部 API 模式...")
            result_external = get_entities_from_ner_service(test_text, prefer_local=False)
            
            print(f"✅ 本地模式: {result_local.get('service_used')} - {len(result_local.get('entities', []))} 個實體")
            print(f"✅ 外部模式: {result_external.get('service_used')} - {len(result_external.get('entities', []))} 個實體")
            
            self.test_results.append({
                "test": "回退機制",
                "status": "PASS",
                "details": {
                    "local_service": result_local.get('service_used'),
                    "external_service": result_external.get('service_used')
                }
            })
            
        except Exception as e:
            print(f"❌ 回退機制測試失敗: {e}")
            self.test_results.append({
                "test": "回退機制",
                "status": "FAIL",
                "error": str(e)
            })
    
    def test_gemini_analysis(self):
        """Gemini 分析測試"""
        print("\n🤖 測試 6: Gemini 分析")
        print("-" * 50)
        
        try:
            # 準備測試數據
            test_text = "Apple Inc. and Microsoft are leading technology companies in the United States."
            test_query = "technology companies"
            
            # 先進行 NER
            ner_result = get_entities_from_ner_service(test_text)
            entities = ner_result.get('entities', [])
            
            if entities:
                print(f"使用 {len(entities)} 個實體進行 Gemini 分析...")
                
                # 執行 Gemini 分析
                analysis = analyze_entities_with_gemini(entities, test_text, test_query)
                
                print(f"✅ Gemini 分析完成")
                print(f"   實體數量: {analysis.get('entity_count', 0)}")
                print(f"   分析長度: {len(analysis.get('gemini_analysis', ''))} 字符")
                
                self.test_results.append({
                    "test": "Gemini分析",
                    "status": "PASS",
                    "details": {
                        "entities_analyzed": analysis.get('entity_count', 0),
                        "analysis_length": len(analysis.get('gemini_analysis', ''))
                    }
                })
            else:
                print("⚠️ 無實體可供分析")
                self.test_results.append({
                    "test": "Gemini分析",
                    "status": "WARNING",
                    "details": "無實體可分析"
                })
                
        except Exception as e:
            print(f"❌ Gemini 分析測試失敗: {e}")
            self.test_results.append({
                "test": "Gemini分析",
                "status": "FAIL",
                "error": str(e)
            })
    
    def test_performance(self):
        """性能測試"""
        print("\n⚡ 測試 7: 性能測試")
        print("-" * 50)
        
        try:
            # 準備長文本
            long_text = """
            Apple Inc. is an American multinational technology company headquartered in Cupertino, California. 
            Apple is the world's largest technology company by revenue and the world's most valuable company. 
            The company was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. 
            Apple's hardware products include the iPhone smartphone, the iPad tablet computer, the Mac personal computer, 
            the iPod portable media player, the Apple Watch smartwatch, the Apple TV digital media player, 
            and the HomePod smart speaker. Apple's software includes the macOS and iOS operating systems, 
            the iTunes media player, the Safari web browser, and the iLife and iWork creativity and productivity suites.
            """ * 3  # 重複 3 次增加長度
            
            print(f"測試文本長度: {len(long_text)} 字符")
            
            # 計時測試
            start_time = time.time()
            result = get_entities_from_ner_service(long_text)
            end_time = time.time()
            
            processing_time = end_time - start_time
            entities_count = len(result.get('entities', []))
            
            print(f"✅ 處理時間: {processing_time:.2f} 秒")
            print(f"✅ 識別實體: {entities_count} 個")
            print(f"✅ 處理速度: {len(long_text)/processing_time:.0f} 字符/秒")
            
            self.test_results.append({
                "test": "性能測試",
                "status": "PASS",
                "details": {
                    "text_length": len(long_text),
                    "processing_time": processing_time,
                    "entities_found": entities_count,
                    "chars_per_second": len(long_text)/processing_time
                }
            })
            
        except Exception as e:
            print(f"❌ 性能測試失敗: {e}")
            self.test_results.append({
                "test": "性能測試",
                "status": "FAIL",
                "error": str(e)
            })
    
    def generate_test_report(self):
        """生成測試報告"""
        print("\n" + "=" * 80)
        print("📊 測試報告")
        print("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['status'] == 'PASS')
        failed_tests = sum(1 for result in self.test_results if result['status'] == 'FAIL')
        warning_tests = sum(1 for result in self.test_results if result['status'] == 'WARNING')
        
        total_time = time.time() - self.start_time
        
        print(f"📈 測試統計:")
        print(f"   總測試數: {total_tests}")
        print(f"   ✅ 通過: {passed_tests}")
        print(f"   ❌ 失敗: {failed_tests}")
        print(f"   ⚠️ 警告: {warning_tests}")
        print(f"   ⏱️ 總時間: {total_time:.2f} 秒")
        print(f"   📊 成功率: {(passed_tests/total_tests)*100:.1f}%")
        
        print(f"\n📋 詳細結果:")
        for result in self.test_results:
            status_icon = {"PASS": "✅", "FAIL": "❌", "WARNING": "⚠️"}[result['status']]
            print(f"   {status_icon} {result['test']}: {result['status']}")
            if 'error' in result:
                print(f"      錯誤: {result['error']}")
        
        # 保存詳細報告到文件
        report_filename = f"ner_test_report_{int(time.time())}.json"
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {
                    "total_tests": total_tests,
                    "passed": passed_tests,
                    "failed": failed_tests,
                    "warnings": warning_tests,
                    "success_rate": (passed_tests/total_tests)*100,
                    "total_time": total_time
                },
                "detailed_results": self.test_results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 詳細報告已保存到: {report_filename}")
        print("=" * 80)

def main():
    """主測試函數"""
    print("🧪 NER 系統測試套件")
    print("版本: 1.0")
    print("目的: 全面測試 NER 系統的各項功能")
    print("=" * 80)
    
    # 創建測試套件並執行
    test_suite = NERTestSuite()
    test_suite.run_all_tests()
    
    print("\n🎉 測試套件執行完成！")

if __name__ == "__main__":
    main()
