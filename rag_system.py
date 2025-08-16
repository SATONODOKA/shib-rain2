#!/usr/bin/env python3
"""
営業ナレッジベース RAGシステム - 簡潔版
法人向け研修事業の営業支援AIアシスタント
"""

import os
import glob
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
import requests
import datetime
from typing import List, Dict, Any

# 設定
LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
FILE_PATTERNS = ["*.md", "*.txt", "*.docx.md"]

class RAGSystem:
    def __init__(self):
        """RAGシステムの初期化"""
        self.setup_embedding_model()
        self.setup_chroma_db()
        self.check_lm_studio_connection()
        self.auto_load_documents()
        
    def setup_embedding_model(self):
        """埋め込みモデルの初期化"""
        try:
            self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-small')
            self.embedding_status = "✅ 埋め込みモデル初期化完了"
            return True
        except Exception as e:
            self.embedding_status = f"❌ 埋め込みモデル初期化失敗: {e}"
            return False
    
    def setup_chroma_db(self):
        """ChromaDBの初期化"""
        try:
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            try:
                self.collection = self.chroma_client.get_collection("sales_knowledge")
                self.db_status = f"📚 既存ナレッジベース読み込み完了（{self.collection.count()}件）"
            except:
                self.collection = self.chroma_client.create_collection(
                    name="sales_knowledge",
                    metadata={"description": "営業ナレッジベース"}
                )
                self.db_status = "📚 新規ナレッジベース作成完了"
            return True
        except Exception as e:
            self.db_status = f"❌ ChromaDB初期化失敗: {e}"
            return False
    
    def check_lm_studio_connection(self):
        """LM Studio接続状況の確認"""
        try:
            response = requests.get("http://localhost:1234/v1/models", timeout=5)
            if response.status_code == 200:
                models = response.json()
                if models.get("data"):
                    model_details = []
                    for model in models["data"]:
                        model_id = model.get("id", "unknown")
                        if "gpt-oss" in model_id.lower():
                            model_details.append(f"✅ {model_id} (推奨)")
                        else:
                            model_details.append(f"📋 {model_id}")
                    
                    self.lm_studio_status = f"✅ 接続済み - {', '.join(model_details)}"
                else:
                    self.lm_studio_status = "⚠️ 接続済み - モデルが見つかりません"
                return True
            else:
                self.lm_studio_status = f"❌ 接続エラー - ステータス: {response.status_code}"
                return False
        except requests.exceptions.ConnectionError:
            self.lm_studio_status = "❌ 未接続 - LM Studioを起動してください"
            return False
        except Exception as e:
            self.lm_studio_status = f"❌ 接続確認エラー: {e}"
            return False
    
    def auto_load_documents(self):
        """初期化時に自動でドキュメントを読み込み"""
        try:
            current_count = self.collection.count()
            if current_count > 0:
                self.db_status = f"✅ {current_count}件のドキュメントが利用可能です"
                return True
            
            documents = self.get_documents()
            if documents:
                self.reset_collection()
                total_chunks = sum(self.process_document(doc_path) for doc_path in documents)
                self.db_status = f"✅ {len(documents)}個のファイルから{total_chunks}件のドキュメントを読み込みました"
            else:
                self.db_status = "⚠️ sample_documentsフォルダが見つかりません"
        except Exception as e:
            self.db_status = f"❌ ドキュメント自動読み込みエラー: {e}"
    
    def get_documents(self, document_path: str = "sample_documents") -> List[str]:
        """ドキュメントファイルの取得"""
        if not os.path.exists(document_path):
            return []
        
        documents = []
        for pattern in FILE_PATTERNS:
            files = glob.glob(os.path.join(document_path, pattern))
            documents.extend(files)
        
        return sorted(documents)
    
    def reset_collection(self):
        """コレクションをリセット"""
        try:
            self.chroma_client.delete_collection("sales_knowledge")
        except:
            pass
        self.collection = self.chroma_client.create_collection(
            name="sales_knowledge",
            metadata={"description": "営業ナレッジベース"}
        )
    
    def process_document(self, file_path: str) -> int:
        """個別ドキュメントの処理"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                return 0
            
            filename = os.path.basename(file_path)
            if filename.endswith('.md'):
                filename = filename[:-3]
            elif filename.endswith('.txt'):
                filename = filename[:-4]
            
            chunks = self.split_text(content)
            chunks_added = 0
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    chunk_id = f"{filename}#chunk-{i+1}"
                    
                    # 重複チェック
                    existing = self.collection.query(
                        query_embeddings=[self.embedding_model.encode(chunk).tolist()],
                        n_results=1,
                        where={"chunk_id": chunk_id}
                    )
                    
                    if not existing["documents"][0]:
                        self.collection.add(
                            documents=[chunk],
                            metadatas=[{
                                "source": filename,
                                "chunk_id": chunk_id,
                                "file_path": file_path,
                                "chunk_index": i,
                                "timestamp": datetime.datetime.now().isoformat()
                            }],
                            ids=[f"{filename}_{i}"]
                        )
                        chunks_added += 1
            
            return chunks_added
            
        except Exception as e:
            if 'st' in globals():
                st.error(f"❌ ファイル処理エラー {file_path}: {e}")
            return 0
    
    def split_text(self, text: str) -> List[str]:
        """テキストの分割"""
        if len(text) <= CHUNK_SIZE:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + CHUNK_SIZE
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # 文の境界で分割を試行
            split_point = text.rfind('。', start, end)
            if split_point > start:
                end = split_point + 1
            
            chunks.append(text[start:end])
            start = end - CHUNK_OVERLAP
            
            if start >= len(text):
                break
        
        return chunks
    
    def search_similar_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """類似ドキュメントの検索"""
        try:
            if self.collection.count() == 0:
                if 'st' in globals():
                    st.warning("⚠️ ナレッジベースが空です。")
                return []
            
            query_embedding = self.embedding_model.encode(query).tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.collection.count()),
                include=["documents", "metadatas", "distances"]
            )
            
            if not results["documents"] or not results["documents"][0]:
                if 'st' in globals():
                    st.warning("⚠️ 検索結果が見つかりませんでした。")
                return []
            
            return [
                {
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "source": results["metadatas"][0][i]["source"]
                }
                for i in range(len(results["documents"][0]))
            ]
            
        except Exception as e:
            if 'st' in globals():
                st.error(f"❌ 検索エラー: {e}")
            return []
    
    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """LM Studioを使用して回答生成"""
        try:
            # コンテキストを簡潔にまとめる
            sorted_docs = sorted(context_docs, key=lambda x: x['distance'])
            context_summary = "\n".join([
                f"{doc['source']}: {doc['content'][:100]}"
                for doc in sorted_docs[:2]
            ])
            
            # 超軽量プロンプト
            prompt = f"""質問: {query}
参考: {context_summary}
回答:"""

            data = {
                "model": "gpt-oss-20b",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 150,
                "top_p": 0.8,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "stream": False
            }
            
            response = requests.post(
                LM_STUDIO_API_URL, 
                headers={"Content-Type": "application/json"}, 
                json=data, 
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    if len(content.strip()) < 5:
                        return self._generate_simple_answer(query, context_docs)
                    return content
                else:
                    return "❌ LM Studioからの応答形式が不正です"
            else:
                return f"❌ LM Studio APIエラー: {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return "❌ LM Studioに接続できません。LM Studioが起動していることを確認してください。"
        except requests.exceptions.Timeout:
            return "❌ LM Studioからの応答がタイムアウトしました。"
        except Exception as e:
            return f"❌ 回答生成エラー: {e}"
    
    def _generate_simple_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """シンプルな回答生成（フォールバック用）"""
        if not context_docs:
            return "申し訳ございませんが、関連する情報が見つかりませんでした。"
        
        best_doc = min(context_docs, key=lambda x: x['distance'])
        
        return f"""## 【回答】
質問: {query}

## 【関連情報】
{best_doc['source']}より:
{best_doc['content'][:300]}...

## 【アドバイス】
上記の情報を基に、営業活動に活用してください。"""
    
    def query(self, question: str) -> tuple[str, List[Dict[str, Any]]]:
        """RAGシステムのメイン処理"""
        self.check_lm_studio_connection()
        
        search_results = self.search_similar_documents(question, n_results=5)
        
        if not search_results:
            return "申し訳ございませんが、関連する情報が見つかりませんでした。", []
        
        # LM Studio接続チェックと回答生成
        try:
            response = requests.get("http://localhost:1234/v1/models", timeout=3)
            if response.status_code == 200:
                answer = self.generate_answer(question, search_results)
            else:
                answer = self._generate_simple_answer(question, search_results)
        except:
            answer = self._generate_simple_answer(question, search_results)
        
        return answer, search_results


def get_custom_css():
    """カスタムCSS"""
    return """
    <style>
    .main .block-container {
        padding-top: 0.5rem;
        padding-bottom: 2rem;
        max-width: 50rem;
        margin: 0 auto;
    }
    
    .main-title {
        text-align: left;
        color: #1f2937;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    
    .main-subtitle {
        text-align: left;
        color: #6b7280;
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
    }
    
    .stTextArea textarea {
        border: 1px solid #d1d5db;
        border-radius: 0.75rem;
        padding: 1rem;
        font-size: 0.95rem;
        line-height: 1.5;
        background-color: #ffffff;
        color: #111827 !important;
        resize: none;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    
    .stButton > button {
        background-color: #10b981;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.625rem 1.5rem;
        font-weight: 600;
        font-size: 0.875rem;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background-color: #059669;
        transform: translateY(-1px);
    }
    
    .ai-response {
        background-color: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 1rem 0;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #374151;
        max-height: 70vh;
        overflow-y: auto;
    }
    </style>
    """


def main():
    st.set_page_config(
        page_title="営業ナレッジベース RAGシステム",
        page_icon="💼",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # ヘッダー
    st.markdown('<h1 class="main-title">💼 営業ナレッジベース</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">法人向け研修事業の営業支援AIアシスタント</p>', unsafe_allow_html=True)
    
    # RAGシステムの初期化
    if 'rag_system' not in st.session_state:
        with st.spinner("システムを初期化中..."):
            st.session_state.rag_system = RAGSystem()
    
    rag_system = st.session_state.rag_system
    
    # サイドバー
    with st.sidebar:
        st.markdown("#### ⚙️ システム状態")
        
        if hasattr(rag_system, 'embedding_status'):
            status_text = "❌ システムエラー" if "❌" in rag_system.embedding_status else "✅ システム準備完了"
            st.caption(status_text)
        
        if hasattr(rag_system, 'db_status') and "件のドキュメント" in rag_system.db_status:
            import re
            match = re.search(r'(\d+)件のドキュメント', rag_system.db_status)
            if match:
                st.caption(f"📚 {match.group(1)}件のドキュメント利用可能")
        else:
            st.caption("📚 ドキュメント読み込み中...")
        
        if hasattr(rag_system, 'lm_studio_status'):
            if "✅" in rag_system.lm_studio_status:
                st.caption(f"🤖 {rag_system.lm_studio_status.split(' - ')[1] if ' - ' in rag_system.lm_studio_status else 'LM Studio接続済み'}")
            else:
                st.caption(f"❌ {rag_system.lm_studio_status}")
        
        if st.button("🔄 接続状況更新", use_container_width=True):
            rag_system.check_lm_studio_connection()
            st.rerun()
        
        st.markdown("---")
        
        # 会話履歴
        col_title, col_new = st.columns([2, 1])
        with col_title:
            st.markdown("### 💬 会話履歴")
        with col_new:
            if st.button("➕", help="新規チャット", use_container_width=True):
                st.session_state.chat_history = []
                if 'reuse_question' in st.session_state:
                    del st.session_state.reuse_question
                st.session_state.clear_input = True
                st.rerun()
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if st.session_state.chat_history:
            for i, chat in enumerate(st.session_state.chat_history[-5:]):
                display_text = f"💭 {chat['question'][:30]}..." if len(chat['question']) > 30 else f"💭 {chat['question']}"
                if st.button(display_text, key=f"history_{len(st.session_state.chat_history)-5+i}", use_container_width=True):
                    st.session_state.reuse_question = chat['question']
                    st.rerun()
        else:
            st.caption("まだ会話履歴がありません")
        
        if st.session_state.chat_history:
            if st.button("🗑️ 履歴をクリア", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        st.markdown("---")
        
        with st.expander("💡 使い方"):
            st.markdown("""
            1. メインエリアに営業に関する質問を入力
            2. AIが関連情報を検索して回答
            3. 会話履歴から過去の質問を再利用可能
            """)
        
        with st.expander("📝 質問例"):
            st.markdown("""
            **価格・料金系**
            - 新任管理職研修の料金は？
            - 研修の費用対効果は？
            
            **顧客情報系**
            - トヨタ自動車様の課題は？
            - 大成建設様への提案内容は？
            
            **競合・差別化系**
            - 競合他社との差別化ポイントは？
            - 建設業界向けの研修内容は？
            """)
    
    # メインコンテンツエリア
    st.markdown('<div style="margin-top: 3rem;"></div>', unsafe_allow_html=True)
    
    # 入力処理
    initial_question = ""
    if 'reuse_question' in st.session_state:
        initial_question = st.session_state.reuse_question
        del st.session_state.reuse_question
    
    form_key_suffix = ""
    if 'clear_input' in st.session_state:
        initial_question = ""
        if 'form_reset_counter' not in st.session_state:
            st.session_state.form_reset_counter = 0
        st.session_state.form_reset_counter += 1
        form_key_suffix = f"_{st.session_state.form_reset_counter}"
        del st.session_state.clear_input
    
    # 質問入力フォーム
    with st.form(f"question_form{form_key_suffix}", clear_on_submit=False):
        user_question = st.text_area(
            "質問を入力してください",
            value=initial_question,
            height=120,
            placeholder="営業に関することは何でもお答えします",
            help="💡 下のボタンをクリックして質問を送信してください",
            key=f"question_input{form_key_suffix}"
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            ask_button = st.form_submit_button(
                "🔍 質問する", 
                type="primary", 
                use_container_width=True
            )
    
    # 回答処理
    if ask_button:
        if user_question.strip():
            st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
            
            with st.spinner("💭 回答を生成中..."):
                answer, search_results = rag_system.query(user_question)
                
                # 会話履歴に保存
                chat_entry = {
                    "question": user_question,
                    "answer": answer,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "sources": [result['source'] for result in search_results] if search_results else []
                }
                
                st.session_state.chat_history.append(chat_entry)
                if len(st.session_state.chat_history) > 20:
                    st.session_state.chat_history = st.session_state.chat_history[-20:]
                
                # AI回答表示
                st.markdown("### 🤖 AI回答")
                st.markdown(f'<div class="ai-response">{answer}</div>', unsafe_allow_html=True)
                
                # 参考情報
                if search_results:
                    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
                    with st.expander("📋 参考にした情報", expanded=False):
                        for i, result in enumerate(search_results, 1):
                            st.markdown(f"**📄 参考情報 {i}**")
                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.markdown(f"**出典:** {result['source']}")
                            with col_b:
                                st.markdown(f"**類似度:** {1 - result['distance']:.3f}")
                            
                            st.markdown("**内容抜粋:**")
                            content_preview = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
                            st.markdown(f"_{content_preview}_")
                            
                            if i < len(search_results):
                                st.markdown("---")
        else:
            st.warning("質問を入力してください。", icon="⚠️")


if __name__ == "__main__":
    main() 