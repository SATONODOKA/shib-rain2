import os
import glob
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
import requests
import datetime
from typing import List, Dict, Any
import re

# 設定
LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
FILE_PATTERNS = ["*.md", "*.txt", "*.docx.md"]  # 対応ファイル形式を一箇所で定義

class RAGSystem:
    def __init__(self):
        self.embedding_status = ""
        self.db_status = ""
        self.lm_studio_status = "未確認"
        self.setup_embedding_model()
        self.setup_chroma_db()
        self.check_lm_studio_connection()
        self.auto_load_documents()
        
    def setup_embedding_model(self):
        """埋め込みモデルの初期化"""
        try:
            self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-small')
            self.embedding_status = "✅ 埋め込みモデルを初期化しました"
            return True
        except Exception as e:
            self.embedding_status = f"❌ 埋め込みモデルの初期化に失敗: {e}"
            return False
    
    def setup_chroma_db(self):
        """ChromaDBの初期化"""
        try:
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            try:
                self.collection = self.chroma_client.get_collection("sales_knowledge")
                self.db_status = f"📚 既存のナレッジベースを読み込みました（{self.collection.count()}件のドキュメント）"
            except:
                self.collection = self.chroma_client.create_collection(
                    name="sales_knowledge",
                    metadata={"description": "営業ナレッジベース"}
                )
                self.db_status = "📚 新しいナレッジベースを作成しました"
            return True
        except Exception as e:
            self.db_status = f"❌ ChromaDBの初期化に失敗: {e}"
            return False
    
    def check_lm_studio_connection(self):
        """LM Studio接続状況の確認"""
        try:
            response = requests.get("http://localhost:1234/v1/models", timeout=3)
            if response.status_code == 200:
                models = response.json()
                if models.get("data"):
                    model_names = [model.get("id", "unknown") for model in models["data"]]
                    self.lm_studio_status = f"✅ 接続済み - 利用可能モデル: {', '.join(model_names)}"
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
    
    def reset_collection(self):
        """コレクションをリセット（重複処理の統一化）"""
        try:
            self.chroma_client.delete_collection("sales_knowledge")
        except:
            pass
        self.collection = self.chroma_client.create_collection(
            name="sales_knowledge",
            metadata={"description": "営業ナレッジベース"}
        )
    
    def get_documents(self, document_path: str = "sample_documents") -> List[str]:
        """ドキュメントファイルの取得（共通処理）"""
        if not os.path.exists(document_path):
            return []
        
        documents = []
        for pattern in FILE_PATTERNS:
            files = glob.glob(os.path.join(document_path, pattern))
            documents.extend(files)
        
        return sorted(documents)  # 安定した処理順序
    
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
    
    def load_documents(self, document_path: str = "sample_documents"):
        """UI付きドキュメント読み込み（手動実行用）"""
        documents = self.get_documents(document_path)
        if not documents:
            if 'st' in globals():
                st.error(f"❌ ドキュメントフォルダが見つかりません: {document_path}")
            return False
        
        try:
            self.reset_collection()
            total_chunks = 0
            
            progress_bar = None
            if 'st' in globals():
                progress_bar = st.progress(0)
                st.info(f"📚 {len(documents)}個のファイルを処理中...")
            
            for i, doc_path in enumerate(documents):
                chunks_added = self.process_document(doc_path)
                total_chunks += chunks_added
                
                if progress_bar:
                    progress_bar.progress((i + 1) / len(documents))
            
            if progress_bar:
                progress_bar.empty()
                st.success(f"✅ {len(documents)}個のファイルから{total_chunks}件のドキュメントを読み込みました")
            
            return True
            
        except Exception as e:
            if 'st' in globals():
                st.error(f"❌ ドキュメント読み込みエラー: {e}")
            return False
    
    def process_document(self, file_path: str) -> int:
        """個別ドキュメントの処理"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                return 0
            
            # ファイル名から拡張子を除去
            filename = os.path.basename(file_path)
            if filename.endswith('.md'):
                filename = filename[:-3]
            elif filename.endswith('.txt'):
                filename = filename[:-4]
            
            chunks = self.split_text(content)
            chunks_added = 0
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # 空のチャンクをスキップ
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
        """類似ドキュメントの検索（結果数を増加）"""
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
        """LM Studioを使用して高度な回答生成"""
        try:
            # コンテキストの整理と優先順位付け
            sorted_docs = sorted(context_docs, key=lambda x: x['distance'])
            
            context = "\n\n".join([
                f"【出典: {doc['source']}】\n{doc['content']}"
                for doc in sorted_docs[:3]  # 上位3件のみ使用
            ])
            
            # より詳細で実用的なプロンプト
            prompt = f"""あなたは法人向け研修事業の営業支援AIアシスタントです。以下の情報を基に、営業担当者が顧客に説明する際に使える実用的で具体的な回答を提供してください。

# 質問
{query}

# 参考情報
{context}

# 回答形式
以下の形式で回答してください：

## 【結論】
質問に対する明確で簡潔な答え（1-2行）

## 【根拠・詳細】
- 参考情報から抜粋した具体的な根拠
- 実用的なアドバイスや提案
- 具体的な数値や事例（可能な場合）
- 営業活動での活用ポイント

## 【出典】
参考にした文書名（形式：ファイル名#チャンク番号）

## 【営業活用のポイント】
- 顧客への説明時の注意点
- 提案時の強調ポイント
- よくある質問への対応

回答は日本語で、営業担当者が即座に活用できる実用的な内容にしてください。専門用語は分かりやすく説明し、具体的な数値や事例があれば積極的に活用してください。"""

            data = {
                "model": "gpt-oss-20b",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1500,  # トークン数を増加
                "top_p": 0.9,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1
            }
            
            response = requests.post(LM_STUDIO_API_URL, 
                                   headers={"Content-Type": "application/json"}, 
                                   json=data, timeout=60)  # タイムアウトを延長
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                else:
                    return "❌ LM Studioからの応答形式が不正です"
            else:
                return f"❌ LM Studio APIエラー: {response.status_code} - {response.text}"
                
        except requests.exceptions.ConnectionError:
            return "❌ LM Studioに接続できません。LM Studioが起動していることを確認してください。"
        except requests.exceptions.Timeout:
            return "❌ LM Studioからの応答がタイムアウトしました。モデルの読み込み状況を確認してください。"
        except Exception as e:
            return f"❌ 回答生成エラー: {e}"
    
    def generate_fallback_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """LM Studio未接続時のフォールバック回答生成（改善版）"""
        if not context_docs:
            return "申し訳ございませんが、関連する情報が見つかりませんでした。"
        
        # 類似度でソート
        sorted_docs = sorted(context_docs, key=lambda x: x['distance'])
        
        answer = f"""## 【検索結果サマリー】
質問: {query}

## 【関連情報（類似度順）】
"""
        
        for i, doc in enumerate(sorted_docs, 1):
            similarity = 1 - doc['distance']
            answer += f"""
### {i}. {doc['source']} (類似度: {similarity:.3f})
{doc['content'][:400]}{"..." if len(doc['content']) > 400 else ""}

"""
        
        answer += """## 【出典】
"""
        for doc in sorted_docs:
            answer += f"- {doc['metadata']['chunk_id']}\n"
        
        answer += """
## 【より詳細な回答を得るには】
1. **LM Studioを起動**してください
2. **Local Serverタブ**で「Start Server」をクリック
3. **gpt-oss-20bモデル**を読み込んでください
4. ページを再読み込みして質問し直してください

現在は検索結果のみを表示していますが、AIによる動的回答生成により、より実用的で具体的なアドバイスが得られます。"""
        
        return answer
    
    def query(self, question: str) -> tuple[str, List[Dict[str, Any]]]:
        """RAGシステムのメイン処理（改善版）"""
        # 接続状況を再確認
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
                answer = self.generate_fallback_answer(question, search_results)
        except:
            answer = self.generate_fallback_answer(question, search_results)
        
        return answer, search_results


def get_custom_css():
    """カスタムCSSを返す（UIスタイル定義）"""
    return """
    <style>
    /* 基本フォント設定 */
    .main {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* メインコンテナ */
    .main .block-container {
        padding-top: 0.5rem;
        padding-bottom: 2rem;
        max-width: 50rem;
        margin: 0 auto;
    }
    
    /* タイトル */
    .main-title {
        text-align: left;
        color: #1f2937;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        letter-spacing: -0.025em;
    }
    
    .main-subtitle {
        text-align: left;
        color: #6b7280;
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
        font-weight: 400;
    }
    
    /* テキストエリア */
    .stTextArea textarea {
        border: 1px solid #d1d5db;
        border-radius: 0.75rem;
        padding: 1rem;
        font-size: 0.95rem;
        line-height: 1.5;
        background-color: #ffffff;
        color: #111827 !important;
        caret-color: #111827 !important;
        resize: none;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }
    
    .stTextArea textarea:focus {
        outline: none;
        border-color: #10b981;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
        color: #111827 !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #9ca3af !important;
        opacity: 1;
    }
    
    /* ボタン */
    .stButton > button {
        background-color: #10b981;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.625rem 1.5rem;
        font-weight: 600;
        font-size: 0.875rem;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    
    .stButton > button:hover {
        background-color: #059669;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* AI回答エリア */
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
    
    .ai-response h2, .ai-response h3 {
        color: #1f2937;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    .ai-response h2:first-child, .ai-response h3:first-child {
        margin-top: 0;
    }
    
    /* サイドバー */
    .css-1d391kg {
        background-color: #f9fafb;
        border-right: 1px solid #e5e7eb;
    }
    
    .css-1d391kg .stButton > button {
        width: 100%;
        font-size: 0.8rem;
        padding: 0.5rem 1rem;
    }
    
    .css-1d391kg .stCaption {
        font-size: 0.7rem !important;
        color: #6b7280 !important;
        margin-bottom: 0.25rem !important;
        line-height: 1.2 !important;
    }
    
    /* 新規チャットボタン */
    .css-1d391kg .stButton > button[title="新規チャット"] {
        background-color: #059669 !important;
        color: white !important;
        border: none !important;
        border-radius: 0.375rem !important;
        padding: 0.25rem 0.5rem !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        margin-top: 0.25rem !important;
    }
    
    .css-1d391kg .stButton > button[title="新規チャット"]:hover {
        background-color: #047857 !important;
    }
    
    /* アラート */
    .stAlert {
        border-radius: 0.375rem;
        margin-bottom: 0.5rem;
        font-size: 0.75rem;
        border: none;
        padding: 0.5rem 0.75rem;
        font-weight: 500;
    }
    
    .stSuccess {
        background-color: #ecfdf5;
        color: #065f46;
        border-left: 4px solid #10b981;
    }
    
    .stInfo {
        background-color: #eff6ff;
        color: #1e40af;
        border-left: 4px solid #3b82f6;
    }
    
    .stWarning {
        background-color: #fffbeb;
        color: #92400e;
        border-left: 4px solid #f59e0b;
    }
    
    .stError {
        background-color: #fef2f2;
        color: #991b1b;
        border-left: 4px solid #ef4444;
    }
    
    /* その他の共通スタイル */
    input, textarea, select {
        color: #111827 !important;
    }
    
    .stTextInput input {
        color: #111827 !important;
    }
    
    hr {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 2rem 0;
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
    
    # カスタムCSS適用
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
        # システム状態
        st.markdown("#### ⚙️ システム状態")
        
        # 埋め込みモデル状態
        if hasattr(rag_system, 'embedding_status'):
            status_text = "❌ システムエラー" if "❌" in rag_system.embedding_status else "✅ システム準備完了"
            st.caption(status_text)
        
        # ドキュメント状態
        if hasattr(rag_system, 'db_status') and "件のドキュメント" in rag_system.db_status:
            match = re.search(r'(\d+)件のドキュメント', rag_system.db_status)
            if match:
                st.caption(f"📚 {match.group(1)}件のドキュメント利用可能")
        else:
            st.caption("📚 ドキュメント読み込み中...")
        
        # LM Studio接続状態
        if hasattr(rag_system, 'lm_studio_status'):
            if "✅" in rag_system.lm_studio_status:
                st.caption(f"🤖 {rag_system.lm_studio_status.split(' - ')[1] if ' - ' in rag_system.lm_studio_status else 'LM Studio接続済み'}")
            elif "⚠️" in rag_system.lm_studio_status:
                st.caption(f"⚠️ {rag_system.lm_studio_status}")
            else:
                st.caption(f"❌ {rag_system.lm_studio_status}")
        
        # 接続状況の更新ボタン
        if st.button("🔄 接続状況更新", use_container_width=True, key="refresh_connection"):
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
        
        # 会話履歴の初期化と表示
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
        
        # 履歴クリアボタン
        if st.session_state.chat_history:
            if st.button("🗑️ 履歴をクリア", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        st.markdown("---")
        
        # 情報セクション
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
        
        with st.expander("📊 システム詳細"):
            st.markdown(f"""
            **埋め込みモデル:** multilingual-e5-small  
            **ベクトルDB:** ChromaDB  
            **LLM:** LM Studio (gpt-oss-20b)  
            **チャンクサイズ:** {CHUNK_SIZE}文字  
            **検索精度:** Top-5
            **LM Studio接続:** {'✅ 接続済み' if '✅' in rag_system.lm_studio_status else '❌ 未接続'}
            """)
    
    # メインコンテンツエリア
    st.markdown('<div style="margin-top: 3rem;"></div>', unsafe_allow_html=True)
    
    # 入力処理
    initial_question = ""
    if 'reuse_question' in st.session_state:
        initial_question = st.session_state.reuse_question
        del st.session_state.reuse_question
    
    # 新規チャット処理
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