import os
import glob
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
import requests
import json
from typing import List, Dict, Any
import re

# 設定
LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

class RAGSystem:
    def __init__(self):
        self.embedding_status = ""
        self.db_status = ""
        self.setup_embedding_model()
        self.setup_chroma_db()
        
    def setup_embedding_model(self):
        """埋め込みモデルの初期化"""
        try:
            # 日本語対応の軽量モデルを使用
            self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-small')
            self.embedding_status = "✅ 埋め込みモデルを初期化しました"
        except Exception as e:
            self.embedding_status = f"❌ 埋め込みモデルの初期化に失敗: {e}"
            return False
        return True
    
    def setup_chroma_db(self):
        """ChromaDBの初期化"""
        try:
            # ChromaDBクライアントの作成
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            
            # コレクションの作成または取得
            try:
                self.collection = self.chroma_client.get_collection("sales_knowledge")
                self.db_status = f"📚 既存のナレッジベースを読み込みました（{self.collection.count()}件のドキュメント）"
            except:
                self.collection = self.chroma_client.create_collection(
                    name="sales_knowledge",
                    metadata={"description": "営業ナレッジベース"}
                )
                self.db_status = "📚 新しいナレッジベースを作成しました"
                
        except Exception as e:
            self.db_status = f"❌ ChromaDBの初期化に失敗: {e}"
            return False
        return True
    
    def load_documents(self, document_path: str = "sample_documents"):
        """サンプルドキュメントの読み込み"""
        if not os.path.exists(document_path):
            st.error(f"❌ ドキュメントフォルダが見つかりません: {document_path}")
            return False
        
        # 既存のドキュメントをクリア（重複を避けるため）
        try:
            self.chroma_client.delete_collection("sales_knowledge")
            self.collection = self.chroma_client.create_collection(
                name="sales_knowledge",
                metadata={"description": "営業ナレッジベース"}
            )
            st.info("🔄 既存のナレッジベースをクリアしました")
        except:
            pass
            
        # 対応ファイル形式
        file_patterns = [
            "*.md", "*.txt", "*.docx.md"  # 今回はマークダウンとテキストファイルのみ
        ]
        
        documents = []
        for pattern in file_patterns:
            files = glob.glob(os.path.join(document_path, pattern))
            documents.extend(files)
        
        # ファイルを名前でソート（処理順序を安定化）
        documents.sort()
        
        if not documents:
            st.warning("⚠️ 読み込み可能なドキュメントが見つかりません")
            return False
            
        st.info(f"📄 {len(documents)}個のファイルを発見しました")
        
        # ドキュメントの処理
        processed_count = 0
        total_chunks = 0
        progress_bar = st.progress(0)
        
        for i, file_path in enumerate(documents):
            try:
                st.text(f"📖 処理中: {os.path.basename(file_path)}")
                chunks_added = self.process_document(file_path)
                if chunks_added > 0:
                    processed_count += 1
                    total_chunks += chunks_added
                    st.text(f"   ✅ {chunks_added}個のチャンクを追加")
                else:
                    st.text(f"   ⚠️ 処理をスキップ")
                progress_bar.progress((i + 1) / len(documents))
            except Exception as e:
                st.error(f"❌ ファイル処理エラー ({os.path.basename(file_path)}): {e}")
        
        st.success(f"✅ {processed_count}/{len(documents)}個のファイルを処理しました")
        st.info(f"📊 合計 {total_chunks}個のチャンクを保存しました")
        return processed_count > 0
    
    def process_document(self, file_path: str) -> int:
        """個別ドキュメントの処理"""
        try:
            # ファイルの読み込み
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                return 0
                
            # ファイル名から出典情報を取得
            filename = os.path.basename(file_path)
            
            # チャンク分割
            chunks = self.split_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
            
            if not chunks:
                return 0
            
            # 各チャンクを埋め込みベクトル化してDBに保存
            chunks_added = 0
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                    
                chunk_id = f"{filename}#chunk-{i+1}"
                
                # 重複チェック
                try:
                    existing = self.collection.get(ids=[chunk_id])
                    if existing['ids']:
                        continue  # 既に存在する場合はスキップ
                except:
                    pass
                
                # 埋め込みベクトルの計算
                embedding = self.embedding_model.encode(chunk).tolist()
                
                # メタデータの作成
                metadata = {
                    "source": filename,
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "file_path": file_path
                }
                
                # ChromaDBに保存
                self.collection.add(
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    ids=[chunk_id]
                )
                chunks_added += 1
            
            return chunks_added
            
        except Exception as e:
            st.error(f"ドキュメント処理エラー: {e}")
            return 0
    
    def split_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """テキストをチャンクに分割"""
        # シンプルな文区切りでの分割
        sentences = re.split(r'[。！？\n]+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # チャンクサイズを超える場合は新しいチャンクを開始
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk)
                # オーバーラップを考慮
                overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
                current_chunk = overlap_text + sentence
            else:
                current_chunk += sentence + "。"
        
        # 最後のチャンクを追加
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def search_similar_documents(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """類似ドキュメントの検索"""
        try:
            # コレクションが空でないかチェック
            if self.collection.count() == 0:
                st.warning("⚠️ ナレッジベースが空です。「ドキュメント読み込み」を実行してください。")
                return []
            
            # クエリの埋め込みベクトル化
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # ChromaDBで類似検索
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.collection.count()),
                include=["documents", "metadatas", "distances"]
            )
            
            # 結果が空でないかチェック
            if not results["documents"] or not results["documents"][0]:
                st.warning("⚠️ 検索結果が見つかりませんでした。")
                return []
            
            # 結果の整形
            search_results = []
            for i in range(len(results["documents"][0])):
                search_results.append({
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "source": results["metadatas"][0][i]["source"]
                })
            
            return search_results
            
        except Exception as e:
            st.error(f"❌ 検索エラー: {e}")
            st.info("💡 解決策: サイドバーの「ドキュメント読み込み」ボタンをクリックしてナレッジベースを再構築してください。")
            return []
    
    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """LM Studioを使用して回答生成"""
        try:
            # コンテキストの構築
            context = "\n\n".join([
                f"【出典: {doc['source']}】\n{doc['content']}"
                for doc in context_docs
            ])
            
            # プロンプトの構築
            prompt = f"""あなたは法人向け研修事業の営業支援AIアシスタントです。以下の情報を基に、正確で具体的な回答を提供してください。

# 質問
{query}

# 参考情報
{context}

# 回答形式
以下の形式で回答してください：

## 【結論】
質問に対する明確で簡潔な答え

## 【根拠・詳細】
参考情報から抜粋した具体的な根拠や詳細説明

## 【出典】
参考にした文書名（形式：ファイル名#チャンク番号）

回答は日本語で、営業担当者が顧客に説明する際に使える実用的な内容にしてください。"""

            # LM Studio APIへのリクエスト
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-oss-20b",  # LM Studioで読み込んだモデル名
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            response = requests.post(LM_STUDIO_API_URL, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"❌ LM Studio APIエラー: {response.status_code}\n{response.text}"
                
        except requests.exceptions.ConnectionError:
            return "❌ LM Studioに接続できません。LM Studioが起動していることを確認してください。"
        except requests.exceptions.Timeout:
            return "❌ LM Studioからの応答がタイムアウトしました。"
        except Exception as e:
            return f"❌ 回答生成エラー: {e}"
    
    def query(self, question: str) -> tuple[str, List[Dict[str, Any]]]:
        """RAGシステムのメイン処理"""
        # 1. 類似ドキュメントの検索
        search_results = self.search_similar_documents(question, n_results=3)
        
        if not search_results:
            return "申し訳ございませんが、関連する情報が見つかりませんでした。", []
        
        # 2. LLMによる回答生成（LM Studio接続チェック）
        try:
            response = requests.get("http://localhost:1234/v1/models", timeout=2)
            if response.status_code == 200:
                answer = self.generate_answer(question, search_results)
            else:
                answer = self.generate_fallback_answer(question, search_results)
        except:
            answer = self.generate_fallback_answer(question, search_results)
        
        return answer, search_results
    
    def generate_fallback_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """LM Studio未接続時のフォールバック回答生成"""
        # シンプルな構造化回答を生成
        answer = f"""## 【検索結果】
質問: {query}

## 【関連情報】
"""
        
        for i, doc in enumerate(context_docs, 1):
            similarity = 1 - doc['distance']
            answer += f"""
### {i}. {doc['source']} (類似度: {similarity:.3f})
{doc['content'][:300]}{"..." if len(doc['content']) > 300 else ""}

"""
        
        answer += """
## 【出典】
"""
        for i, doc in enumerate(context_docs, 1):
            answer += f"- {doc['metadata']['chunk_id']}\n"
        
        answer += """
ℹ️ **より詳細な回答を得るには**: LM Studioを起動してgpt-oss-20bモデルを読み込んでください。"""
        
        return answer


def main():
    st.set_page_config(
        page_title="営業ナレッジベース RAGシステム",
        page_icon="💼",
        layout="wide"
    )
    
    st.title("💼 営業ナレッジベース RAGシステム")
    st.markdown("法人向け研修事業の営業支援AIアシスタント")
    
    # RAGシステムの初期化
    if 'rag_system' not in st.session_state:
        with st.spinner("RAGシステムを初期化中..."):
            st.session_state.rag_system = RAGSystem()
    
    rag_system = st.session_state.rag_system
    
    # 初期化ステータスを表示
    if hasattr(rag_system, 'embedding_status'):
        if "❌" in rag_system.embedding_status:
            st.error(rag_system.embedding_status)
        else:
            st.success(rag_system.embedding_status)
    
    if hasattr(rag_system, 'db_status'):
        if "❌" in rag_system.db_status:
            st.error(rag_system.db_status)
        else:
            st.info(rag_system.db_status)
    
    # サイドバー
    with st.sidebar:
        st.header("🔧 システム管理")
        
        # ドキュメント読み込み
        if st.button("📚 ドキュメント読み込み", type="primary"):
            with st.spinner("ドキュメントを読み込み中..."):
                rag_system.load_documents()
        
        # ナレッジベース情報
        try:
            doc_count = rag_system.collection.count()
            st.metric("保存済みドキュメント", f"{doc_count}件")
        except:
            st.metric("保存済みドキュメント", "0件")
        
        st.markdown("---")
        st.markdown("### 💡 使い方")
        st.markdown("""
        1. 「ドキュメント読み込み」でナレッジベース構築
        2. 下記に営業に関する質問を入力
        3. AIが関連情報を検索して回答
        """)
        
        st.markdown("### 📝 質問例")
        st.markdown("""
        - 新任管理職研修の料金は？
        - トヨタ自動車様の課題は？
        - 競合他社との差別化ポイントは？
        - 建設業界向けの研修内容は？
        """)
    
    # メインエリア
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 質問・相談")
        
        # 質問入力
        user_question = st.text_area(
            "営業に関する質問を入力してください：",
            height=100,
            placeholder="例：新任管理職研修の価格と内容について教えて"
        )
        
        if st.button("🔍 質問する", type="primary", disabled=not user_question):
            with st.spinner("回答を生成中..."):
                answer, search_results = rag_system.query(user_question)
                
                st.markdown("## 🤖 AI回答")
                st.markdown(answer)
                
                # 検索結果の表示
                if search_results:
                    with st.expander("📋 参考にした情報", expanded=False):
                        for i, result in enumerate(search_results, 1):
                            st.markdown(f"### 参考情報 {i}")
                            st.markdown(f"**出典:** {result['source']}")
                            st.markdown(f"**類似度:** {1 - result['distance']:.3f}")
                            st.markdown(f"**内容:**")
                            st.text(result['content'][:300] + "..." if len(result['content']) > 300 else result['content'])
                            st.markdown("---")
    
    with col2:
        st.header("⚙️ システム状態")
        
        # LM Studio接続確認
        try:
            response = requests.get("http://localhost:1234/v1/models", timeout=2)
            if response.status_code == 200:
                models = response.json()
                model_count = len(models.get('data', []))
                st.success(f"✅ LM Studio 接続OK ({model_count}モデル)")
            else:
                st.warning("⚠️ LM Studio 接続エラー (検索機能のみ利用可能)")
        except requests.exceptions.ConnectionError:
            st.warning("⚠️ LM Studio 未起動 (検索機能のみ利用可能)")
            st.info("💡 LM Studioを起動してポート1234でサーバーを開始してください")
        except Exception as e:
            st.warning(f"⚠️ LM Studio 状態不明: {e}")
        
        # システム情報
        st.markdown("### 📊 システム情報")
        st.info(f"""
        **埋め込みモデル:** multilingual-e5-small  
        **ベクトルDB:** ChromaDB  
        **LLM:** LM Studio (gpt-oss-20b)  
        **チャンクサイズ:** {CHUNK_SIZE}文字  
        """)


if __name__ == "__main__":
    main() 