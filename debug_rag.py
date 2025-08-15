#!/usr/bin/env python3
"""
RAGシステムのデバッグ用スクリプト
ドキュメント読み込みと検索機能をテストします
"""

import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer
import re
from typing import List, Dict, Any

# 設定
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def setup_embedding_model():
    """埋め込みモデルの初期化"""
    try:
        print("🔄 埋め込みモデルを初期化中...")
        model = SentenceTransformer('intfloat/multilingual-e5-small')
        print("✅ 埋め込みモデルを初期化しました")
        return model
    except Exception as e:
        print(f"❌ 埋め込みモデルの初期化に失敗: {e}")
        return None

def setup_chroma_db():
    """ChromaDBの初期化"""
    try:
        print("🔄 ChromaDBを初期化中...")
        client = chromadb.PersistentClient(path="./chroma_db")
        
        try:
            collection = client.get_collection("sales_knowledge")
            count = collection.count()
            print(f"📚 既存のナレッジベースを読み込みました（{count}件のドキュメント）")
        except:
            collection = client.create_collection(
                name="sales_knowledge",
                metadata={"description": "営業ナレッジベース"}
            )
            print("📚 新しいナレッジベースを作成しました")
            
        return client, collection
    except Exception as e:
        print(f"❌ ChromaDBの初期化に失敗: {e}")
        return None, None

def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """テキストをチャンクに分割"""
    sentences = re.split(r'[。！？\n]+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk)
            overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
            current_chunk = overlap_text + sentence
        else:
            current_chunk += sentence + "。"
    
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

def load_documents(embedding_model, collection, document_path: str = "sample_documents"):
    """サンプルドキュメントの読み込み"""
    print(f"🔄 ドキュメントフォルダを確認中: {document_path}")
    
    if not os.path.exists(document_path):
        print(f"❌ ドキュメントフォルダが見つかりません: {document_path}")
        return False
        
    # 対応ファイル形式
    file_patterns = ["*.md", "*.txt", "*.docx.md"]
    
    documents = []
    for pattern in file_patterns:
        files = glob.glob(os.path.join(document_path, pattern))
        documents.extend(files)
    
    if not documents:
        print("⚠️ 読み込み可能なドキュメントが見つかりません")
        return False
        
    print(f"📄 {len(documents)}個のファイルを発見しました")
    
    # ドキュメントの処理
    processed_count = 0
    total_chunks = 0
    
    for i, file_path in enumerate(documents):
        try:
            print(f"📖 処理中 ({i+1}/{len(documents)}): {os.path.basename(file_path)}")
            
            # ファイルの読み込み
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                print(f"⚠️ 空のファイル: {file_path}")
                continue
                
            # ファイル名から出典情報を取得
            filename = os.path.basename(file_path)
            
            # チャンク分割
            chunks = split_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
            print(f"   📝 {len(chunks)}個のチャンクに分割")
            
            # 各チャンクを埋め込みベクトル化してDBに保存
            for j, chunk in enumerate(chunks):
                chunk_id = f"{filename}#chunk-{j+1}"
                
                # 埋め込みベクトルの計算
                embedding = embedding_model.encode(chunk).tolist()
                
                # メタデータの作成
                metadata = {
                    "source": filename,
                    "chunk_id": chunk_id,
                    "chunk_index": j,
                    "file_path": file_path
                }
                
                # ChromaDBに保存
                collection.add(
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    ids=[chunk_id]
                )
            
            processed_count += 1
            total_chunks += len(chunks)
            
        except Exception as e:
            print(f"❌ ファイル処理エラー ({file_path}): {e}")
    
    print(f"✅ {processed_count}/{len(documents)}個のファイルを処理しました")
    print(f"📊 合計 {total_chunks}個のチャンクを保存しました")
    return processed_count > 0

def test_search(embedding_model, collection, query: str = "新任管理職研修の料金"):
    """検索テスト"""
    print(f"\n🔍 検索テスト: '{query}'")
    
    try:
        # クエリの埋め込みベクトル化
        query_embedding = embedding_model.encode(query).tolist()
        
        # ChromaDBで類似検索
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
        
        print(f"📋 {len(results['documents'][0])}件の関連ドキュメントを発見")
        
        for i in range(len(results["documents"][0])):
            content = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            
            print(f"\n--- 結果 {i+1} ---")
            print(f"出典: {metadata['source']}")
            print(f"類似度: {1 - distance:.3f}")
            print(f"内容: {content[:200]}...")
            
    except Exception as e:
        print(f"❌ 検索エラー: {e}")

def main():
    print("🚀 RAGシステム デバッグテスト開始")
    print("=" * 50)
    
    # 1. 埋め込みモデルの初期化
    embedding_model = setup_embedding_model()
    if not embedding_model:
        return
    
    # 2. ChromaDBの初期化
    client, collection = setup_chroma_db()
    if not client or not collection:
        return
    
    # 3. 現在のドキュメント数を確認
    current_count = collection.count()
    print(f"📊 現在のドキュメント数: {current_count}")
    
    # 4. ドキュメントが空の場合は読み込み
    if current_count == 0:
        print("\n🔄 ドキュメント読み込み開始")
        success = load_documents(embedding_model, collection)
        if not success:
            print("❌ ドキュメント読み込みに失敗しました")
            return
    else:
        print("📚 既存のドキュメントを使用します")
    
    # 5. 検索テスト
    test_queries = [
        "新任管理職研修の料金",
        "トヨタ自動車様の課題",
        "競合他社との差別化ポイント",
        "建設業界向けの研修内容"
    ]
    
    for query in test_queries:
        test_search(embedding_model, collection, query)
    
    print("\n🎉 デバッグテスト完了")

if __name__ == "__main__":
    main() 