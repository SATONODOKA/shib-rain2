#!/usr/bin/env python3
"""
LM Studio APIサーバー起動スクリプト
"""

import subprocess
import time
import requests
import json
import os

def start_lm_studio_api():
    """LM StudioのAPIサーバーを起動する"""
    
    print("🚀 LM Studio APIサーバーを起動中...")
    
    # LM Studioアプリのパス
    lm_studio_path = "/Applications/LM Studio.app"
    
    if not os.path.exists(lm_studio_path):
        print("❌ LM Studioアプリが見つかりません")
        return False
    
    try:
        # LM Studioを起動
        print("📱 LM Studioアプリを起動中...")
        subprocess.run(["open", "-a", "LM Studio"], check=True)
        
        # 起動を待つ
        print("⏳ アプリの起動を待機中...")
        time.sleep(10)
        
        # APIサーバーの起動を確認
        print("🔍 APIサーバーの起動を確認中...")
        for i in range(30):  # 最大30回試行
            try:
                response = requests.get("http://localhost:1234/v1/models", timeout=2)
                if response.status_code == 200:
                    print("✅ LM Studio APIサーバーが起動しました！")
                    
                    # 利用可能なモデルを表示
                    models = response.json()
                    if models.get("data"):
                        print("📋 利用可能なモデル:")
                        for model in models["data"]:
                            print(f"  - {model.get('id', 'unknown')}")
                    else:
                        print("⚠️ モデルが見つかりません")
                    
                    return True
                else:
                    print(f"⚠️ APIサーバー応答: {response.status_code}")
            except requests.exceptions.ConnectionError:
                print(f"⏳ 接続待機中... ({i+1}/30)")
            except Exception as e:
                print(f"⚠️ エラー: {e}")
            
            time.sleep(2)
        
        print("❌ APIサーバーの起動がタイムアウトしました")
        return False
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

def check_api_status():
    """APIサーバーの状態を確認する"""
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print("✅ APIサーバーは起動中です")
            if models.get("data"):
                print("📋 利用可能なモデル:")
                for model in models["data"]:
                    print(f"  - {model.get('id', 'unknown')}")
            return True
        else:
            print(f"❌ APIサーバーエラー: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 接続エラー: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("LM Studio APIサーバー起動ツール")
    print("=" * 50)
    
    # 現在の状態を確認
    print("\n🔍 現在のAPIサーバー状態を確認中...")
    if check_api_status():
        print("✅ 既に起動しています")
    else:
        print("❌ 起動していません")
        
        # 起動を試行
        print("\n🚀 APIサーバーの起動を試行中...")
        if start_lm_studio_api():
            print("\n🎉 起動完了！")
        else:
            print("\n❌ 起動に失敗しました")
            print("\n📋 手動での設定手順:")
            print("1. LM Studioアプリを開く")
            print("2. Local Serverタブをクリック")
            print("3. Start Serverボタンをクリック")
            print("4. ポート1234で起動することを確認")
    
    print("\n" + "=" * 50) 