tell application "LM Studio"
    activate
    delay 5
    
    -- Local Serverタブをクリック
    tell application "System Events"
        tell process "LM Studio"
            -- メインウィンドウが表示されるまで待つ
            repeat until (exists window 1)
                delay 1
            end repeat
            
            -- Local Serverタブを探してクリック
            try
                click button "Local Server" of tab group 1 of window 1
                delay 2
                
                -- Start Serverボタンをクリック
                click button "Start Server" of window 1
                log "✅ Start Serverボタンをクリックしました"
                
                -- 起動を待つ
                delay 10
                
                -- 状態を確認
                if exists text field "Running" of window 1 then
                    log "✅ APIサーバーが起動しました"
                else
                    log "⚠️ APIサーバーの起動状態を確認できません"
                end if
                
            on error errMsg
                log "❌ エラー: " & errMsg
                log "📋 手動での設定が必要です:"
                log "1. Local Serverタブをクリック"
                log "2. Start Serverボタンをクリック"
            end try
        end tell
    end tell
end tell 