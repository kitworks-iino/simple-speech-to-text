Windows 11 x64 向けの、マイクで話して文字起こしするだけのアプリです。

1. `simple-speech-to-text-windows-x64.zip` をダウンロードして新しいフォルダーに展開します。
2. `SimpleSpeechToText.exe` または `Start.cmd` を起動します。
3. 案内後に話し、約 1 秒黙ると文字起こしが表示されます。

Python 3.13、Whisper small ONNX、Silero VAD、必要な音声ライブラリを同梱しています。API キー、アカウント登録、モデルの追加ダウンロードは不要です。

FFmpeg / PyAV / faster-whisper を使用せず、マイクの PCM 音声をローカルで処理します。配布 ZIP 内に依存関係検査結果とライセンスを同梱しています。Windows でのテストと、最終 ZIP を展開した EXE のモデル推論をビルド時に確認しています。実機のマイク・社内ポリシーとの適合や日本語認識精度は利用環境で確認してください。

対応環境と使い方は同梱 README を参照してください。SHA-256 は `SHA256SUMS.txt` に掲載しています。
