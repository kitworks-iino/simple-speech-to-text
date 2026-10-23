# simple-speech-to-text

**起動 → 喋る → 喋り終わる → 文字起こし。**

FFmpegを使わずに、どのくらい手軽に音声を文字にできるかを試す、小さなコンソールアプリです。
[giji-automemo](https://github.com/kitworks-iino/giji-automemo) v0.5と同じ
**sherpa-onnx + 多言語Whisper smallのint8 ONNXモデル**を使います。
マイク音声をPC内で処理し、日本語の文章を表示します。APIキー、アカウント登録、GPUは不要です。

現在の`main`とiPhone試用版は、議事録アプリv0.5の文字起こしコードをそのまま呼び出します。
公開済みWindows ZIPの**v0.1.0は、この処理経路統一前の版**です。
エンジンとモデルは同じですが、音声の分割方法などが異なります。

## Windowsですぐ使う

1. [Releases](https://github.com/kitworks-iino/simple-speech-to-text/releases)から
   `simple-speech-to-text-windows-x64.zip`をダウンロード。
2. ZIP全体を新しいフォルダーに展開し、`Start.cmd`をダブルクリック。
3. 「話してください」と表示されたら話します。
4. 約1秒黙ると録音が終了し、「文字起こし中」の後に文章が表示されます。

Windows 11 x64、利用可能なマイク、Windowsのマイク利用許可が必要です。
Python 3.13の実行環境・音声処理ライブラリ・WhisperモデルはZIPに同梱します。
ZIP版の起動・文字起こしにインターネット接続は不要です。
録音開始前にモデルの整合性確認をするため、起動に少し時間がかかります。
展開後のexeだけを別の場所へ移動せず、同梱ファイルを一緒に置いてください。

```text
準備しています…
話してください。約1秒の無音で終了します（中止: Ctrl+C）。
録音中…
文字起こし中…

明日の打ち合わせは午前十時からお願いします。
```

1回の起動で1回分の発話を処理します。文章は端末からコピーしてください。
音声・文字を履歴として保存せず、サーバーにも送信しません。
現在のソース版は認識中だけ一時PCM WAVを作成し、処理後に削除します。
30秒間発話がなければ終了し、1回の録音は最大120秒です。
話の途中でも約1秒黙ると終了します。長い発話はWhisperに渡す前に28秒以内へ分割します。
CPUの性能・発話の長さによって、録音後の処理に時間がかかります。
認識誤りや長い発話の分割境界での抜け・重複は起こり得ます。

## 仕組み

```text
マイク → PortAudio / sounddevice → PCM（メモリー内）
       → Silero VADで発話終了を検知
       → 一時PCM WAV
       → 議事録アプリv0.5のPCM読み込み・発話区間分割
       → 同じWhisper small ONNX推論・時刻情報生成 → 文字を表示
```

マイクが16 kHz入力に対応しない場合は、デバイスの既定サンプルレートで録音し、
SciPyで16 kHzへ変換します。録音・発話検出・認識はすべてローカルです。
WAVを受け取った後は、[議事録アプリv0.5から変更せず取り込んだコード](src/simple_speech_to_text/_vendor/giji_v050/)
の`Pipeline._transcribe`を実行します。取り込み元のコミットとファイルハッシュは
同ディレクトリの`PROVENANCE.json`に記録しています。
録音方法と録音終了の操作は、Windows版とiPhone試用版で異なります。
FFmpeg実行ファイル、PyAV、faster-whisper、動画・圧縮音声デコーダーは使いません。
Whisperの学習済みモデルを使いますが、FFmpegに依存する`openai-whisper`パッケージは導入しません。

FFmpegのライセンスと、各コーデックの特許は別の話です。
このリポジトリは「FFmpegを音声処理経路・配布物に入れない構成」を試すもので、
あらゆるソフトウェアの特許問題がなくなることを保証するものではありません。
検査対象・依存バージョン・ライセンスは[依存関係の説明](docs/dependencies.md)を参照してください。

## Python 3.13から動かす

Windows用ZIPを使う場合、この操作は不要です。
ソース版は通常のPython 3.13 x64と、新しい仮想環境を使います。

```powershell
git clone https://github.com/kitworks-iino/simple-speech-to-text.git
cd simple-speech-to-text
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\python.exe -m simple_speech_to_text
```

ソース版のみ、初回起動時に固定版モデルをダウンロードし、SHA-256を検証します。
保存先は`%LOCALAPPDATA%\simple-speech-to-text\models\whisper-small`です。
次回以降はネット接続不要です。音声を外部へ送る処理はありません。

マイクを選ぶ場合や英語を試す場合:

```powershell
.\.venv\Scripts\python.exe -m simple_speech_to_text --list-devices
.\.venv\Scripts\python.exe -m simple_speech_to_text --device 1
.\.venv\Scripts\python.exe -m simple_speech_to_text --language en
```

既定は日本語です。`--language auto`で言語を自動判定できます。
利用可能な入力デバイスがないときは、マイクの接続・Windowsの入力設定・マイクの利用許可を確認してください。
このアプリは会議アプリとの連携、システム音声の録音、話者分離、要約を行いません。

## iPhoneからの一時試用

開発用の`Temporary phone demo (30 minutes)`ワークフローで、一時的なHTTPSの試用ページを起動できます。
iPhoneのSafariで共有リンクを開き、「話す」を押してマイクを許可します。
話し終わったら「停止して文字起こし」を押してください。無音による自動停止はありません。
一時試用は1回最大30秒で終了します。Safariを前面にしたままお試しください。

この試用では、SafariのAudioWorkletで取得したPCMを、Cloudflare Quick Tunnel経由で
GitHub Actionsの一時サーバーへ送り、PC版と同じWhisper smallで処理します。
**iPhone内だけで推論する機能ではありません。** サーバー側では、議事録アプリv0.5と同じ
PCM読み込み・16 kHz変換・Silero VADによる分割・Whisper推論・時刻情報生成を実行します。
VADは停止後の音声を分割するために使い、録音を無音で終了させる用途には使いません。
認識中だけ一時PCM WAVを作成し、処理後に削除します。Linuxでは`/dev/shm`のメモリー領域を優先し、
利用できない場合はOSの一時フォルダーを使います。音声・文字をアクセスログや履歴には残しません。
サーバーとトンネルは約30分で停止します。

起動者は十分に長いランダムな接続トークンを用意し、そのSHA-256だけを`token_sha256`入力へ渡します。
接続先はワークフローの`temporary-demo-connection-*`成果物で取得し、
`https://接続先/#t=接続トークン`の形で試用者に共有します。トークンそのものを
GitHubの入力・ログ・ソースへ記載しないでください。画面はURL断片を読み取って除去し、
APIにはAuthorizationヘッダーで渡します。有効期限後はリンクを使用できません。

画面とサーバーのコードは[`demo/`](demo/)と[`scripts/demo_server.py`](scripts/demo_server.py)にあります。
Windows用ZIPは、ローカルで動くコンソールアプリのみを同梱します。

## 開発・検証

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe scripts/audit_dependencies.py --environment
.\.venv\Scripts\python.exe -m simple_speech_to_text --download-model
.\.venv\Scripts\python.exe -m simple_speech_to_text --self-test
```

`--self-test`はマイクを開かず、モデルのハッシュ照合・VAD・Whisperの実際のCPU推論を確認します。
実機マイクの録音成功や、日本語の認識精度を保証するテストではありません。
Windows用配布物はGitHub Actionsでビルドし、展開したexeと同梱モデルの動作・配布物の依存関係を検査します。

アプリ本体はMIT Licenseです。再配布するライブラリ・モデルの通知は配布ZIPに同梱します。
