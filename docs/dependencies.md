# FFmpeg を含まない音声処理と検査範囲

このアプリでは、マイクから取り込んだ PCM をローカルの音声認識エンジンへ渡します。
FFmpeg を外部コマンドとして呼ぶ経路、FFmpeg を内包する PyAV / faster-whisper の経路、
ブラウザーのメディアデコード、音声認識 API は使いません。API キーも不要です。

## 処理経路

```text
録音完了した PCM → 一時 PCM16 WAV
                → giji-automemo v0.5 の Pipeline._transcribe
                → WAV を1秒単位で読み、SciPy で 16 kHz へ変換
                → Silero VAD で発話区間を分割
                → Whisper small (ONNX / int8 / CPU) → 時刻付き文字 → 表示
```

現在の `main` と iPhone 試用版は、[giji-automemo v0.5.0](https://github.com/kitworks-iino/giji-automemo/tree/d5745a3152e7c022704e64d2421196892ebfbe49)
の `pipeline.py` と `models.py` を変更せず同梱し、元の文字起こしメソッドを呼び出します。
由来と SHA-256 は [PROVENANCE.json](../src/simple_speech_to_text/_vendor/giji_v050/PROVENANCE.json) に記録します。
Silero VAD は閾値 0.5・無音区間 0.5 秒・最短発話 0.25 秒・最大区間 28 秒、
Whisper は CPU 4 スレッド・セグメント時刻情報あり、という元の設定をそのまま使います。
会議用システム音声の代わりに空の PCM WAV を渡すため、処理対象はマイク音声だけです。

Windows のマイク取得・録音終了検出と、Safari の AudioWorklet による録音・停止ボタンは別の入口です。
iPhone では元のサンプルレートの PCM をサーバーへ送信し、上記の共通処理に渡します。
サーバーの VAD は録音停止後に実行するため、無音で録音を自動終了させません。
公開済み Windows ZIP v0.1.0 は処理経路統一前の版で、この説明の対象は現在のソースと試用版です。

入力はマイクです。MP3・AAC・動画の取り込みや、圧縮音声への変換はありません。
認識中だけ PCM WAV を一時作成し、成功・例外のいずれの場合も終了時に削除します。
Linux は `/dev/shm` を優先し、利用できない場合と Windows は OS の一時フォルダーを使います。
録音・文字の履歴保存はありません。PC 版は音声を外部へ送信せず、iPhone 試用版のみ一時サーバーへ送信します。
配布 ZIP にはモデルを同梱し、通常起動でのダウンロードを不要にします。
ソース版でモデルを初めて取得するときだけ、公開されたモデル配布先へ接続します。

| パッケージ | 固定版 | 用途 |
| --- | --- | --- |
| sounddevice | 0.5.3 | PortAudio によるマイク入力 |
| cffi | 2.0.0 | sounddevice からネイティブ関数を呼ぶ |
| pycparser | 2.23 | cffi の Python 依存 |
| NumPy | 2.3.5 | PCM の数値列 |
| SciPy | 1.16.3 | サンプルレート変換 |
| sherpa-onnx | 1.13.8 | VAD・Whisper 推論の Python バインディング |
| sherpa-onnx-core | 1.13.8 | sherpa-onnx と ONNX Runtime のネイティブライブラリ |
| httpx | 0.28.1 | 変更せず同梱した元モジュールの import に必要。文字起こし処理では通信しない |

httpx の依存には httpcore・h11・anyio・certifi・idna などがあります。音声コーデックは含みません。
元モジュール内のモデル取得・LLM 通信・話者分離は、この文字起こし経路では呼び出しません。
httpx とその依存も起動時・配布時の依存検査、およびライセンス収集の対象です。
Whisper のモデル・トークンと Silero VAD は、モデル管理コードに固定した SHA-256 を確認して取得します。
OpenAI の `whisper` Python パッケージはインストールしません。
Whisper 推論は実行中の PC / 一時サーバー内で完結します。

## Windows 用 wheel の確認

2026年10月23日に、以下の CPython 3.13 / Windows x64 用 wheel を PyPI から取得し、
PyPI 公開情報の SHA-256 とダウンロードしたファイルのハッシュが一致することを確認しました。

| wheel | SHA-256 |
| --- | --- |
| `sounddevice-0.5.3-py3-none-win_amd64.whl` | `f55ad20082efc2bdec06928e974fbcae07bc6c405409ae1334cefe7d377eb687` |
| `cffi-2.0.0-cp313-cp313-win_amd64.whl` | `19f705ada2530c1167abacb171925dd886168931e0a7b78f5bffcae5c6b5be75` |
| `pycparser-2.23-py3-none-any.whl` | `e5c6e8d3fbad53479cab09ac03729e0a9faf2bee3db8208a550daf5af81a5934` |
| `numpy-2.3.5-cp313-cp313-win_amd64.whl` | `00dc4e846108a382c5869e77c6ed514394bdeb3403461d25a829711041217d5b` |
| `scipy-1.16.3-cp313-cp313-win_amd64.whl` | `062246acacbe9f8210de8e751b16fc37458213f124bef161a5a02c7a39284304` |
| `sherpa_onnx-1.13.8-cp313-cp313-win_amd64.whl` | `195455d8d6cc49f616e9d459b7d08f10a32daf2ddcbe90d4d479a8ba9aab28bf` |
| `sherpa_onnx_core-1.13.8-py3-none-win_amd64.whl` | `5579e80196d516e6dae23c8f629292ce3142ab8869925d32b94612fd86f93733` |

7 個の wheel の非空ファイル 3,010 個と、ネイティブファイル 285 個を検査しました。
DLL の通常・遅延 import、禁止パッケージの依存メタデータ、禁止ライブラリ名と
FFmpeg の主要なネイティブ識別子は検出されませんでした。
ネイティブファイル数には、重複配置された DLL、リンク用 `.lib`、元 wheel の ASIO 版も含みます。
SciPy wheel 内の空ファイルなど、0 バイトのファイルは名前のみ確認しています。

sounddevice の通常版 `libportaudio64bit.dll` の SHA-256 は
`433b77811ed6a2509cd15d9af826335c458cd0ef5f424b02b97597c74e170462` です。
通常・遅延 import の参照先は `ADVAPI32.dll`、`KERNEL32.dll`、`ole32.dll`、
`SETUPAPI.dll`、`USER32.dll`、`WINMM.dll` でした。
配布 ZIP では不要な ASIO 版 DLL を除き、通常の Windows マイク入力を使います。
Windows 起動時はこの同梱 DLL の絶対パスを指定します。PATH 上の別の PortAudio や
conda の DLL へはフォールバックしません。同梱 DLL がない場合や、別の DLL が既に
読み込まれた Python プロセスでは停止します。Linux のソース実行ではシステムの PortAudio を使うため、
上記の Windows DLL の検査結果は適用されません。

元の sounddevice wheel には ASIO 版も含まれるため、ソース版の `pip install` ではそのファイルが
仮想環境へ入ります。ASIO 版も上記の FFmpeg 検査は通過していますが、このアプリはそれを使いません。

## 上流の構成とライセンス

確認に使用した上流ソースは次のとおりです。

- [giji-automemo v0.5.0](https://github.com/kitworks-iino/giji-automemo/tree/d5745a3152e7c022704e64d2421196892ebfbe49)：MIT。元のライセンスと由来を同梱。
- [httpx 0.28.1](https://github.com/encode/httpx/tree/0.28.1)：BSD-3-Clause。依存パッケージの原文通知も収集。
- [sounddevice 0.5.3](https://github.com/spatialaudio/python-sounddevice/tree/0.5.3)：MIT。
  同版の `portaudio-binaries` サブモジュールは `083325660f3e002371213eef8f6a4bd579b647d6`。
- [PortAudio のビルド定義](https://github.com/spatialaudio/portaudio-binaries/blob/083325660f3e002371213eef8f6a4bd579b647d6/.github/workflows/build-libs.yml)：
  PortAudio v19.7.0 を取得し、通常版は `PA_USE_ASIO=OFF` でビルド。PortAudio は MIT。
- [sherpa-onnx 1.13.8 の Windows ビルド](https://github.com/k2-fsa/sherpa-onnx/blob/11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf/.github/workflows/build-wheels-win-x64.yaml)：
  sherpa-onnx は Apache-2.0。
- [sherpa-onnx のリンク定義](https://github.com/k2-fsa/sherpa-onnx/blob/11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf/sherpa-onnx/csrc/CMakeLists.txt)と
  [ONNX Runtime の取得定義](https://github.com/k2-fsa/sherpa-onnx/blob/11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf/cmake/onnxruntime-win-x64.cmake)：
  固定版の FFmpeg ビルド・リンク依存は見つかりませんでした。Windows x64 用 ONNX Runtime は 1.28.2。
- [Whisper のライセンス](https://github.com/openai/whisper/blob/31243bad24cc746f07d4c8bfdd2d974872cb1803/LICENSE)：MIT。
- [Silero VAD のライセンス](https://github.com/snakers4/silero-vad/blob/6478567951ae5c9979ad7b234185b5515f4be7a1/LICENSE)：MIT。
- [ONNX Runtime のライセンス](https://github.com/microsoft/onnxruntime/blob/33ca9628233dc8f002435e868d4c2e9f82766ca1/LICENSE)と
  [第三者ライセンス通知](https://github.com/microsoft/onnxruntime/blob/33ca9628233dc8f002435e868d4c2e9f82766ca1/ThirdPartyNotices.txt)。

配布物には使用したパッケージとモデルのライセンス通知を同梱します。
FFmpeg を除外することと、あらゆる特許・ライセンス上の権利関係を保証することは別です。
この文書は採用構成と検査結果の記録であり、特許非侵害の保証ではありません。

## 再検査と限界

起動前の [依存チェック](../src/simple_speech_to_text/dependency_policy.py) は、
インストール済みパッケージのメタデータを標準ライブラリだけで読み、
FFmpeg 系パッケージや PyAV 等を検出すると、音声ライブラリを読み込む前に停止します。
既存環境のパッケージを勝手に削除しません。専用の新しい仮想環境を使ってください。

[ビルド時の検査](../scripts/audit_dependencies.py) では、メタデータに加えて配布ファイル名、
通常・遅延 PE import、ネイティブ識別子、内包アーカイブを調べます。
検査対象のハッシュと import は検査 JSON に残ります。
Windows の最終 ZIP でもこの検査を通す構成です。

```powershell
.\.venv\Scripts\python.exe scripts/audit_dependencies.py --environment
.\.venv\Scripts\python.exe scripts/audit_dependencies.py --archive dist/simple-speech-to-text-windows-x64.zip
```

wheel の検査と、最終 ZIP の検査は別です。上記の wheel の記録だけを、
別バージョン・独自ビルド・未検査の ZIP の検査結果として扱わないでください。
ソースから上流バイナリを再現ビルドしたわけではなく、検査対象の識別子に一致しない
任意の改変まで完全に証明する検査でもありません。依存を更新するときは再確認が必要です。
利用者の PC にある他のアプリ・オーディオドライバー・OS 全体の構成は、このアプリの検査対象外です。
