# ソースコードをクローンしてくる
サブモジュールを含むので、以下のコマンドでクローンする。
```
git clone --recursive git@github.com:Shiiya0418/minipro_music.git
```

サブモジュール無しでクローンしちゃったら以下のコマンドでサブモジュールをクローンする。
```
git submodule update --init --recursive
```
# 基本の前処理
lmd_matchを手元に落として解凍 -> zip作成
```
wget http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz
tar -xzvf lmd_matched.tar.gz
zip -r lmd_matched.zip lmd_matched
```
これでカレントディレクトリに以下3つファイル/ディレクトリができている
```
- lmd_matched
- lmd_matched.tar.gz
- lmd_matched.zip
```

## Octuple MIDI の前処理
以下のコマンドを実行する。
```
python -u preprocess/midi_to_octuple.py
```
標準入力が求められるので、次のように入力する。
```
Dataset zip path: lmd_matched.zip
OctupleMIDI output path: lmd_matched_octuple
```
すると、`lmd_matched_octuple` にOctupleに変換されたtxtファイルがいっぱいできる。

## REMI の前処理
以下のコマンドを実行する。
```
python preprocess/midi_to_remi.py
```
すると、 `lmd_matched_REMI` にいっぱいREMI+形式に変換されたjsonファイルがいっぱいできる。

# 参考
- [microsft/muzic/musicbert](https://github.com/microsoft/muzic/tree/main/musicbert)
- [Miditok](https://miditok.readthedocs.io/en/latest/)