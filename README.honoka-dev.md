# About

## 結局，どのタスクを回せば良いですか？
* **huggingface.datasets のタスク**
    * test setのラベルが公開されていないタスクも手元で性能評価したい -> Cross-Validation必須 -> huggingface.datasets のみ対応
* 具体的なタスク名
    * GLUE
        - mnli
        - qqp
        - qnli
        - sst2
        - cola
        - stsb
        - mrpc
        - rte
        - wnli
    * SuperGLUE
        - rte
        - cb
        - copa
        - boolq
    * その他
        - arc_easy
        - arc_challenge
        - commonsenseqa
        - cosmosqa
        - hellaswag
        - quoref
        - scitail
        - snli
        - socialqa
        - swag

## test setのmetricsを計測するために．
* test setのmetricsを計測することができる条件
    - Task classに`_get_test_labels`が正しく実装されている．
        * 最上位クラス`jiant.tasks.core.Task` で`_get_test_labels`が実装されているが，サブクラスの実装に依存しているため，動かない場合もある．その場合，サブクラスでオーバーライドする必要がある．
    - test setのラベルが公開されていること．
* test setのラベルが公開されていないタスク(例: GLUE)の場合，"Cross Validationする"に記載しているように，ラベルが公開されているvalidation setをtest setとして使えば良い．
* `_get_test_labels`が実装されているタスク

## Cross Validationするために．
* huggingface datasets を通して呼び出しているタスクは cross-validationが使える．
* 特別なタスク名でプログラムを呼べばよい．
    * 例: `mrpc__cv-5-0`
        - 5-fold Cross Validation
        - CV fold は，`jiant train`から作成される．
        - jiant train: CV train
        - jiant val  : CV val (0-fold目のvalidation)
        - jiant test : **jiant val**

## 新規モデルタイプの追加
- `grep "Bert" ./jiant` で出てくる全てのファイルに，新規クラス(ex. GPT2)を追加する必要がある．
- ただ，上記殆どのファイルは，一部のタスクからしか呼ばれない．それ意外のタスクのみでよいなら，以下のファイルを編集すれば良いと思われる：
    * jiant/jiant/proj/main/export_model.py
    * jiant/jiant/shared/model_resolution.py
    * jiant/jiant/proj/main/modeling/heads.py
    * jiant/jiant/proj/main/modeling/model_setup.py
* 利用可能なモデル
    - `jiant/jiant/proj/main/export_model.py`




# Known Issues
* test data での性能値が異常な値になる．
    - TODO: fix
    - test dataのラベルが無いデータセットに対して，評価してしまっていることによる．
