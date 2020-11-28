# About

## test setのmetricsを計測する．
* Task classに`_get_test_labels`を実装する．`jiant/tasks/lib/mrpc.py` を参考にすること．
* ただし，そもそもtest setのラベルが公開されていないタスク(GLUE等)は，metricを計測することができない．その場合は，"Cross Validationする"のように，ラベルが公開されているvalidation setをtest setとして使えば良い．
* 現状，`_get_test_labels`が使えるタスク．
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
        - cb -> load_metric はdatasetsが対応していない．
        - copa -> load_metric はdatasetsが対応していない．
        - boolq -> load_metric はdatasetsが対応していない．
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

/home/acb11878tj/work/projects/jiant/jiant/tasks/lib/socialiqa.py

## Cross Validationする．
* huggingface datasets を通して呼び出しているタスクは cross-validationが使える．
* 特別なタスク名でプログラムを呼べばよい．
    * 例: `mrpc__cv-5-0`
        - 5-fold Cross Validation
        - CV fold は，`jiant train`から作成される．
        - jiant train: CV train
        - jiant val  : CV val (0-fold目のvalidation)
        - jiant test : **jiant val**

## モデルの追加
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
