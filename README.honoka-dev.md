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
        - arc_easy:
            * 却下．４択の問題と，５択の問題が混在しているため．実装側で対応できる可能性はあるが，面倒くさい．(stacking等)
        - arc_challenge
        - commonsenseqa
        - cosmosqa
        - hellaswag
        - quoref
        - scitail
        - snli
        - socialiqa

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
1. `grep "Bert|BERT" ./jiant` で出てくるモジュールが，transformer のタイプ依存の処理を行っていると予測できる．
2. 一方で，上記モジュールの一部は，特定のタスクのみから利用される"専用"モジュールのようになっている．
3. "専用"モジュールでないモジュールは，以下くらいだと思う:
    * jiant/proj/main/export_model.py
    * jiant/proj/main/modeling/heads.py
    * jiant/proj/main/modeling/model_setup.py
    * jiant/proj/main/modeling/taskmodels.py
    * jiant/shared/model_resolution.py
    * jiant/utils/transformer_utils.py

## hyperparameterは何がよいですか？
* [BERT論文](https://arxiv.org/pdf/1810.04805.pdf)を参考にすること．
    - lrate: [2e-5, 5e-5]
    - epochs: 3
    - ただし，「小さいデータセットだとハイパラ依存性が高いので，bestなところを探した」，とも書いてある．
* lrate:
    - `3e-5 >> 5e-5`
        * 特に，5e-5だと高すぎて学習曲線が発狂するタスクもある．
- epochs: lrate=3e-5で，5epoch．
    - 少し足りていないかも．






# Known Issues
* test data での性能値が異常な値になる．
    - TODO: fix
    - test dataのラベルが無いデータセットに対して，評価してしまっていることによる．
