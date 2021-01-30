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

## Jiantのモデル構造
1. `jiant/shared/model_resolution.py`
    - BERTなど，各モデルタイプの型や・出力特徴量の型を定義する．
2. `jiant/proj/main/export_model.py`
    - transformersのモデルをダウンロードする．
    - このときのモデルは，`BertForPreTraining`など，pure encoder + top-layer から構築される．
    - このtop-layerは，次段のモジュールによって，jinatが定義するtop-layerに取り替えられる．
3. `jiant/proj/main/modeling/model_setup.py`
    - jiantのモデルを作成する．
    - このモジュールで，transformersのtop-layerが排除される:
        ```
            ancestor_model = get_ancestor_model(
                transformers_class_spec=transformers_class_spec, model_config_path=model_config_path,
            )
            encoder = get_encoder(model_arch=model_arch, ancestor_model=ancestor_model)
        ```
    - そして，jiantが定義するtop-layerが追加される．
        ```
        if task.TASK_TYPE == TaskTypes.CLASSIFICATION:
            assert taskmodel_kwargs is None
            classification_head = heads.ClassificationHead(
                hidden_size=hidden_size,
                hidden_dropout_prob=hidden_dropout_prob,
                num_labels=len(task.LABELS),
            )
            taskmodel = taskmodels.ClassificationModel(
                encoder=encoder, classification_head=classification_head,
            )
        ```

## 新規モデルタイプの追加
* `grep "Bert|BERT" ./jiant` で出てくるモジュールに，モデルタイプごとの処理を加えればよいと考えられる．
* ただし，以下のモジュールは，特定のタスクのみから利用される"専用"モジュールのようになっている．よって，当該タスクを必要としないなら，実装しなくてもよい．
    * jiant/proj/main/export_model.py
    * jiant/proj/main/modeling/heads.py
    * jiant/proj/main/modeling/model_setup.py
    * jiant/proj/main/modeling/taskmodels.py
    * jiant/shared/model_resolution.py
    * jiant/utils/transformer_utils.py




# Known Issues
* test data での性能値が異常な値になる．
    - TODO: fix
    - test dataのラベルが無いデータセットに対して，評価してしまっていることによる．
