# narrative_graph_emnlp2020
This repository contains implementations and trained models for reproducing results reported in the follow paper:
> I-Ta Lee, Maria Lenonor Pacheco, and Dan Goldwsser, "Weakly-Supervised Modeling of Contextualized Event Embedding for Discourse Relations," Findings of EMNLP 2020

## Dependencies

- Python 3.6.x
- OpenJDK 11


Install python packages in `requirements.txt`:
```
pip install -r requirements.txt
```

## Data

The data used in the paper can be downloaded from here: 

  https://drive.google.com/file/d/1cTXGlf2Z5DBkEqsYDlQg6pG8Ol-HGNLo/view?usp=sharing

Simply decompress it to the repository folder. The default folder name is `v{0-9}_released`. The subfolder `data` contains pre-processed dev and test sets, and `model` contains all the trained models. Note that the pre-processed training set is too huge to share. If you need it, please follow `Run Pre-training from Scratch` section to run the pre-processing code on the raw training. The raw training data can be found in the links below:
- Englishg Gigaword: https://catalog.ldc.upenn.edu/LDC2011T07
  - We use the NYT section and the data splits can be found here: https://mark.granroth-wilding.co.uk/papers/what_happens_next/
- CoNLL 2016 Shared Task: https://www.cs.brandeis.edu/~clp/conll16st/


## Use Pre-trained Models

We provide commands that evaluate our pre-trained NG model in each evaluation task to reproduce the reported results. For running other baseline models, simply check `--help` menu to see how to specify them in the command. Usually, you only need to change the model-specific command options,  like `--model_name`, `--model_dir`, `--model_file`, or `--model_config`. If you find anything that doesn't work, please report them in Issues page.


### Intrinsic: Multiple-Choice Narrative Cloze

> python bin/pretraining/eval_mcnc_8.py configs/config_narrative_graph.json v1_released/data/narrative_graph_exps/ng_v10_chain9_test_mcnc_sampled_small tmpout --model_name ng --model_config configs/config_model_rgcn_ng_pp_mclass_dropout01.json --model_dir v1_released/model/out_ng_v6_noent20_h128_lr24_wd095_mclass_dropout01 -v -g 0 --test_n_graphs 10000

### Intrinsic: Triplet Classification

> python bin/pretraining/eval_intrinsic_5.py triplet configs/config_narrative_graph.json v1_released/data/ng_v6_data_noent_min20_test_sampled v1_released/data/ng_v6_data_noent_min20_test_intrinsic_fixed_small_10ch tmpout --model_name ng --model_config configs/config_model_rgcn_ng_pp_mclass_dropout01.json --model_dir v1_released/model/out_ng_v6_noent20_h128_lr24_wd095_mclass_dropout01 -v -g 0 --test_n_graphs 5000


### Intrinsic: Predict Coreferent Next Event

> python bin/pretraining/eval_intrinsic_5.py pp_coref_next configs/config_narrative_graph.json v1_released/data/ng_v6_data_noent_min20_test_sampled_small v1_released/data/narrative_graph_exps/ng_v6_data_noent_min20_test_intrinsic_fixed_small_10ch tmpout --model_name ng --model_config configs/config_model_rgcn_ng_pp_mclass_dropout01.json --model_dir v1_released/model/out_ng_v6_noent20_h128_lr24_wd095_mclass_dropout01 -v -g 0 --test_n_graphs 5000

### Intrinsic: Predict Discourse Next Event

> python bin/pretraining/eval_intrinsic_5.py pp_discourse_link configs/config_narrative_graph.json v1_released/data/ng_v6_data_noent_min20_test_sampled_small v1_released/data/narrative_graph_exps/ng_v6_data_noent_min20_test_intrinsic_fixed_small_10ch tmpout --model_name ng --model_config configs/config_model_rgcn_ng_pp_mclass_dropout01.json --model_dir v1_released/model/out_ng_v6_noent20_h128_lr24_wd095_mclass_dropout01 -v -g 0 --test_n_graphs 5000

### Extrinsic: Implicit Discourse Sense Classification

#### Val
> python bin/task/implicit_discourse/train_3.py tmpout test --test_dir v1_released/data/features_conll16v4_ng284000_dev --test_rel v1_released/data/conll16_pdtb_dev/relations_dev.json --model_file v1_released/model/out_conll16v4_ng284000_mclass_lr13_wd095_cw_p20_19777/best_model.pt -v -g 0 --use_ng

#### Test
> python bin/task/implicit_discourse/train_3.py tmpout test --test_dir v1_released/data/features_conll16v4_ng284000_test --test_rel v1_released/data/conll16_pdtb_test/relations_test.json --model_file v1_released/model/out_conll16v4_ng284000_mclass_lr13_wd095_cw_p20_19777/best_model.pt -v -g 0 --use_ng

#### Blind-Test
> python bin/task/implicit_discourse/train_3.py tmpout test --test_dir v1_released/data/features_conll16v4_ng284000_blind_test --test_rel v1_released/data/conll16_pdtb_blind/relations_blind.json --model_file v1_released/model/out_conll16v4_ng284000_mclass_lr13_wd095_cw_p20_19777/best_model.pt -v -g 0 --use_ng


## Run Pre-training from Scratch

### For Narrative Graph Pre-training

We will prepare a script to do the steps below, except Step 1. Here we still write down the pipeline flow for reference.

1. Parse raw text of Gigaword NYT using Stanford CoreNLP and save the parsing output in the json format (one document in one line). We mainly need the dependency tree and coreference resolution outputs. We provide an example output json file that contains two documents: v{0-9}_released/data/example_parsed
2. run `bin/pretraining/prepare_ng_1.py` to prepare narrative graphs for each split. Note that the corpus folder is specified in the config file. The default example input is `v1_released/data/nyt_parsed_examples`. There are only two documents in each split for demoonstrating.  Run this command for *train*, *dev*, and *test* splits.
  > python bin/pretraining/prepare_ng_1.py configs/config_narrative_graph.json ng_features --target_split dev -v
3. run `bin/pretraining/sample_triplets_2.py` to sample negative edges for *train*, *dev* and *test* splits (the train split will be sampled during training).
  > python bin/pretraining/sample_triplets_2.py configs/config_narrative_graph.json ng_features/dev dev neg_sample_dev -v
4. run `bin/pretraining/train_3.py` for training.
  > python3 bin/pretraining/train_3.py configs/config_narrative_graph.json configs/config_model_rgcn_ng_pp_mclass_dropout01.json out_model --train_dir neg_sample_train --dev_dir neg_sample_dev --model_name ng -v -g 0
  There are of course more detailed options. Please refer to the help menu of each command.

## Other Notes
- Sample intrinsic evaluations' questions: run `bin/pretraining/sample_intrinsic_4.py`.
- Sample MCNC questions: run `bin/pretraining/sample_mcnc_6.py`.


