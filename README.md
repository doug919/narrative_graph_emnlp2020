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



## Use Pre-trained Models

We provide commands that evaluate our pre-trained NG model in each evaluation task to reproduce the reported results. For running other baseline models, simply check `--help` menu to see how to specify them in the command. Usually, you only need to change `--model_name` and `--model_config`. If you find anything that doesn't work, please report them in the Issue page.


### Intrinsic: Multiple-Choice Narrative Cloze



### Intrinsic: Triplet Classification

> python bin/pretraining/eval_intrinsic_5.py triplet configs/config_narrative_graph.json v1_released/data/ng_v6_data_noent_min20_test_sampled v1_released/data/ng_v6_data_noent_min20_test_intrinsic_fixed_small_10ch tmpout --model_name ng --model_config configs/config_model_rgcn_ng_pp_mclass_dropout01.json --model_dir v1_released/model/out_ng_v6_noent20_h128_lr24_wd095_mclass_dropout01 -v -g 0 --test_n_graphs 5000


### Intrinsic: Predict Coreferent Next Event

> python bin/pretraining/eval_intrinsic_5.py pp_coref_next configs/config_narrative_graph.json v1_released/data/ng_v6_data_noent_min20_test_sampled_small v1_released/data/narrative_graph_exps/ng_v6_data_noent_min20_test_intrinsic_fixed_small_10ch tmpout --model_name ng --model_config configs/config_model_rgcn_ng_pp_mclass_dropout01.json --model_dir v1_released/model/out_ng_v6_noent20_h128_lr24_wd095_mclass_dropout01 -v -g 0 --test_n_graphs 5000

### Intrinsic: Predict Discourse Next Event

> python bin/pretraining/eval_intrinsic_5.py pp_discourse_link configs/config_narrative_graph.json v1_released/data/ng_v6_data_noent_min20_test_sampled_small v1_released/data/narrative_graph_exps/ng_v6_data_noent_min20_test_intrinsic_fixed_small_10ch tmpout --model_name ng --model_config configs/config_model_rgcn_ng_pp_mclass_dropout01.json --model_dir v1_released/model/out_ng_v6_noent20_h128_lr24_wd095_mclass_dropout01 -v -g 0 --test_n_graphs 5000

### Extrinsic: Implicit Discourse Sense Classification




## Run Pre-training from Scratch



