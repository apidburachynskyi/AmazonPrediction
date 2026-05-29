# Amazon Product Recommendation

Neural Collaborative Filtering models for product recommendation on Amazon Reviews 2023.

All the results are saved in the `Runs` folder.

This project uses the `Appliances` category and evaluates recommendation models with ranking metrics:

- HR@10
- NDCG@10

Implemented models:

- GMF
- MLP
- NeuMF
- NeuMF + GRU item-text extension

Training objectives:

- Binary Cross-Entropy loss with negative sampling
- Bayesian Personalized Ranking loss

---

## Execution commands

Run all commands from the project root.

```powershell
cd C:\Users\apidb\OneDrive\Bureau\AmazonPrediction-main
```

---

### 1. Download data

```powershell
python download_data.py --category Appliances --out_dir Data/Raw
```

Expected file:

```text
Data/Raw/Appliances.jsonl
```

---

### 2. Prepare data

This step binarizes ratings, filters the dataset, creates train / validation / test splits, and saves user/item mappings.

Rating binarization:

```text
4-5 stars -> label 1
1-2 stars -> label 0
3 stars   -> ignored
```

```powershell
python prepare_amazon_data.py --input Data/Raw/Appliances.jsonl --out_dir Data/Processed_Appliances
```

Expected output:

```text
Data/Processed_Appliances/train.csv
Data/Processed_Appliances/val.csv
Data/Processed_Appliances/test.csv
Data/Processed_Appliances/train_positives.csv
Data/Processed_Appliances/user_mapping.csv
Data/Processed_Appliances/item_mapping.csv
```

---

### 3. Negative sampling

The main BCE setting uses 2 negative items per positive interaction.

```powershell
python negative_sampling.py --processed_dir Data/Processed_Appliances --num_negatives 2
```

Expected output:

```text
Data/Processed_Appliances/train_neg_sampled.csv
```

Expected label counts:

```text
label 0: 770754
label 1: 385377
```

---

### 4. Random baseline

The random baseline is evaluated with the same ranking protocol as the trained models:

```text
1 positive item + 99 negative items
```

```powershell
python evaluate_random_baseline.py --processed_dir Data/Processed_Appliances --output_dir Runs/Appliances/Baselines --k 10 --num_negatives 99
```

Expected output:

```text
Runs/Appliances/Baselines/random_baseline_k10.csv
Runs/Appliances/Baselines/baseline_ranking_k10.csv
```

---

### 5. Train BCE models

Train GMF, MLP and NeuMF with BCE loss.

Main configuration:

```text
dropout = 0.4
weight_decay = 1e-6
learning_rate = 0.0001
batch_size = 512
selection_metric = HR@10
```

```powershell
python train_bce.py --model all --processed_dir Data/Processed_Appliances --output_dir Runs/Appliances/BCE_regularized --epochs 30 --batch_size 512 --lr 0.0001 --layers 64,32,16,8 --dropout 0.4 --weight_decay 0.000001 --patience 10 --selection_metric hr
```

Evaluate ranking metrics:

```powershell
python evaluate_ranking.py --model all --processed_dir Data/Processed_Appliances --runs_dir Runs/Appliances/BCE_regularized --k 10 --num_negatives 99
```

---

### 6. Train BPR models

Train GMF, MLP and NeuMF with BPR loss.

```powershell
python train_bpr.py --model all --processed_dir Data/Processed_Appliances --output_dir Runs/Appliances/BPR_regularized --epochs 30 --batch_size 512 --lr 0.0001 --layers 64,32,16,8 --dropout 0.4 --weight_decay 0.000001 --patience 10 --selection_metric hr
```

Evaluate ranking metrics:

```powershell
python evaluate_ranking.py --model all --processed_dir Data/Processed_Appliances --runs_dir Runs/Appliances/BPR_regularized --k 10 --num_negatives 99
```

---

### 7. Regularization experiments

These experiments are done with NeuMF + BCE only.

#### No regularization

```powershell
python train_bce.py --model neumf --processed_dir Data/Processed_Appliances --output_dir Runs/Appliances/Ablation_Reg_None --epochs 30 --batch_size 512 --lr 0.0001 --layers 64,32,16,8 --dropout 0.0 --weight_decay 0.0 --patience 10 --selection_metric hr
python evaluate_ranking.py --model neumf --processed_dir Data/Processed_Appliances --runs_dir Runs/Appliances/Ablation_Reg_None --k 10 --num_negatives 99
```

#### Dropout only

```powershell
python train_bce.py --model neumf --processed_dir Data/Processed_Appliances --output_dir Runs/Appliances/Ablation_Reg_DropoutOnly --epochs 30 --batch_size 512 --lr 0.0001 --layers 64,32,16,8 --dropout 0.4 --weight_decay 0.0 --patience 10 --selection_metric hr
python evaluate_ranking.py --model neumf --processed_dir Data/Processed_Appliances --runs_dir Runs/Appliances/Ablation_Reg_DropoutOnly --k 10 --num_negatives 99
```

#### L2 / weight decay only

```powershell
python train_bce.py --model neumf --processed_dir Data/Processed_Appliances --output_dir Runs/Appliances/Ablation_Reg_L2Only --epochs 30 --batch_size 512 --lr 0.0001 --layers 64,32,16,8 --dropout 0.0 --weight_decay 0.000001 --patience 10 --selection_metric hr
python evaluate_ranking.py --model neumf --processed_dir Data/Processed_Appliances --runs_dir Runs/Appliances/Ablation_Reg_L2Only --k 10 --num_negatives 99
```

#### Dropout + L2

```powershell
python train_bce.py --model neumf --processed_dir Data/Processed_Appliances --output_dir Runs/Appliances/Ablation_Reg_DropoutL2 --epochs 30 --batch_size 512 --lr 0.0001 --layers 64,32,16,8 --dropout 0.4 --weight_decay 0.000001 --patience 10 --selection_metric hr
python evaluate_ranking.py --model neumf --processed_dir Data/Processed_Appliances --runs_dir Runs/Appliances/Ablation_Reg_DropoutL2 --k 10 --num_negatives 99
```

---

### 8. MLP depth experiments

These experiments are done with NeuMF + BCE only.

#### One hidden layer

```powershell
python train_bce.py --model neumf --processed_dir Data/Processed_Appliances --output_dir Runs/Appliances/Ablation_Depth_BCE_16_8 --epochs 30 --batch_size 512 --lr 0.0001 --layers 16,8 --dropout 0.4 --weight_decay 0.000001 --patience 10 --selection_metric hr
python evaluate_ranking.py --model neumf --processed_dir Data/Processed_Appliances --runs_dir Runs/Appliances/Ablation_Depth_BCE_16_8 --k 10 --num_negatives 99
```

#### Two hidden layers

```powershell
python train_bce.py --model neumf --processed_dir Data/Processed_Appliances --output_dir Runs/Appliances/Ablation_Depth_BCE_32_16_8 --epochs 30 --batch_size 512 --lr 0.0001 --layers 32,16,8 --dropout 0.4 --weight_decay 0.000001 --patience 10 --selection_metric hr
python evaluate_ranking.py --model neumf --processed_dir Data/Processed_Appliances --runs_dir Runs/Appliances/Ablation_Depth_BCE_32_16_8 --k 10 --num_negatives 99
```

#### Three hidden layers

```powershell
python train_bce.py --model neumf --processed_dir Data/Processed_Appliances --output_dir Runs/Appliances/Ablation_Depth_BCE_64_32_16_8 --epochs 30 --batch_size 512 --lr 0.0001 --layers 64,32,16,8 --dropout 0.4 --weight_decay 0.000001 --patience 10 --selection_metric hr
python evaluate_ranking.py --model neumf --processed_dir Data/Processed_Appliances --runs_dir Runs/Appliances/Ablation_Depth_BCE_64_32_16_8 --k 10 --num_negatives 99
```

---

### 9. Learning rate experiments

These experiments are done with NeuMF + BCE only.

#### Learning rate 0.00005

```powershell
python train_bce.py --model neumf --processed_dir Data/Processed_Appliances --output_dir Runs/Appliances/Ablation_LR_BCE_00005 --epochs 30 --batch_size 512 --lr 0.00005 --layers 64,32,16,8 --dropout 0.4 --weight_decay 0.000001 --patience 10 --selection_metric hr
python evaluate_ranking.py --model neumf --processed_dir Data/Processed_Appliances --runs_dir Runs/Appliances/Ablation_LR_BCE_00005 --k 10 --num_negatives 99
```

#### Learning rate 0.0005

```powershell
python train_bce.py --model neumf --processed_dir Data/Processed_Appliances --output_dir Runs/Appliances/Ablation_LR_BCE_0005 --epochs 30 --batch_size 512 --lr 0.0005 --layers 64,32,16,8 --dropout 0.4 --weight_decay 0.000001 --patience 10 --selection_metric hr
python evaluate_ranking.py --model neumf --processed_dir Data/Processed_Appliances --runs_dir Runs/Appliances/Ablation_LR_BCE_0005 --k 10 --num_negatives 99
```

#### Learning rate 0.001

```powershell
python train_bce.py --model neumf --processed_dir Data/Processed_Appliances --output_dir Runs/Appliances/Ablation_LR_BCE_001 --epochs 30 --batch_size 512 --lr 0.001 --layers 64,32,16,8 --dropout 0.4 --weight_decay 0.000001 --patience 10 --selection_metric hr
python evaluate_ranking.py --model neumf --processed_dir Data/Processed_Appliances --runs_dir Runs/Appliances/Ablation_LR_BCE_001 --k 10 --num_negatives 99
```

#### Learning rate 0.0015

```powershell
python train_bce.py --model neumf --processed_dir Data/Processed_Appliances --output_dir Runs/Appliances/Ablation_LR_BCE_0015 --epochs 30 --batch_size 512 --lr 0.0015 --layers 64,32,16,8 --dropout 0.4 --weight_decay 0.000001 --patience 10 --selection_metric hr
python evaluate_ranking.py --model neumf --processed_dir Data/Processed_Appliances --runs_dir Runs/Appliances/Ablation_LR_BCE_0015 --k 10 --num_negatives 99
```

#### Learning rate 0.01

```powershell
python train_bce.py --model neumf --processed_dir Data/Processed_Appliances --output_dir Runs/Appliances/Ablation_LR_BCE_01 --epochs 30 --batch_size 512 --lr 0.01 --layers 64,32,16,8 --dropout 0.4 --weight_decay 0.000001 --patience 10 --selection_metric hr
python evaluate_ranking.py --model neumf --processed_dir Data/Processed_Appliances --runs_dir Runs/Appliances/Ablation_LR_BCE_01 --k 10 --num_negatives 99
```

---

### 10. Negative sampling experiments

These experiments are done with NeuMF + BCE only.

#### Original train distribution

```powershell
Copy-Item Data/Processed_Appliances/train.csv Data/Processed_Appliances/train_neg_sampled.csv -Force
python train_bce.py --model neumf --processed_dir Data/Processed_Appliances --output_dir Runs/Appliances/Ablation_Neg_BCE_original --epochs 30 --batch_size 512 --lr 0.0001 --layers 64,32,16,8 --dropout 0.4 --weight_decay 0.000001 --patience 10 --selection_metric hr
python evaluate_ranking.py --model neumf --processed_dir Data/Processed_Appliances --runs_dir Runs/Appliances/Ablation_Neg_BCE_original --k 10 --num_negatives 99
```

#### One negative per positive

```powershell
python negative_sampling.py --processed_dir Data/Processed_Appliances --num_negatives 1
python train_bce.py --model neumf --processed_dir Data/Processed_Appliances --output_dir Runs/Appliances/Ablation_Neg_BCE_1 --epochs 30 --batch_size 512 --lr 0.0001 --layers 64,32,16,8 --dropout 0.4 --weight_decay 0.000001 --patience 10 --selection_metric hr
python evaluate_ranking.py --model neumf --processed_dir Data/Processed_Appliances --runs_dir Runs/Appliances/Ablation_Neg_BCE_1 --k 10 --num_negatives 99
```

#### Two negatives per positive

```powershell
python negative_sampling.py --processed_dir Data/Processed_Appliances --num_negatives 2
python train_bce.py --model neumf --processed_dir Data/Processed_Appliances --output_dir Runs/Appliances/Ablation_Neg_BCE_2 --epochs 30 --batch_size 512 --lr 0.0001 --layers 64,32,16,8 --dropout 0.4 --weight_decay 0.000001 --patience 10 --selection_metric hr
python evaluate_ranking.py --model neumf --processed_dir Data/Processed_Appliances --runs_dir Runs/Appliances/Ablation_Neg_BCE_2 --k 10 --num_negatives 99
```

#### Four negatives per positive

```powershell
python negative_sampling.py --processed_dir Data/Processed_Appliances --num_negatives 4
python train_bce.py --model neumf --processed_dir Data/Processed_Appliances --output_dir Runs/Appliances/Ablation_Neg_BCE_4 --epochs 30 --batch_size 512 --lr 0.0001 --layers 64,32,16,8 --dropout 0.4 --weight_decay 0.000001 --patience 10 --selection_metric hr
python evaluate_ranking.py --model neumf --processed_dir Data/Processed_Appliances --runs_dir Runs/Appliances/Ablation_Neg_BCE_4 --k 10 --num_negatives 99
```

Restore the main sampling setting after negative sampling experiments:

```powershell
python negative_sampling.py --processed_dir Data/Processed_Appliances --num_negatives 2
```

---

### 11. GRU item-text extension

The GRU extension keeps the same recommendation task.

Instead of using the target user's test review text, item-level text is built from training reviews only. This avoids data leakage.

#### Prepare item text

```powershell
python prepare_item_text_gru_data.py --raw_jsonl Data/Raw/Appliances.jsonl --processed_dir Data/Processed_Appliances --out_dir Data/ProcessedText_Appliances_ItemGRU --max_vocab_size 20000 --max_len 50 --max_reviews_per_item 5
```

Expected output:

```text
Data/ProcessedText_Appliances_ItemGRU/item_tokens.npy
Data/ProcessedText_Appliances_ItemGRU/vocab.json
Data/ProcessedText_Appliances_ItemGRU/summary.json
```

#### Train NeuMF + GRU recommender

Restore the main negative sampling setting first:

```powershell
python negative_sampling.py --processed_dir Data/Processed_Appliances --num_negatives 2
```

Then train the model:

```powershell
python train_item_gru_recommender.py --processed_dir Data/Processed_Appliances --text_dir Data/ProcessedText_Appliances_ItemGRU --output_dir Runs/Appliances/NeuMF_GRU_BCE --epochs 30 --batch_size 512 --lr 0.0001 --layers 64,32,16,8 --dropout 0.4 --weight_decay 0.000001 --patience 10 --selection_metric hr
```

The model is evaluated as a recommender with HR@10 and NDCG@10.

---

## Expected dataset statistics

After preprocessing `Appliances`:

```text
Number of users: 226758
Number of items: 33699
Validation shape: 54687
Test shape: 54688
```

After negative sampling with 2 negatives per positive:

```text
Train sampled shape: 1156131
label 0: 770754
label 1: 385377
```

---

## Notes

- `Data/` is used for raw and processed datasets.
- `Runs/` is used for training outputs and evaluation results.
- Large files are ignored by Git.
- The main evaluation protocol is `1 positive + 99 negatives`.
- The main checkpoint selection metric is validation HR@10.
- The main reported metrics are HR@10 and NDCG@10.
