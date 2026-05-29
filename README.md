# Amazon Product Recommendation
All the results are saved in Runs folder
## Execution commands

### 1. Download data
```powershell
python download_data.py --category All_Beauty --out_dir Data/Raw
```
### 2. Prepare data
```powershell
python prepare_amazon_data.py --input Data/Raw/All_Beauty.jsonl --out_dir Data/Processed
```
### 3. Negative sampling
```powershell
python negative_sampling.py --processed_dir Data/Processed --num_negatives 4
```
### 4. Test models
```powershell
python test_models.py --processed_dir Data/Processed
```
### 5. Train BCE models
```powershell
python train_bce.py --model all --processed_dir Data/Processed --output_dir Runs/BCE --epochs 30 --batch_size 256 --lr 0.001 --patience 5
```
### 6. Evaluate ranking metrics
```powershell
python evaluate_ranking.py --model all --processed_dir Data/Processed --runs_dir Runs/BCE --k 10 --num_negatives 99
```
### 7. Plot BCE learning curves
```powershell
python plot_history.py --runs_dir Runs/BCE
```
### 8. Train BPR models
```powershell
python train_bpr.py --model all --processed_dir Data/Processed --output_dir Runs/BPR --epochs 10 --batch_size 256 --lr 0.001
python evaluate_ranking.py --model all --processed_dir Data/Processed --runs_dir Runs/BPR --k 10 --num_negatives 99
python plot_bpr_history.py --runs_dir Runs/BPR
```
### 9. Regularization experiments

Baseline with early stopping:
```powershell
python train_bce.py --model neumf --processed_dir Data/Processed --output_dir Runs/BCE_baseline_es --epochs 30 --batch_size 256 --lr 0.001 --patience 5
python evaluate_ranking.py --model neumf --processed_dir Data/Processed --runs_dir Runs/BCE_baseline_es --k 10 --num_negatives 99
```
Dropout:
```powershell
python train_bce.py --model neumf --processed_dir Data/Processed --output_dir Runs/BCE_dropout --epochs 30 --batch_size 256 --lr 0.001 --dropout 0.2 --patience 5
python evaluate_ranking.py --model neumf --processed_dir Data/Processed --runs_dir Runs/BCE_dropout --k 10 --num_negatives 99
```
L2 regularization:
```powershell
python train_bce.py --model neumf --processed_dir Data/Processed --output_dir Runs/BCE_l2 --epochs 30 --batch_size 256 --lr 0.001 --weight_decay 0.0001 --patience 5
python evaluate_ranking.py --model neumf --processed_dir Data/Processed --runs_dir Runs/BCE_l2 --k 10 --num_negatives 99
```
Dropout + L2:
```powershell
python train_bce.py --model neumf --processed_dir Data/Processed --output_dir Runs/BCE_dropout_l2 --epochs 30 --batch_size 256 --lr 0.001 --dropout 0.2 --weight_decay 0.0001 --patience 5
python evaluate_ranking.py --model neumf --processed_dir Data/Processed --runs_dir Runs/BCE_dropout_l2 --k 10 --num_negatives 99
```
### 10. MLP depth experiments
```powershell
python train_bce.py --model neumf --processed_dir Data/Processed --output_dir Runs/Depth_64_32 --epochs 30 --batch_size 256 --lr 0.001 --layers 64,32 --patience 5
python evaluate_ranking.py --model neumf --processed_dir Data/Processed --runs_dir Runs/Depth_64_32 --k 10 --num_negatives 99
```
```powershell
python train_bce.py --model neumf --processed_dir Data/Processed --output_dir Runs/Depth_64_32_16 --epochs 30 --batch_size 256 --lr 0.001 --layers 64,32,16 --patience 5
python evaluate_ranking.py --model neumf --processed_dir Data/Processed --runs_dir Runs/Depth_64_32_16 --k 10 --num_negatives 99
```
```powershell
python train_bce.py --model neumf --processed_dir Data/Processed --output_dir Runs/Depth_64_32_16_8 --epochs 30 --batch_size 256 --lr 0.001 --layers 64,32,16,8 --patience 5
python evaluate_ranking.py --model neumf --processed_dir Data/Processed --runs_dir Runs/Depth_64_32_16_8 --k 10 --num_negatives 99
```

### 11. Learning rate experiments
```powershell
python train_bce.py --model neumf --processed_dir Data/Processed --output_dir Runs/LR_0001 --epochs 30 --batch_size 256 --lr 0.0001 --layers 64,32,16,8 --patience 5
python evaluate_ranking.py --model neumf --processed_dir Data/Processed --runs_dir Runs/LR_0001 --k 10 --num_negatives 99
```
```powershell
python train_bce.py --model neumf --processed_dir Data/Processed --output_dir Runs/LR_001 --epochs 30 --batch_size 256 --lr 0.001 --layers 64,32,16,8 --patience 5
python evaluate_ranking.py --model neumf --processed_dir Data/Processed --runs_dir Runs/LR_001 --k 10 --num_negatives 99
```
```powershell
python train_bce.py --model neumf --processed_dir Data/Processed --output_dir Runs/LR_005 --epochs 30 --batch_size 256 --lr 0.005 --layers 64,32,16,8 --patience 5
python evaluate_ranking.py --model neumf --processed_dir Data/Processed --runs_dir Runs/LR_005 --k 10 --num_negatives 99
```
### 12. Negative sampling experiments

1 negative per positive:
```powershell
python negative_sampling.py --processed_dir Data/Processed --num_negatives 1
python train_bce.py --model neumf --processed_dir Data/Processed --output_dir Runs/Neg_1 --epochs 30 --batch_size 256 --lr 0.001 --layers 64,32,16,8 --patience 5
python evaluate_ranking.py --model neumf --processed_dir Data/Processed --runs_dir Runs/Neg_1 --k 10 --num_negatives 99
```
4 negatives per positive:
```powershell
python negative_sampling.py --processed_dir Data/Processed --num_negatives 4
python train_bce.py --model neumf --processed_dir Data/Processed --output_dir Runs/Neg_4 --epochs 30 --batch_size 256 --lr 0.001 --layers 64,32,16,8 --patience 5
python evaluate_ranking.py --model neumf --processed_dir Data/Processed --runs_dir Runs/Neg_4 --k 10 --num_negatives 99
```
8 negatives per positive:
```powershell
python negative_sampling.py --processed_dir Data/Processed --num_negatives 8
python train_bce.py --model neumf --processed_dir Data/Processed --output_dir Runs/Neg_8 --epochs 30 --batch_size 256 --lr 0.001 --layers 64,32,16,8 --patience 5
python evaluate_ranking.py --model neumf --processed_dir Data/Processed --runs_dir Runs/Neg_8 --k 10 --num_negatives 99
```
### 13. GRU text extension
```powershell
python prepare_amazon_text_data.py --input Data/Raw/All_Beauty.jsonl --out_dir Data/ProcessedText
```
```powershell
python train_text_gru.py --processed_dir Data/ProcessedText --output_dir Runs/TextGRU --epochs 10 --batch_size 256 --lr 0.001 --max_vocab_size 20000 --max_len 50
```
