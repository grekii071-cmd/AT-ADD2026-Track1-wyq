python main_train.py --model ft-w2v2aasist --gpu 0 --num_epochs 1 --batch_size 8 --num_workers 2 --subset 2000 --save_best_by f1 --out_fold ./ablation_A_baseline 2>&1 | tee ablation_A.log

python main_train.py --model ft-w2v2aasist --gpu 0 --num_epochs 1 --batch_size 8 --num_workers 2 --subset 2000 --save_best_by f1 --use_rawboost True --out_fold ./ablation_B_rawboost 2>&1 | tee ablation_B.log

python main_train.py --model ft-w2v2aasist --gpu 0 --num_epochs 1 --batch_size 8 --num_workers 2 --subset 2000 --save_best_by f1 --use_lora True --lora_r 8 --out_fold ./ablation_C_lora 2>&1 | tee ablation_C.log

python main_train.py --model ft-w2v2aasist --gpu 0 --num_epochs 1 --batch_size 8 --num_workers 2 --subset 2000 --save_best_by f1 --use_lora True --lora_r 8 --use_rawboost True --out_fold ./ablation_D_lora_rawboost 2>&1 | tee ablation_D.log
