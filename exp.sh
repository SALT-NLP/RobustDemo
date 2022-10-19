tagging=IO
k=5shots
task=ner
dataset=conll2003
for mode in standard standard_wrong standard_no_l random_totally random_support
do
  for id in 1 2 3 4 5
  do
    for seed in 11 42 55
    do
      python3 train.py --gpu 0 \
      --task ${task} \
      --dataset ${dataset} \
      --tagging ${tagging} \
      --mode ${mode} \
      --do_train \
      --do_eval \
      --do_predict \
      --evaluate_during_training \
      --evaluate_period 2 \
      --batch_size 4 \
      --learning_rate 2e-5 \
      --num_train_epochs 50 \
      --max_seq_length 300 \
      --output_dir out/${task}-${k}/${dataset}-${tagging}/${mode}/${id}_${seed} \
      --few_shot_dir data/${task}/${dataset}/${k}/${id} \
      --seed $seed
    done
  done
done

for mode in no
do
  for id in 1 2 3 4 5
  do
    for seed in 11 42 55
    do
      python3 train.py --gpu 0 \
      --task ${task} \
      --dataset ${dataset} \
      --tagging ${tagging} \
      --mode ${mode} \
      --do_train \
      --do_eval \
      --do_predict \
      --evaluate_during_training \
      --evaluate_period 2 \
      --batch_size 4 \
      --learning_rate 2e-5 \
      --num_train_epochs 50 \
      --max_seq_length 256 \
      --output_dir out/${task}-${k}/${dataset}-${tagging}/${mode}/${id}_${seed} \
      --few_shot_dir data/${task}/${dataset}/${k}/${id} \
      --seed $seed
    done
  done
done