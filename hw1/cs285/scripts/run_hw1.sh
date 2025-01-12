HW_DIR="TODO"

ep_len=1000
eval_batch_size=$((5 * ep_len))         # eval data collected (in the env) for logging metrics
train_batch_size=100                    # number of sampled data points to be used per gradient/train step
num_agent_train_steps_per_iter=1000     # number of gradient steps for training policy (per iter in n_iter)

n_layers=2                              # depth, of policy to be learned
size=64                                 # width of each layer, of policy to be learned
learning_rate=5e-3                      # LR for supervised learning

video_log_freq=-1                       # -1, 5

############################################
# Behavior Cloning: 2+ Tasks
############################################

method="bc"
n_iter=1

for seed in {0..9}; do
    gpu=0

    for env_name in "Ant" "Walker2d" "HalfCheetah" "Hopper"; do
        EXPERT_POLICY_FILE="${HW_DIR}/policies/experts/${env_name}.pkl"
        EXPERT_DATA_FILE="${HW_DIR}/expert_data/expert_data_${env_name}-v4.pkl"

        CUDA_VISIBLE_DEVICES=$gpu python ${HW_DIR}/scripts/run_hw1.py \
            -epf ${EXPERT_POLICY_FILE} -ed ${EXPERT_DATA_FILE} \
            -env "${env_name}-v4" -exp "${method}_${env_name}" --ep_len $ep_len \
            --num_agent_train_steps_per_iter $num_agent_train_steps_per_iter -n $n_iter \
            --eval_batch_size $eval_batch_size --train_batch_size $train_batch_size \
            --n_layers $n_layers --size $size -lr $learning_rate \
            --video_log_freq $video_log_freq --seed $seed &

        ((gpu++))
    done

    wait
done

############################################
# Behavior Cloning: Hyperparameter Tuning
############################################

env_name="Hopper"
EXPERT_POLICY_FILE="${HW_DIR}/policies/experts/${env_name}.pkl"
EXPERT_DATA_FILE="${HW_DIR}/expert_data/expert_data_${env_name}-v4.pkl"

train_batch_size=100                    # number of sampled data points to be used per gradient/train step

method="bc"
n_iter=1

seed=3

for num_agent_train_steps_per_iter in 1 10 100 1000 10000 100000; do
    CUDA_VISIBLE_DEVICES=0 python ${HW_DIR}/scripts/run_hw1.py \
        -epf ${EXPERT_POLICY_FILE} -ed ${EXPERT_DATA_FILE} \
        -env "${env_name}-v4" -exp "${method}_${env_name}" --ep_len $ep_len \
        --num_agent_train_steps_per_iter $num_agent_train_steps_per_iter -n $n_iter \
        --eval_batch_size $eval_batch_size --train_batch_size $train_batch_size \
        --n_layers $n_layers --size $size -lr $learning_rate \
        --video_log_freq $video_log_freq --seed $seed

    wait
done

############################################
# DAgger
############################################

method="dagger"
n_iter=10
batch_size=$((1 * ep_len))              # training data collected (in the env) during each iteration

for seed in {0..9}; do
    gpu=0

    for env_name in "Walker2d" "Hopper"; do
        EXPERT_POLICY_FILE="${HW_DIR}/policies/experts/${env_name}.pkl"
        EXPERT_DATA_FILE="${HW_DIR}/expert_data/expert_data_${env_name}-v4.pkl"

        CUDA_VISIBLE_DEVICES=$gpu python ${HW_DIR}/scripts/run_hw1.py \
            -epf ${EXPERT_POLICY_FILE} -ed ${EXPERT_DATA_FILE} \
            -env "${env_name}-v4" -exp "${method}_${env_name}" --ep_len $ep_len \
            --num_agent_train_steps_per_iter $num_agent_train_steps_per_iter -n $n_iter \
            --batch_size $batch_size --eval_batch_size $eval_batch_size --train_batch_size $train_batch_size \
            --n_layers $n_layers --size $size -lr $learning_rate \
            --video_log_freq $video_log_freq --seed $seed \
            --do_dagger &

        ((gpu++))
    done

    wait
done