#!/bin/bash
# track the close demos, build the datasets and train the diffusion models, one job per gpu
# usage: bash train_close.sh [seeds...]
# set DEMOS (raw demos) and PRIV (tracked demos) to your own data
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
SEEDS=${@:-0}
GPUS=${GPUS:-0 1}
STEPS=${STEPS:-100000}
N_TOTAL=${N_TOTAL:-300}
SETS=${SETS:-0 1 2 3}
MODELS=${MODELS:-0 1 2 3}
NAME=${NAME:-close_v1}
DEMOS=${DEMOS:-/media/$USER/T7/CoData/inner-product/inner_product/close/train}
PRIV=${PRIV:-${DEMOS%/}_privileged}
WANDB_ARG=${WANDB:+--wandb $WANDB}
OBJECTS=("handle kcup" "handle kcup apple" "handle kcup apple orange" "handle kcup apple orange banana")
mkdir -p job_outputs

[ -d "$DEMOS" ] && python track_close.py --demo_dir "$DEMOS" --out_dir "$PRIV" --static 15
if [ ! -d demos/$NAME ]; then
    python lowdim_demos.py --priv_dir "$PRIV" --out_dir demos/$NAME --n_total $N_TOTAL | tail -2
fi
for d in $SETS; do
    [ -f datasets/${NAME}_cc_d$d.pkl ] || python process_demos.py --priv_dir demos/$NAME --state_space cartesian \
        --action_space cartesian_abs --target_steps 5 --n_phases 2 --objects ${OBJECTS[$d]} --no_yaw handle kcup \
        --relative handle kcup --chunk 1 --chunk_relative --rot_relative --gripper_lead 3 --out datasets/${NAME}_cc_d$d.pkl | tail -1
done

JOBS=()
for seed in $SEEDS; do
    for d in $SETS; do
        for k in $MODELS; do
            [ $d = 0 ] && [ $k != 0 ] && continue
            JOBS+=("$d $seed $k")
        done
    done
done

CLAIMS=$(mktemp -d)
worker() {
    for job in "${JOBS[@]}"; do
        read d seed k <<< "$job"
        SET=${NAME}_cc_d$d
        LOG=job_outputs/local_${SET}_s${seed}_diffusion${k}.out
        grep -q "Step: $STEPS |" "$LOG" 2>/dev/null && continue
        mkdir "$CLAIMS/$d-$seed-$k" 2>/dev/null || continue
        echo "$(date +%H:%M:%S) gpu $1: $SET seed $seed model $k"
        CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=2 python train_diffusion_policy.py --dataset datasets/$SET.pkl \
            --save_dir models/$SET/seed$seed --seed $seed --models $k --steps $STEPS --ema 0.999 --cosine $WANDB_ARG > "$LOG" 2>&1 \
            || echo "failed: $SET seed $seed model $k, see $LOG"
    done
}
for gpu in $GPUS; do
    worker $gpu &
done
wait
rm -rf "$CLAIMS"
echo "$(date +%H:%M:%S) done: models/${NAME}_cc_d0/seed<N>/diffusion_weights0, models/${NAME}_cc_d{1,2,3}/seed<N>/diffusion_weights{0,1,2,3}"
