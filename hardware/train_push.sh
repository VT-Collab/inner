#!/bin/bash
# build the push datasets and train the diffusion models, one job per gpu
# GPUS lists a gpu per worker
# usage: bash train_push.sh [seeds...]
# set PRIV (tracked demos) and RAW (raw demos) to build from your own data
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
SEEDS=${@:-0}
GPUS=${GPUS:-0 1}
STEPS=${STEPS:-100000}
N_TOTAL=${N_TOTAL:-100}
N_PHASES=${N_PHASES:-2}
TARGET_STEPS=${TARGET_STEPS:-5}
NAME=${NAME:-push_v4}
WANDB_ARG=${WANDB:+--wandb $WANDB}
OBJECTS=("pot" "pot apple" "pot apple orange" "pot apple orange banana")
mkdir -p job_outputs

if [ ! -f demos/$NAME/demo_$((N_TOTAL - 1)).hdf5 ]; then
    if [ -n "$PRIV" ]; then
        python lowdim_demos.py --priv_dir "$PRIV" ${RAW:+--raw_dir "$RAW"} --out_dir demos/$NAME --n_total $N_TOTAL | tail -2
    else
        python lowdim_demos.py --from_lowdim demos/push_v3 --out_dir demos/$NAME --n_total $N_TOTAL | tail -2
    fi
fi
for d in 0 1 2 3; do
    [ -f datasets/${NAME}_cc_d$d.pkl ] || python process_demos.py --priv_dir demos/$NAME --state_space cartesian \
        --action_space cartesian_abs --target_steps $TARGET_STEPS --n_phases $N_PHASES --objects ${OBJECTS[$d]} \
        --out datasets/${NAME}_cc_d$d.pkl | tail -1
done

JOBS=()
for seed in $SEEDS; do
    for d in 0 1 2 3; do
        for k in 0 1 2 3; do
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
            --save_dir models/$SET/seed$seed --seed $seed --models $k --steps $STEPS $WANDB_ARG > "$LOG" 2>&1 \
            || echo "failed: $SET seed $seed model $k, see $LOG"
    done
}
for gpu in $GPUS; do
    worker $gpu &
done
wait
rm -rf "$CLAIMS"
echo "$(date +%H:%M:%S) done: models/${NAME}_cc_d0/seed<N>/diffusion_weights0, models/${NAME}_cc_d{1,2,3}/seed<N>/diffusion_weights{0,1,2,3}"
