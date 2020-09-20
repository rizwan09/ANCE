gpu_no=8

# model type
model_type="dpr"
seq_length=256
triplet="--triplet --optimizer lamb" # set this to empty for non triplet model

# hyper parameters
batch_size=16
gradient_accumulation_steps=1
learning_rate=1e-5
warmup_steps=1000

# input/output directories
base_data_dir="/checkpoint/mdrizwanparvez/data/DPR_PUBLIC/data/QA_NQ_data/"
base_data_dir="/checkpoint/mdrizwanparvez/data/DPR_PUBLIC/data/QA_NQ_data_ctx/"
#base_data_dir="/checkpoint/mdrizwanparvez/data/DPR_PUBLIC/data/QA_NQ_data_singleset_ctx/"
#base_data_dir="/checkpoint/mdrizwanparvez/data/DPR_PUBLIC/data/QA_TRIVIAQ_data/"
job_name="ann_NQ_test_singleset"
job_name="ann_NQ_test_singleset_ctx"
#job_name="ann_TRIVIAQ_test"
model_dir="${base_data_dir}${job_name}/"
model_ann_data_dir="${model_dir}ann_data/"
#pretrained_checkpoint_dir="/checkpoint/mdrizwanparvez/data/DPR_PUBLIC/data/retriever/multiset/bert-base-encoder.cp"
#pretrained_checkpoint_dir="/checkpoint/mdrizwanparvez/data/DPR_PUBLIC/data/retriever/singleset/hf_bert_base.cp"
#pretrained_checkpoint_dir="/checkpoint/mdrizwanparvez/srun/hf_bert/ctx_to_question_loss_from_scratch/dpr_biencoder.44.460"
pretrained_checkpoint_dir="/checkpoint/mdrizwanparvez/data/dpr_ctx_losss.cp"


train_cmd="\
python -m torch.distributed.launch --nproc_per_node=$gpu_no /private/home/mdrizwanparvez/ANCE/drivers/run_ann_dpr.py --model_type $model_type \
--model_name_or_path $pretrained_checkpoint_dir --task_name MSMarco $triplet --data_dir $base_data_dir \
--ann_dir $model_ann_data_dir --max_seq_length $seq_length --per_gpu_train_batch_size=$batch_size \
--gradient_accumulation_steps $gradient_accumulation_steps --learning_rate $learning_rate --output_dir $model_dir \
--warmup_steps $warmup_steps --logging_steps 100 --save_steps 1000 --log_dir "~/tensorboard/${SLURM_JOB_ID}/logs/${job_name}" \
--logging_steps 460 \
"

echo $train_cmd
#eval $train_cmd







SWEEP_NAME=ANCE_DPR_TRAIN_SINGLESET_FROM_CTX_LOSS
queue=learnfair
queue=dev
num_nodes=1
port=55188
JOBSCRIPTS=slrum_scripts
mkdir -p ${JOBSCRIPTS}


WORLD_SIZE=$gpu_no


MY_CHECKPOINT_BASE_DIR=/checkpoint/mdrizwanparvez

SAVE_STDOUT=${MY_CHECKPOINT_BASE_DIR}/stdout/
SAVE_STDERR=${MY_CHECKPOINT_BASE_DIR}/stderr/



JNAME=${SWEEP_NAME}
SCRIPT=${JOBSCRIPTS}/run.${JNAME}.sh
SLURM=${JOBSCRIPTS}/run.${JNAME}.slrm
echo ${JNAME}
echo "#!/bin/sh" > ${SCRIPT}
echo "#!/bin/sh" > ${SLURM}
echo "#SBATCH --job-name=$JNAME" >> ${SLURM}
echo "#SBATCH --output=${SAVE_STDOUT}/${JNAME}.%j" >> ${SLURM}
echo "#SBATCH --error=${SAVE_STDERR}/${JNAME}.%j" >> ${SLURM}
echo "#SBATCH --signal=USR1@120" >> ${SLURM}
echo "#SBATCH --partition=${queue}" >> ${SLURM}
echo "#SBATCH --nodes=${num_nodes}" >> ${SLURM}
echo "#SBATCH --ntasks-per-node=1" >> ${SLURM}
echo "#SBATCH --mem=500000" >> ${SLURM}
#echo "#SBATCH --gres=gpu:8" >> ${SLURM}
echo "#SBATCH --gpus-per-node=volta:8" >> ${SLURM}
#echo "#SBATCH --gpus=8" >> ${SLURM}
echo "#SBATCH --cpus-per-task=80" >> ${SLURM}
echo "#SBATCH --time=4300" >> ${SLURM}
echo "#SBATCH --constraint=volta32gb" >> ${SLURM}
echo "srun sh ${SCRIPT}" >> ${SLURM}
echo "echo \$SLURM_JOB_ID >> jobs" >> ${SCRIPT}
echo "{ " >> ${SCRIPT}
echo "echo $SWEEP_NAME " >> ${SCRIPT}
echo "cd $CODE_ROOT" >> ${SCRIPT}

echo $train_cmd>> ${SCRIPT}
echo "kill -9 \$\$" >> ${SCRIPT}
echo "} & " >> ${SCRIPT}
echo "child_pid=\$!" >> ${SCRIPT}
echo "trap \"echo 'TERM Signal received';\" TERM" >> ${SCRIPT}
echo "trap \"echo 'Signal received'; if [ \"\$SLURM_PROCID\" -eq \"0\" ]; then sbatch ${SLURM}; fi; kill -9 \$child_pid; \" USR1" >> ${SCRIPT}
echo "while true; do     sleep 1; done" >> ${SCRIPT}
sbatch ${SLURM}













echo "copy current script to model directory"
cp $0 $model_dir