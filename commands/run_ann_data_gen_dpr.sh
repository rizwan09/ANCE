#!/bin/bash
# tokenization
wiki_dir="/checkpoint/mdrizwanparvez/data/DPR_PUBLIC/data/wikipedia_split/" # path for psgs_w100.tsv downloaded with DPR code
ans_dir="/checkpoint/mdrizwanparvez/data/DPR_PUBLIC/data/retriever/qas/" # path for DPR question&answer csv files
#ans_dir="/checkpoint/mdrizwanparvez/data/DPR_PUBLIC/data/retriever/DEV_AS_TEST/qas/" # path for DPR question&answer csv files
#ans_dir="/checkpoint/mdrizwanparvez/data/DPR_PUBLIC/data/retriever/TRAIN_AS_TEST/qas/" # path for DPR question&answer csv files
question_dir="/checkpoint/mdrizwanparvez/data/DPR_PUBLIC/data/retriever/" # path for DPR training data
data_type=0 #0 is nq, 1 is trivia, 2 is both
#out_data_dir="/checkpoint/mdrizwanparvez/data/DPR_PUBLIC/data/QA_TRIVIAQ_data/" # change this for different data_type
out_data_dir="/checkpoint/mdrizwanparvez/data/DPR_PUBLIC/data/QA_NQ_data_ctx/" # change this for different data_type
model_out_data_dir=$out_data_dir # change this for different data_type
#model_out_data_dir="/checkpoint/mdrizwanparvez/data/DPR_PUBLIC/data/QA_NQ_data_singleset/" # change this for different data_type
#model_out_data_dir="/checkpoint/mdrizwanparvez/data/DPR_PUBLIC/data/QA_NQ_data_singleset/" # change this for different data_type
#out_data_dir="/checkpoint/mdrizwanparvez/data/DPR_PUBLIC/data/QA_NQ_data_singleset_DEV_AS_TEST/" # change this for different data_type
#out_data_dir="/checkpoint/mdrizwanparvez/data/DPR_PUBLIC/data/QA_NQ_data_singleset_TRAIN_AS_TEST/" # change this for different data_type

tokenization_cmd="\
python ../data/DPR_data.py --wiki_dir $wiki_dir --question_dir $question_dir --data_type $data_type --answer_dir $ans_dir \
--out_data_dir $out_data_dir \
"

echo $tokenization_cmd
eval $tokenization_cmd


gpu_no=8

# model type
model_type="dpr"
seq_length=256

# ann parameters
batch_size=16
ann_topk=200
ann_negative_sample=100

# input/output directories
base_data_dir="${out_data_dir}"
base_model_dir="${model_out_data_dir}"


#job_name="ann_NQ_test"
#job_name="ann_NQ_test_singleset"
job_name="ann_NQ_test_singleset_ctx"
#job_name="ann_TRIVIAQ_test"

prediction_output_file=/private/home/mdrizwanparvez/ANCE/commands/NQ_${prediction_dataset}_prediction_ouput_file_${job_name}.json
triviaqa_prediction_output_file=/private/home/mdrizwanparvez/ANCE/commands/TRIVIAQ_${prediction_dataset}_prediction_ouput_file_${job_name}.json


model_dir="${base_model_dir}${job_name}/"
model_ann_data_dir="${model_dir}ann_data/"
#pretrained_checkpoint_dir="/checkpoint/mdrizwanparvez/data/DPR_PUBLIC/data/retriever/multiset/bert-base-encoder.cp"
#pretrained_checkpoint_dir="/checkpoint/mdrizwanparvez/data/DPR_PUBLIC/data/retriever/multiset/bert-base-encoder.cp"
pretrained_checkpoint_dir="/checkpoint/mdrizwanparvez/srun/hf_bert/ctx_to_question_loss_from_scratch/dpr_biencoder.44.460"
passage_path=$wiki_dir
test_qa_path=$ans_dir
trivia_test_qa_path=$ans_dir


data_gen_cmd="\
python -m torch.distributed.launch --nproc_per_node=$gpu_no /private/home/mdrizwanparvez/ANCE/drivers/run_ann_data_gen_dpr.py --training_dir $model_dir \
--init_model_dir $pretrained_checkpoint_dir --model_type $model_type --output_dir $model_ann_data_dir \
--cache_dir "${model_ann_data_dir}cache/" --data_dir $base_data_dir --max_seq_length $seq_length \
--per_gpu_eval_batch_size $batch_size --topk_training $ann_topk --negative_sample $ann_negative_sample \
--passage_path $passage_path --test_qa_path $test_qa_path --trivia_test_qa_path $trivia_test_qa_path \
"
#--do_predict_only --prediction_output_file $prediction_output_file --prediction_dataset ${prediction_dataset}
# --do_predict_only --triviaqa_prediction_output_file $triviaqa_prediction_output_file \
#--do_debug

echo $data_gen_cmd


SWEEP_NAME=ANCE_DPR_DATA_GEN_SINGLESET_FROM_CTX_LOSS
queue=learnfair
queue=dev
num_nodes=1
port=54187
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

echo $data_gen_cmd>> ${SCRIPT}
echo "kill -9 \$\$" >> ${SCRIPT}
echo "} & " >> ${SCRIPT}
echo "child_pid=\$!" >> ${SCRIPT}
echo "trap \"echo 'TERM Signal received';\" TERM" >> ${SCRIPT}
echo "trap \"echo 'Signal received'; if [ \"\$SLURM_PROCID\" -eq \"0\" ]; then sbatch ${SLURM}; fi; kill -9 \$child_pid; \" USR1" >> ${SCRIPT}
echo "while true; do     sleep 1; done" >> ${SCRIPT}
sbatch ${SLURM}
