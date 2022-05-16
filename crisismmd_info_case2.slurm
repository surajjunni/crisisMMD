#!/bin/bash
 
#SBATCH -n 3                        # number of cores
#SBATCH -t 99:00:00                  # wall time (D-HH:MM)
##SBATCH -A sjyothiu             # Account hours will be pulled from (commented out with double # in front)
#SBATCH -o slurm_case1.%j.out             # STDOUT (%j = JobId)
#SBATCH -e slurm_case1.%j.err             # STDERR (%j = JobId)
#SBATCH --mail-type=ALL             # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=sjyothiu@asu.edu # send-to address

module purge    # Always purge modules to ensure a consistent environment

run_dir=/home/sjyothiu/crisismmd

cd $run_dir

source /packages/7x/anaconda3/5.3.0/bin/activate crisis

which python
python3 crisismmd_case2.py -i task_informative_text_img_agreed_lab_train.tsv -v task_informative_text_img_agreed_lab_dev.tsv -t task_informative_text_img_agreed_lab_test.tsv -m models/case2/info_multimodal_paired_agreed_lab.hd5 -o results/case2/info_multimodal_results_cnn_paired_agreed_lab.txt --w2v_checkpoint w2v_checkpoint/data_w2v_info_paired_agreed_lab.model --lambda 0.75 
