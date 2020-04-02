Follow the steps down below to compute bleu score. 
I put the testing data in the directory MLDS_hw2_1_data/testing_data (assume that you are already in the directory of hw2/hw2_1). 
So before following the steps, please copy testing data to the submission folder. 

chmod u+r+x hw2_seq2seq.sh
./hw2_seq2seq.sh MLDS_hw2_1_data/testing_data output_new.txt
python bleu_eval.py ouput_new.txt 