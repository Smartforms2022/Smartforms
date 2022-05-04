
python ./code/train_embedding.py \
       --image_folder_path=./data/form2 \
       --output_path=./weights/form2 \
       --ground_truth_path=./data/form2_gt.txt \
       --split_path=./data/form2_split3.pkl

python ./code/train_classifier.py \
       --image_folder_path=./data/form2 \
       --output_path=./weights/form2 \
       --ground_truth_path=./data/form2_gt.txt \
       --split_path=./data/form2_split3.pkl \
       --weight_path=./weights/form2/embedding.h5

