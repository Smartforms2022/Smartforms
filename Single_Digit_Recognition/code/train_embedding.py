import argparse
import os
import tensorflow as tf
from cnn_model import *
import tensorflow_addons as tfa
from data import *
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import pickle


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_folder_path",
        default="data/form1",
        metavar="FILE",
        help="path to input folder"
    )
    parser.add_argument(
        "--output_path",
        default="weights/form1/",
        metavar="FILE",
        help="path to output folder"
    )
    parser.add_argument(
        "--ground_truth_path",
        default="data/form1_gt.txt",
        metavar="FILE",
        help="path to ground truth file"
    )
    parser.add_argument(
        "--split_path",
        default=None,
        metavar="FILE",
        help='path to train-val-test split pickle file'
    )
    
    return parser


def make_train_test_val_split(split_path,img_root,labels_filepath,img_type):
    gt = pd.read_csv(labels_filepath, dtype=str)
    img_list = os.listdir(img_root)
    num_images = len(img_list)
    idx_list = np.arange(num_images)
    np.random.shuffle(idx_list)

    combined_digits = ''.join(gt['phoneNumber'].tolist())
    total_num_digits = len(combined_digits) - combined_digits.count('X')
    print("total digits= %d"%(total_num_digits))
    val_set_size = int(total_num_digits * 0.2)
    test_set_size = int(total_num_digits*0.2)
    split_list = []
    train_list = []
    val_list = []
    test_list = []
    count = 0
    flag = 0
    for idx in idx_list:
        imgid = img_list[idx].replace('.'+img_type, '')
        labels = gt[gt['fileName'] == imgid]
        combined_digits = ''.join(labels['phoneNumber'].tolist())
        num_digits_img = len(combined_digits) - combined_digits.count('X')

        count += num_digits_img
        if flag == 0 and count >= val_set_size:
            flag = 1
        elif flag == 1 and count >= (val_set_size+test_set_size):
            flag = 2
        
        if flag == 0:
            val_list.append(imgid)
        elif flag == 1:
            test_list.append(imgid)
        else:
            train_list.append(imgid)
 
    print("train set size=%d, val_set_size=%d, test_set_size=%d"%(len(train_list),len(val_list),len(test_list)))
    split_list.append(train_list)
    split_list.append(val_list)
    split_list.append(test_list)

    print(len(split_list[0]), len(val_list))
    with open(split_path, 'wb') as f:
        pickle.dump(split_list, f)

    return split_list



def get_backbone(img_size, weight_path):
    obj = Model1(img_size, 10)
    model = obj.get_backbone()
    if weight_path != None:
        model.load_weights(weight_path)
    model.summary()
    return model
    
def train_model(model, train_gen, val_gen, log_path, save_path):
    # Compile the model
    model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss = tfa.losses.TripletSemiHardLoss(margin=0.1))

    callbacks = [TensorBoard(log_dir=log_path, update_freq='epoch', write_graph=False, profile_batch=0),
                ModelCheckpoint('form2_cluster_train.h5', 
                                     save_weights_only=True, period=100),
                ReduceLROnPlateau(monitor='loss', factor=0.1, patience=4, verbose=1, min_lr=1e-6),
                EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, min_delta=0.0001)]
    
    
    # Train the network
    history = model.fit(
    train_gen,
    verbose=1,
    validation_data=val_gen,
    validation_steps=len(val_gen),
    callbacks=callbacks,
    steps_per_epoch=len(train_gen),
    epochs=45)

    model.save_weights(os.path.join(args.output_path, 'embedding.h5'))
    return history

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    h,w,c = 30,30,1
    weight_path = None
    model = get_backbone((h,w,c), weight_path)
    data_path = args.image_folder_path
    label_path = args.ground_truth_path
    split_path = args.split_path


    dir1 = next(os.walk(data_path))
    img_type = dir1[2][0].split('.')[1]
    
    
    if os.path.isfile(split_path):
        with open(split_path,'rb') as f:
            img_list = pickle.load(f)
    else:
        img_list = make_train_test_val_split(split_path,data_path,label_path,img_type)
 
    train_gen = DigitDataset(img_list[0],img_type,256,data_path,label_path,random_crop=False,isClassifier=False)
    val_gen = DigitDataset(img_list[1],img_type,256, data_path,label_path,random_crop=False, isClassifier=False)

    print('training starting-')
    history = train_model(model,train_gen,val_gen,'log','weights')
    
    
