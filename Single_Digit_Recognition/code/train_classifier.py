import argparse
import os
import tensorflow as tf
from cnn_model import *
import tensorflow_addons as tfa
from data import *
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import pickle 



def build_parser():
    parser = argparse.ArgumentParser(description="Image folder")
    parser.add_argument(
        "--image_folder_path",
        default="../data/form1",
        metavar="FILE",
        help="path to input folder"
    )
    parser.add_argument(
        "--output_path",
        default="../weights/form1/",
        metavar="FILE",
        help="path to output folder"
    )
    parser.add_argument(
        "--ground_truth_path",
        default="../data/form1_gt.txt",
        metavar="FILE",
        help="path to ground truth file"
    )
    parser.add_argument(
        "--split_path",
        default='../data/form1_split.pkl',
        metavar="FILE",
        help='path to train-val-test split pickle file'
    )
    parser.add_argument(
        "--weight_path",
        default="../weights/form1/embedding_weights.h5",
        metavar="FILE",
        help='path to embedding model weights file'
    )

    return parser


def load_model(img_size, load_path):
    obj = Model1(img_size, 10)
    model = obj.get_model(freeze_backbone=True)
    model.summary()    
    model.load_weights(load_path, by_name=True)
    return model
    
def train_model(model, train_gen, val_gen, log_path, save_path):
    # Compile the model
    model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='categorical_crossentropy',
    metrics = ['accuracy'])

    callbacks = [TensorBoard(log_dir=log_path, update_freq='epoch', write_graph=False, profile_batch=0),
                ModelCheckpoint(save_path, save_weights_only=True),
                ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=1, min_lr=1e-6),
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, min_delta=0.0001)]
    
    
    # Train the network
    history = model.fit(
    x=train_gen,
    verbose=1,
    validation_data=val_gen,
    validation_steps=len(val_gen),
    callbacks=callbacks,
    steps_per_epoch=len(train_gen),
    epochs=25,
    workers=1
    )
    
    model.save(os.path.join(args.output_path,'classifier.h5'))
    return history

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    h,w,c = 30,30,1
    weight_path = args.weight_path
    data_path = args.image_folder_path
    label_path = args.ground_truth_path
    split_path = args.split_path
    img_list = []
    dir1 = next(os.walk(data_path))
    img_type = dir1[2][0].split('.')[1]

    model = load_model((h,w,c), weight_path)

    if split_path == None:
        img_list = make_train_val_split(data_path,label_path,img_type)
    else:
        with open(split_path,'rb') as f:
            img_list = pickle.load(f)
    #print(img_list[1])
    train_gen = DigitDataset(img_list[0],img_type,256,data_path,label_path,random_crop=False, isClassifier=True)
    val_gen = DigitDataset(img_list[2],img_type,256,data_path,label_path,random_crop=False, isClassifier=True)
    # print(len(img_list[0]), len(img_list[1]))
    history = train_model(model,train_gen,val_gen,'log_classifier','weights_classifier')
    
    
   
