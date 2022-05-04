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
        help="path to image folder"
    )
    parser.add_argument(
        "--weight_path",
        default="../weights/form1/classifier.h5",
        metavar="FILE",
        help="path to classifier weights"
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

    return parser


def get_model(img_size, load_path):
    obj = Model1(img_size, 10)
    model = obj.get_model(freeze_backbone=True)
    model.load_weights(load_path, by_name=True)
    return model


def test_model(model, test_gen):
    # test
    confi = model.predict(test_gen)
    return confi


    

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    h,w,c = 30,30,1
    weight_path = args.weight_path
    data_path = args.image_folder_path
    label_path = args.ground_truth_path 
    split_path = args.split_path

    model = get_model((h,w,c),weight_path)
    dir1 = next(os.walk(data_path))
    img_type = dir1[2][0].split('.')[1]
    img_list = [img_name.split('.')[0] for img_name in dir1[2]]
    
    with open(split_path,'rb') as f:
        img_list = pickle.load(f)
        img_list = img_list[2]

    print(len(img_list))
    test_gen = DigitDataset(img_list,img_type,256,data_path,label_path,random_crop=False, isClassifier=True)
    X,y = test_gen.get_labels()
    print(X.shape, y.shape)
    confi1 = test_model(model,test_gen)
    
    confi = confi1 #* 0.1 + confi2 * 0.9
    pred = np.argmax(confi,axis=1)
    labels = np.argmax(y,axis=1)
    correct = np.sum(pred==labels)
    print(correct, pred.shape[0])
    print(correct/pred.shape[0]) 
