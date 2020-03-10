# Action-Proposal-Networks

## Preparation


First of all, clone the code
```
https://github.com/samerogers/Action-Proposal-Networks.git
```

Then, create a folder:
```
cd faster-rcnn.pytorch && mkdir data
```

### prerequisites

* Python 2.7 or 3.6
* Pytorch 1.0 
* CUDA 8.0 or higher

### Data Preparation

* **VIRAT ACTEV Dataset**: Create list files for training,validation,testing dataset from the datasplits
```
Format: 
       /path_to_frames/ num_of_frames 
```

1. Datasplits are available under DataPreparation folder
```
cd DataPreparation 
```
2. Set the path to the split files to create the train/val/test list files in dataset_list.py
```
HOME_DIR = 'set the path to list file'
train_list_file='path to train split file'
val_list_file='path to val split file'
test_list_file='path to test split file'
```
3.Set the list files path in dataset.py 
```
ROOT_DATASET='root path where dataset is located'
filename_imglist_train ='path to the training list file'
filename_imglist_val = 'path to the validation list file'
filename_imglist_test = 'path to test list file'
train_data  = 'path to train dataset'
val_data = 'path to val dataset'
test_data = 'path to test dataset'
```
### Compilation

Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

Compile the cuda dependencies using following simple commands:

```
cd lib
python setup.py build develop
```

It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Align and ROI_Crop. The default version is compiled with Python 2.7, please compile by yourself if you are using a different python version.

## Train

Before training, set the right directory to save and load the trained models. Change the arguments "save_dir" and "load_dir" in trainval_net.py and test.py to adapt to your environment.

To train a faster R-CNN model with resnet50 on virat, simply run:
```
CUDA_VISIBLE_DEVICES=$GPU_ID python3 trainval_net.py \
                   RGB --dataset virat --net res50 --bs $BATCH_SIZE --nw $NUM_WORKERS --lr $LEARNING_RATE \
                   --lr_decay_step $DECAY_STEP --cuda --shift --shift_div 8 --shift_place blockres --num_segments 8 \
                   --use_tfb  --s $SESSION_NUM --epochs $MAX_EPOCHS --dense_sample --vis &

```
where 'bs' is the batch size with default 1,'s' is the session number to differentiate the training session,'epochs' is the value the maximum epoch should be, 'vis' to enable visualisation on validation data ,'dense_sample' to enable dense sampling ,it can be uniform_sampling,strided_sampling,random sampling  **V100 GPU accomodated batch size of 3 (24 frames) at lr_rate 0.001 , lr_decay_step of 4**

## Test

If you want to evlauate the detection performance of a pre-trained res50 model on virat test set, simply run
```
python test.py --dataset virat RGB --net res50 --num_segments 8 --nw $NUM_WORKERS --bs 1 \
                --shift --shift_div 8 --shift_place blockres --lr $LEARNING_RATE \
                --lr_decay_step $DECAY_STEP --cuda --use_tfb --epochs $MAX_EPOCHS \
                --dense_sample --checksession $CHECKSESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT
```
Specify the specific model session, chechepoch and checkpoint, e.g., SESSION=1, EPOCH=6, CHECKPOINT=416.


