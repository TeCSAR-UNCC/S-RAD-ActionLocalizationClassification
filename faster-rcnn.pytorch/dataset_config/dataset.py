import os

ROOT_DATASET = '/mnt/AI_RAID/'  


def return_ucf101(modality):
    filename_categories = 'UCF101/labels/classInd.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'UCF101/jpg'
        filename_imglist_train = 'UCF101/file_list/ucf101_rgb_train_split_1.txt'
        filename_imglist_val = 'UCF101/file_list/ucf101_rgb_val_split_1.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'UCF101/jpg'
        filename_imglist_train = 'UCF101/file_list/ucf101_flow_train_split_1.txt'
        filename_imglist_val = 'UCF101/file_list/ucf101_flow_val_split_1.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_hmdb51(modality):
    filename_categories = 51
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'HMDB51/images'
        filename_imglist_train = 'HMDB51/splits/hmdb51_rgb_train_split_1.txt'
        filename_imglist_val = 'HMDB51/splits/hmdb51_rgb_val_split_1.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'HMDB51/images'
        filename_imglist_train = 'HMDB51/splits/hmdb51_flow_train_split_1.txt'
        filename_imglist_val = 'HMDB51/splits/hmdb51_flow_val_split_1.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_kinetics(modality):
    filename_categories = 400
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'kinetics/images'
        filename_imglist_train = 'kinetics/labels/train_videofolder.txt'
        filename_imglist_val = 'kinetics/labels/val_videofolder.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_virat(modality):
    filename_categories = 4
    if modality == 'RGB':
        train_data = ROOT_DATASET + 'VIRAT/actev-data-repo/dataset/train/'
        val_data = ROOT_DATASET + 'VIRAT/actev-data-repo/dataset/val/'
        test_data = ROOT_DATASET + 'VIRAT/actev-data-repo/dataset/test/'
        filename_imglist_train = '/home/malar/git_sam/Action-Proposal-Networks/faster-rcnn.pytorch/train_list.txt'
        filename_imglist_val = '/home/malar/git_sam/Action-Proposal-Networks/faster-rcnn.pytorch/val_list.txt'
        filename_imglist_test = '/home/malar/git_sam/Action-Proposal-Networks/faster-rcnn.pytorch/test_list.txt'
        #prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val,filename_imglist_test,train_data,val_data,test_data



def return_dataset(dataset, modality):
    dict_single = {'ucf101': return_ucf101, 'hmdb51': return_hmdb51,
                   'kinetics': return_kinetics,'virat': return_virat }
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val,file_imglist_test,train_data,val_data,test_data = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val,file_imglist_test, train_data,val_data,test_data