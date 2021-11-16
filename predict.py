import glob
import random
import json
import os

import cv2
import numpy as np
from tqdm import tqdm
from keras.models import load_model

from .train import find_latest_checkpoint
from .data_utils.data_loader import get_image_array, get_segmentation_array, DATA_LOADER_SEED, class_colors , get_pairs_from_paths
from .models.config import IMAGE_ORDERING
from . import metrics
import six
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score,matthews_corrcoef

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


random.seed(DATA_LOADER_SEED)

def model_from_checkpoint_path(checkpoints_path):

    from .models.all_models import model_from_name
    assert (os.path.isfile(checkpoints_path+"_config.json")
            ), "Checkpoint not found."
    model_config = json.loads(
        open(checkpoints_path+"_config.json", "r").read())
    latest_weights = find_latest_checkpoint(checkpoints_path)
    assert (latest_weights is not None), "Checkpoint not found."
    model = model_from_name[model_config['model_class']](
        model_config['n_classes'], input_height=model_config['input_height'],
        input_width=model_config['input_width'])
    print("loaded weights ", latest_weights)
    model.load_weights(latest_weights)
    return model


def predict(model=None, inp=None, out_fname=None, checkpoints_path=None):

    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)

    assert (inp is not None)
    assert((type(inp) is np.ndarray) or isinstance(inp, six.string_types)
           ), "Inupt should be the CV image or the input file name"

    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp)

    assert len(inp.shape) == 3, "Image should be h,w,3 "
    orininal_h = inp.shape[0]
    orininal_w = inp.shape[1]

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = get_image_array(inp, input_width, input_height, ordering=IMAGE_ORDERING)
    pr = model.predict(np.array([x]))[0]

    pr = pr.reshape((output_height,  output_width, n_classes)).argmax(axis=2)

    seg_img = np.zeros((output_height, output_width, 3))
    colors = class_colors

    for c in range(n_classes):
        seg_img[:, :, 0] += ((pr[:, :] == c)*(colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c)*(colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c)*(colors[c][2])).astype('uint8')

    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    if out_fname is not None:
        cv2.imwrite(out_fname, seg_img)

    return pr


def predict_multiple(model=None, inps=None, inp_dir=None, out_dir=None,
                     checkpoints_path=None):

    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)

    if inps is None and (inp_dir is not None):
        inps = glob.glob(os.path.join(inp_dir, "*.jpg")) + glob.glob(
            os.path.join(inp_dir, "*.png")) + \
            glob.glob(os.path.join(inp_dir, "*.jpeg"))

    assert type(inps) is list

    all_prs = []

    for i, inp in enumerate(tqdm(inps)):
        if out_dir is None:
            out_fname = None
        else:
            if isinstance(inp, six.string_types):
                out_fname = os.path.join(out_dir, os.path.basename(inp))
            else:
                out_fname = os.path.join(out_dir, str(i) + ".jpg")

        pr = predict(model, inp, out_fname)
        all_prs.append(pr)

    return all_prs



def evaluate( model=None , inp_images=None , annotations=None,inp_images_dir=None ,annotations_dir=None , checkpoints_path=None ):

    if model is None:
        assert (checkpoints_path is not None) , "Please provide the model or the checkpoints_path"
        model = model_from_checkpoint_path(checkpoints_path)

    if inp_images is None:
        assert (inp_images_dir is not None) , "Please privide inp_images or inp_images_dir"
        assert (annotations_dir is not None) , "Please privide inp_images or inp_images_dir"

        paths = get_pairs_from_paths(inp_images_dir , annotations_dir )
        paths = list(zip(*paths))
        inp_images = list(paths[0])
        annotations = list(paths[1])

    assert type(inp_images) is list
    assert type(annotations) is list

    # tp = np.zeros( model.n_classes  )
    # fp = np.zeros( model.n_classes  )
    # fn = np.zeros( model.n_classes  )
    # n_pixels = np.zeros( model.n_classes  )

    tp = np.zeros( model.n_classes -1 )
    tn = np.zeros( model.n_classes -1 )
    fp = np.zeros( model.n_classes -1 )
    fn = np.zeros( model.n_classes -1 )
    n_pixels = np.zeros( model.n_classes -1 )

    Y_pred = []
    Y_gt = []


    for inp , ann   in tqdm( zip( inp_images , annotations )):
        pr = predict(model , inp )
        gt = get_segmentation_array( ann , model.n_classes ,  model.output_width , model.output_height , no_reshape=True  )
        gt = gt.argmax(-1)
        pr = pr.flatten()
        gt = gt.flatten()
        Y_pred.append(pr)
        Y_gt.append(gt)
        # for cl_i in range(1,model.n_classes ):
        #     tp[ cl_i ] += np.sum( (pr == cl_i) * (gt == cl_i) )
        #     fp[ cl_i ] += np.sum( (pr == cl_i) * ((gt != cl_i)) )
        #     fn[ cl_i ] += np.sum( (pr != cl_i) * ((gt == cl_i)) )
        #     n_pixels[ cl_i ] += np.sum( gt == cl_i  )
        for cl_i in range(1,model.n_classes ):
            tp[ cl_i-1 ] += np.sum( (pr == cl_i) * (gt == cl_i ) )
            tn[ cl_i-1 ] += np.sum( (pr != cl_i) * (gt != cl_i ) )
            fp[ cl_i-1 ] += np.sum( (pr == cl_i) * (gt != cl_i ) )
            fn[ cl_i-1 ] += np.sum( (pr != cl_i) * (gt == cl_i ) )
            n_pixels[ cl_i-1 ] += np.sum( gt == cl_i  )
    # import pdb; pdb.set_trace()

    precision = tp/(tp+fp+1e-12)
    recall = tp/(tp+fn+1e-12)
    accuracy = (tp+tn)/(tp+tn+fp+fn+1e-12)
    sensitivity = tp/(tp+fn+1e-12)
    specificity = tn/(tn+fp+1e-12)
    f1_score = 2*(precision*recall)/(precision+recall+1e-12)
    cl_wise_score = tp / ( tp + fp + fn + 0.000000000001 )
    n_pixels_norm = n_pixels /  np.sum(n_pixels)

    Se = np.sum(sensitivity*n_pixels_norm)
    Sp = np.sum(specificity*n_pixels_norm)
    Pr = np.sum(precision*n_pixels_norm)
    Acc = np.sum(accuracy*n_pixels_norm)
    F1 = np.sum(f1_score*n_pixels_norm)
    frequency_weighted_IU = np.sum(cl_wise_score*n_pixels_norm)

    print('model_name:{}\tcheckpoints_path:{}\nSensitivity = {:.4f},Specificity = {:.4f},Presicion = {:.4f},Accuracy = {:.4f},F1_score = {:.4f},FW_IoU:{:.4f}'
    .format(model.model_name,checkpoints_path.split('\\')[0]+' '+inp_images_dir.split('\\')[-1],Se,Sp,Pr,Acc,F1,frequency_weighted_IU))
    with open('result.txt','a') as f:
        f.writelines('\nmodel_name:{}\tcheckpoints_path:{}\nSensitivity = {:.4f}\tSpecificity = {:.4f}\tPresicion = {:.4f}\nAccuracy = {:.4f}\tF1_score = {:.4f}\tFW_IoU:{:.4f}\n'
        .format(model.model_name,checkpoints_path.split('\\')[0]+' '+inp_images_dir.split('\\')[-1],Se,Sp,Pr,Acc,F1,frequency_weighted_IU))
    # import pdb; pdb.set_trace()
    cl_wise_score = tp / ( tp + fp + fn + 0.000000000001 )
    n_pixels_norm = n_pixels /  np.sum(n_pixels)
    frequency_weighted_IU = cl_wise_score*n_pixels_norm
    mean_IU = np.mean(cl_wise_score)
    print("\nfrequency_weighted_IU:{}\nmean_IU{}\nclass_wise_IU{}".format(frequency_weighted_IU,mean_IU,cl_wise_score))



    import pdb; pdb.set_trace()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    import pdb; pdb.set_trace()
    for i in range(model.n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_gt[:, i], Y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_valid.ravel(), Y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nb_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= nb_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    print("--- %s seconds ---" % (time.time() - start_time))


    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(nb_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig("./ROC/ROC_3分类.png")
    plt.show()

    with open('result.txt','a') as f:
        f.writelines('\nmodel_name:{}\nfrequency_weighted_IU:{:.4f}\tmean_IU:{:.4f}\nclass_IoU:{}\n'.format(model.model_name,frequency_weighted_IU,mean_IU,cl_wise_score))
    return {"frequency_weighted_IU":frequency_weighted_IU , "mean_IU":mean_IU , "class_wise_IU":cl_wise_score }
