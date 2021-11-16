import argparse
import json
from .data_utils.data_loader import image_segmentation_generator, \
    verify_segmentation_dataset
import os
import glob
import six
from tensorflow.keras.losses import categorical_crossentropy,binary_crossentropy
from keras.callbacks import  ModelCheckpoint,EarlyStopping,ReduceLROnPlateau,Callback
from keras.optimizers import Adam,SGD
from keras import backend as K
import tensorflow as tf



def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true) # 将 y_true 拉伸为一维.
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)
    # return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed

# def bce_logdice_loss(y_true, y_pred):
#     return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_coef_loss(y_true, y_pred))
    # + focal_loss(gamma=2., alpha=.25)(y_true, y_pred)

from keras.losses import binary_crossentropy
import keras.backend as K
import tensorflow as tf

epsilon = 1e-5
smooth = 1

def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def confusion(y_true, y_pred):
    smooth=1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    prec = (tp + smooth)/(tp+fp+smooth)
    recall = (tp+smooth)/(tp+fn+smooth)
    return prec, recall

def tp(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    tp = (K.sum(y_pos * y_pred_pos) + smooth)/ (K.sum(y_pos) + smooth)
    return tp

def tn(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth )
    return tn

def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)


def Active_Contour_Loss(y_true, y_pred):

	#y_pred = K.cast(y_pred, dtype = 'float64')

	"""
	lenth term
	"""

	x = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:] # horizontal and vertical directions
	y = y_pred[:,:,:,1:] - y_pred[:,:,:,:-1]

	delta_x = x[:,:,1:,:-2]**2
	delta_y = y[:,:,:-2,1:]**2
	delta_u = K.abs(delta_x + delta_y)

	epsilon = 0.00000001 # where is a parameter to avoid square root is zero in practice.
	w = 1
	lenth = w * K.sum(K.sqrt(delta_u + epsilon)) # equ.(11) in the paper

	"""
	region term
	"""

	C_1 = np.ones((256, 256))
	C_2 = np.zeros((256, 256))

	region_in = K.abs(K.sum( y_pred[:,0,:,:] * ((y_true[:,0,:,:] - C_1)**2) ) ) # equ.(12) in the paper
	region_out = K.abs(K.sum( (1-y_pred[:,0,:,:]) * ((y_true[:,0,:,:] - C_2)**2) )) # equ.(12) in the paper

	lambdaP = 1 # lambda parameter could be various.

	loss =  lenth + lambdaP * (region_in + region_out)

	return loss

ce_w = 0.5
ce_d_w = 0.5
e = K.epsilon()
smooth = 1
'''
ce_w values smaller than 0.5 penalize false positives more while values larger than 0.5 penalize false negatives more
ce_d_w is level of contribution of the cross-entropy loss in the total loss.
'''

def Combo_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    d = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    y_pred_f = K.clip(y_pred_f, e, 1.0 - e)
    out = - (ce_w * y_true_f * K.log(y_pred_f)) + ((1 - ce_w) * (1.0 - y_true_f) * K.log(1.0 - y_pred_f))
    weighted_ce = K.mean(out, axis=-1)
    combo = (ce_d_w * weighted_ce) - ((1 - ce_d_w) * (1 - d))
    return combo


def find_latest_checkpoint(checkpoints_path, fail_safe=True):

    def get_epoch_number_from_path(path):
        return path.replace(checkpoints_path, "").strip(".")

    # Get all matching files
    all_checkpoint_files = glob.glob(checkpoints_path + ".*")
    # Filter out entries where the epoc_number part is pure number
    all_checkpoint_files = list(filter(lambda f: get_epoch_number_from_path(f).isdigit(), all_checkpoint_files))
    if not len(all_checkpoint_files):
        # The glob list is empty, don't have a checkpoints_path
        if not fail_safe:
            raise ValueError("Checkpoint path {0} invalid".format(checkpoints_path))
        else:
            return None

    # Find the checkpoint file with the maximum epoch
    latest_epoch_checkpoint = max(all_checkpoint_files, key=lambda f: int(get_epoch_number_from_path(f)))
    return latest_epoch_checkpoint


def train(model,
          train_images,
          train_annotations,
          input_height=None,
          input_width=None,
          n_classes=None,
          verify_dataset=True,
          checkpoints_path=None,
          epochs=5,
          batch_size=2,
          validate=False,
          val_images=None,
          val_annotations=None,
          val_batch_size=2,
          auto_resume_checkpoint=False,
          load_weights=None,
          steps_per_epoch=512,
          optimizer_name='adadelta'
          ):

    from .models.all_models import model_from_name
    # check if user gives model name instead of the model object
    if isinstance(model, six.string_types):
        # create the model from the name
        assert (n_classes is not None), "Please provide the n_classes"
        if (input_height is not None) and (input_width is not None):
            model = model_from_name[model](
                n_classes, input_height=input_height, input_width=input_width)
        else:
            model = model_from_name[model](n_classes)

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    if validate:
        assert val_images is not None
        assert val_annotations is not None

    from .loss import ce_jaccard_loss
    from .metrics import iou
    Metrics = ['accuracy', iou]
    if optimizer_name is not None:
        model.compile(metrics=[dsc],optimizer=Adam(lr=1e-3),loss=bce_dice_loss)  #[focal_loss(alpha=.25, gamma=2)]   改四处，上2下2
        # loss='categorical_crossentropy',                             #bce_logdice_loss
                    # metrics=['accuracy']，

    # import pdb; pdb.set_trace()
    model.summary()
    import time
    time.sleep(5)
    if checkpoints_path is not None:
        with open(checkpoints_path+"_config.json", "w") as f:
            json.dump({
                "model_class": model.model_name,
                "n_classes": n_classes,
                "input_height": input_height,
                "input_width": input_width,
                "output_height": output_height,
                "output_width": output_width
            }, f)

    if load_weights is not None and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    if auto_resume_checkpoint and (checkpoints_path is not None):
        latest_checkpoint = find_latest_checkpoint(checkpoints_path)
        if latest_checkpoint is not None:
            print("Loading the weights from latest checkpoint ",
                  latest_checkpoint)
            model.load_weights(latest_checkpoint)

    if verify_dataset:
        print("Verifying training dataset")
        verified = verify_segmentation_dataset(train_images, train_annotations, n_classes)
        assert verified
        if validate:
            print("Verifying validation dataset")
            verified = verify_segmentation_dataset(val_images, val_annotations, n_classes)
            assert verified
    print('Load train data')
    train_gen = image_segmentation_generator(
        train_images, train_annotations,  batch_size,  n_classes,
        input_height, input_width, output_height, output_width,do_augment=False)

    if validate:
        print('Load validate data')
        val_gen = image_segmentation_generator(
            val_images, val_annotations,  val_batch_size,
            n_classes, input_height, input_width, output_height, output_width,do_augment=False)

    # if not validate:
    #     for ep in range(epochs):
    #         print("Starting Epoch ", ep)
    #         model.fit_generator(train_gen, steps_per_epoch, epochs=1)
    #         if checkpoints_path is not None:
    #             model.save_weights(checkpoints_path + "." + str(ep))
    #             print("saved ", checkpoints_path + ".model." + str(ep))
    #         print("Finished Epoch", ep)
    # else:

    # for ep in range(epochs):
    #     print("Starting Epoch ", ep)
    #     model.fit_generator(train_gen, steps_per_epoch,
    #                         validation_data=val_gen,
    #                         validation_steps=200,  epochs=1)
    #     if checkpoints_path is not None:
    #         model.save_weights(checkpoints_path + "." + str(ep))
    #         print("saved ", checkpoints_path + ".model." + str(ep))
    #     print("Finished Epoch", ep)
    #
    tr_steps = len(os.listdir(train_images))*2
    val_steps = len(os.listdir(val_images))*2
    print(model.model_name)
    weights_checkpoints = ModelCheckpoint(checkpoints_path + "." + "{epoch:01d}",
                                          period=1, save_weights_only=True)
                                         # 需要提前建好文件夹
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, mode='auto', factor=0.1)  # 当评价指标不在提升时，减少学习率
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    model.fit_generator(train_gen, tr_steps,
                        validation_data=val_gen,
                        validation_steps=val_steps,  epochs=epochs,
                        callbacks=[weights_checkpoints, early_stop,reduce_lr])
    # if checkpoints_path is not None:
    #     model.save_weights(checkpoints_path + "." + str(ep))
    # print("saved ", checkpoints_path + ".model." + str(ep))
