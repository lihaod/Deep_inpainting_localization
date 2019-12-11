import os
import sys
import time
import warnings
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
# from tensorflow.contrib.slim.nets import resnet_v2 # pylint: disable=E0611
from tensorflow.contrib.layers.python.layers import layers # pylint: disable=E0611
from shutil import rmtree
from operator import itemgetter
from skimage import io
import utils
vgg_mfcn = utils.vgg_mfcn 

FLAGS = tf.flags.FLAGS
# dataset
tf.flags.DEFINE_string('data_dir', './data/full/?/jpg75/TOG/', 'path to dataset')
tf.flags.DEFINE_integer('subset', None, 'Use a subset of the whole dataset')
tf.flags.DEFINE_string('img_size', None, 'size of input image')
tf.flags.DEFINE_bool('img_aug', False, 'apply image augmentation')
# running configuration
tf.flags.DEFINE_string('mode', 'train', 'Mode: train / test / visual')
tf.flags.DEFINE_integer('epoch', 10, 'No. of epoch to run')
tf.flags.DEFINE_float('train_ratio', 0.9, 'Trainning ratio')
tf.flags.DEFINE_string('restore', None, 'Explicitly restore checkpoint')
tf.flags.DEFINE_bool('reset_global_step', False, 'Reset global step')
# learning configuration
tf.flags.DEFINE_integer('batch_size', 1, 'batch size')
tf.flags.DEFINE_string('optimizer', 'Momentum', 'GradientDescent / Adadelta / Momentum / Adam / Ftrl / RMSProp')
tf.flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate for Optimizer')
tf.flags.DEFINE_float('lr_decay', 1.0, 'Decay of learning rate')
tf.flags.DEFINE_float('lr_decay_freq', 1.0, 'Epochs that the lr is reduced once')
tf.flags.DEFINE_string('loss', 'wxent', 'Loss function type')
tf.flags.DEFINE_float('weight_decay', 5e-4, 'Learning rate for Optimizer')
tf.flags.DEFINE_integer('shuffle_seed', None, 'Seed for shuffling images')
# logs
tf.flags.DEFINE_string('logdir', None, 'path to logs directory')
tf.flags.DEFINE_integer('verbose_time', 10, 'verbose times in each epoch')
tf.flags.DEFINE_integer('valid_time', 1, 'validation times in each epoch')
tf.flags.DEFINE_integer('keep_ckpt', 1, 'num of checkpoint files to keep')
# outputs
tf.flags.DEFINE_string('visout_dir', 'visual/', 'path to output directory')

OPTIMIZERS = {
    'GradientDescent': {'func': tf.train.GradientDescentOptimizer, 'args': {}},
    'Adadelta': {'func': tf.train.AdadeltaOptimizer, 'args': {}},
    'Momentum': {'func': tf.train.MomentumOptimizer, 'args': {'momentum': 0.9}},
    'Adam': {'func': tf.train.AdamOptimizer, 'args': {}},
    'Ftrl': {'func': tf.train.FtrlOptimizer, 'args': {}},
    'RMSProp': {'func': tf.train.RMSPropOptimizer, 'args': {}}
    }
LOSS = {
    'wxent': {'func': utils.losses.sparse_weighted_softmax_cross_entropy_with_logits, 'args': {}},
    'xent':  {'func': utils.losses.sparse_softmax_cross_entropy_with_logits, 'args': {}}
    }

def model(images, weight_decay, is_training, num_classes=2):
    with slim.arg_scope(vgg_mfcn.vgg_arg_scope(weight_decay=weight_decay)):
        return vgg_mfcn.vgg_mfcn(images, is_training)

def main(argv=None):
    if '?' in FLAGS.data_dir:
        if FLAGS.mode == 'train':
            FLAGS.data_dir = FLAGS.data_dir.replace('?','train')
        else:
            FLAGS.data_dir = FLAGS.data_dir.replace('?','test')
    
    if FLAGS.logdir is None:
        sys.stderr.write('Log dir not specified.\n')
        return None
    if FLAGS.mode == 'train':
        write_log_mode = 'w'
        if not os.path.isdir(FLAGS.logdir):
            os.makedirs(FLAGS.logdir)
        else:
            if os.listdir(FLAGS.logdir):
                sys.stderr.write('Log dir is not empty, continue? [yes(y)/remove(r)/no(n)]: ')
                chioce = input('')
                if (chioce == 'y' or chioce == 'Y'):
                    write_log_mode = 'a'
                elif (chioce == 'r' or chioce == 'R'):
                    rmtree(FLAGS.logdir)
                else:
                    sys.stderr.write('Abort.\n')
                    return None
        tee_print = utils.tee_print.TeePrint(filename=FLAGS.logdir+'.log', mode=write_log_mode)
        print_func = tee_print.write
    else:
        print_func = print
    
    print_func(sys.argv[0])
    print_func('--------------FLAGS--------------')
    for name, val in sorted(FLAGS.flag_values_dict().items(), key=itemgetter(0)):
        if not ('help' in name or name == 'h'):
            print_func('{}: {}'.format(name,val))
    print_func('---------------------------------')

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Setting up dataset
    shuffle_seed = FLAGS.shuffle_seed or int(time.time()*256)
    print_func('Seed={}'.format(shuffle_seed))
    if 'jpg' in FLAGS.data_dir:
        pattern = '*.jpg'
        msk_rep = [[['jpg','msk'],['.jpg','.png']],[['jpg','edg'],['.jpg','.png']]]
    else:
        pattern = '*.png'
        msk_rep = [['png','msk']]
    dataset, instance_num = utils.read_dataset.read_dataset_with2msk(FLAGS.data_dir, pattern=pattern, msk_replace=msk_rep, shuffle_seed=shuffle_seed,subset=FLAGS.subset)
    
    def map_func(*args):
        return utils.read_dataset.read_image_with2msk(*args, outputsize=[int(v) for v in reversed(FLAGS.img_size.split('x'))] if FLAGS.img_size else None, random_flip=FLAGS.img_aug)

    if FLAGS.mode == 'train':
        dataset_trn = dataset.take(int(np.ceil(instance_num*FLAGS.train_ratio))).shuffle(buffer_size=10000).map(map_func).batch(FLAGS.batch_size).repeat()
        dataset_vld = dataset.skip(int(np.ceil(instance_num*FLAGS.train_ratio))).map(map_func).batch(FLAGS.batch_size)

        iterator_trn = dataset_trn.make_one_shot_iterator()
        iterator_vld = dataset_vld.make_initializable_iterator()
    elif FLAGS.mode == 'test' or FLAGS.mode == 'visual':
        dataset_vld = dataset.map(map_func).batch(FLAGS.batch_size)
        iterator_vld = dataset_vld.make_initializable_iterator()

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, dataset_vld.output_types, dataset_vld.output_shapes)
    next_element = iterator.get_next()
    images = next_element[0]
    labels_msk = tf.squeeze(next_element[1],axis=3)
    labels_edg = tf.squeeze(next_element[2],axis=3)
    imgnames = next_element[3]
    
    is_training = tf.placeholder(tf.bool,[])
    logits_msk, logits_edg, preds_msk, preds_edg, preds_msk_map, preds_edg_map = model(images, FLAGS.weight_decay, is_training)
    
    loss = LOSS[FLAGS.loss]['func'](logits=logits_msk,labels=labels_msk,**LOSS[FLAGS.loss]['args']) \
        + LOSS[FLAGS.loss]['func'](logits=logits_edg,labels=labels_edg,**LOSS[FLAGS.loss]['args']) \
        + tf.add_n(tf.losses.get_regularization_losses())

    global_step = tf.Variable(0, trainable=False, name='global_step')
    itr_per_epoch = int(np.ceil(instance_num*FLAGS.train_ratio)/FLAGS.batch_size)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,global_step,decay_steps=int(itr_per_epoch*FLAGS.lr_decay_freq),decay_rate=FLAGS.lr_decay,staircase=True)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = OPTIMIZERS[FLAGS.optimizer]['func'](learning_rate,**OPTIMIZERS[FLAGS.optimizer]['args']).\
                    minimize(loss, global_step=global_step, var_list=tf.trainable_variables())
    
    with tf.name_scope('metrics'): 
        tp_count  = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(labels_msk,1),tf.equal(preds_msk,1))),name='true_positives')
        tn_count  = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(labels_msk,0),tf.equal(preds_msk,0))),name='true_negatives')
        fp_count  = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(labels_msk,0),tf.equal(preds_msk,1))),name='false_positives')
        fn_count  = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(labels_msk,1),tf.equal(preds_msk,0))),name='false_negatives')
        metrics_count = tf.Variable(0.0, name='metrics_count', trainable = False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
        recall_sum    = tf.Variable(0.0, name='recall_sum', trainable = False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
        precision_sum = tf.Variable(0.0, name='precision_sum', trainable = False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
        accuracy_sum  = tf.Variable(0.0, name='accuracy_sum', trainable = False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
        loss_sum      = tf.Variable(0.0, name='loss_sum', trainable = False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
        update_recall_sum = tf.assign_add(recall_sum, tp_count/(tp_count+fn_count))
        update_precision_sum = tf.assign_add(precision_sum, tf.cond(tf.equal(tp_count+fp_count,0), \
                                                                    lambda: 0.0, \
                                                                    lambda: tp_count/(tp_count+fp_count)))
        update_accuracy_sum = tf.assign_add(accuracy_sum, (tp_count+tn_count)/(tp_count+tn_count+fp_count+fn_count))
        update_loss_sum = tf.assign_add(loss_sum, loss)
        with tf.control_dependencies([update_recall_sum, update_precision_sum, update_accuracy_sum, update_loss_sum]):
            update_metrics_count = tf.assign_add(metrics_count, 1.0)
        mean_recall = recall_sum/metrics_count
        mean_precision = precision_sum/metrics_count
        mean_accuracy = accuracy_sum/metrics_count
        mean_loss = loss_sum/metrics_count
    
    config=tf.ConfigProto(log_device_placement=False)
    # config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    local_vars_metrics = [v for v in tf.local_variables() if 'metrics/' in v.name]

    saver = tf.train.Saver(max_to_keep= FLAGS.keep_ckpt+1 if FLAGS.keep_ckpt else 1000000)
    model_checkpoint_path = ''
    if FLAGS.restore and 'ckpt' in FLAGS.restore:
        model_checkpoint_path = FLAGS.restore
    else:
        ckpt = tf.train.get_checkpoint_state(FLAGS.restore or FLAGS.logdir)
        if ckpt and ckpt.model_checkpoint_path:
            model_checkpoint_path = ckpt.model_checkpoint_path

    if model_checkpoint_path:
        saver.restore(sess, model_checkpoint_path)
        print_func('Model restored from {}'.format(model_checkpoint_path))

    if FLAGS.mode == 'train':
        summary_op = tf.summary.merge([tf.summary.scalar('loss', mean_loss),
                                       tf.summary.scalar('lr', learning_rate)])
        summary_writer_trn = tf.summary.FileWriter(FLAGS.logdir + '/train', sess.graph)
        summary_writer_vld = tf.summary.FileWriter(FLAGS.logdir + '/validation')

        handle_trn = sess.run(iterator_trn.string_handle())
        handle_vld = sess.run(iterator_vld.string_handle())
        best_metric = 0.0
        if FLAGS.reset_global_step:
            sess.run(tf.variables_initializer([global_step]))
        for itr in range(itr_per_epoch*FLAGS.epoch):
            _, step, _, = sess.run([train_op, global_step, update_metrics_count], feed_dict={handle: handle_trn, is_training: True})
            if step % (itr_per_epoch/FLAGS.verbose_time) == 0:
                mean_loss_, mean_accuracy_, mean_recall_, mean_precision_, summary_str = sess.run([mean_loss, mean_accuracy, mean_recall, mean_precision, summary_op])
                print_func('epoch: {:d} step: {:d} loss: {:0.6f} ACC: {:0.6f} Recall: {:0.6f} Precision: {:0.6f}'.format(\
                            int(step/itr_per_epoch),step,mean_loss_,mean_accuracy_,mean_recall_,mean_precision_))
                summary_writer_trn.add_summary(summary_str, step)
                sess.run(tf.variables_initializer(local_vars_metrics))
            if step > 0 and step % (itr_per_epoch/FLAGS.valid_time) == 0:
                sess.run(iterator_vld.initializer)
                sess.run(tf.variables_initializer(local_vars_metrics))
                TNR, F1, MCC, IoU, Recall, Prec = [], [], [], [], [], []
                warnings.simplefilter('ignore',RuntimeWarning)
                while True:
                    try:
                        labels_, preds_, _ = sess.run([labels_msk, preds_msk, update_metrics_count], feed_dict={handle: handle_vld, is_training: False})
                        for i in range(labels_.shape[0]):
                            recall,tnr,prec,f1,mcc,iou  = utils.metrics.get_metrics(labels_[i],preds_[i])
                            TNR.append(tnr)
                            F1.append(f1)
                            MCC.append(mcc)
                            IoU.append(iou)
                            Recall.append(recall)
                            Prec.append(prec)
                    except tf.errors.OutOfRangeError:
                        break
                mean_loss_, mean_accuracy_, summary_str = sess.run([mean_loss, mean_accuracy, summary_op])
                if np.mean(F1) > best_metric:
                    best_metric = np.mean(F1)
                print_func('validation loss: {:0.6f} ACC: {:0.6f} Recall: {:0.6f} Prec: {:0.6f} TNR: {:0.6f} \033[1;31mF1: {:0.6f}\033[0m MCC: {:0.6f} IoU: {:0.6f} best_metric: {:0.6f}'.format( \
                            mean_loss_,mean_accuracy_,np.mean(Recall),np.mean(Prec),np.mean(TNR),np.mean(F1),np.mean(MCC),np.mean(IoU),best_metric))
                summary_writer_vld.add_summary(summary_str, step)
                sess.run(tf.variables_initializer(local_vars_metrics))

                saver.save(sess, '{}/model.ckpt-{:0.6f}'.format(FLAGS.logdir, np.mean(F1)), int(step/itr_per_epoch))
                saver._last_checkpoints = sorted(saver._last_checkpoints, key=lambda x: x[0].split('/')[-1].split('-')[1])
                if FLAGS.keep_ckpt and len(saver._last_checkpoints) > FLAGS.keep_ckpt:
                    saver._checkpoints_to_be_deleted.append(saver._last_checkpoints.pop(0))
                    saver._MaybeDeleteOldCheckpoints()
                tf.train.update_checkpoint_state(save_dir=FLAGS.logdir, \
                    model_checkpoint_path=saver.last_checkpoints[-1], \
                    all_model_checkpoint_paths=saver.last_checkpoints)

    elif FLAGS.mode == 'test':
        handle_vld = sess.run(iterator_vld.string_handle())
        sess.run(iterator_vld.initializer)
        TNR, F1, MCC, IoU, Recall, Prec = [], [], [], [], [], []
        warnings.simplefilter('ignore',(UserWarning, RuntimeWarning))
        while True:
            try:
                labels_, preds_, _ = sess.run([labels_msk, preds_msk, update_metrics_count], feed_dict={handle: handle_vld, is_training: False})
                for i in range(labels_.shape[0]):
                    recall,tnr,prec,f1,mcc,iou  = utils.metrics.get_metrics(labels_[i],preds_[i])
                    TNR.append(tnr)
                    F1.append(f1)
                    MCC.append(mcc)
                    IoU.append(iou)
                    Recall.append(recall)
                    Prec.append(prec)
            except tf.errors.OutOfRangeError:
                break
        mean_loss_, mean_accuracy_ = sess.run([mean_loss, mean_accuracy])
        print_func('testing loss: {:0.6f} ACC: {:0.6f} Recall: {:0.6f} Prec: {:0.6f} TNR: {:0.6f} \033[1;31mF1: {:0.6f}\033[0m MCC: {:0.6f} IoU: {:0.6f}'.format( \
                    mean_loss_,mean_accuracy_,np.mean(Recall),np.mean(Prec),np.mean(TNR),np.mean(F1),np.mean(MCC),np.mean(IoU)))

    elif FLAGS.mode == 'visual':
        handle_vld = sess.run(iterator_vld.string_handle())
        sess.run(iterator_vld.initializer)
        warnings.simplefilter('ignore',(UserWarning, RuntimeWarning))
        if not os.path.exists(FLAGS.visout_dir):
            os.makedirs(FLAGS.visout_dir)
        index = 0
        while True:
            try:
                labels_, preds_, preds_map_, preds_edg_map_, imgnames_, images_ = sess.run([labels_msk, preds_msk, preds_msk_map, preds_edg_map, imgnames, images], feed_dict={handle: handle_vld, is_training: False})
                for i in range(FLAGS.batch_size):
                    imgname = imgnames_[i].decode().split('/')[-1]
                    vis_out_msk = preds_map_[i]
                    vis_out_edg =  preds_edg_map_[i]
                    io.imsave(os.path.join(FLAGS.visout_dir,imgname.replace('.jpg','_msk_pred.png')), np.uint8(np.round(vis_out_msk*255.0)))
                    io.imsave(os.path.join(FLAGS.visout_dir,imgname.replace('.jpg','_edg_pred.png')), np.uint8(np.round(vis_out_edg*255.0)))
                    # vis_out = labels_[i]
                    # io.imsave(os.path.join(FLAGS.visout_dir,imgname.replace('.jpg','_gt.png')), np.uint8(np.round(vis_out*255.0)))
                    # vis_out = images_[i]
                    # io.imsave(os.path.join(FLAGS.visout_dir,imgname.replace('.jpg','_img.png')), np.uint8(np.round(vis_out*255.0)))
                    recall,tnr,prec,f1,mcc,iou  = utils.metrics.get_metrics(labels_[i],preds_[i])
                    print('{}: {} '.format(index,imgname),end='')
                    print('Recall: {:0.6f} Prec: {:0.6f} TNR: {:0.6f} \033[1;31mF1: {:0.6f}\033[0m MCC: {:0.6f} IoU: {:0.6f}'.format(recall,prec,tnr,f1,mcc,iou),end='')
                    index += 1
                    print('')
            except tf.errors.OutOfRangeError:
                break

    else:
        print_func('Mode not defined: '+FLAGS.mode)
        return None

if __name__ == '__main__':
    tf.app.run()
