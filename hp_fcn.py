import os
import sys
import time
import warnings
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
from tensorflow.contrib.slim.nets import resnet_v2 # pylint: disable=E0611
from tensorflow.contrib.layers.python.layers import layers # pylint: disable=E0611
from shutil import rmtree
from operator import itemgetter
from skimage import io
import utils
from utils.bilinear_upsample_weights import bilinear_upsample_weights

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
tf.flags.DEFINE_string('optimizer', 'Adam', 'GradientDescent / Adadelta / Momentum / Adam / Ftrl / RMSProp')
tf.flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate for Optimizer')
tf.flags.DEFINE_float('lr_decay', 0.5, 'Decay of learning rate')
tf.flags.DEFINE_float('lr_decay_freq', 1.0, 'Epochs that the lr is reduced once')
tf.flags.DEFINE_string('filter_type', 'd1', 'Filter kernel type')
tf.flags.DEFINE_bool('filter_learnable', True, 'Learnable filter kernel')
tf.flags.DEFINE_string('loss', 'focal', 'Loss function type')
tf.flags.DEFINE_float('focal_gamma', 2.0, 'gamma of focal loss')
tf.flags.DEFINE_float('weight_decay', 1e-5, 'Learning rate for Optimizer')
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
    'focal':  {'func': utils.losses.focal_loss, 'args': {'gamma': FLAGS.focal_gamma}},
    'xent':  {'func': utils.losses.sparse_softmax_cross_entropy_with_logits, 'args': {}}
    }
FILTERS = {
    'd1': [
        np.array([[0., 0., 0.], [0., -1., 0.], [0., 1., 0.]]),
        np.array([[0., 0., 0.], [0., -1., 1.], [0., 0., 0.]]),
        np.array([[0., 0., 0.], [0., -1., 0.], [0., 0., 1.]])],
    'd2': [
        np.array([[0., 1., 0.], [0., -2., 0.], [0., 1., 0.]]),
        np.array([[0., 0., 0.], [1., -2., 1.], [0., 0., 0.]]),
        np.array([[1., 0., 0.], [0., -2., 0.], [0., 0., 1.]])],
    'd3': [
        np.array([[0., 0., 0., 0., 0.], [0., 0., -1., 0., 0.], [0., 0., 3., 0., 0.], [0., 0., -3., 0., 0.], [0., 0., 1., 0., 0.]]),
        np.array([[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., -1., 3., -3., 1.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]]),
        np.array([[0., 0., 0., 0., 0.], [0., -1., 0., 0., 0.], [0., 0., 3., 0., 0.], [0., 0., 0., -3., 0.], [0., 0., 0., 0., 1.]])],
    'd4': [
        np.array([[0., 0., 1., 0., 0.], [0., 0., -4., 0., 0.], [0., 0., 6., 0., 0.], [0., 0., -4., 0., 0.], [0., 0., 1., 0., 0.]]),
        np.array([[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [1., -4., 6., -4., 1.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]]),
        np.array([[1., 0., 0., 0., 0.], [0., -4., 0., 0., 0.], [0., 0., 6., 0., 0.], [0., 0., 0., -4., 0.], [0., 0., 0., 0., 1.]])],
    }

def get_residuals(image, filter_type='d1', filter_trainable=True, image_channel=3):

    if filter_type == 'none':
        return image - np.array([123.68, 116.78, 103.94])/255.0
    
    residuals = []
    
    if filter_type == 'random':
        for kernel_index in range(3):
            kernel_variable = tf.get_variable(name='root_filter{}'.format(kernel_index),shape=[3,3,image_channel,1], \
                                        initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            image_filtered = tf.nn.depthwise_conv2d_native(image, kernel_variable, strides=[1, 1, 1, 1], padding='SAME')
            residuals.append(image_filtered)
    else:
        kernel_index = 0
        for filter_kernel in FILTERS[filter_type]:
            kernel_variable = tf.Variable(np.repeat(filter_kernel[:,:,np.newaxis,np.newaxis],image_channel,axis=2), \
                                            trainable=filter_trainable, dtype='float', name='root_filter{}'.format(kernel_index))
            image_filtered = tf.nn.depthwise_conv2d_native(image, kernel_variable, strides=[1, 1, 1, 1], padding='SAME')
            residuals.append(image_filtered)
            kernel_index += 1

    return tf.concat(residuals, 3)


def resnet_small(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 include_root_block=True,
                 reuse=None,
                 scope='resnet_small'):
    blocks = [
        resnet_v2.resnet_v2_block('block1', base_depth=32, num_units=2, stride=2),
        resnet_v2.resnet_v2_block('block2', base_depth=64, num_units=2, stride=2),
        resnet_v2.resnet_v2_block('block3', base_depth=128, num_units=2, stride=2),
        resnet_v2.resnet_v2_block('block4', base_depth=256, num_units=2, stride=2),
    ]
    return resnet_v2.resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                             global_pool=global_pool, output_stride=output_stride,
                             include_root_block=include_root_block,
                             reuse=reuse, scope=scope)    

def model(images, filter_type, filter_trainable, weight_decay, batch_size, is_training, num_classes=2):
    with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
        inputs = get_residuals(images, filter_type, filter_trainable)
        _, end_points = resnet_small(inputs, 
                                    num_classes=None, 
                                    is_training=is_training, 
                                    global_pool=False, 
                                    output_stride=None,
                                    include_root_block=False)
        net = end_points['resnet_small/block4']
        net = tf.nn.conv2d_transpose(net, tf.Variable(bilinear_upsample_weights(4,64,1024),dtype=tf.float32,name='bilinear_kernel0'), \
                                     [batch_size, tf.shape(end_points['resnet_small/block2'])[1], tf.shape(end_points['resnet_small/block2'])[2], 64], strides=[1, 4, 4, 1], padding="SAME")
        end_points['upsample1'] = net
        net = tf.nn.conv2d_transpose(net, tf.Variable(bilinear_upsample_weights(4,4,64),dtype=tf.float32,name='bilinear_kernel1'), \
                                     [batch_size, tf.shape(inputs)[1], tf.shape(inputs)[2], 4], strides=[1, 4, 4, 1], padding="SAME")
        end_points['upsample2'] = net
        net = layers.batch_norm(net, activation_fn=tf.nn.relu, is_training=is_training, scope='post_norm')
        logits = slim.conv2d(net, num_classes, [5, 5], activation_fn=None, normalizer_fn=None, scope='logits')
        preds = tf.cast(tf.argmax(logits,3),tf.int32)
        preds_map = tf.nn.softmax(logits)[:,:,:,1]

        return logits, preds, preds_map, net, end_points, inputs

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
        msk_rep = [['jpg','msk'],['.jpg','.png']]
    else:
        pattern = '*.png'
        msk_rep = [['png','msk']]
    dataset, instance_num = utils.read_dataset.read_dataset_withmsk(FLAGS.data_dir, pattern=pattern, msk_replace=msk_rep, shuffle_seed=shuffle_seed,subset=FLAGS.subset)

    def map_func(x, y):
            return utils.read_dataset.read_image_withmsk(x, y, outputsize=[int(v) for v in reversed(FLAGS.img_size.split('x'))] if FLAGS.img_size else None, random_flip=FLAGS.img_aug)

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
    labels = next_element[1]
    labels = tf.squeeze(labels,axis=3)
    imgnames = next_element[2]
    
    is_training = tf.placeholder(tf.bool,[])
    logits, preds, preds_map, net, end_points, img_res = model(images, FLAGS.filter_type, FLAGS.filter_learnable, FLAGS.weight_decay, FLAGS.batch_size, is_training)
    
    loss = LOSS[FLAGS.loss]['func'](logits=logits,labels=labels,**LOSS[FLAGS.loss]['args']) + tf.add_n(tf.losses.get_regularization_losses())

    global_step = tf.Variable(0, trainable=False, name='global_step')
    itr_per_epoch = int(np.ceil(instance_num*FLAGS.train_ratio)/FLAGS.batch_size)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,global_step,decay_steps=int(itr_per_epoch*FLAGS.lr_decay_freq),decay_rate=FLAGS.lr_decay,staircase=True)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = OPTIMIZERS[FLAGS.optimizer]['func'](learning_rate,**OPTIMIZERS[FLAGS.optimizer]['args']).\
                    minimize(loss, global_step=global_step, var_list=tf.trainable_variables())
    
    with tf.name_scope('metrics'): 
        tp_count  = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(labels,1),tf.equal(preds,1))),name='true_positives')
        tn_count  = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(labels,0),tf.equal(preds,0))),name='true_negatives')
        fp_count  = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(labels,0),tf.equal(preds,1))),name='false_positives')
        fn_count  = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(labels,1),tf.equal(preds,0))),name='false_negatives')
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
                        labels_, preds_, _ = sess.run([labels, preds, update_metrics_count], feed_dict={handle: handle_vld, is_training: False})
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
                saver._last_checkpoints = sorted(saver._last_checkpoints, key=lambda x: x[0].split('-')[1])
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
                labels_, preds_, _ = sess.run([labels, preds, update_metrics_count], feed_dict={handle: handle_vld, is_training: False})
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
                labels_, preds_, preds_map_, imgnames_, images_ = sess.run([labels, preds, preds_map, imgnames, images], feed_dict={handle: handle_vld, is_training: False})
                for i in range(FLAGS.batch_size):
                    imgname = imgnames_[i].decode().split('/')[-1]
                    vis_out = preds_map_[i]
                    io.imsave(os.path.join(FLAGS.visout_dir,imgname.replace('.jpg','_pred.png')), np.uint8(np.round(vis_out*255.0)))
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
