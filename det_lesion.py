"""
Original code from OSVOS (https://github.com/scaelles/OSVOS-TensorFlow)
Sergi Caelles (scaelles@vision.ee.ethz.ch)

Modified code for liver and lesion segmentation:
Miriam Bellver (miriam.bellver@bsc.es)
"""

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.layers.python.layers import initializers
import sys
from datetime import datetime
import os
import scipy.misc
from PIL import Image
slim = tf.contrib.slim
from tensorflow.contrib.slim.nets import resnet_v1
import scipy.io
import scipy.misc

DTYPE = tf.float32


# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(variables):
    interp_tensors = []
    for v in variables:
        if '-up' in v.name:
            h, w, k, m = v.get_shape()
            tmp = np.zeros((m, k, h, w))
            if m != k:
                print 'input + output channels need to be the same'
                raise
            if h != w:
                print 'filters need to be square'
                raise
            up_filter = upsample_filt(int(h))
            tmp[range(m), range(k), :, :] = up_filter
            interp_tensors.append(tf.assign(v, tmp.transpose((2, 3, 1, 0)), validate_shape=True, use_locking=True))
    return interp_tensors


def det_lesion_arg_scope(weight_decay=0.0002):
    """Defines the arg scope.
    Args:
    weight_decay: The l2 regularization coefficient.
    Returns:
    An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.convolution2d_transpose],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.random_normal_initializer(stddev=0.001),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer,
                        biases_regularizer=None,
                        padding='SAME') as arg_sc:
        return arg_sc
        
        
def binary_cross_entropy(output, target, epsilon=1e-8, name='bce_loss'):
    """Defines the binary cross entropy loss
    Args:
    output: the output of the network
    target: the ground truth
    Returns:
    A scalar with the loss, the output and the target
    """
    target = tf.cast(target, tf.float32)
    output = tf.cast(tf.squeeze(output), tf.float32)
    
    with tf.name_scope(name):
        return tf.reduce_mean(-(target * tf.log(output + epsilon) +
                              (1. - target) * tf.log(1. - output + epsilon))), output, target
                              

def preprocess_img(image, x_bb, y_bb, ids=None):
    """Preprocess the image to adapt it to network requirements
    Args:
    Image we want to input the network (W,H,3) numpy array
    Returns:
    Image ready to input the network (1,W,H,3)
    """
    if ids == None:

        ids = np.ones(np.array(image).shape[0])

    images = [[] for i in range(np.array(image).shape[0])]
    
    for j in range(np.array(image).shape[0]):
        for i in range(3):
            aux = np.array(scipy.io.loadmat(image[j])['section'], dtype=np.float32)
            crop = aux[int(float(x_bb[j])):int((float(x_bb[j])+80)), int(float(y_bb[j])): int((float(y_bb[j])+80))]
            """Different data augmentation options
                """
            if id == '2':
                crop = np.fliplr(crop)
            elif id == '3':
                crop = np.fliphr(crop)
            elif id == '4':
                crop = np.fliphr(crop)
                crop = np.fliplr(crop)
            elif id == '5':
                crop = np.rot90(crop)
            elif id == '6':
                crop = np.rot90(crop, 2)
            elif id == '7':
                crop = np.fliplr(crop)
                crop = np.rot90(crop)
            elif id == '8':
                crop = np.fliplr(crop)
                crop = np.rot90(crop, 2)

            images[j].append(crop)
    in_ = np.array(images)
    in_ = in_.transpose((0,2,3,1))
    in_ = np.subtract(in_, np.array((104.00699, 116.66877, 122.67892), dtype=np.float32))

    return in_
    

def preprocess_labels(label):
    """Preprocess the labels to adapt them to the loss computation requirements
    Args:
    Label corresponding to the input image (W,H) numpy array
    Returns:
    Label ready to compute the loss (1,W,H,1)
    """
    labels = [[] for i in range(np.array(label).shape[0])]  
    
    for j in range(np.array(label).shape[0]):
        if type(label) is not np.ndarray:
            for i in range(3):
                aux = np.array(Image.open(label[j][i]), dtype=np.uint8)
                crop = aux[int(float(x_bb[j])):int((float(x_bb[j])+80)), int(float(y_bb[j])): int((float(y_bb[j])+80))]
                labels[j].append(crop)
            
    label = np.array(labels[0])
    label = label.transpose((1,2,0))
    max_mask = np.max(label) * 0.5
    label = np.greater(label, max_mask)
    label = np.expand_dims(label, axis=0)

    return label
        
        
def det_lesion_resnet(inputs, is_training_option=False, scope='det_lesion'):
    """Defines the network
    Args:
    inputs: Tensorflow placeholder that contains the input image
    scope: Scope name for the network
    Returns:
    net: Output Tensor of the network
    end_points: Dictionary with all Tensors of the network
    """

    with tf.variable_scope(scope, 'det_lesion', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):

            net, end_points = resnet_v1.resnet_v1_50(inputs, is_training=is_training_option)
            net = slim.flatten(net, scope='flatten5')
            net = slim.fully_connected(net, 1, activation_fn=tf.nn.sigmoid,
                                       weights_initializer=initializers.xavier_initializer(), scope='output')
            utils.collect_named_outputs(end_points_collection, 'det_lesion/output', net)

    end_points = slim.utils.convert_collection_to_dict(end_points_collection)
    return net, end_points


def load_resnet_imagenet(ckpt_path):
    """Initialize the network parameters from the Resnet-50 pre-trained model provided by TF-SLIM
    Args:
    Path to the checkpoint
    Returns:
    Function that takes a session and initializes the network
    """
    reader = tf.train.NewCheckpointReader(ckpt_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    vars_corresp = dict()
    
    for v in var_to_shape_map:
        if "bottleneck_v1" in v or "conv1" in v:
            vars_corresp[v] = slim.get_model_variables(v.replace("resnet_v1_50", "det_lesion/resnet_v1_50"))[0]
    init_fn = slim.assign_from_checkpoint_fn(ckpt_path, vars_corresp)
    return init_fn


def my_accuracy(output, target, name='accuracy'):
    """Accuracy for detection
    Args:
    The output and the target
    Returns:
    The accuracy based on the binary cross entropy
    """


    target = tf.cast(target, tf.float32)
    output = tf.squeeze(output)
    with tf.name_scope(name):
        return tf.reduce_mean((target * output) + (1. - target) * (1. - output))


def train(dataset, initial_ckpt, learning_rate, logs_path, max_training_iters, save_step, display_step,
           global_step, iter_mean_grad=1, batch_size=1, momentum=0.9, resume_training=False, config=None, finetune=1):

    """Train network
    Args:
    dataset: Reference to a Dataset object instance
    initial_ckpt: Path to the checkpoint to initialize the network (May be parent network or pre-trained Imagenet)
    supervison: Level of the side outputs supervision: 1-Strong 2-Weak 3-No supervision
    learning_rate: Value for the learning rate. It can be number or an instance to a learning rate object.
    logs_path: Path to store the checkpoints
    max_training_iters: Number of training iterations
    save_step: A checkpoint will be created every save_steps
    display_step: Information of the training will be displayed every display_steps
    global_step: Reference to a Variable that keeps track of the training steps
    iter_mean_grad: Number of gradient computations that are average before updating the weights
    batch_size:
    momentum: Value of the momentum parameter for the Momentum optimizer
    resume_training: Boolean to try to restore from a previous checkpoint (True) or not (False)
    config: Reference to a Configuration object used in the creation of a Session
    finetune: Use to select to select type of training, 0 for the parent network and 1 for finetunning
    Returns:
    """
    model_name = os.path.join(logs_path, "det_lesion.ckpt")
    if config is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

    tf.logging.set_verbosity(tf.logging.INFO)

    # Prepare the input data
    input_image = tf.placeholder(tf.float32, [batch_size, 80, 80, 3])
    input_label = tf.placeholder(tf.float32, [batch_size])
    is_training = tf.placeholder(tf.bool, shape=())
    
    tf.summary.histogram('input_label', input_label)

    # Create the network
    with slim.arg_scope(det_lesion_arg_scope()):
        net, end_points = det_lesion_resnet(input_image, is_training_option=is_training)

    # Initialize weights from pre-trained model
    if finetune == 0:
        init_weights = load_resnet_imagenet(initial_ckpt)

    # Define loss
    with tf.name_scope('losses'):
        loss, output, target = binary_cross_entropy(net, input_label)
        total_loss = loss + tf.add_n(tf.losses.get_regularization_losses())
        tf.summary.scalar('losses/total_loss', total_loss)
        tf.summary.histogram('losses/output', output)
        tf.summary.histogram('losses/target', target)

    # Define optimization method
    with tf.name_scope('optimization'):
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
        grads_and_vars = optimizer.compute_gradients(total_loss)
        with tf.name_scope('grad_accumulator'):
            grad_accumulator = []
            for ind in range(0, len(grads_and_vars)):
                if grads_and_vars[ind][0] is not None:
                    grad_accumulator.append(tf.ConditionalAccumulator(grads_and_vars[0][0].dtype))
        with tf.name_scope('apply_gradient'):
            grad_accumulator_ops = []
            for ind in range(0, len(grad_accumulator)):
                if grads_and_vars[ind][0] is not None:
                    var_name = str(grads_and_vars[ind][1].name).split(':')[0]
                    var_grad = grads_and_vars[ind][0]

                    if "weights" in var_name:
                        aux_layer_lr = 1.0
                    elif "biases" in var_name:
                        aux_layer_lr = 2.0
                    
                    grad_accumulator_ops.append(grad_accumulator[ind].apply_grad(var_grad*aux_layer_lr,
                                                                                 local_step=global_step))
        with tf.name_scope('take_gradients'):
            mean_grads_and_vars = []
            for ind in range(0, len(grad_accumulator)):
                if grads_and_vars[ind][0] is not None:
                    mean_grads_and_vars.append((grad_accumulator[ind].take_grad(iter_mean_grad), grads_and_vars[ind][1]))
            apply_gradient_op = optimizer.apply_gradients(mean_grads_and_vars, global_step=global_step)

    with tf.name_scope('metrics'):
        acc_op = my_accuracy(net, input_label)
        tf.summary.scalar('metrics/accuracy', acc_op)
        
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        tf.logging.info('Gathering update_ops')
        with tf.control_dependencies(tf.tuple(update_ops)):
            total_loss = tf.identity(total_loss)
       
    merged_summary_op = tf.summary.merge_all()

    # Initialize variables
    init = tf.global_variables_initializer()

    with tf.Session(config=config) as sess:
        print 'Init variable'
        sess.run(init)

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logs_path + '/train', graph=tf.get_default_graph())
        test_writer = tf.summary.FileWriter(logs_path + '/test')

        # Create saver to manage checkpoints
        saver = tf.train.Saver(max_to_keep=None)

        last_ckpt_path = tf.train.latest_checkpoint(logs_path)
        if last_ckpt_path is not None and resume_training:
            # Load last checkpoint
            print('Initializing from previous checkpoint...')
            saver.restore(sess, last_ckpt_path)
            step = global_step.eval() + 1
        else:
            # Load pre-trained model
            if finetune == 0:
                print('Initializing from pre-trained imagenet model...')
                init_weights(sess)
            else:
                print('Initializing from pre-trained model...')
                # init_weights(sess)
                var_list = []
                for var in tf.global_variables():
                    var_type = var.name.split('/')[-1]
                    if 'weights' in var_type or 'bias' in var_type:
                        var_list.append(var)
                saver_res = tf.train.Saver(var_list=var_list)
                saver_res.restore(sess, initial_ckpt)
            step = 1
        sess.run(interp_surgery(tf.global_variables()))
        print('Weights initialized')

        print 'Start training'
        while step < max_training_iters + 1:
            # Average the gradient
            for iter_steps in range(0, iter_mean_grad):
                batch_image, batch_label, x_bb_train, y_bb_train, ids_train = dataset.next_batch(batch_size, 'train', 0.5)
                batch_image_val, batch_label_val, x_bb_val, y_bb_val, ids_val = dataset.next_batch(batch_size, 'val', 0.5)
                image = preprocess_img(batch_image, x_bb_train, y_bb_train, ids_train)
                label = batch_label
                val_image = preprocess_img(batch_image_val, x_bb_val, y_bb_val)
                label_val = batch_label_val
                run_res = sess.run([total_loss, merged_summary_op, acc_op] + grad_accumulator_ops,
                                   feed_dict={input_image: image, input_label: label, is_training: True})
                batch_loss = run_res[0]
                summary = run_res[1]
                acc = run_res[2]
                if step % display_step == 0:
                    val_run_res = sess.run([total_loss, merged_summary_op, acc_op],
                                           feed_dict={input_image: val_image, input_label: label_val, is_training: False})
                    val_batch_loss = val_run_res[0]
                    val_summary = val_run_res[1]
                    val_acc = val_run_res[2]

            # Apply the gradients
            sess.run(apply_gradient_op)

            # Save summary reports
            summary_writer.add_summary(summary, step)
            if step % display_step == 0:
                test_writer.add_summary(val_summary, step)

            # Display training status
            if step % display_step == 0:
                print >> sys.stderr, "{} Iter {}: Training Loss = {:.4f}".format(datetime.now(), step, batch_loss)
                print >> sys.stderr, "{} Iter {}: Validation Loss = {:.4f}".format(datetime.now(), step, val_batch_loss)
                print >> sys.stderr, "{} Iter {}: Training Accuracy = {:.4f}".format(datetime.now(), step, acc)
                print >> sys.stderr, "{} Iter {}: Validation Accuracy = {:.4f}".format(datetime.now(), step, val_acc)

            # Save a checkpoint
            if step % save_step == 0:
                save_path = saver.save(sess, model_name, global_step=global_step)
                print "Model saved in file: %s" % save_path

            step += 1

        if (step-1) % save_step != 0:
            save_path = saver.save(sess, model_name, global_step=global_step)
            print "Model saved in file: %s" % save_path

        print('Finished training.')


def validate(dataset, checkpoint_path, result_path, number_slices=1, config=None):
    """Test one sequence
    Args:
    dataset: Reference to a Dataset object instance
    checkpoint_path: Path of the checkpoint to use for the evaluation
    result_path: Path to save the output images
    config: Reference to a Configuration object used in the creation of a Session
    Returns:
    net:
    """
    if config is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        config.allow_soft_placement = True
    tf.logging.set_verbosity(tf.logging.INFO)

    # Input data
    batch_size = 64
    number_of_slices = number_slices
    depth_input = number_of_slices
    if number_of_slices < 3:
        depth_input = 3

    pos_size = dataset.get_val_pos_size()
    neg_size = dataset.get_val_neg_size()
        
    input_image = tf.placeholder(tf.float32, [batch_size, None, None, depth_input])

    # Create the cnn
    with slim.arg_scope(det_lesion_arg_scope()):
        net, end_points = det_lesion_resnet(input_image, is_training_option=False)
    probabilities = end_points['det_lesion/output']
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create a saver to load the network
    saver = tf.train.Saver([v for v in tf.global_variables() if '-up' not in v.name and '-cr' not in v.name])

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(interp_surgery(tf.global_variables()))
        saver.restore(sess, checkpoint_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        
        results_file_soft = open(os.path.join(result_path, 'soft_results.txt'), 'w')
        results_file_hard = open(os.path.join(result_path, 'hard_results.txt'), 'w')
        
        # Test positive windows
        count_patches = 0
        for frame in range(0, pos_size/batch_size + (pos_size % batch_size > 0)):
            img, label, x_bb, y_bb = dataset.next_batch(batch_size, 'val', 1)
            curr_ct_scan = img[0]
            print 'Testing ' + curr_ct_scan
            image = preprocess_img(img, x_bb, y_bb)
            res = sess.run(probabilities, feed_dict={input_image: image})
            label = np.array(label).astype(np.float32).reshape(batch_size, 1)
           
            for i in range(0, batch_size):
                count_patches +=1
                img_part = img[i]
                res_part = res[i][0]
                label_part = label[i][0]
                if count_patches < (pos_size + 1):
                    results_file_soft.write(img_part.split('images_volumes/')[-1] + ' ' + str(x_bb[i]) + ' ' +
                                            str(y_bb[i]) + ' ' + str(res_part) + ' ' + str(label_part) + '\n')
                    if res_part > 0.5:
                        results_file_hard.write(img_part.split('images_volumes/')[-1] + ' ' +
                                                str(x_bb[i]) + ' ' + str(y_bb[i]) + '\n')

        # Test negative windows
        count_patches = 0
        for frame in range(0, neg_size/batch_size + (neg_size % batch_size > 0)):
            img, label, x_bb, y_bb = dataset.next_batch(batch_size, 'val', 0)
            curr_ct_scan = img[0]
            print 'Testing ' + curr_ct_scan
            image = preprocess_img(img, x_bb, y_bb)
            res = sess.run(probabilities, feed_dict={input_image: image})
            label = np.array(label).astype(np.float32).reshape(batch_size, 1)
           
            for i in range(0, batch_size):
                count_patches += 1
                img_part = img[i]
                res_part = res[i][0]
                label_part = label[i][0]
                if count_patches < (neg_size + 1):
                    results_file_soft.write(img_part.split('images_volumes/')[-1] + ' ' +
                                            str(x_bb[i]) + ' ' + str(y_bb[i]) + ' ' + str(res_part) + ' ' +
                                            str(label_part) + '\n')
                    if res_part > 0.5:
                        results_file_hard.write(img_part.split('images_volumes/')[-1] + ' ' +
                                                str(x_bb[i]) + ' ' + str(y_bb[i]) + '\n')
        
        results_file_soft.close()
        results_file_hard.close()


def test(dataset, checkpoint_path, result_path, number_slices=1, volume=False, config=None):
    """Test one sequence
    Args:
    dataset: Reference to a Dataset object instance
    checkpoint_path: Path of the checkpoint to use for the evaluation
    result_path: Path to save the output images
    config: Reference to a Configuration object used in the creation of a Session
    Returns:
    net:
    """
    if config is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        config.allow_soft_placement = True
    tf.logging.set_verbosity(tf.logging.INFO)

    # Input data
    batch_size = 64
    number_of_slices = number_slices
    depth_input = number_of_slices
    if number_of_slices < 3:
        depth_input = 3

    total_size = dataset.get_val_pos_size()
        
    input_image = tf.placeholder(tf.float32, [batch_size, None, None, depth_input])

    # Create the cnn
    with slim.arg_scope(det_lesion_arg_scope()):
        net, end_points = det_lesion_resnet(input_image, is_training_option=False)
    probabilities = end_points['det_lesion/output']
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create a saver to load the network
    saver = tf.train.Saver([v for v in tf.global_variables() if '-up' not in v.name and '-cr' not in v.name])

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(interp_surgery(tf.global_variables()))
        saver.restore(sess, checkpoint_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        results_file_soft = open(os.path.join(result_path, 'soft_results.txt'), 'w')
        results_file_hard = open(os.path.join(result_path, 'hard_results.txt'), 'w')
        
        # Test all windows
        count_patches = 0
        for frame in range(0, total_size/batch_size + (total_size % batch_size > 0)):
            img, x_bb, y_bb = dataset.next_batch(batch_size, 'test', 1)
            curr_ct_scan = img[0]
            print 'Testing ' + curr_ct_scan
            image = preprocess_img(img, x_bb, y_bb)
            res = sess.run(probabilities, feed_dict={input_image: image})

            for i in range(0, batch_size):
                count_patches += 1
                img_part = img[i]
                res_part = res[i][0]
                if count_patches < (total_size + 1):
                    results_file_soft.write(img_part.split('images_volumes/')[-1] + ' ' + str(x_bb[i]) + ' ' +
                                            str(y_bb[i]) + ' ' + str(res_part) + '\n')
                    if res_part > 0.5:
                        results_file_hard.write(img_part.split('images_volumes/')[-1] + ' ' + str(x_bb[i]) + ' ' +
                                                str(y_bb[i]) + '\n')
        
        results_file_soft.close()
        results_file_hard.close()
