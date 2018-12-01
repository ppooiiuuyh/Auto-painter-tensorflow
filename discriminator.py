import tensorflow as tf
from ops import *
from utils import *


class Discriminator(object):
# ==========================================================
# class initializer
# ==========================================================
    def __init__(self, sess, args, reuse=False, inputsA = None, inputsB= None):
        self.sess = sess
        self.args = args
        self.batch_size = args.batch_size
        self.img_size = args.img_size
        self.reuse = reuse
        self.model_name = args.model_name_dis
        self.inputsA = inputsA
        self.inputsB = inputsB
        self.model()
        self.init_model()
    
# ==========================================================
# preprocessing
# ==========================================================
    def preprocess(self):
        pass
    
# ==========================================================
# build model
# ==========================================================
    def model(self):
        with tf.variable_scope(self.model_name, reuse=self.reuse):
            """ Input layer"""
            if self.inputsA is None :
                self.inputsA = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, 3], name='inputsA')
            if self.inputsB is None :
                self.inputsA = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, 3], name='inputsB')
            self.inputs = tf.concat([self.inputsA, self.inputsB],axis=-1)

            self.is_training = tf.placeholder(tf.bool, name='is_training')
            self.preds = self.inner_model()
    
    
    def inner_model(self):
        channel = 64
        """ hidden layers """
        x = conv(self.inputs, channel, kernel=4, stride=2, pad=1, scope='first_conv')  # NO BATCH NORM
        x = lrelu(x, 0.2)
    
        for i in range(2):
            x = conv(x, channel * 2, kernel=4, stride=2, pad=1, scope='conv_' + str(i))
            x = batch_norm(x, self.is_training, scope='batch_' + str(i))
            x = lrelu(x, 0.2)
            channel = channel * 2
        
        x = conv(x, channel, kernel=3, stride=1, pad=1, scope='conv_2')
        x = batch_norm(x, self.is_training, scope='batch_2')
        x = lrelu(x, 0.2)
   
        """ output layers """
        x = conv(x, channels=1, kernel=3, stride=1, pad=1, scope='last_conv')

        return x
    
# ============================================================
# other tensors related with training
# ============================================================
    def init_model(self):
        if self.reuse == True:
            print("reused")
        else:
            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_name)
            self.sess.run(tf.variables_initializer(var_list=vars))
            self.saver = tf.train.Saver(var_list=vars, max_to_keep=0)
            if self.ckpt_load( getattr(self.args, "checkpoint_itr_" + self.model_name)):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        show_variables(scope=self.model_name, print_infor=False)


    def ckpt_save(self, step):
        model_name = "checks.model"
        checkpoint_dir = getattr(self.args, "checkpoint_dir_" + self.model_name)
        check_folder(checkpoint_dir)
        
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
    
    
    def ckpt_load(self, checkpoint_itr):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = getattr(self.args, "checkpoint_dir_" + self.model_name)
        model_name = "checks.model"
        if checkpoint_itr == 0:
            print("train from scratch")
            return True
        
        elif checkpoint_itr == -1:
            ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        
        else:
            ckpt = os.path.join(checkpoint_dir, "checks.model-" + str(checkpoint_itr))
        
        print(ckpt)
        if ckpt:
            self.saver.restore(self.sess, ckpt)
            return True
        else:
            return False

