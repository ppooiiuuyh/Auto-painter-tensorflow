import tensorflow as tf
from ops import *
from utils import *


class Generator(object):
# ==========================================================
# class initializer
# ==========================================================
    def __init__(self, sess, args, reuse=False, inputs = None):
        self.sess = sess
        self.args = args
        self.batch_size = args.batch_size
        self.img_size = args.img_size
        self.reuse = reuse
        self.inputs = inputs
        self.model_name = args.model_name_gen
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
            if self.inputs is None :
                self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, 3], name='inputs')
            self.is_training = tf.placeholder(tf.bool, name='is_training')
            self.preds = self.inner_model()
    
    def inner_model(self):
        encoderList = []
        input_ch = 64
        encoder_channelList = [128,256,512,512,512,512]
        decoder_channelList = [512, 512, 512, 256, 128, 64]
        output_ch = 3
        
        """ Input layer """
        x = conv(self.inputs, input_ch, kernel=4, stride=2, pad=1, scope='conv_input')
        x = batch_norm(x, self.is_training, scope='conv_input_batch')
        x = relu(x)
        encoderList.append(x)

        """ Encoder """
        for e,ch in enumerate(encoder_channelList):
            x = conv(x, ch, kernel=4, stride=2, pad=1, scope='conv_input_'+str(e))
            x = batch_norm(x, self.is_training, scope='conv_enc_batch_'+str(e))
            x = relu(x)
            encoderList.append(x)

        """ Bottleneck """
        x = conv(x, 512, kernel=4, stride=2, pad=1, scope='conv_bot')
        x = batch_norm(x, self.is_training, scope='conv_bot_batch')
        x = relu(x)
        x = deconv(x, 512, kernel=4, stride=2, scope='deconv_bot')
        x = batch_norm(x, self.is_training, scope='deconv_batch_bot')
        x = relu(x)

        """ Decoder """
        for e,ch in enumerate(decoder_channelList):
            x = tf.concat([x,encoderList.pop()],axis=-1)
            x = deconv(x, ch, kernel=4, stride=2, scope='deconv_dec_' + str(e))
            x = batch_norm(x, self.is_training, scope='deconv_batch_dec_' + str(e))
            x = relu(x)

        """ Output layer """
        #U-Net
        x = tf.concat([x, encoderList.pop()], axis=-1)
        #Fusion-Net
        #x + encoderList.pop() #fusionnet
        x = deconv(x, output_ch, kernel=4, stride=2, scope='deconv_out')
        x = batch_norm(x, self.is_training, scope='deconv_batch_out')
        x = tanh(x)

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
            if self.ckpt_load(getattr(self.args,"checkpoint_itr_" + self.model_name)):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        show_variables(scope=self.model_name, print_infor=False)
    
    def ckpt_save(self, step):
        model_name = "checks.model"
        checkpoint_dir = getattr(self.args,"checkpoint_dir_"+self.model_name)
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

