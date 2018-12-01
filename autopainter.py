from ops import *
from utils import *
from glob import glob
import time
from discriminator import Discriminator
from generator import Generator
import matplotlib.pyplot as plt
from vgg16_pretrained import VGG16_Pretrained
from tqdm import tqdm


class AutoPainter(object):
# ======================================================================================================================
# Initializer
# ======================================================================================================================
    def __init__(self, sess, args):
        """ configuration """
        self.model_name = args.model_name
        self.sess = sess
        self.args = args
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.sample_dir = args.sample_dir
        self.dataset_name = args.dataset
        self.path_trainA = args.path_trainA
        self.path_trainB = args.path_trainB
        self.path_testA = args.path_testA

        self.epoch = args.epoch # 100000
        self.batch_size = args.batch_size
        self.print_freq = args.print_freq

        """ Weight """
        self.L1_weight = args.L1_weight
        self.FL_weight = args.FL_weight
        self.TV_weight = args.TV_weight
        self.lr = args.lr

        self.img_size = args.img_size

        self.gray_to_RGB = args.gray_to_RGB
        if self.gray_to_RGB :
            self.input_ch = 1
            self.output_ch = 3
        else :
            self.input_ch = 3
            self.output_ch = 3


        """ init model"""
        self.preprocess()
        self.build_model()
        self.other_tensors()
        self.init_model()


# ======================================================================================================================
# Preprocess (parse dataset)
# ======================================================================================================================
    def preprocess(self):
        """ load dat """
        print("prepare data")
        self.trainA = prepare_data(path=self.path_trainA, size=self.img_size, len=-1,color=True)
        self.trainB = prepare_data(path=self.path_trainB, size=self.img_size, len=-1,color=True)
        self.testA = prepare_data(path=self.path_testA, size=self.img_size, len=100,color=True)
        self.testA_touched = np.copy(self.testA)
        print(self.trainA.shape, self.trainB.shape, self.testA.shape)
        
        '''
        """ touch data """
        print("touch data")
        for e,(a,b,c) in tqdm(enumerate(zip(self.trainA,self.trainB,self.testA))):
            self.trainA[e] = random_touch(a,b,100,1)
            if(e<min(self.trainB.shape[0],self.testA.shape[0])):
                self.testA_touched[e] = random_touch(c,b,30,1)
        '''
        
        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        check_folder(self.sample_dir)
        
    

# ======================================================================================================================
# Build models
# ======================================================================================================================
    def build_model(self):
        """ place holer """
        self.real_A = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, self.input_ch], name='real_A')  # gray
        self.real_B = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, self.output_ch], name='real_B')  # rgb
    
        """ build generator """
        self.G_fake = Generator(sess = self.sess, args = self.args, inputs= self.real_A)
        self.fake_B = self.G_fake.preds
        
        """ build discriminator """
        self.D_real = Discriminator(sess = self.sess, args = self.args, inputsA = self.real_A, inputsB=self.real_B)
        self.D_fake = Discriminator(sess = self.sess, args = self.args, reuse=True, inputsA = self.real_A, inputsB = self.G_fake.preds)
        
        """ build VGG16 for perceptual loss (feature loss) """
        self.vgg16_4_real = VGG16_Pretrained(sess = self.sess, inputs = self.real_B)
        self.vgg16_4_fake = VGG16_Pretrained(sess = self.sess, inputs = self.fake_B)
        
        '''
        for n in tf.get_default_graph().as_graph_def().node:
            print(n.name)
        '''
        
        self.model_list = [self.D_real, self.D_fake, self.G_fake]


# ======================================================================================================================
# Other tensors
# ======================================================================================================================
    def other_tensors(self):
        with tf.variable_scope("trainer"):
            """ Loss Function """
            self.d_loss = discriminator_loss(real=self.D_real.preds, fake=self.D_fake.preds)
            # G_loss, pixel_loss, feature_loss, total_variation_loss
            self.g_loss = generator_loss(fake=self.D_fake.preds) \
                          + self.L1_weight * L1_loss(self.real_B, self.G_fake.preds) \
                          + self.FL_weight * L2_loss(self.vgg16_4_real.preds, self.vgg16_4_fake.preds) \
                          + self.TV_weight * TV_loss(self.G_fake.preds)
    
            """ Training """
            t_vars = tf.trainable_variables()
            #self.real_B = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.output_ch], name='real_B') # rgb
            G_vars = [var for var in t_vars if self.G_fake.model_name in var.name]
            D_vars = [var for var in t_vars if self.D_fake.model_name in var.name]
            print(len(G_vars))
            print(len(D_vars))
    
            self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.g_loss, var_list=G_vars)
            self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.d_loss, var_list=D_vars)
    
    
            """" Summary """
            self.G_loss_summary = tf.summary.scalar("Generator_loss", self.g_loss)
            self.D_loss_summary = tf.summary.scalar("Discriminator_loss", self.d_loss)
    
    
            """ Test """
            self.test_real_A = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, self.input_ch], name='test_real_A')

            # summary writer
            self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)



    def init_model(self):
        vars_trainer = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="trainer")
        self.sess.run(tf.variables_initializer(var_list=vars_trainer))

        



    def train(self):
        # loop for epoch
        counter=0
        start_time = time.time()
        startepoch = 0
        for epoch in range(startepoch,self.epoch+1):
            
            self.trainA, self.trainB = shuffle(self.trainA, self.trainB)
            self.testA, self.testA_touched = shuffle(self.testA,self.testA_touched)
            
            self.num_batches = max(len(self.trainA), len(self.trainB)) // self.batch_size
            for idx in range(self.num_batches):
                batch_A_images = self.trainA[idx * self.batch_size : (idx + 1) * self.batch_size]
                batch_B_images = self.trainB[idx * self.batch_size : (idx + 1) * self.batch_size]
                batch_test_images = self.testA[0 : self.batch_size]
                batch_test_touched_images = self.testA_touched[0 : self.batch_size]

                # Update D
                # self.d_loss = discriminator_loss(real=self.D_real.preds, fake=self.D_fake.preds)
                feed_dict_d_loss = {
                    self.real_A: batch_A_images,
                    self.real_B: batch_B_images,
                    self.D_real.is_training: True,
                    self.D_fake.is_training: True,
                    self.G_fake.is_training: True
                }
                _, d_loss, summary_str = self.sess.run([self.D_optim, self.d_loss, self.D_loss_summary],feed_dict=feed_dict_d_loss)
                self.writer.add_summary(summary_str, counter)


                # Update G
                feed_dict_g_loss = {
                    self.real_A: batch_A_images,
                    self.real_B: batch_B_images,
                    self.D_fake.is_training: True,
                    self.G_fake.is_training: True
                }
                fake_B,_, g_loss, summary_str = self.sess.run([self.fake_B,self.G_optim, self.g_loss, self.G_loss_summary], feed_dict=feed_dict_g_loss)
                self.writer.add_summary(summary_str, counter)


                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))

                
                    
                # After an epoch, start_batch_id is set to zero
                # non-zero value is only for the first epoch after loading pre-trained model
                start_batch_id = 0

                # save model
                # self.save(self.checkpoint_dir, counter)

                # save model for final step
            
            #if np.mod(counter, self.print_freq) == 0:

            feed_dict_sample = {
                self.real_A: batch_test_touched_images,
                self.G_fake.is_training: False
            }
            samples = self.sess.run(self.fake_B, feed_dict=feed_dict_sample)
            stacked_result_train = np.concatenate([batch_A_images, batch_B_images, fake_B], axis=2)
            stacked_result_test = np.concatenate([batch_test_images, batch_test_touched_images, samples], axis=2)
            stacked_result = np.concatenate([stacked_result_train,stacked_result_test],axis=1)

            save_images(stacked_result, [self.batch_size, 1],
                        './{}/real_A_{:03d}_{:04d}.jpg'.format(self.sample_dir, epoch, idx))


            self.D_real.ckpt_save(epoch)
            self.G_fake.ckpt_save(epoch)


    @property
    def model_dir(self):
        return "{}_{}".format(self.model_name, self.dataset_name)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()
        test_A_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testA'))

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load :
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        for sample_file  in test_A_files : # A -> B
            print('Processing A image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file, size=self.img_size, gray_to_RGB=self.gray_to_RGB))
            image_path = os.path.join(self.result_dir,'{0}'.format(os.path.basename(sample_file)))

            fake_img = self.sess.run(self.fake_B, feed_dict = {self.test_real_A : sample_image})

            save_images(fake_img, [1, 1], image_path)
            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                '..' + os.path.sep + sample_file), self.img_size, self.img_size))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                '..' + os.path.sep + image_path), self.img_size, self.img_size))
            index.write("</tr>")
            index.close()