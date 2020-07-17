import argparse
def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='vgg16', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=20, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  parser.add_argument('--acc_step',help='Gradient Accumalation step', default=1,
                      type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')                      
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')

#temporal shift module
  parser.add_argument('--shift', default=False, action="store_true", help='use shift for models')
  parser.add_argument('--shift_div', default=8, type=int, help='number of div for shift (default: 8)')
  parser.add_argument('--shift_place', default='blockres', type=str, help='place for shift (default: stageres)')
  parser.add_argument('--pretrain', type=str, default='imagenet')
  parser.add_argument('--tune_from', type=str, default=None, help='fine-tune from checkpoint')

# dataset arguments from temporal segment networks
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='virat', type=str)
  parser.add_argument('--num_segments', type=int, default=3)
  parser.add_argument('--input_sampling', default=True, action="store_true", 
  help='use segment sampling techniques for video dataset if set false takes in all the frames')
  #loss type
  parser.add_argument('--loss_type',type=str,default='sigmoid',help="""\
      Loss type for training the network ('softmax', 'sigmoid', 'focal').\
      """)
  parser.add_argument('--splits', type=str,default="JHMDB_RGB_1_split_0")

  # config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr_type', default='step', type=str,
                    metavar='LRtype', help='learning rate type')
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int) 
# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and diaplay
  parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      action='store_true')
#visualisation 
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')

  args = parser.parse_args()
  return args