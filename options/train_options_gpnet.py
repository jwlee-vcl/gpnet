from .base_options_gpnet import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--niter', type=int, default=1000, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=1000, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--lr_decay_epoch', type=int, default=5, help='# of epoch to linearly decay learning rate to zero')
        self.parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy: lambda|step|plateau')

        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of Adam')
        self.parser.add_argument('--lr', type=float, default=0.0004, help='initial learning rate for Adam')

        self.parser.add_argument('--n_frames', type=int, default=32, help='# of frame candidates')

        self.isTrain = True
