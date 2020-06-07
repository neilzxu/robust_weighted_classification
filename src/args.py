from absl import flags


def common_flags():
    flags.DEFINE_string('name', None, 'Experiment name')
    flags.mark_flag_as_required('name')
    flags.DEFINE_string('tags', "", 'Experiment tags')
    flags.DEFINE_enum('mode', None, ['train', 'test'], 'Mode to run in')
    flags.mark_flag_as_required('mode')
    flags.DEFINE_boolean('cuda', False, 'Flag for enabling GPU usage')
    flags.DEFINE_integer('cuda_device', 0,
                         'Index of GPU device to use if using GPU')
    flags.DEFINE_string('preprocess_mode', 'rdc', 'Format of input data')
    flags.DEFINE_boolean(
        'enable_binary', False,
        'Whether to enable specific binary versions of loss and model')


def train_flags():
    flags.DEFINE_string('train_path', None, 'Path to training file')
    flags.DEFINE_string('dev_path', None, 'Path to dev file')
    flags.DEFINE_integer('dev_interval', 5,
                         'Number of epochs between checking dev performance')
    flags.DEFINE_string('loss_type', 'ce',
                        'Loss function to train network with')
    flags.DEFINE_boolean('smote', False,
                         'Whether to use SMOTE data augmentation')
    flags.DEFINE_float(
        'post_factum_wt', None,
        'Whether to do reweighting during inference, and temperature (exponent of class freq to use for balancing)'
    )
    flags.DEFINE_integer('epochs', 300, 'Number of epochs to train for')
    flags.DEFINE_float('hcvar_temp', None,
                       'temperature for hcvar loss class weights for alpha')
    flags.DEFINE_float('hcvar_alpha', None,
                       'multiplier for class weights for hcvar')
    flags.DEFINE_float('cvar_alpha', None,
                       'singular alpha threshold for cvar loss')
    flags.DEFINE_string('cvar_lambda_strat', 'max',
                        'Strategy for calculating lambda in cvar dual form')
    flags.DEFINE_float(
        'var_reg_C', None,
        'complexity constant for variance regularization term in variance regularization versions of cvar'
    )
    flags.DEFINE_float('var_reg_delta', None,
                       'delta constant for cvar variance regularization term')
    flags.DEFINE_integer('tune_epochs', 100,
                         'Number of epochs to tune on reweighted LDAM loss')
    flags.DEFINE_integer('hidden_dim', 300,
                         'Size of hidden/embedding dimension')
    flags.DEFINE_float('margin', 0.0, 'Margin for cos loss')


def optim_flags():
    flags.DEFINE_float('lr', 0.1, 'Learning rate')
    flags.DEFINE_float('momentum', 0.0, 'Momentum weighting for gradient')
    flags.DEFINE_integer('batch_size', 512, 'Batch size')

    flags.DEFINE_string('lr_decay_type', None, 'Linear learning rate decay')
    flags.DEFINE_float('lr_decay_factor', 0.1,
                       'Decay lr by this factor when plateau')
    flags.DEFINE_integer('lr_decay_patience', 10,
                         'How many epochs to wait before decay learning rate')
    flags.DEFINE_float(
        'lr_decay_threshold', 1e-4,
        'Threshold that decay metric must change by to be not counted against patience'
    )
    flags.DEFINE_integer(
        'lr_decay_cooldown', 0,
        'Number of epochs to wait to count patience after a change in learning rate'
    )
    flags.DEFINE_float('lr_decay_min', 0, 'Minimum lr to decay to')
    flags.DEFINE_string('optimizer', 'SGD', 'Name of optimizer')


def test_flags():
    flags.DEFINE_string('test_path', None, 'Path to test file')
    flags.DEFINE_string('model_run_path', None,
                        'wandb run path id to fetch model from')
    flags.DEFINE_string('model_metric', None,
                        'best metric from which to restore model from')


def synth_flags():
    flags.DEFINE_string('synth_mode', None,
                        'Type of synthetic data to generate')
    flags.DEFINE_integer('synth_dim', None, 'Dimension of synthetic data')
    flags.DEFINE_integer('synth_train_ct', 100,
                         'number of samples to generate')
    flags.DEFINE_integer('synth_dev_ct', 10, 'Number of samples to generate')
    flags.DEFINE_integer('synth_test_ct', 10, 'Number of samples to generate')
    flags.DEFINE_integer('synth_seed', None,
                         'Seed for randomly generating samples')
    flags.DEFINE_float('power_one_prob', 0.5, 'P(Y = 1)')
    flags.DEFINE_float('power_scalar', 1.0, 'scalar on power curve')


common_flags()
train_flags()
optim_flags()
test_flags()
synth_flags()
