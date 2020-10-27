# standard modules
import sys
assert sys.version_info >= (3, 5), "Python 3.5 or greater required"
import argparse
import os
import time
import json
from packaging import version

import logging
logger   = logging.getLogger('train_ganomaly')
debug    = logger.debug
info     = logger.info
warning  = logger.warning
error    = logger.error
critical = logger.critical

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# external modules
import tensorflow as tf
import numpy as np

# package modules
from datasets.mvtec_ad import get_labeled_dataset
from datasets.common import get_dataset
from models.ganomaly import GANomaly
from models.cae import CAE
from models.cnae import CNAE
from models.cvae import CVAE
from utils.callbacks import ADModelEvaluator
from utils.datasets import create_anomaly_dataset

default_args = {
    # training params
    'epochs': 1,
    'batch_size': 64,
    'learning_rate': 0.0002,
    'early_stopping': 100,

    # tf.data piepline params
    'dataset_name': 'mnist',
    'cache_path': None,
    'abnormal_class': 2,  # only valid for mnist, fashion_mnist, cifar10, cifar100 and stl10
    'image_size': 32,
    'image_channels': 0,  # only valid for MVTec AD
    'buffer_size': 1000,
    'shuffle': True,
    'prefetch': True,
    'random_flip': False,
    'random_crop': False,
    'random_brightness': False,
    'repeat_dataset': None,

    # model params
    'model_name': 'ganomaly',
    'latent_size': 100,
    'intermediate_size': 100,   # only valid for cvae
    'n_filters': 64,
    'n_extra_layers': 0,
    'w_adv': 1,                 # only valid for GANomaly
    'w_rec': 50,                # only valid for GANomaly
    'w_enc': 1,                 # only valid for GANomaly

    # debugging params
    'train_steps': None,
    'eval_steps': None,
    'log_level': 'info',
    'debug': False,

    # input/output dir params
    'data_dir': './trainig/data',
    'model_dir': './trainig/model',
    'output_data_dir': './trainig/output'
}


def build_model(args) -> tf.keras.Model:
    image_shape = (args.image_size, args.image_size, args.image_channels)

    def build_default(model_class, **kwargs):
        return model_class(
            input_shape=image_shape,
            latent_size=args.latent_size,
            n_filters=args.n_filters,
            n_extra_layers=args.n_extra_layers,
            **kwargs
        )

    def compile_default(model):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=args.learning_rate),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[
                tf.keras.losses.MeanAbsoluteError(),
                tf.keras.losses.BinaryCrossentropy()
            ]
        )
        return model

    def build_ganomaly():
        model = build_default(GANomaly)
        model.compile(
            loss={
                'adv': tf.keras.losses.MeanSquaredError(),
                'rec': tf.keras.losses.MeanAbsoluteError(),
                'enc': tf.keras.losses.MeanSquaredError(),
                'dis': tf.keras.losses.BinaryCrossentropy()
            },
            loss_weights={
                'adv': args.w_adv,
                'rec': args.w_rec,
                'enc': args.w_enc
            },
            optimizer={
                'gen': tf.keras.optimizers.Adam(
                    learning_rate=args.learning_rate,
                    beta_1=0.5, beta_2=0.999),
                'dis': tf.keras.optimizers.Adam(
                    learning_rate=args.learning_rate,
                    beta_1=0.5, beta_2=0.999)
            }
        )
        return model

    def switcher_default():
        warning("Unknown model_name, using 'ganomaly' as default!")
        return build_ganomaly()

    switcher = {
        'ganomaly': build_ganomaly,
        'cae': lambda: compile_default(build_default(CAE)),
        'cnae': lambda: compile_default(build_default(CNAE)),
        'cvae': lambda: compile_default(build_default(CVAE,
                            intermediate_size=args.intermediate_size))
    }
    model = switcher.get(args.model_name, switcher_default)()

    model.build((None, *image_shape))
    return model


def get_prepared_datasets(args):
    # get dataset by name with simple try an error
    try:
        train_ds = get_labeled_dataset(
            category=args.dataset_name, split='train', image_channels=args.image_channels, binary_labels=True)
        test_ds = get_labeled_dataset(
            category=args.dataset_name, split='test', image_channels=args.image_channels, binary_labels=True)
        args.image_channels = 3 if args.image_channels == 0 else args.image_channels
    except ValueError:
        try:
            (train_images, train_labels), (test_images, test_labels) = create_anomaly_dataset(
                dataset=get_dataset(args.dataset_name), abnormal_class=args.abnormal_class)
            train_ds = tf.data.Dataset.from_tensor_slices(
                (train_images, train_labels))
            test_ds = tf.data.Dataset.from_tensor_slices(
                (test_images, test_labels))
            args.image_channels = train_images.shape[-1]
        except ValueError:
            raise ValueError(
                "{} isn't a valid dataset".format(args.dataset_name))

    def resize_image(image, label):
        image = tf.image.resize(image, (args.image_size, args.image_size))
        return image, label
    train_ds = train_ds.map(
        resize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.map(
        resize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if args.cache_path:
        cache_dir = os.path.join(args.cache_path, 'tfdata_cache_{}_{}_{}'.format(
            args.dataset_name, args.image_size, args.image_channels))
        os.makedirs(cache_dir, exist_ok=True)
        train_ds = train_ds.cache(os.path.join(cache_dir, 'train'))
        test_ds = test_ds.cache(os.path.join(cache_dir, 'test'))

    if args.repeat_dataset:
        train_ds = train_ds.repeat(args.repeat_dataset)

    if args.random_flip or args.random_crop or args.random_brightness:
        def augment_image(image, label):
            if args.random_flip:
                image = tf.image.random_flip_left_right(image)
                image = tf.image.random_flip_up_down(image)
            if args.random_crop:
                image_shape = (args.image_size, args.image_size,
                               args.image_channels)
                image = tf.image.resize_with_crop_or_pad(
                    image, image_shape[-3] + 6, image_shape[-2] + 6)
                image = tf.image.random_crop(image, size=image_shape)
            if args.random_brightness:
                image = tf.image.random_brightness(image, max_delta=0.5)
                image = tf.clip_by_value(image, 0, 1)
            return image, label
        train_ds = train_ds.map(
            augment_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if args.shuffle:
        train_ds = train_ds.shuffle(args.buffer_size)

    if args.prefetch:
        train_ds = train_ds.prefetch(args.buffer_size)
        test_ds = test_ds.prefetch(args.buffer_size)

    return train_ds, test_ds


def main(args):
    train_ds, test_ds = get_prepared_datasets(args)
    train_count = tf.data.experimental.cardinality(train_ds).numpy()
    test_count = tf.data.experimental.cardinality(test_ds).numpy()
    info("dataset: train_count: {}, test_count: {}".format(train_count, test_count))

    model = build_model(args)
    model.summary(print_fn=info)
    #model.net_gen.summary(print_fn=info) # TODO call it from summary() of GANomaly
    #model.net_dis.summary(print_fn=info)

    #model.load_weights('./no/valid/path')

    adme = ADModelEvaluator(
        test_count=test_count if args.eval_steps is None else args.eval_steps * args.batch_size,
        model_dir=args.sm_model_dir or args.model_dir,
        early_stopping=args.early_stopping
    )

    results = model.fit(
        x=train_ds.batch(args.batch_size),
        validation_data=test_ds.batch(args.batch_size),
        callbacks=[adme],
        epochs=args.epochs,
        steps_per_epoch=args.train_steps,
        validation_steps=args.eval_steps,
        verbose=0
    )

    # remove the useless per image losses and labels and add test results
    del results.history['val_losses']
    del results.history['val_labels']
    results.history['val_auroc'] = adme.test_results

    # https://stackoverflow.com/questions/23613426/write-dictionary-of-lists-to-a-csv-file
    info("results: {}".format(json.dumps(
        results.history, indent=4, sort_keys=True, default=str)))

    critical("END OF SCRIPT REACHED")


def parse_args():
    """
    https://docs.python.org/3.6/library/argparse.html
    https://sagemaker.readthedocs.io/en/stable/using_tf.html#prepare-a-script-mode-training-script
    https://github.com/aws/sagemaker-containers#list-of-provided-environment-variables-by-sagemaker-containers
    """

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def str2logging(v):
        return {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }.get(v, logging.INFO)
    
    def str2posint(v):
        v = int(v)
        return v if v > 0 else None

    parser = argparse.ArgumentParser()

    # training params
    parser.add_argument('--epochs', type=int, default=default_args['epochs'])
    parser.add_argument('--batch_size', type=int,
                        default=default_args['batch_size'])
    parser.add_argument('--learning_rate', type=float,
                        default=default_args['learning_rate'])
    parser.add_argument('--early_stopping', type=int,
                        default=default_args['early_stopping'])

    # tf.data piepline options
    parser.add_argument('--dataset_name', type=str,
                        default=default_args['dataset_name'])
    parser.add_argument('--cache_path', type=str,
                        default=default_args['cache_path'])
    parser.add_argument('--abnormal_class', type=int,
                        default=default_args['abnormal_class'])
    parser.add_argument('--image_size', type=int,
                        default=default_args['image_size'])
    parser.add_argument('--image_channels', type=int,
                        default=default_args['image_channels'])
    parser.add_argument('--buffer_size', type=int,
                        default=default_args['buffer_size'])
    parser.add_argument('--shuffle', type=str2bool, nargs='?',
                        const=True, default=default_args['shuffle'])
    parser.add_argument('--prefetch', type=str2bool, nargs='?',
                        const=True, default=default_args['prefetch'])
    parser.add_argument('--random_flip', type=str2bool, nargs='?',
                        const=True, default=default_args['random_flip'])
    parser.add_argument('--random_crop', type=str2bool, nargs='?',
                        const=True, default=default_args['random_crop'])
    parser.add_argument('--random_brightness', type=str2bool, nargs='?',
                        const=True, default=default_args['random_brightness'])
    parser.add_argument('--repeat_dataset', type=str2posint,
                        default=default_args['repeat_dataset'])

    # model params
    parser.add_argument('--model_name', type=str,
                        default=default_args['model_name'])
    parser.add_argument('--latent_size', type=int,
                        default=default_args['latent_size'])
    parser.add_argument('--intermediate_size', type=int,
                        default=default_args['intermediate_size'])
    parser.add_argument('--n_filters', type=int,
                        default=default_args['n_filters'])
    parser.add_argument('--n_extra_layers', type=int,
                        default=default_args['n_extra_layers'])
    parser.add_argument('--w_adv', type=int,
                        default=default_args['w_adv'])
    parser.add_argument('--w_rec', type=int,
                        default=default_args['w_rec'])
    parser.add_argument('--w_enc', type=int,
                        default=default_args['w_enc'])

    # debugging params
    parser.add_argument('--train_steps', type=str2posint,
                        default=default_args['train_steps'])
    parser.add_argument('--eval_steps', type=str2posint,
                        default=default_args['eval_steps'])
    parser.add_argument('--log_level', type=str2logging,
                        default=default_args['log_level'])
    parser.add_argument('--debug', type=str2bool, nargs='?',
                        const=True, default=default_args['debug'])

    # input/output dir params
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_DATA_DIR') or default_args['data_dir'])
    parser.add_argument('--model_dir', type=str,
                        default=default_args['model_dir'])
    parser.add_argument('--sm_model_dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output_data_dir', type=str,
                        default=os.environ.get('SM_OUTPUT_DATA_DIR') or default_args['output_data_dir'])

    return parser.parse_known_args()


if __name__ == '__main__':

    args, unknown = parse_args()

    # setup logging
    logging.basicConfig(stream=sys.stdout, # SageMaker doesn't log the default stderr
                        level=args.log_level,
                        # https://docs.python.org/3.8/library/logging.html#logrecord-attributes
                        format='[%(asctime)s | %(name)s | %(levelname)s] %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S')

    # print info about script params and env values
    debug('Know args: {}'.format(args))
    if unknown:
        debug('Unknown args: {}'.format(unknown))
    sm_env_vals = ['{}="{}"'.format(env, val)
        for env, val in os.environ.items() if env.startswith('SM_')]
    if sm_env_vals:
        debug('ENV: {}'.format(', '.join(sm_env_vals)))

    # use eager execution for debugging
    if args.debug:
        assert version.parse('2.3') <= version.parse(tf.version.VERSION), "Tensorflow 2.3 or geater required"
        tf.config.run_functions_eagerly(True)

    main(args)
