import os
import tensorflow as tf

file_url_base = "ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection"
files = {
    "bottle": {        # 148 MB
        "fname": "bottle.tar.xz",
        "sha256": "726512129cb3b1f47d4f6cd7c7fc9170db7fc209132249c788fec06029cb5dc3"
    },
    "cable": {         # 481 MB
        "fname": "cable.tar.xz",
        "sha256": "1e3d59b80a6a8da5adf32fb0cf311c7b964cde146d6920aef9ca66380999a9cd"
    },
    "capsule": {       # 385 MB
        "fname": "capsule.tar.xz",
        "sha256": "f6a41cb11f1589d552888fe9c43a1adcbcae15b70073e19093322f836d00a2b6"
    },
    "carpet": {        # 705 MB
        "fname": "carpet.tar.xz",
        "sha256": "d9dd5064515a20bd75cf24d223c570579141abd72388beef892780b74dbaa85d"
    },
    "grid": {          # 153 MB
        "fname": "grid.tar.xz",
        "sha256": "d94a9a651cbfaee22c94d06cf8fa8204a590c872362de0489eb34f1645328de8"
    },
    "hazelnut": {      # 588 MB
        "fname": "hazelnut.tar.xz",
        "sha256": "a7f4e7d65a60a235def7236a46fcb854d62e6c6917c16b1ec9cb4ff089d3b0da"
    },
    "leather": {       # 500 MB
        "fname": "leather.tar.xz",
        "sha256": "704ffa48b1dc2fc44824aa50442d9febc399dd92a5d6fd8156f8aa0373d15f24"
    },
    "metal_nut": {     # 157 MB
        "fname": "metal_nut.tar.xz",
        "sha256": "86e3c8e163ceb19146ad3ddc408e85cc34ae02e0da430e07a73bac6bbb48532a"
    },
    "pill": {          # 262 MB
        "fname": "pill.tar.xz",
        "sha256": "ac6ce9ec4ec2a5d67c906d82147bd7431980cf00b78f25d63201a5f17e3a2038"
    },
    "screw": {         # 186 MB
        "fname": "screw.tar.xz",
        "sha256": "0a27a365a21e472d19b9fb48cb8e4002b49fabe083bff7f4c16fa57d78154f10"
    },
    "tile": {          # 335 MB
        "fname": "tile.tar.xz",
        "sha256": "392228153b96d0db9f179d3f7c2d63e131e907154883b57309fcbf798643173c"
    },
    "toothbrush": {    # 104 MB
        "fname": "toothbrush.tar.xz",
        "sha256": "fe72a34b4b0d074f84e844be7ef8c4f64f7f02f20e25f19c9e3b4a231c8bea77"
    },
    "transistor": {    # 384 MB
        "fname": "transistor.tar.xz",
        "sha256": "146b2166c35a1d0cf37ded091366ac01a50338b4ac704632f1239890eaca4449"
    },
    "wood": {          # 474 MB
        "fname": "wood.tar.xz",
        "sha256": "3757297ac2a266072eb4f25a97f805f5b37fd583492b20bd709f310736c1ea47"
    },
    "zipper": {        # 152 MB
        "fname": "zipper.tar.xz",
        "sha256": "7ab46e0e195da145db30052baea3d3a5d14c5f0e710137c968e0423f02cb942d"
    }
}


def _get_extracted_ds_root(category):
    try:
        file = files[category]
    except KeyError:
        raise ValueError("{} is not a valid category name".format(category))

    # tensorflow doesn't honor KERAS_HOME
    # https://github.com/tensorflow/tensorflow/issues/38831
    def get_cache_dir():
        return os.getenv('KERAS_HOME') or os.path.join(os.path.expanduser('~'), '.keras')

    try:
        path = tf.keras.utils.get_file(
            fname = file["fname"],
            origin = os.path.join(file_url_base, file["fname"]),
            file_hash = file["sha256"],
            extract = True,
            cache_dir = get_cache_dir()
        )
    except PermissionError as pe:
        # The content of the .tar.xz files aren't writeable by default,
        # so permission errors arises if dataset is already extracted.
        # Manual fix with shell:
        #  $ find ${KERAS_HOME:-~/.keras}/datasets -type d -exec chmod 775 {} \;
        #  $ find ${KERAS_HOME:-~/.keras}/datasets -type f -exec chmod 664 {} \;
        #  $ find ${KERAS_HOME:-~/.keras}/datasets -maxdepth 1 -mindepth 1 -type d -exec rm -rf {} \;
        path = os.path.join(get_cache_dir(), "datasets", file["fname"])

    ds_root = os.path.join(os.path.dirname(path), category)
    if not os.path.isdir(ds_root):
        raise NotADirectoryError(ds_root)

    return ds_root


def get_labeled_dataset(category, split='train', image_channels=0, binary_labels=False):
    ds_root = _get_extracted_ds_root(category)

    # everything except 'TRAIN' and 'train' results in 'test'
    split = split.lower()
    if not isinstance(split, str) or split != 'train':
        split = 'test'

    ds_list = tf.data.Dataset.list_files(os.path.join(ds_root, split, '*', '*.png'))

    def process_path(file_path):
        label = tf.strings.split(file_path, os.path.sep)[-2]
        image = tf.io.read_file(file_path)
        image = tf.io.decode_png(image, channels=image_channels)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image, label

    ds = ds_list.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if binary_labels:# and split == 'test':
        ds = ds.map(lambda x, y: (x, 0 if y == 'good' else 1))

    return ds
