import numpy as np


def merge_and_split_train_test(*data, test_percentage=10):
    merged = data[0]
    for d in data[1:]:
        merged = np.concatenate((merged, d), axis=0)
    split = int((len(merged) / 100) * (100 - test_percentage))
    return merged[:split], merged[split:]


def normalize_dataset(dataset):
    (train_images, train_labels), (test_images, test_labels) = dataset

    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')

    max_value = max(np.max(train_images), np.max(test_images))
    train_images /= max_value
    test_images /= max_value

    return (train_images, train_labels), (test_images, test_labels)


def normalize_images(*images, dtype='float32'):
    # variable length arguments is always a tuple and
    # tuples are read-only, so put it into a list
    images = list(images)

    # fancy python generator expression:
    # determine the max value in all numpy arrays
    max_value = max(np.max(i) for i in images)

    # fancy map and lamda combo:
    # convert to dtype (default: float32) and normalize to interval [0,1]
    # ??? is this memory friendly ???
    images = map(lambda x: x.astype(dtype) / max_value, images)
    # also not more memory friendly
    #for i in range(len(images)):
    #    images[i] = images[i].astype(dtype, copy=False)
    #    images[i] /= max_value

    return tuple(images)


def reshape_dataset(dataset, colors=1):
    (train_images, train_labels), (test_images, test_labels) = dataset

    train_images = train_images.reshape(-1, train_images.shape[1], train_images.shape[2], colors)
    test_images = test_images.reshape(-1, test_images.shape[1], test_images.shape[2], colors)

    return (train_images, train_labels), (test_images, test_labels)


def reduce_to_one_category(images, labels, category):
    # result = filter(lambda x: x[1] == category, zip(images, labels))
    # tmp_images, tmp_labels = zip(*result)
    tmp_images = np.empty(images.shape)
    i = 0
    for j in range(min(len(images), len(labels))):
        if labels[j] == category:
            tmp_images[i] = images[j]
            i += 1
    return tmp_images[:i]


def create_anomaly_dataset(dataset, abnormal_class=0, normal_test_percentage=20, shuffle=True, seed=-1):
    (trn_img, trn_lbl), (tst_img, tst_lbl) = dataset

    img = np.concatenate([trn_img, tst_img], axis=0)
    lbl = np.concatenate([trn_lbl, tst_lbl], axis=0)

    if shuffle:
        idx = np.arange(len(lbl))
        if seed != -1:
            np.random.seed(seed)
        np.random.shuffle(idx)
        img = img[idx]
        lbl = lbl[idx]

    nrm_idx = np.where(lbl != abnormal_class)[0]
    abn_idx = np.where(lbl == abnormal_class)[0]

    nrm_img = img[nrm_idx]
    nrm_lbl = lbl[nrm_idx]
    abn_img = img[abn_idx]
    abn_lbl = lbl[abn_idx]

    # normal=0, abnormal=1
    #nrm_lbl[:] = 0
    nrm_lbl = np.zeros_like(nrm_lbl).reshape((-1, 1))
    #abn_lbl[:] = 0
    abn_lbl = np.ones_like(abn_lbl).reshape((-1, 1))

    # split off test percentage from normal data
    tst_end = int((len(nrm_img) * normal_test_percentage) / 100)
    trn_img = nrm_img[tst_end:]
    trn_lbl = nrm_lbl[tst_end:]
    tst_img = nrm_img[:tst_end]
    tst_lbl = nrm_lbl[:tst_end]

    tst_img = np.concatenate([tst_img, abn_img], axis=0)
    tst_lbl = np.concatenate([tst_lbl, abn_lbl], axis=0)

    return (trn_img, trn_lbl), (tst_img, tst_lbl)


def find_abnormal_start_index(labels, abnormal_label=1, normal_label=0):
    prev = 0
    for i, l in enumerate(labels):
        if prev == normal_label and l == abnormal_label:
            print("abnormal_start =", i)
            return i
        prev = l
    raise ValueError("No start of abnormal data found")
