import os
    for dictionary in dict_list:
        merged.update(dictionary)
    return merged


def format_string(template_string, *dict_args):
    merged_dict = merge_dictionaries(*dict_args)
    return string.Formatter().vformat(template_string, [], SafeFormatter(merged_dict))


def get_elapsed_time(start_time):
    elapsed = time.time() - start_time
    minutes, seconds = divmod(elapsed, 60)
    hours, minutes = divmod(minutes, 60)
    return "%d:%02d:%02d" % (hours, minutes, seconds)

# ---------------- PARALLEL EXECUTION ----------------

def run_with_progress(function, items, message=None, num_jobs=1):

    results = []
    print(f"Starting pool of {num_jobs} jobs")

    total_items = len(items)

    if num_jobs == 1:
        for idx, item in enumerate(items, start=1):
            results.append(function(item))
            if message:
                sys.stdout.write(f"\r{message.format(current=idx, total=total_items)}")
                sys.stdout.flush()

    else:
        pool = multiprocessing.Pool(processes=num_jobs)
        for item in items:
            pool.apply_async(function, args=(item,), callback=results.append)

        while len(results) < total_items:
            if message:
                sys.stdout.write(f"\r{message.format(current=len(results), total=total_items)}")
                sys.stdout.flush()
            time.sleep(0.5)

        pool.close()
        pool.join()

    return results

# ---------------- MODEL HELPERS ----------------

def to_one_hot(num_classes, class_index):
    one_hot = [0.0] * num_classes
    one_hot[int(class_index)] = 1.0
    return one_hot


def load_autoencoder_encoder(input_dim, latent_dim, checkpoint_path):
    model = ae(input_dim, latent_dim)
    init = tf.global_variables_initializer()

    try:
        with tf.Session() as session:
            session.run(init)

            saver = tf.train.Saver(model["params"], write_version=tf.train.SaverDef.V2)

            if os.path.isfile(checkpoint_path):
                print("Restoring", checkpoint_path)
                saver.restore(session, checkpoint_path)

            trained_params = session.run(model["params"])

            return {
                "W_enc": trained_params["W_enc"],
                "b_enc": trained_params["b_enc"]
            }
    finally:
        reset_graph()


def compute_sparsity_penalty(activation, sparsity_target, weight):
    avg_activation = tf.reduce_mean(tf.abs(activation), 0)
    kl_divergence = sparsity_target * tf.log(sparsity_target / avg_activation) + \
        (1 - sparsity_target) * tf.log((1 - sparsity_target) / (1 - avg_activation))
    return weight * tf.reduce_sum(kl_divergence)

# ---------------- SITE BALANCED SPLIT ----------------

from sklearn.model_selection import StratifiedGroupKFold


def site_balanced_kfold(dataset, num_splits=5, random_seed=42):
    """Site-balanced splitting for ABIDE dataset"""

    site_list, label_list = [], []

    for key in dataset.keys:
        patient = dataset.patients[key]
        site_list.append(patient.attrs["site"])
        label_list.append(patient.attrs["y"])

    site_array = np.array(site_list)
    label_array = np.array(label_list)
    indices = np.arange(len(label_array))

    splitter = StratifiedGroupKFold(
        n_splits=num_splits,
        shuffle=True,
        random_state=random_seed
    )

    folds = []
    for train_idx, test_idx in splitter.split(indices, label_array, groups=site_array):
        folds.append((train_idx, test_idx))

    return folds
