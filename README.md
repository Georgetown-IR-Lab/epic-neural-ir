# Expansion via Prediction of Importance with Contextualization

Below are the steps to reproduce the primary results from Expansion via Prediction of Importance with Contextualization. Sean  MacAvaney, Franco Maria Nardini, Raffaele Perego, Nicola Tonellotto, Nazli Goharian, and Ophir Frieder. SIGIR 2020 (short). [pdf](https://arxiv.org/pdf/2004.14245.pdf)

tl;dr EPIC is an effective and fast neural re-ranking model.

![diagram](overview.pdf)

## Getting started

The code to reproduce the paper is incorporated into OpenNIR. Refer to instructions at https://opennir.net/ to download and configure.

Start by initializing the MS-MARCO dataset, which we'll use for training and evaluation. This downloads, indexes, and prepares necessary files.

```bash
scripts/init_dataset.sh dataset=msmarco
[snip]
dataset is initialized (8841824 documents)
```

## Training

If you'd like to train your own model from scratch, run the following command. It will train a pretty EPIC model using the MS-MARCO training pairs. Grab a coffee, go for a run, take a power nap; this took us about 2.5 hours using a GeForce GTX 1080ti GPU.

```bash
scripts/pipeline.sh config/epic/train_validate
[snip]
validation mrr@10=0.2830
```

If you'd rather use a pre-trained model, you can download one [here](#Files). If you do so, set `PRETRAINED_MODEL=/path/to/model.p`.

## Tune re-ranking thresholds

The re-ranking threshold is an important parameter. We'll tune this threshold on the validation set. These values are used below.

```bash
# OK to leave this if not using a pre-trained model
ARGS="pipeline.file_path=$PRETRAINED_MODEL pipeline.epoch=50"

# For standard index
scripts/pipeline.sh config/epic/tune_threshold $ARGS
[snip]
validation top_threshold=99 mrr@10=0.3275

# For docTTTTTquery index
scripts/pipeline.sh config/epic/tune_threshold $ARGS test_ds.index=doctttttquery
[snip]
validation top_threshold=15 mrr@10=0.3565
```

## Building the document vector cache

If we tried to use EPIC as a typical re-ranking method, it would take about 1 second per query on the standard index and 150ms per query on the docTTTTTquery index (differences due to the above re-ranking thresholds). Luckily, since the document vectors in EPIC do not depend on the query vectors, document vectors can be pre-computed ahead of time, which means we're able to reduce that time considerably: 20x faster on the standard index, and 2x faster on the docTTTTTquery index.

Now we'll run all the documents in the collection through the model to build up their vector representations. This will also take some time -- the collection is about 9M documents. How about watching a good documentary or a dozen. This takes us about 14-16 hours.

```bash
PRUNE=1000
VECS=msmarco.$PRUNE.v
ARGS="pipeline.file_path=$PRETRAINED_MODEL test_pred.prune=$PRUNE pipeline.output_vecs=$VECS"

scripts/pipeline.sh config/epic/build_vectors $ARGS
```

You can adjust `PRUNE` as you wish-- in our experiments, 1000 offered a good trade-off between effectiveness (virtually the same), and storage/runtime (much better). If you want the full vectors, use `PRUNE=0`; but beware, the file will be pretty big (~500GB). It's also probably a good idea to set `VECS` to some place on a SSD because the access pattern in this file will be pretty random come ranking time.

If you're too excited and cannot wait, you can grab our pre-computed representations for MS-MARCO [here](#Files).

## EPIC Effectiveness

Let's first establish some baselines: BM25 on MS-MARCO "Dev" set with standard and [docTTTTTquery](https://github.com/castorini/docTTTTTquery) indexes:

```bash
scripts/pipeline.sh config/trivial/bm25 config/msmarco pipeline.test=True
[snip]
mrr@10=0.1866

scripts/pipeline.sh config/trivial/bm25 config/msmarco pipeline.test=True test_ds.index=doctttttquery
[snip]
mrr@10=0.2748
```

That's not bad, but EPIC can do better!

```bash
THRESHOLD=99 # from tuning above
THRESHOLD_T5=15 # from tuning above
ARGS="pipeline.dvec_file=$VECS pipeline.prune=$PRUNE pipeline.file_path=$PRETRAINED_MODEL"
scripts/pipeline.sh config/epic/test $ARGS pipeline.rerank_threshold=$THRESHOLD
[snip]
mrr@10=0.2722

scripts/pipeline.sh config/epic/test $ARGS pipeline.rerank_threshold=$THRESHOLD_T5 test_ds.index=doctttttquery
[snip]
mrr@10=0.3029
```

## EPIC Efficiency

The previous run was really fast (for us, over 100 queries per second for docTTTTTquery re-ranking), but that's not a fair representation actual query latency because queries were processed in batches and the initial retrieval results were cached.

So for a fair comparison, we clear disk caches, run both initial retrieval and re-ranking, and process one query at a time. We'll warm up with 1000 queries, then take the average times of the subsequent 1000 queries. Baselines:

```bash
sync; echo 1 | sudo tee /proc/sys/vm/drop_caches # clears page cache
scripts/pipeline.sh config/epic/time pipeline.rerank=False
[snip]
<DurationTimer initial_retrieval=21ms total=21ms>

sync; echo 1 | sudo tee /proc/sys/vm/drop_caches # clears page cache
scripts/pipeline.sh config/epic/time pipeline.rerank=False test_ds.index=doctttttquery
[snip]
<DurationTimer initial_retrieval=62ms total=62ms>
```

So, although the docTTTTTquery index is much more effective, it's more than twice as slow.

And now EPIC:

```bash
sync; echo 1 | sudo tee /proc/sys/vm/drop_caches # clears page cache
scripts/pipeline.sh config/epic/time $ARGS pipeline.rerank_threshold=$THRESHOLD
[snip]
<DurationTimer query_vec=18ms initial_retrieval=24ms doc_vec_lookup=4ms rerank=1ms total=34ms>

sync; echo 1 | sudo tee /proc/sys/vm/drop_caches # clears page cache
scripts/pipeline.sh config/epic/time $ARGS pipeline.rerank_threshold=$THRESHOLD_T5 test_ds.index=doctttttquery
[snip]
<DurationTimer initial_retrieval=63ms doc_vec_lookup=1ms query_vectors=19ms rerank=1ms total=67ms>
```

## Summary of Results

| Method | MRR@10 (Dev) | Total Query Latency |
| ------ | -------------:| -------------:|
| BM25 | 0.1866 | 21ms |
| BM25 + EPIC (pruned @ 1000) | 0.2722 | 34ms |
| BM25 on docTTTTTquery | 0.2748 | 62ms |
| BM25 on docTTTTTquery + EPIC (pruned @ 1000) | 0.3029 | 67ms |

\* Note: These numbers differ slightly from what was reported in the paper. From what we can tell, this is a result of changes made to [Anserini](https://github.com/castorini/anserini) (which OpenNIR uses for indexing and first-stage retrieval), which resulted in a drop in MS-MARCO ranking performance. This happened somewhere between versions 0.3.1 (used to run original experiments) and 0.8.0 (version being used by OpenNIR at the time of writing). However, this resulted in no appreciable difference in EPIC performance after re-ranking.

## Files

| File | Description | MD5 | Link |
| ------------- | ------------- | ------------- | ------------- |
| `epic.p` (510MB) | EPIC model trained on MS-MARCO | `df756649ba3c449cf0f462083ef6082d` | [Google Drive][epic.p]  |
| `msmarco.1000.v` (33GB) | Document vectors for MS-MARCO passages from `epic.p`, pruned to 1000 top dimensions | `9ea6f37cf79cf426112cc9586835781f` | [Google Drive][msmarco.1000.v]  |

*([Tip for downloading large Google Drive files using wget][wget])*

## Citation

If you use this work, please cite:

```
@inproceedings{macavaney:sigir2020-epic,
  author = {MacAvaney, Sean and Nardini, Franco Maria and Perego, Raffaele and Tonellotto, Nicola and Goharian, Nazli and Frieder, Ophir},
  title = {Expansion via Prediction of Importance with Contextualization},
  booktitle = {SIGIR},
  year = {2020}
}
```

[wget]: https://medium.com/@acpanjan/download-google-drive-files-using-wget-3c2c025a8b99
[epic.p]: https://drive.google.com/file/d/1d2OODfIv0PLE_KmtwEVAaGbhqvfvqlzx/view?usp=sharing
[msmarco.1000.v]: https://drive.google.com/file/d/1WXhFZv8y_Cs_abzjfWaDJAtp2pUs72qF/view?usp=sharing
