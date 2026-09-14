# My LLM: Google Colab Training Guide

This guide covers the complete path from a fresh Google Colab runtime to a
pretrained checkpoint and then a LoRA conversation adapter.

Repository: <https://github.com/Mr-Mcleon/My-LLM>

## 1. What to expect

This project currently defines a GPT-style model of approximately 505 million
parameters with a 64-token context window.

The first Colab run should be treated as an infrastructure test, not as a
quality training run:

1. Run a 20-step pretraining smoke test.
2. Confirm that the checkpoint is written to Google Drive.
3. Start a longer pretraining run.
4. Only after a usable base checkpoint exists, run LoRA conversation tuning.

A few hundred megabytes of text can verify the pipeline, but it is not enough
to train a capable 505M-parameter general language model. A useful training
run needs a much larger, clean, legally usable corpus and substantially more
optimizer steps. Colab sessions can disconnect, so the Drive checkpoint
directory is part of the training setup, not an optional backup.

## 2. Before opening Colab

Have these ready:

- A Google account with enough Drive space for datasets and several large
  checkpoints.
- The GitHub repository URL above.
- A GPU runtime. In Colab, choose **Runtime → Change runtime type → T4 GPU**
  or another available CUDA GPU.
- A dataset whose license permits your intended use.
- A short record of every dataset used: source URL, dataset version or
  snapshot, license, download date, and transformations performed.

Do not upload model checkpoints or large datasets to this GitHub repository.
Keep them in Google Drive or another storage location intended for large
artifacts.

## 3. Dataset choices

### Pretraining text

Use a mixture of clean English text rather than relying only on classic
literature.

**Good starting options:**

1. **Project Gutenberg public-domain books**
   - <https://www.gutenberg.org/>
   - The repository includes a builder for a small public-domain starter
     corpus.
   - The public-domain status is generally based on United States law. Check
     the copyright position in your own country and preserve each book's
     license/notice. Project Gutenberg also asks bulk users to use mirrors or
     offline catalogs rather than repeatedly hitting the main site.
   - This is useful for a smoke test, but the included list of books is not a
     sufficient general pretraining corpus for a 505M model.

2. **FineWeb-Edu samples**
   - <https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu>
   - The dataset card currently exposes manageable sample configurations and
     describes an ODC-BY distribution. Read the current card and its source
     terms before downloading; web-crawl data can have additional
     source-level restrictions.
   - Use a sample or a deliberately selected shard first. Do not attempt to
     download the entire dataset into a free Colab runtime.

3. **Wikimedia/Wikipedia dumps**
   - <https://dumps.wikimedia.org/>
   - Check the current Wikimedia dump and attribution terms. Wikipedia content
     is generally distributed under a share-alike Creative Commons license,
     which may affect redistribution of a derived corpus or model.

For a first serious experiment, use a manageable FineWeb-Edu sample or a
carefully reviewed public-domain/permissioned corpus, then add conversation
data only in the fine-tuning stage.

### Conversation fine-tuning data

The current `ai-model/build_chat_dataset.py` can combine:

- OpenAssistant OASST1:
  <https://huggingface.co/datasets/OpenAssistant/oasst1>
- Databricks Dolly 15k:
  <https://huggingface.co/datasets/databricks/databricks-dolly-15k>
- Alpaca:
  <https://huggingface.co/datasets/tatsu-lab/alpaca>

Read the current dataset card before using or redistributing any of them.
Dolly's card identifies CC BY-SA 3.0. Alpaca has non-commercial
attribution/share-alike restrictions, so do not use it for a commercial
training run unless your intended use is compatible with its current terms.
For a public or commercial project, prefer sources whose terms you have
reviewed and avoid mixing in a dataset with incompatible restrictions.

## 4. The file formats this project expects

### Pretraining

Place a UTF-8 text file at:

```text
My Drive/my-llm/datasets/training_data.txt
```

The loader reads the whole file as text and trains on the resulting token
stream. Plain text with documents separated by blank lines is suitable. Do
not upload a PDF, HTML page, compressed archive, or Parquet file under this
name without converting it first.

### Conversation tuning

Place a UTF-8 JSON Lines file at:

```text
My Drive/my-llm/datasets/chat_dataset.jsonl
```

Each line must be one JSON object with a `messages` array:

```json
{"messages":[{"role":"user","content":"Explain recursion simply."},{"role":"assistant","content":"Recursion is when a function calls itself on a smaller version of the same problem."}]}
```

Use `user` and `assistant` roles. Multiple turns are allowed. The training
code calculates loss on assistant answers and uses the preceding conversation
as context.

## 5. Google Drive layout

The notebook creates this layout automatically:

```text
My Drive/
└── my-llm/
    ├── datasets/
    │   ├── training_data.txt
    │   └── chat_dataset.jsonl
    └── checkpoints/
        ├── pretrain_500m_latest.pth
        ├── pretrain_500m_best.pth
        ├── small_llm_pretrained_500m.pth
        ├── lora_500m_latest.pth
        ├── lora_500m_best.pth
        └── small_llm_lora.pth
```

You do not need both dataset files for every run:

- `MODE = "pretrain"` uses `training_data.txt`.
- `MODE = "finetune"` or `MODE = "lora"` uses `chat_dataset.jsonl`.

## 6. Open and configure the notebook

Open
`colab/Colab_Train_My_LLM.ipynb` from the GitHub repository in Google Colab.
The repository URL is already configured.

The safe first-run settings are:

```python
MODE = "pretrain"
STEPS = 20
EPOCHS = 1
CHECKPOINT_EVERY = 10
DTYPE = "auto"
GENERATE_PUBLIC_DOMAIN_DATA = False
```

Run the notebook cells in order:

1. Configuration
2. Install dependencies, mount Drive, and clone or update the repository
3. GPU check
4. Dataset check
5. Training
6. Checkpoint listing

The notebook now stops with a clear error if the dataset is missing. Put the
file in the Drive path above before running the training cell.

## 7. Creating the starter public-domain corpus

If you want the notebook to create a small starter corpus automatically:

```python
GENERATE_PUBLIC_DOMAIN_DATA = True
MODE = "pretrain"
```

The notebook runs:

```bash
python build_dataset.py --target_mb 500
```

This builder downloads a fixed list of public-domain Gutenberg books and may
finish below the requested target because the target is an upper goal, not a
guarantee. Always inspect the printed size. Use it for a smoke test or
pipeline validation, not as evidence that the 505M model has been adequately
pretrained.

## 8. Converting a Hugging Face text sample into the expected file

For a manageable FineWeb-Edu sample, install the notebook dependencies first,
then use a streaming subset and write only the amount you can store and train.
The dataset card should be checked first because configuration names and terms
can change.

Example pattern:

```python
from datasets import load_dataset
from pathlib import Path

output_path = Path("/content/drive/MyDrive/my-llm/datasets/training_data.txt")
output_path.parent.mkdir(parents=True, exist_ok=True)
target_bytes = 500 * 1024 * 1024
written = 0

stream = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True,
)

with output_path.open("w", encoding="utf-8") as output:
    for row in stream:
        text = str(row.get("text", "")).replace("\x00", " ").strip()
        if not text:
            continue
        chunk = (text + "\n\n").encode("utf-8")
        output.write(chunk.decode("utf-8"))
        written += len(chunk)
        if written >= target_bytes:
            break

print(f"Wrote {written / (1024 * 1024):.1f} MB to {output_path}")
```

If the selected configuration is unavailable, choose a current sample listed
on the FineWeb-Edu dataset card rather than guessing a configuration name.
Keep the downloaded subset's metadata and license information beside your
private Drive copy.

## 9. Pretraining sequence

### 9.1 Smoke test

Use:

```python
MODE = "pretrain"
STEPS = 20
CHECKPOINT_EVERY = 10
```

Run the notebook. The output should show:

- CUDA is available and a GPU name is printed.
- The dataset path exists and has a non-zero size.
- The model and tokenizer load.
- Training loss is printed.
- `pretrain_500m_latest.pth`, `pretrain_500m_best.pth`, and
  `small_llm_pretrained_500m.pth` appear under the Drive checkpoint directory.

If the smoke test fails, do not start a long run. Fix the reported issue first.

### 9.2 Longer base-model run

After the smoke test, keep `MODE = "pretrain"` and increase `STEPS`. For
example:

```python
STEPS = 5000
CHECKPOINT_EVERY = 100
```

This project treats `STEPS` as optimizer updates, not as a guaranteed full
pass over the corpus. A 5,000-step run is a short experiment for this model
size. Increase the run only after checking loss behavior, GPU memory, runtime
cost, and checkpoint growth.

If Colab disconnects, remount Drive, run the notebook again with the same
mode, dataset, checkpoint directory, dtype, and step target. The latest
checkpoint resumes automatically.

Do not delete `pretrain_500m_latest.pth` while a run is in progress. Use
`--no-resume` only when intentionally starting a new run from the beginning.

## 10. LoRA conversation fine-tuning

Do not start with LoRA on a fresh Drive. It requires:

```text
My Drive/my-llm/checkpoints/pretrain_500m_latest.pth
```

First create or upload `chat_dataset.jsonl` to the datasets directory. To
generate it from the supported Hugging Face sources, run this in a notebook
cell:

```python
%cd /content/my-llm/ai-model
!python build_chat_dataset.py
!cp chat_dataset.jsonl /content/drive/MyDrive/my-llm/datasets/chat_dataset.jsonl
```

Then change the configuration to:

```python
MODE = "lora"
STEPS = 2000
EPOCHS = 1
CHECKPOINT_EVERY = 100
LORA_RANK = 16
LORA_ALPHA = 32.0
LORA_DROPOUT = 0.05
```

Run the dataset-check and training cells again. The notebook automatically
passes the base checkpoint to `colab_train.py`. The final adapter is saved as:

```text
My Drive/my-llm/checkpoints/small_llm_lora.pth
```

When resuming LoRA, keep the same rank, alpha, dropout, target modules, base
checkpoint, and checkpoint directory. Changing the LoRA configuration while
resuming is intentionally rejected.

## 11. Full fine-tuning versus LoRA

- `pretrain`: creates the base language model from plain text.
- `finetune`: updates all model weights using conversation JSONL and needs the
  base checkpoint.
- `lora`: freezes the base model and trains a small adapter. It uses much less
  optimizer memory and is the recommended conversation-tuning path for this
  model in Colab.

Run full conversation fine-tuning only when you deliberately want a new full
checkpoint and have enough storage and GPU memory.

## 12. Loading the trained model for chat

To use the adapter with the Flask chat app, make both files available to the
app and set:

```bash
LLM_BASE_CHECKPOINT=/path/to/pretrain_500m_latest.pth
LLM_ADAPTER_CHECKPOINT=/path/to/small_llm_lora.pth
```

The adapter is not a standalone model. It must be loaded together with the
matching base checkpoint and matching LoRA configuration.

## 13. Troubleshooting

### `CUDA available: False`

Stop. Select a GPU runtime, reconnect, and rerun the setup cells.

### Dataset not found

Use the exact filenames and paths:

```text
/content/drive/MyDrive/my-llm/datasets/training_data.txt
/content/drive/MyDrive/my-llm/datasets/chat_dataset.jsonl
```

Remember that pretraining and LoRA use different files.

### LoRA says the pretrained checkpoint is missing

Complete pretraining first and confirm that
`pretrain_500m_latest.pth` is in the Drive checkpoint directory. Do not point
LoRA at the final adapter file.

### Out-of-memory error

Keep the 64-token context window, use `DTYPE = "auto"`, and run batch size 1
as configured by the large-model path. If the selected GPU still cannot fit
the model, use a larger Colab GPU or reduce the model architecture in a
separate experiment.

### The loss is unstable or output is incoherent

Check the dataset size and cleanliness first. A tiny or repetitive corpus will
not adequately train a 505M model. Also verify that you are testing the base
model only after pretraining and the adapter together with its matching base
checkpoint after LoRA.

### The runtime disconnects

Reconnect, remount Drive, and rerun with the same configuration. Checkpoints
are written atomically and the latest checkpoint is designed for resuming.

## 14. Final pretraining checklist

Before starting the long run, confirm:

- [ ] GPU is visible in Colab.
- [ ] `training_data.txt` is UTF-8 and large enough for the experiment.
- [ ] Dataset licenses and source records are saved.
- [ ] Dataset is in `My Drive/my-llm/datasets/`.
- [ ] `MODE = "pretrain"` for the base run.
- [ ] The 20-step smoke test completed successfully.
- [ ] Checkpoints are appearing in `My Drive/my-llm/checkpoints/`.
- [ ] You have enough Drive storage for multiple checkpoint copies.
- [ ] You will keep the same checkpoint directory when resuming.
- [ ] You will run LoRA only after `pretrain_500m_latest.pth` exists.