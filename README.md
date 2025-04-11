# Small-Language-Model

A small language model based on a multi-head self-attention Transformer architecture.


## 🧠 Model Features

- Designed as a **document completer**: Given a prompt, it generates a continuation of the text.
- Total parameters: **25.28M**
- Uses a **character-level tokenizer**
- **Block size**: 512 (maximum input prompt length)



<br>

## 🚀 Usage Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/seangao1/SG-GPT-SLM.git
   cd SG-GPT-SLM

2. **Run the training notebook**

   - Open `Experiment.ipynb`
   - Run it in a GPU-enabled environment (e.g., Google Colab, Kaggle, or local machine with CUDA)
   - ⚠️ **Do NOT run on CPU** — the training is compute-intensive and may overheat or crash your machine.
   - Training duration depends on your setup (typically a few hours).
  
3. **Locate the model checkpoint**

    After training, you’ll find a `.pth` file saved under:

    ```
    base_line_GPT_SG stats/
    └── base_line_GPT_SG weights/
        └── model_checkpoint.pth
    ```

4. **Run the text generator**
    - Open `generator.ipynb`

    - This notebook will use the saved weights to generate text continuations based on your prompt.

<br>

## ⚠️ Caveats
 - This model does not support question-answering like ChatGPT.

 - Output may contain incoherent or non-sensical text due to:

    - Limited model capacity (25M params)

    - Character-level tokenization


