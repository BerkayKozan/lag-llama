# Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting

![lag-llama-architecture](images/lagllama.webp)

Lag-Llama is the <b>first open-source foundation model for time series forecasting</b>!

[[Tweet Thread](https://twitter.com/arjunashok37/status/1755261111233114165)] 

[[Model Weights](https://huggingface.co/time-series-foundation-models/Lag-Llama)] [[Colab Demo 1: Zero-Shot Forecasting](https://colab.research.google.com/drive/1DRAzLUPxsd-0r8b-o4nlyFXrjw_ZajJJ?usp=sharing)] [[Colab Demo 2: (Preliminary Finetuning)](https://colab.research.google.com/drive/1uvTmh-pe1zO5TeaaRVDdoEWJ5dFDI-pA?usp=sharing)]

[[Paper](https://arxiv.org/abs/2310.08278)]

[[Video](https://www.youtube.com/watch?v=Mf2FOzDPxck)]
____

## Forked Repo Notes

This repository is a **fork** of the original Lag-Llama. We have **extended** it to research **predictive maintenance** (PdM) scenarios. Our code and instructions focus on how to train and evaluate Lag-Llama with different data preprocessing choices (normalization), zero-shot modes, and RoPE scaling.

### Abstract (Our Paper)

> Modern engineering systems experience progressive deterioration due to wear and varying operating conditions, necessitating robust strategies for timely maintenance interventions. Predictive Maintenance (PdM) stands out among traditional approaches by leveraging Remaining Useful Life (RUL) estimations to schedule repairs only when genuinely necessary. In this work, we simulate real-life degradation by constructing a hierarchical gamma-process framework that generates univariate time-series trajectories. These synthetic data reflect realistic deterioration patterns, providing a controlled environment for evaluating predictive maintenance approaches. We then explore both statistical and deep-learning time-series forecasting methods, ultimately focusing on Lag-Llama—a foundation-model-based architecture adapted for univariate forecasting tasks.
> Lag-Llama’s extensive pretraining allows for strong generalization, while fine-tuning refines its ability to model complex deterioration patterns. To evaluate how effectively such forecasts translate into cost-saving maintenance decisions, we adopt a cost-centric metric that captures the economic impact of early versus late repairs. We systematically compare zero-shot and fine-tuned settings, the effect of RoPE (Rotary Position Embedding) scaling, and the role of normalized versus unnormalized data.
> Our results demonstrate that, under the right conditions, Lag-Llama not only produces accurate forecasts but also achieves favorable scores on our cost-based metric, reflecting well-timed maintenance actions. These findings offer valuable insights into how PdM strategies can be optimized by combining foundation-model capabilities with a rigorous, cost-sensitive evaluation framework.

---

## Repository Structure and How to Use This Fork

1. **`main.py`**  
   - Primary entry point to **train** or **evaluate** the model in our PdM scenario.
   - Key command-line arguments:
     - **`-m` / `--mode`** *(Required)*: `train` or `evaluate`.
     - **`-n` / `--normalize`** *(Default: `True`)*: Whether to normalize data.
     - **`-r` / `--rope_scaling`** *(Default: `False`)*: Enable RoPE scaling.
     - **`-z` / `--zero_shot`** *(Default: `False`)*: Zero-shot mode (no model finetuning).
   - **Example usage**:
     ```bash
     # Train with normalization (default)
     python main.py -m train

     # Evaluate with normalization (default)
     python main.py -m evaluate

     # Train with rope scaling on
     python main.py -m train -r True

     # Evaluate in zero-shot mode with normalization
     python main.py -m evaluate -z True -n True
     ```

2. **`data_loader.py`**  
   - Loads and processes time-series data.

3. **`lag_llama_trainer.py`**  
   - Wraps the training logic for Lag-Llama. Also references the code under `lag_llama/`.

4. **`callbacks_handler.py`**, **`wandb_handler.py`**  
   - Manages callbacks and Weights & Biases logging.

5. **`deterioration_probability_calculator.py`**, **`forecast_cost_estimator.py`**  
   - Modules specific to our predictive maintenance setting.  
   - They compute probabilities of deterioration/failure and the associated cost metrics.

6. **`constants.py`**  
   - Centralized configuration for file paths, hyperparameters, default cost values, etc.

## Deterioration Dataset Generator

`deterioration_dataset_generator.py` is a standalone script for creating synthetic deterioration data.  
It simulates a univariate time series that reflects progressive wear-and-tear, making it ideal for testing and benchmarking predictive maintenance strategies.

### Usage

You can run it directly:
```bash
python deterioration_dataset_generator.py
```
You can change given parameters in the class. By default, the script generates and saves two Parquet files (long and wide formats) in:
```bash
datasets/deterioration/
```

## Installation and Setup

1. **Clone or download** this fork:
   ```bash
   git clone https://github.com/BerkayKozan/lag-llama.git
   cd lag-llama
   ```
2. **Create a virtual environment**:
   ```bash
      python3 -m venv .venv
      source .venv/bin/activate  # On macOS/Linux
      # .venv\Scripts\activate   # On Windows
   ```
3. **Install Dependencies**:
   ```bash
      pip install --upgrade pip setuptools wheel
      pip install -r requirements.txt
   ```
4. **Run**: 
   ```bash
   python main.py -m train
   # or
   python main.py -m evaluate
   ```
   • Refer to Usage above for more argument options.

## Best Practices of Our Work (From Original + Our PdM Work):
1. **Context Length**:
      - For our dataset, larger context lengths (i.e. 64, 128) are preferred, for better results.
2. **Learning Rate**:
      - For our dataset, smaller learning rates (i.e. $10^{-5}$) works best, resulting in smaller validation loss.
4. **RoPE Scaling**:
      - If you need a context length beyond the model’s training size, use --rope_scaling (i.e., -r True).
5. **Normalization**:
      - By default, -n True is enabled. If your data is already scaled or you prefer raw values, disable it with -n False.
7. **Fine-tuning vs zero-shot**:
      - Zero-shot can be fast to test on the new data. Since we can generate our data, fine-tuning boosts the performance, increasing accuracy, and resulting in a better metric score, if used on a correct configurations and hyperparameter tuning.
8. **Early Stopping (training)**:
      - By default it is enabled. This option lets us save time, when validation loss does not decrease in several epochs.
9. **PdM-Specific Metric**:
      - We incorporate a cost-based metric to repair times for each component, determined based on the forecasts from our model. See our code in forecast_cost_estimator.py for details.
   
____
## Back to Original Lag Llama Repo:

<b>Updates</b>:
* **27-June-2024**: Fixed critical issues in the kv_cache implementation, improving forecast accuracy. The fixes include: resetting the self.y_cache flag globally, using causal attention correctly during kv_cache initialization, and adjusting rotary embeddings post-concatenation. Contribution by [@KelianM](https://github.com/KelianM).
* **16-Apr-2024**: Released pretraining and finetuning scripts to replicate the experiments in the paper. See [Reproducing Experiments in the Paper](https://github.com/time-series-foundation-models/lag-llama?tab=readme-ov-file#reproducing-experiments-in-the-paper) for details.
* **9-Apr-2024**: We have released a 15-minute video 🎥 on Lag-Llama on [YouTube](https://www.youtube.com/watch?v=Mf2FOzDPxck).
* **5-Apr-2024**: Added a [section](https://colab.research.google.com/drive/1DRAzLUPxsd-0r8b-o4nlyFXrjw_ZajJJ?authuser=1#scrollTo=Mj9LXMpJ01d7&line=6&uniqifier=1) in Colab Demo 1 on the importance of tuning the context length for zero-shot forecasting. Added a [best practices section](https://github.com/time-series-foundation-models/lag-llama?tab=readme-ov-file#best-practices) in the README; added recommendations for finetuning. These recommendations will be demonstrated with an example in [Colab Demo 2](https://colab.research.google.com/drive/1uvTmh-pe1zO5TeaaRVDdoEWJ5dFDI-pA?usp=sharing) soon.
* **4-Apr-2024**: We have updated our requirements file with new versions of certain packages. Please update/recreate your environments if you have previously used the code locally.
* **7-Mar-2024**: We have released a preliminary [Colab Demo 2](https://colab.research.google.com/drive/1uvTmh-pe1zO5TeaaRVDdoEWJ5dFDI-pA?usp=sharing) for finetuning. Please note this is a preliminary tutorial. We recommend taking a look at the best practices if you are finetuning the model or using it for benchmarking.
* **17-Feb-2024**: We have released a new updated [Colab Demo 1](https://colab.research.google.com/drive/1DRAzLUPxsd-0r8b-o4nlyFXrjw_ZajJJ?usp=sharing) for zero-shot forecasting that shows how one can load time series of different formats.
* **7-Feb-2024**: We released Lag-Llama, with open-source model checkpoints and a Colab Demo for zero-shot forecasting.

____

**Current Features**:

💫 <b>Zero-shot forecasting</b> on a dataset of <b>any frequency</b> for <b>any prediction length</b>, using <a href="https://colab.research.google.com/drive/1DRAzLUPxsd-0r8b-o4nlyFXrjw_ZajJJ?usp=sharing" target="_blank">Colab Demo 1.</a><br/>

💫 <b>Finetuning</b> on a dataset using [Colab Demo 2](https://colab.research.google.com/drive/1uvTmh-pe1zO5TeaaRVDdoEWJ5dFDI-pA?usp=sharing).

💫 <b>Reproducing</b> experiments in the paper using the released scripts. See [Reproducing Experiments in the Paper](https://github.com/time-series-foundation-models/lag-llama?tab=readme-ov-file#reproducing-experiments-in-the-paper) for details. 

**Note**: Please see the [best practices section](https://github.com/time-series-foundation-models/lag-llama?tab=readme-ov-file#best-practices) when using the model for zero-shot prediction and finetuning.

____

## Reproducing Experiments in the Paper

To replicate the pretraining setup used in the paper, please see [the pretraining script](scripts/pretrain.sh). Once a model is pretrained, instructions to finetune it with the setup in the paper can be found in [the finetuning script](scripts/finetune.sh).


## Best Practices

Here are some general tips in using Lag-Llama. 
<!-- We recommend reading the [paper](https://arxiv.org/abs/2310.08278) for all details about the model. -->

### General Information

* Lag-Llama is a **probabilistic** forecasting model trained to output a probability distribution for each timestep to be predicted. For your own specific use-case, we would recommend benchmarking the zero-shot performance of the model on your data first, and then finetuning if necessary. As we show in our paper, Lag-Llama has strong zero-shot capabilities, but performs best when finetuned. The more data you finetune on, the better. For specific tips on applying on model zero-shot or on finetuning, please refer to the sections below.

#### Zero-Shot Forecasting

* Importantly, we recommend trying different **context lengths** (starting from $32$ which it was trained on) and identifying what works best for your data. As we show in [this section of the zero-shot forecasting demo](https://colab.research.google.com/drive/1DRAzLUPxsd-0r8b-o4nlyFXrjw_ZajJJ?authuser=1#scrollTo=Mj9LXMpJ01d7&line=6&uniqifier=1), the model's zero-shot performance improves as the context length is increased, until a certain context length which may be specific to your data. Further, we recommend enabling RoPE scaling for the model to work well with context lengths larger than what it was trained on.

#### Fine-Tuning

If you are trying to **benchmark** the performance of the model under finetuning, or trying to obtain maximum performance from the model: 

* We recommend tuning two important hyperparameters for each dataset that you finetune on: the **context length** (suggested values: $32$, $64$, $128$, $256$, $512$, $1024$) and the **learning rate** (suggested values: $10^{-2}$, $5 * 10^{-3}$, $10^{-3}$, $5 * 10^{-3}$, $1 * 10^{-4}$, $5 * 10^{-4}$). 
* We also highly recommend using a validation split of your dataset to early stop your model, with an early stopping patience of 50 epochs. 

## Contact

We are dedicated to ensuring the reproducility of our results, and would be happy to help clarify questions about benchmarking our model or about the experiments in the paper.
The quickest way to reach us would be by email. Please email **both**: 
1. [Arjun Ashok](https://ashok-arjun.github.io/) - arjun [dot] ashok [at] servicenow [dot] com
2. [Kashif Rasul](https://scholar.google.de/citations?user=cfIrwmAAAAAJ&hl=en) - kashif [dot] rasul [at] gmail [dot] com

If you have questions about the model usage (or) code (or) have specific errors (eg. using it with your own dataset), it would be best to create an issue in the GitHub repository.

## Citing this work

Please use the following Bibtex entry to cite Lag-Llama.

```
@misc{rasul2024lagllama,
      title={Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting}, 
      author={Kashif Rasul and Arjun Ashok and Andrew Robert Williams and Hena Ghonia and Rishika Bhagwatkar and Arian Khorasani and Mohammad Javad Darvishi Bayazi and George Adamopoulos and Roland Riachi and Nadhir Hassen and Marin Biloš and Sahil Garg and Anderson Schneider and Nicolas Chapados and Alexandre Drouin and Valentina Zantedeschi and Yuriy Nevmyvaka and Irina Rish},
      year={2024},
      eprint={2310.08278},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```




