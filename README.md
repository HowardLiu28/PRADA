<div align="center">
  <h1><b>PRADA: Prompt-guided Representation Alignment and Dynamic Adaption for Time Series Forecasting </b></h1>
</div>

<div align="center">
  <a href="https://github.com/HowardLiu28">Yinhao Liu</a>, <a href="https://github.com/KZYYYY">Zhenyu Kuang</a>, Hongyang Zhang, <a href="https://github.com/lichen0620">Chen Li</a>, Feifei Li, Xinghao Ding
</div>

---
>
> 🙋 Please let us know if you find out a mistake or have any suggestions!
> 
> 🙏 If you find this resource helpful, please consider to star this repository and cite our research:

```
@article{liu2025prada,
  title={PRADA: Prompt-guided Representation Alignment and Dynamic Adaption for time series forecasting},
  author={Liu, Yinhao and Kuang, Zhenyu and Zhang, Hongyang and Li, Chen and Li, Feifei and Ding, Xinghao},
  journal={Knowledge-Based Systems},
  pages={113478},
  year={2025},
  publisher={Elsevier}
}
```

## 🏆 Updates/News:

🚩 **News** (April.1 2025): PRADA has been accpeted by Knowledge-Based Systems, 2025.

🚩 **News** (April.22 2025): The paper is availble at <a href="https://www.sciencedirect.com/science/article/abs/pii/S0950705125005246">here</a>.

🚩 **News** (April.27 2025): The code is now released.

## 📰 Introduction:

PRADA is a novel framework for time series forecasting based on LLMs. It decomposes time series into trend, seasonal, and residual terms, designs learnable textual prompts aligned with each component by the multi-view TSAA, which further mitigates the gap between natural language and time series. Additionally, a Time-Frequency Dual Constraint is applied to capture overlooked label autocorrelations.
<p align="center">
<img src="pic/main_model.png" height = "360" alt="" align=center />
</p>

## 🤗 Basic preparation:

**Datasets:** You can access the well pre-processed datasets from [[Google Drive]](https://drive.google.com/file/d/1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP/view?usp=sharing) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy), then place the downloaded contents under `./all_datasets`.

**Pretrained GTP-2:** We use the pre-trained GPT-2 as our backbone network. You can download its weight from [here](https://huggingface.co/openai-community/gpt2) and place it under `./gpt2`.

## 🛠️ Requirements and Installation:

Ensure your environment meets the following requirements under python 3.11.0:

- einops==0.8.1
- joblib==1.4.2
- matplotlib==3.10.1
- numpy==1.23.5
- pandas==2.2.3
- peft==0.13.2
- scikit_learn==1.6.1
- statsmodels==0.14.4
- torch==2.1.0
- torchprofile==0.0.4
- tqdm==4.66.5
- transformers==4.45.2
- xlrd==2.0.1

For convenience, you can run:
```bash
conda create -n prada python=3.11.0 -y
conda activate prada
pip install -r requirements.txt
```

## 🚀 Getting Started:

**Long-term forecasting:** We provide all experimental scripts for 8 different benchmarks including ETTh1, ETTh2, ETTm1, ETTm2, Traffic, Weather, Electricity, and ILI for long-term forecasting task. The forecasting horizon is set to {24, 36, 48, 60} for ILI and {96, 192, 336, 720} for the others. For example, you can evaluate the model by:
```bash
# ETTh1
bash ./scripts/long_term_forecast/ETT_script/PRADA_ETTh1.sh
# Traffic
bash ./scripts/long_term_forecast/Traffic_script/PRADA.sh
# Weather
bash ./scripts/long_term_forecast/Weather_script/PRADA.sh
# Electricity
bash ./scripts/long_term_forecast/ECL_script/PRADA.sh
# ILI
bash ./scripts/long_term_forecast/ECL_script/ILI.sh
```

**Short-term forecasting:** We evaluate our model on M4 dataset for short-term forecasting task. The forecasting horizon is set to {6, 48} under different sampling intervals, including Yearly, Quarterly, Monthly, Weekly, Daily, and Hourly. You can run:
```bash
# M4
bash ./scripts/short_term_forecast/PRADA_M4.sh
```
**Few-shot forecasting:** You can set the parameter `--percent` to evaluate the model's few-shot forecasting performance. For example, for the few-shot task on 10% training data setting, you can set `--percent 5 \` in your experimental script (only for long-term forecasting task).

**Zero-shot forecasting:** ??

## 📈 Train and forecast:


## 🌟 Acknowledgement:
