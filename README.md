# SI-AI-FGSM
> 🚧 WIP

Boosting adversarial attack with AdaGrad, AdaDelta, RMSProp, Adam and more... 
### Requirements 
* Python 3.6.5
* Tensorflow 1.12.0
* Numpy 1.15.4
* opencv-python 3.4.2.16
* scipy 1.1.0

### Experiments
Download the  [data](https://drive.google.com/open?id=1CfobY6i8BfqfWPHL31FKFDipNjqWwAhS) and [pretrained models](https://drive.google.com/open?id=10cFNVEhLpCatwECA6SPB-2g0q5zZyfaw)

#### Running
* `python agi_fgsm.py` to generate adversarial examples for inception_v3 using AGI-FGSM;
* `python ri_fgsm.py` to generate adversarial examples for inception_v3 using RI-FGSM;
* `python ai_fgsm.py` to generate adversarial examples for inception_v3 using AI-FGSM;
* `python si_agi_fgsm.py` to generate adversarial examples for inception_v3 using SI-AGI-FGSM;
* `python si_ri_fgsm.py` to generate adversarial examples for inception_v3 using SI-RI-FGSM;
* `python si_ai_fgsm.py` to generate adversarial examples for inception_v3 using SI-AI-FGSM;
* `python si_agi_ti_dim.py` to generate adversarial examples for inception_v3 using SI-AGI-TI-DIM;
* `python si_ri_ti_dim.py` to generate adversarial examples for inception_v3 using SI-RI-TI-DIM;
* `python si_ai_ti_dim.py` to generate adversarial examples for inception_v3 using SI-AI-TI-DIM;
* `python simple_eval.py`:  evaluate the attack success rate under 8 models including normal training models and adversarial training models.

### Acknowledgements
Code refers to [NAG attack](https://github.com/JHL-HUST/SI-NI-FGSM)
