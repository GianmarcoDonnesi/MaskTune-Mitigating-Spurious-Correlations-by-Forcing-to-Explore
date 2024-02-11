<h1>MaskTune: Mitigating Spurious Correlations by Forcing to Explore</h1>
This repository contains the code developed for the Neural Networks exam project. The work, based on reworking the solution proposed in the original paper titled `MaskTune: Mitigating Spurious Correlations by Forcing to Explore`, focused on investigating and applying strategies to mitigate spurious correlations in deep learning models, drawing inspiration from the recent development of a method called MaskTune. This represents an innovative approach to addressing a fundamental challenge in overparameterized models (i.e., models with a very high number of parameters compared to the number of examples in the training dataset): learning meaningful data representations that yield good performance on a downstream task without overfitting to spurious input features.

<br>
</br>

<h1>How to use</h1>

1. Clone the code.
2. For the `CIFAR-10` dataset, there's no need to load it manually, as it is downloaded and saved in the project's "data" folder automatically.
3. To load and train the `VGG` model on the CIFAR-10 dataset, you need to run the `vgg.py` script. Additionally, you can modify parameters such as `batch_size`, `num_epochs`, `learning_rate`, and `momentum` to customize the training procedure according to your preferences.
4. To evaluate the accuracy of the VGG model before and after fine-tuning, you can launch the `evaluate_vgg.py` script, which provides the network's accuracy on the entire dataset and on each individual class.
5. To run the selective classification task for the VGG model, you need to execute the script `selective_classification_vgg.py`, which evaluates the network's accuracy on the testing dataset.
6. For `CelebA` dataset, please download `img_align_celeba` folder from <a href="https://www.kaggle.com/datasets/jessicali9530/celeba-dataset?resource=download-directory">here</a>. After extracting it, you should see a folder named `archive`. Please cut and paste this folder into the project's `data/celeba` directory to allow the network to load the dataset correctly.
7. To load and train the `AttentionMaskingResNet50` model on the `CelebA` dataset, you need to run the `resnet50.py` script. Additionally, you can modify parameters such as `batch_size`, `num_epochs`, `learning_rate`, `momentum` and `num_classes`to customize the training procedure according to your preferences. To train the model on a subset of the complete dataset, for computational resource availability reasons, you can modify the size of the subset by changing the variable `subset_size`.
8. To run the selective classification task for the AttentionMaskingResNet50 model, you need to execute the script `selective_classification_resnet.py`, which evaluates the network's accuracy on the testing dataset.
