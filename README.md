<h1>MaskTune: Mitigating Spurious Correlations by Forcing to Explore</h1>
This repository hosts the code developed for the Neural Networks exam project. The task, based on reworking the solution proposed in the original paper titled `MaskTune: Mitigating Spurious Correlations by Forcing to Explore`, centered on exploring and implementing techniques to reduce the impact of false correlations in deep learning architectures.

<br>
</br>

**Autor:**  
Gianmarco Donnesi  
Matr. n. 2152311

**Supervisor:**  
Prof. Scardapane

## Abstract
The work analyzed in this report focused on investigating and applying strategies to mitigate spurious correlations in deep learning models, inspired by the recent development of a method called MaskTune. This represents an innovative approach to tackle a fundamental challenge in over-parametrized models (i.e., models that have a very high number of parameters compared to the number of examples in the training dataset): learning meaningful data representations that produce good performance on a downstream task without overfitting to spurious input features. This method proposes a masking strategy that prevents excessive dependency on a limited number of features, forcing the model to explore new ones by masking those previously discovered. To do this, masking is applied during the fine-tuning of a single epoch. This is a technique for adapting a pre-trained model to a new task or dataset, which allows leveraging the model’s existing knowledge, reducing the time and computational resources needed for training, and improving performance on specific tasks compared to training a model from scratch. Finally, an addi- tional selective classification task was implemented, exploiting MaskTune’s ability to promote the learning of more robust representations less dependent on potentially misleading or unreliable features. This would allow the model to recognize situations where the main informative features are absent or masked, opting to abstain from classification rather than risking an inaccurate prediction. To measure the effectiveness of MaskTune in the selective classification task, specific metrics were used to evaluate both the accuracy of the predictions, when the model decides to make them, and the ability to abstain from making decisions when the available information is not sufficient for a reliable prediction.

<h1>How to use</h1>

1. Clone the code.
2. To install the Python packages listed in `requirements.txt` and used within the project's code, open the terminal and execute the following command `pip install -r requirements.txt`.
3. For the `CIFAR-10` dataset, there's no need to load it manually, as it is downloaded and saved in the project's "data" folder automatically.
4. To load and train the `VGG` model on the CIFAR-10 dataset, you need to run the `vgg.py` script. Additionally, you can modify parameters such as `batch_size`, `num_epochs`, `learning_rate`, and `momentum` to customize the training procedure according to your preferences.
5. To evaluate the accuracy of the VGG model before and after fine-tuning, you can launch the `evaluate_vgg.py` script, which provides the network's accuracy on the entire dataset and on each individual class.
6. To run the selective classification task for the VGG model, you need to execute the script `selective_classification_vgg.py`, which evaluates the network's accuracy on the testing dataset.
7. For `CelebA` dataset, please download `img_align_celeba` folder from <a href="https://www.kaggle.com/datasets/jessicali9530/celeba-dataset?resource=download-directory">this link</a>. After extracting it, you should see a folder named `archive`. Please cut and paste this folder into the project's `data/celeba` directory to allow the network to load the dataset correctly.
8. To load and train the `AttentionMaskingResNet50` model on the `CelebA` dataset, you need to run the `resnet50.py` script. Additionally, you can modify parameters such as `batch_size`, `num_epochs`, `learning_rate`, `momentum` and `num_classes`to customize the training procedure according to your preferences. To train the model on a subset of the complete dataset, for computational resource availability reasons, you can modify the size of the subset by changing the variable `subset_size`.
9. To run the selective classification task for the AttentionMaskingResNet50 model, you need to execute the script `selective_classification_resnet.py`, which evaluates the network's accuracy on the testing dataset.
