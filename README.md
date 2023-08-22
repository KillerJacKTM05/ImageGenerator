## ImageGenerator
Image generator GAN project that uses Cifar-10 dataset (for more information: "https://www.cs.toronto.edu/~kriz/cifar.html") as generator model training.
You can enter prompts for using this program. If your prompts are recognizable, GAN will create sample images.

# Version 0.1
Training .py prototype

# Version 0.2
Revised classifier, generator, and discriminator. Because the classifier was worse enough to make an additional approach. Also, there was a conflict between the input sizes of the discriminator and generator.
The test folder of cifar-10 is not in use now. Because of the lack of labeling on the test folder, we could not check whether the assumption was correct.
90-10 split has been done to train images.

# Version 0.3
Classifier needed another revision because it's accuracy capped on approx. 50% on recent trainings. More and incrementing conv layers added, dropouts organized.
New approach shown it's benefits and validation accuracy reached 70% on my pretrainings.
20 epoch weighted score
Precision: 0.7215832630947135
Recall: 0.7184
F1-Score: 0.7088958283317461
