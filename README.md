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

# Version 0.3.5
Some discriminator revision, added batch normalization and additional layer. Minimal revision to classifier trainer code. It now exports the classifier after each epoch.
Splitted classifier/GAN training phases. Moreover, they are now skipable. 35 epoch classifier training:
Precision: 0.7195336983478448
Recall: 0.7018
F1-Score: 0.6911505132916511

# Version 0.4
When transfer learning seemed not improving for a while, a second file contains a typical conditional-NN has been added.
It will mainly be used for understanding where the main problem happens for inability to train transfer NN.

# Version 0.5
GPU accelerator added. Gan monitoring revised simpler but effective approach. Monitor prints sample images after each epoch, with it's labels.

# Version 0.6 - 0.6.5
New generator & Discriminator schemes. Switched to the GPU training. 50 epoch training gave some meaningful images for labels but d_loss : 0.27 , g_loss : 2.5 . They are somewhat not ideal for a proper gan.
Simple image generator program script added. It gives 256*256 image samples with 0 to 0 user input.