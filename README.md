## ImageGenerator
Image generator GAN project that uses Cifar-10 dataset (for more information: "https://www.cs.toronto.edu/~kriz/cifar.html") as generator model training.
You can enter prompts for using this program. If your prompts are recognizable, GAN will create sample images.

# Version 0.1
Training .py prototype

#Version 0.2
Revised classifier, generator and discriminator. Because classifier was worse enough to make additional approach. Also, there was an conflict between the input sizes of discriminator and generator.
Test folder of cifar-10 is not in usage now. Because of the lack of labelling on the test folder, we could not check whether the assumption is correct or not.
90-10 split has been done to train images.