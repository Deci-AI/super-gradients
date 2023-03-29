In this tutorial we will guide you step by step on how to prepare  our DeciYolo to production!
We will leverage DeciYolos architecture which includes quantization friendly blocks, and train a deci yolo model on XXX
in a way that would maximize our throughput without comprimising on the model's accuracy.

The steps will be:
1. Training from scratch on one of the downstream datasets- these will play the role of the users dataset (i.e the one which the model will need to be trained for the user's task)
2. Performing post training quantization
3. Performing quantization aware training

Background:
**Add intro about qat, ptq and QARepvgg blocks

Now, lets get to it.

First, some installations:

```shell

```
