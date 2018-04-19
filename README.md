
FReLU: Flexible Rectified Linear Unit
============================

This project is the original implementation of [FReLU: Flexible Rectified Linear Units for Improving Convolutional Neural Networks](https://arxiv.org/abs/1706.08098) in Torch.

This project is also a clone of [Facebook ResNet implementation using ReLU](https://github.com/facebook/fb.resnet.torch) in Torch.

Other implementations (many thanks for the contributors):
+ [FReLU in caffe](https://github.com/ducha-aiki/caffenet-benchmark/blob/master/Activations.md) by Dmytro Mishkin.

This is my first time to share the experimental codes. It may be a little messy. Any comments/pull requests are appreciated. 

## FAQs

We list some questions about the method here. Some come from the [disscussion on reddit](https://www.reddit.com/r/MachineLearning/comments/6qu57t/r_170608098_flexible_rectified_linear_units_for/?st=jg56gd2o&sh=0335b581#dl1xi5u).

+ The purpose of this work ?
	
	We wanna figure out: 1) the effects of negative values for networks, 2) the compatibility between activation functions and batch normalization. 

+ Layer-wise biases or channel-wise biases ?

	FReLU uses layer-wise biases. We do not suggest to use channel-wise biases, which may be easy to make the training harder. 

+ Compare to SELU ?

	The bias in FReLU is not a constant value. We think it is hard to say which value is the best. 

+ When to use FReLU ?

	FReLU is helpful for smaller networks and is also a good choice when using batch normalization. 

+ Failure cases of FReLU ? 

	By monitoring the value of the bias in FReLU, we observe that positive bias will harm the training. For large networks that have large capacities, FReLU may lose advantages. 

+ About the theory ?

	The current paper gives a little theory analysis. The intuition mainly comes from ELU, normalizing networks and the expressiveness of rectifier networks.

+ Future work ?
	+ Different tasks (e.g. classification & regression) may need different activation functions. The exploration of the task-specific activation function is helpful to understand the corresponding task and network architecture. 
	+ The theory developments about the network architecture and the learning behavior will better guide the design of activation functions. 

We appreciate any comments/disscussions about activation functions. More observations are going to get us closer to the truth. 



## Training

The example training commands are available in the following scripts. Please read the corresponding script before running. More scripts are in the sub folder `scripts`. 

* PReLU: `run-prelu.sh`
* SReLU: `scripts/cifar100-pelu-smallnet-srelu-seed.sh`
* ELU: `run-elu.sh`
* ReLU: `run-relu.sh`
* ReLU with ResNet: `run-resnet.sh`
* FReLU: `run-possrelu.sh`
* FReLU with ResNet: `run-resnet-possrelu.sh`

The implementation of FReLU in torch is `models/frelu/PosSReLU.lua`. 

To monitor the value of biases in FReLU, use `th show.lua -model $your_model_path`.

Codes in `draw` are use to read and plot the curves from log files. 

Run the visulization experiment. Just `cd mnist` and `th *.lua`. FReLU may need several runs. Some initial parameters can lead the dead neuron. 


Model files table :
| File                | Network                | ACT   |
| ------------------- | ---------------------- | ----- |
| resnet-possrelu     | Ori. bottleneck        | FReLU |
| elu-resnet-possrelu | w/o ACT after addition | FReLU |


## Related work

* [Expressiveness of Rectifier Networks](https://arxiv.org/abs/1511.05678v1)
* [Normalization Propagation: A Parametric Technique for Removing Internal Covariate Shift in Deep Networks](https://arxiv.org/abs/1603.01431)
* [FAST AND ACCURATE DEEP NETWORK LEARNING BY EXPONENTIAL LINEAR UNITS (ELUS)](https://arxiv.org/abs/1511.07289)
* [Deep Residual Networks with Exponential Linear Unit](https://arxiv.org/abs/1604.04112)
