
#from keras.applications.resnet import  ResNet152
from torchvision import models
import torch.nn.modules
md=models.resnet152(pretrained=True)
#print(md._modules.items())

def pr(module):
	print(type(module))

for module_pos, module in md._modules.items():
#	if(isinstance(module,torch.nn.modules.container.Sequential)):
#		for i,j in module:
#			print(module)
	module.apply(pr)
