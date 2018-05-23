
#from keras.applications.resnet import  ResNet152
from torchvision import models
import torch.nn.modules
md=models.densenet121(pretrained=True)
#print(md._modules.items())

def pr(module):
	print(type(module))

k=0

print(md.features._modules.items())
for module_pos, module in md.features._modules.items():
#	if(isinstance(module,torch.nn.modules.container.Sequential)):
#		for i,j in module:
#			print(module)
	def reach_layer(m):
		global k
		if type(m)==torch.nn.modules.conv.Conv2d or type(m)==torch.nn.modules.batchnorm.BatchNorm2d or type(m)== torch.nn.modules.activation.ReLU or type(m)==torch.nn.modules.pooling.MaxPool2d:
			print('layer')
			print(k)

			print(type(m))
			k+=1
		else:
			try:
				for i in m.children():
					#print('a')
					#print('',type(i))
					reach_layer(i)
			except:
				for i in len(m):
					#print('abc')
					#print('module',type(m[i]))
					reach_layer(m[i])

	reach_layer(module)
	#print(type(module))
print(k)

#print(k.children())