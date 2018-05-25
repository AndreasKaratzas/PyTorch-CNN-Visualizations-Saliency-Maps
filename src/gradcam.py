"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


from misc_functions import get_params, save_class_activation_on_image


class CamExtractor():
    """
        Extracts cam features from the model
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.flag=0
        self.conv_output=None
        self.k=0

    def save_gradient(self,module,gradin,gradout):


        self.gradients = gradout[0]


    def save_conv(self,module, input, output):

        self.conv_output=output

    def set_hooks(self):
        ''' iterates through seqeuntial and module list using reccursion to reach required layers and assign hook functions to compute
        gradient and save output at the layer '''

        if hasattr(self.model, 'features'):

            for module_pos, module in self.model.features._modules.items():

                def reach_layer(m):
                    if type(m)==torch.nn.modules.conv.Conv2d or type(m)==torch.nn.modules.batchnorm.BatchNorm2d or type(m)== torch.nn.modules.activation.ReLU or type(m)==torch.nn.modules.pooling.MaxPool2d:

                        if self.k == self.target_layer:
                            m.register_forward_hook(self.save_conv)
                            m.register_backward_hook(self.save_gradient)
                        self.k+=1
                    else:
                        try:
                            for i in m.children():
                                reach_layer(i)
                        except:
                            for i in len(m):
                                reach_layer(m[i])

                reach_layer(module)

        else:
            for module_pos, module in self.model._modules.items():

                def reach_layer(m):
                    if type(m)==torch.nn.modules.conv.Conv2d or type(m)==torch.nn.modules.batchnorm.BatchNorm2d or type(m)== torch.nn.modules.activation.ReLU or type(m)==torch.nn.modules.pooling.MaxPool2d:

                        if self.k == self.target_layer:
                            m.register_forward_hook(self.save_conv)
                            m.register_backward_hook(self.save_gradient)
                        self.k+=1
                    else:
                        try:
                            for i in m.children():
                                reach_layer(i)
                        except:
                            for i in len(m):
                                reach_layer(m[i])

                reach_layer(module)


    


    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        self.set_hooks()
        x = self.model(x)
        
        return self.conv_output, x


class GradCam():
    """
        Produces class activation map

       """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None,size=[224,224]):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        #print('output shapes',model_output.shape,conv_output.shape)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        try:
            self.model.features.zero_grad()
            self.model.classifier.zero_grad()
        except:
            self.model.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients

        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        #print(weights.shape)
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        #print('cam-0',cam.shape)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        #print('cam',cam)
        cam = cv2.resize(cam, (size[0], size[1]))
        #print(cam)
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam)+ 10**-6)  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        return cam


if __name__ == '__main__':
    # Get params
    target_example = 2  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_params(target_example)
    # Grad cam
    grad_cam = GradCam(pretrained_model, target_layer=10)
    # Generate cam mask
    cam = grad_cam.generate_cam(prep_img, target_class)
    # Save mask
    save_class_activation_on_image(original_image, cam, file_name_to_export)
    print('Grad cam completed')
