from misc_functions import (get_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            preprocess_image,
                            save_class_activation_on_image
                            )
import cv2
import matplotlib.pyplot as plt

from guided_gradcam import guided_grad_cam
from gradcam import GradCam
from guided_backprop import GuidedBackprop
from torchvision import models
from torchsummary import summary



def vis_grad(model,class_index,layer,image_path,req_max_pool_at_end=False):
	original_image=cv2.imread(image_path,1)
	#plt.imshow(original_image)
	#plt.show()
	prep_img=    prep_img = preprocess_image(original_image)
	file_name_to_export ='model'+'_classindex_'+str(class_index)+'-layer_'+ str(layer)
	print(prep_img.shape)


    # Grad cam
	gcv2 = GradCam(model, target_layer=layer,req_max_pool_at_end=req_max_pool_at_end)
	# Generate cam mask
	cam = gcv2.generate_cam(prep_img, class_index)
	print('Grad cam completed')

    # Guided backprop
	GBP = GuidedBackprop(model)
	# Get gradients
	guided_grads = GBP.generate_gradients(prep_img, class_index)
	print('Guided backpropagation completed')

    # Guided Grad cam
	cam_gb = guided_grad_cam(cam, guided_grads)
	save_gradient_images(cam_gb, file_name_to_export + '_GGrad_Cam')
	grayscale_cam_gb = convert_to_grayscale(cam_gb)
	save_gradient_images(grayscale_cam_gb, file_name_to_export + '_GGrad_Cam_gray')
	print('Guided grad cam completed')

def vis_gradcam(model,class_index,layer,image_path,req_max_pool_at_end=False):


	original_image=cv2.imread(image_path,1)
	#plt.imshow(original_image)
	#plt.show()
	prep_img = preprocess_image(original_image)
	file_name_to_export ='model'+'_classindex_'+str(class_index)+'-layer_'+ str(layer)
	print(prep_img.shape)


    # Grad cam
	gcv2 = GradCam(model, target_layer=layer,req_max_pool_at_end=req_max_pool_at_end)
	# Generate cam mask
	cam = gcv2.generate_cam(prep_img, class_index)
	print('Grad cam completed')

	save_class_activation_on_image(original_image, cam, file_name_to_export)


if __name__ == '__main__':
	md=models.alexnet(pretrained=True)
	md2=models.densenet121(pretrained=True)
	md3=models.resnet152(pretrained=True)
	#print(str(md))
	#print(summary(md,input_size=(3,224,224)))
	#print(dir(md))
	#vis_grad(md2,56,6,'../input_images/snake.jpg')
	vis_gradcam(md3,56,6,'../input_images/snake.jpg',True)




