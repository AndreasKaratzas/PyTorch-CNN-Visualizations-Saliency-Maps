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
import numpy as np 
import os


def act_on_img(org_img, activation_map,size):

    activation_heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_HSV)

    org_img = cv2.resize(org_img, (size[0], size[1]))
    img_with_heatmap = np.float32(activation_heatmap) + np.float32(org_img)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    return img_with_heatmap, activation_heatmap

def trans(gradient):
	gradient = gradient - gradient.min()
	gradient /= gradient.max()
	gradient = np.uint8(gradient * 255).transpose(1, 2, 0)
	return(gradient)
   




def vis_grad(model,class_index,layer,image_path,size=[224,224]):
	original_image=cv2.imread(image_path,1)
	#plt.imshow(original_image)
	#plt.show()
	prep_img = preprocess_image(original_image,size)
	file_name_to_export ='model'+'_classindex_'+str(class_index)+'-layer_'+ str(layer)


    # Grad cam
	gcv2 = GradCam(model, target_layer=layer)
	# Generate cam mask
	cam = gcv2.generate_cam(prep_img, class_index,size)
	print('Grad cam completed')

    # Guided backprop
	GBP = GuidedBackprop(model)
	# Get gradients
	guided_grads = GBP.generate_gradients(prep_img, class_index)
	print('Guided backpropagation completed')

    # Guided Grad cam
	cam_gb = guided_grad_cam(cam, guided_grads)
	#save_gradient_images(cam_gb, file_name_to_export + '_GGrad_Cam')
	grayscale_cam_gb = convert_to_grayscale(cam_gb)
	#save_gradient_images(grayscale_cam_gb, file_name_to_export + '_GGrad_Cam_gray')
	print('Guided grad cam completed')
	cam_gb=trans(cam_gb)
	grayscale_cam_gb=trans(grayscale_cam_gb)



	return cam_gb,grayscale_cam_gb

def vis_gradcam(model,class_index,layer,image_path,size=[224,224]):


	original_image=cv2.imread(image_path,1)
	#plt.imshow(original_image)
	#plt.show()
	prep_img = preprocess_image(original_image,size)
	file_name_to_export ='model'+'_classindex_'+str(class_index)+'-layer_'+ str(layer)


    # Grad cam
	gcv2 = GradCam(model, target_layer=layer)
	# Generate cam mask
	cam = gcv2.generate_cam(prep_img, class_index,size)
	print('Grad cam completed')

	#save_class_activation_on_image(original_image, cam, file_name_to_export)
	img_with_heatmap,activation_heatmap=act_on_img(original_image,cam,size)

	return cam,activation_heatmap,img_with_heatmap

def model_compare(listm,class_index,layer,image_path):

	grey=[]
	grad=[]
	name=[]
	if not os.path.exists('../results'):
		os.makedirs('../results')
	s=''


	for i in listm :

		x,y=vis_grad(i[0],class_index,layer,image_path,i[2])
		grad.append(x)
		grey.append(y)
		name.append(i[1])

	for i in name:
		s=s+'__'+i
	file_name1='guided_gradcam'+s+'_classindex_'+str(class_index)+'-layer_'+ str(layer)
	file_name2='guided_gradcam_grey'+s+'_classindex_'+str(class_index)+'-layer_'+ str(layer)

	path_to_file1 = os.path.join('../results', file_name1 + '.png')
	path_to_file2=os.path.join('../results', file_name2 + '.png')

	fig=plt.figure(figsize=(15, 15))
	fig.suptitle('guided_gradcam', fontsize=20)

	for i in range(0, len(grey)):
		img = grad[i]
		b=fig.add_subplot(1, len(name), i+1)
		plt.imshow(img)
		b.title.set_text(name[i])
		
	
	plt.show()
	fig.savefig(path_to_file1)

	fig=plt.figure(figsize=(15,15))
	fig.suptitle('guided_gradcam_grey', fontsize=20)

	for i in range(0, len(grey)):
		b=fig.add_subplot(1, len(name), i+1)
		b.title.set_text(name[i])
		c=grey[i].reshape((grey[i].shape[0],grey[i].shape[1]))
		plt.imshow(c,cmap='gray')
	
	plt.show()
	fig.savefig(path_to_file2)


def model_compare_cam(listm,class_index,layer,image_path):

	cam=[]
	hmap=[]
	ihmap=[]
	name=[]

	if not os.path.exists('../results'):
		os.makedirs('../results')
	s=''


	
	for i in listm :

		x,y,z=vis_gradcam(i[0],class_index,layer,image_path,i[2])
		cam.append(x)
		hmap.append(y)
		ihmap.append(z)
		name.append(i[1])

	for i in name:
		s=s+'__'+i
	file_name1='gradcam'+s+'_classindex_'+str(class_index)+'-layer_'+ str(layer)
	file_name2='gradcam_heatmap'+s+'_classindex_'+str(class_index)+'-layer_'+ str(layer)
	file_name3='Image_gradcam_heatmap'+s+'_classindex_'+str(class_index)+'-layer_'+ str(layer)

	path_to_file1 = os.path.join('../results', file_name1 + '.png')
	path_to_file2=os.path.join('../results', file_name2 + '.png')
	path_to_file3=os.path.join('../results', file_name3 + '.png')


	fig=plt.figure(figsize=(15,15))
	fig.suptitle('gradcam', fontsize=20)

	for i in range(0, len(cam)):
		img = cam[i]
		b=fig.add_subplot(1, len(name), i+1)
		plt.imshow(img)
		b.title.set_text(name[i])
		
	
	plt.show()
	fig.savefig(path_to_file1)

	fig=plt.figure(figsize=(15, 15))
	fig.suptitle('gradcam-heatmap', fontsize=20)


	for i in range(0, len(cam)):
		img = hmap[i]
		b=fig.add_subplot(1, len(name), i+1)
		plt.imshow(img)
		b.title.set_text(name[i])

	plt.show()
	fig.savefig(path_to_file2)

	fig=plt.figure(figsize=(15,15))
	fig.suptitle('gradcam-heatmap-onimage', fontsize=20)


	for i in range(0, len(cam)):
		img = ihmap[i]
		b=fig.add_subplot(1, len(name), i+1)
		plt.imshow(img)
		b.title.set_text(name[i])

	plt.show()
	fig.savefig(path_to_file3)





if __name__ == '__main__':
	md=models.alexnet(pretrained=True)
	md2=models.densenet121(pretrained=True)
	md3=models.resnet152(pretrained=True)

	#print(str(md))
	#print(summary(md,input_size=(3,224,224)))
	#print(dir(md))
	#vis_grad(md2,56,6,'../input_images/snake.jpg')
	list=[[md,'alexnet',[224,224]]]

	#vis_gradcam(md3,56,6,'../input_images/snake.jpg',True)
	#model_compare(list,56,6,'../input_images/snake.jpg')

	model_compare(list,56,6,'../input_images/snake.jpg')




