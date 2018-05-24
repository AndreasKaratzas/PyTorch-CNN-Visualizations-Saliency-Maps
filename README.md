
# Demo for visualizing CNNs using Grad_cam and Guided_Grad_Gam

###### vis_grad file contains model_compare function which is used to visualize guided_gradcam_back_prop and model_compare_cam perfroms grad_cam


```python
from vis_grad import model_compare_cam , model_compare
```

###### import pretrained models using torch vision models (custom models can be used)


```python
from torchvision import models
```

###### using 3 models , alex net , dense net 121 and resnet 152 


```python
md=models.alexnet(pretrained=True)
md2=models.densenet121(pretrained=True)
md3=models.resnet152(pretrained=True)
```

###### create a list containing (model,if_pool_requried_before_last_layer,'model name to print') for each model 


```python
list=[[md,False,'alexnet'],[md2,True,'densenet121'],[md3,True,'resnet152']]
```

###### pass the list , class number , layer to visualize , input_image to visualize on 


```python
model_compare(list,497,6,'../input_images/church.jpg')
```

    Grad cam completed
    Guided backpropagation completed
    Guided grad cam completed
    Grad cam completed
    Guided backpropagation completed
    Guided grad cam completed
    Grad cam completed
    Guided backpropagation completed
    Guided grad cam completed
    


![png](https://github.com/TejaGollapudi/pytorch-cnn-visualizations/blob/master/display/output_10_1.png?raw=true)



![png](https://github.com/TejaGollapudi/pytorch-cnn-visualizations/blob/master/display/output_10_2.png?raw=true)


###### Images are automatically saved in result folder

###### For visualizing grad_cam


```python
model_compare_cam(list,497,6,'../input_images/church.jpg')
```

    Grad cam completed
    Grad cam completed
    Grad cam completed
    


![png](https://github.com/TejaGollapudi/pytorch-cnn-visualizations/blob/master/display/output_13_1.png?raw=true)



![png](https://github.com/TejaGollapudi/pytorch-cnn-visualizations/blob/master/display/output_13_2.png?raw=true)



![png](https://github.com/TejaGollapudi/pytorch-cnn-visualizations/blob/master/display/output_13_3.png?raw=true)

