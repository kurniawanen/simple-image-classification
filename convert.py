from skimage import io;
from skimage import img_as_float;
from scipy.misc import imresize as resize;
from learning import train,predict
import os

def convert_to_ndarray(image,size):
	#image: string, size: tuple[x,y]
	#read as greyscale and then convert uint16 to float
	arr 		= img_as_float(io.imread(image,as_grey = True));
	#self explain
	avg 		= arr.mean();
	mx 			= arr.max();
	mn 			= arr.min();
	threshold 	= ((3*avg) + mx) / 4;
	#if value>threshold 1 else 0, python magic :v
	arr			= (arr > threshold).astype(float);
	#resize to 56x56
	arr 		= resize(arr,size);
	arr 		= arr.astype(float);
	#normalize to 0 - 1
	arr 		*= 1.0/arr.max();
	#flatten to 1 dimensional array[3136]
	arr 		= arr.flatten(); 
	return arr;

def train_all_image_in_folder(foldername,status):
	#foldername: string, status: one hot [0,1] (yes) or [1,0] (no)
	trlabel = [];
	trfeat = [];
	#input all image file in folder
	for file in os.listdir(foldername):
		if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
			train_array = convert_to_ndarray(foldername + '/' + file,[56,56]);
			trfeat.append(train_array);
			trlabel.append(status);
	train(trfeat, trlabel);
	
def predict_image(image):
	test 	= [];
	array 	= convert_to_ndarray(image,[56,56]);
	test.append(array);
	predict(test);