from skimage import io;
from skimage import img_as_float;
from scipy.misc import imresize as resize;
from learning import train,predict
import os

def convert_to_ndarray(image,size):
	#image: string, size: tuple[x,y]
	arr 		= img_as_float(io.imread(image,as_grey = True));
	avg 		= arr.mean();
	mx 			= arr.max();
	mn 			= arr.min();
	threshold 	= ((3*avg) + mx) / 4;
	arr			= (arr > threshold).astype(float);
	arr 		= resize(arr,[56,56]);
	arr 		= arr.astype(float);
	arr 		*= 1.0/arr.max();
	return arr;

def train_all_image_in_folder(foldername,status):
	#foldername: string, status: one hot [0,1,0,0]
	trlabel = [];
	trfeat = [];
	for file in os.listdir(foldername):
		if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
			train_array = convert_to_ndarray(foldername + '/' + file,[56,56]);
			trfeat.append(train_array.flatten());
			trlabel.append(status);
	train(trfeat, trlabel);
	
def predict_image(image):
	test = [];
	train_array = convert_to_ndarray(image,[56,56]);
	test.append(train_array.flatten());
	predict(test);