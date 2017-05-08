# simple-image-classification

requirement:

python 3.5

scikit-image

tensorflow

numpy

usage:

1. create 2 folder "yes" and "no" (or anything)

2. copy all intended image to "yes", and all other image to "no"

3. run:

      import convert
      
      convert.train_all_image_in_folder('yes',[0,1])
      
      convert.train_all_image_in_folder('no',[1,0])

4. you can predict image using convert.predict_image(image) e.g. convert.predict_image(example.png), the result will be 1 if it's the intended image or 0 if isn't.
      
