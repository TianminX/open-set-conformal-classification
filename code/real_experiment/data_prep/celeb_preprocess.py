import numpy as np
import sys
import os
import cv2

from PIL import Image 
from tqdm import tqdm
from numpy import savez_compressed
from mtcnn.mtcnn import MTCNN
from pathlib import Path


sys.path.insert(0, '../third_party/keras-facenet/code/')
import inception_resnet_v1


#########################
# Experiment parameters #
#########################
if True:
    # Parse input arguments
    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))
    if len(sys.argv) != 3:
        print("Error: incorrect number of parameters.")
        quit()

    batch_num = int(sys.argv[1])
    batch_size = int(sys.argv[2])


# Fixed data parameters
model_path = "../third_party/keras-facenet/model/facenet_keras.h5"
file_path = "../../data/celebrity/identity_CelebA.txt"
image_dir = "../../data/celebrity/img_align_celeba/"
out_dir = "../../data/celebrity/embeddings/"

Path(out_dir).mkdir(parents=True, exist_ok=True)



#####################
# Helper functions  #
#####################

def extract_image(image, img_size=160):
    img1 = Image.open(image)            #open the image
    img1 = img1.convert('RGB')          #convert the image to RGB format 
    pixels = np.asarray(img1)           #convert the image to numpy array
    detector = MTCNN()         #assign the MTCNN detector
    f = detector.detect_faces(pixels)
    #fetching the (x,y)co-ordinate and (width-->w, height-->h) of the image
    if len(f) == 0:
        face_array = []
    else:
        x1,y1,w,h = f[0]['box']             
        x1, y1 = abs(x1), abs(y1)
        x2 = abs(x1+w)
        y2 = abs(y1+h)
        #locate the co-ordinates of face in the image
        store_face = pixels[y1:y2,x1:x2]
        image1 = Image.fromarray(store_face,'RGB')    #convert the numpy array to object
        image1 = image1.resize((img_size, img_size))  #resize the image
        face_array = np.asarray(image1)               #image to array
    return face_array

        
def batch_process(image_dir, file_path, batch_num, batch_size):
    lines = []
    with open(file_path, 'r') as file:
        for line in file: 
            lines.append(line)
    print(f"Total number of images: {len(lines)}, batch size: {batch_size},",\
          f"processing #{batch_num} of the {int(np.ceil(len(lines)/batch_size))} batches.")       
    sys.stdout.flush()
    batch_lines = lines[batch_size*(batch_num-1): batch_size*batch_num]
    batch_paths = []
    batch_labels = []
    batch_names = []

    for line in batch_lines:
        image_name, label = line.split()
        image_path = os.path.join(image_dir, image_name)
        batch_paths.append(image_path)
        batch_labels.append(int(label))
        batch_names.append(image_name)

    return batch_paths, batch_labels, batch_names
    

def load_dataset(file_path, image_dir, batch_num, batch_size):
    x = []
    batch_paths, y, names = batch_process(image_dir, file_path, batch_num, batch_size)
    for i, image_path in enumerate(tqdm(batch_paths)):
        # Split the line into image name and label
        face = extract_image(image_path)
        if len(face) == 0:
            y.pop(i)
            names.pop(i)
        else:
            x.append(face)
    
    return np.asarray(x), np.asarray(y), np.asarray(names)


def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y


def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


def calculate_embs(model, aligned_images, batch_size=1):
    aligned_images = prewhiten(aligned_images)
    pd = []
    for start in tqdm(range(0, len(aligned_images), batch_size)):
        pd.append(model.predict_on_batch(aligned_images[start:start+batch_size]))
    embs = l2_normalize(np.concatenate(pd))

    return embs



########################
# Processing pipelines #
########################

def preprocess(file_path, image_dir, model_path, batch_num, batch_size, out_dir):
    #load the facenet keras model
    model = inception_resnet_v1.InceptionResNetV1()
    model.load_weights(model_path)

    X, Y, image_names = load_dataset(file_path, image_dir, batch_num, batch_size)
    embs = calculate_embs(model, X)

    outfile = os.path.join(out_dir, "batch"+str(batch_num))
    savez_compressed(outfile, X=embs, Y=Y, image_name=image_names)

    print(f"Preprocess for batch {batch_num} complete! Data saved to {outfile}.")
    sys.stdout.flush()

    return

preprocess(file_path, image_dir, model_path, batch_num, batch_size, out_dir)

