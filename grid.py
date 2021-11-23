'''
    Functionality to print images from bitmap arrays 11-23-21
'''

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import pytesseract
import re
import numpy as np

pytesseract.pytesseract.tesseract_cmd = 'C:/Users/whomola/AppData/Local/Programs/Tesseract-OCR/tesseract'

# w, h = 512, 512
# data = np.zeros((h,w,3), dtype=np.uint8)
# A red dot in image center
# data[256, 256] = [255, 0, 0]
# A blue square in top left quadrant
# data[0:256, 0:256] = [0, 0, 255]


def pildraw(): 
    img = Image.fromarray(data, 'RGB')
    img.save('test.png')
    img.show()
    
def mpldraw(data): 
    plt.imshow(data, cmap='Greys', interpolation='nearest')
    plt.show()
   
def readpicture(filename): 
    return pytesseract.image_to_string(Image.open(filename))
    
def getcoords(text): 
    # Find coordinate pairs in the OCR'd text.
    coord_pattern = re.compile(r'\(.*?,.*?\)')
    coord_array = coord_pattern.findall(text)
    # The above is an array of coordinate strings.  We need tuples
    coords = []
    letter = re.compile(".*[a-zA-Z].*")
    for i in coord_array: 
        # Make sure the findall didn't let in any strings with characters in them
        if not re.search(letter, i): 
            coords.append(eval(i))     
    coords = np.array(coords)
    # Since there are negative coordinates, shift everything up into the rop right quadrant.
    coords += np.array((25, 25))
    # coords = np.transpose(coords)
    return coords
    
def getfullcoords(coords, h, w): 
    data = np.zeros((h, w), dtype=np.uint8)
    for i in range(max(h,w)): 
        for j in range(max(h,w)):
            # For mysterious reasons, x in array gives True for any x for numpy arrays
            # The following working syntax is from https://stackoverflow.com/questions/33217660/checking-if-a-numpy-array-contains-another-array/33218744
            if (np.array([i, j]) == coords).all(1).any(): 
                # print(np.array((i,j)))
                data[i][j] = 1
    return data
    

# At present, this draws a turkey
coords = getcoords(readpicture('coords.jpg'))
fullcoords = getfullcoords(coords, 50, 50)
mpldraw(fullcoords)