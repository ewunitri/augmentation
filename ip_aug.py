# -*- coding: utf-8 -*-
"""
Created on Mon Mar 05 17:36:15 2018

@author: 455495
"""

import cv2
import numpy as np
import os 
from os.path import isfile, join
import re



#-------------------------------------------------------------------------------
#   splitStrWithComma()
#   split the input string by ','. return a string list 
#   if there is one item only, return this single item in a list
#-------------------------------------------------------------------------------
def splitStrWithComma(Str, isUnicode=0):
    if (',' in Str):
        if (isUnicode):
            return [unicode(x.strip(),'UTF-8') for x in Str.split(',')]
        else:
            return [x.strip() for x in Str.split(',')]
    else:
        if (isUnicode):
            return [unicode(Str,'UTF-8')]
        else:
            return [Str]
        


#-------------------------------------------------------------------------------
#   MewImgProcessingAug class
#-------------------------------------------------------------------------------
class NewImgProcessingAug:
    
    def __init__(self):
        self.srcPath = ''
        self.dstPath = ''
        self.substr = ''
        self.srcFiles = []
        self.brightness =[None]
        self.noise = [None]
        self.salt = 255
        self.pepper = 0
        self.blur = [None]
        self.perspective = [None]
        self.rotate = [None]
        self.turnLeft = [None]
        
    def printall(self):
        print self.srcPath 
        print self.substr
        print self.srcFiles
        print self.brightness
        print self.noise
        print self.salt
        print self.pepper
        print self.blur
        print self.perspective
        print self.rotate
        print self.turnLeft
    
    
    
    
    
    
#-------------------------------------------------------------------------------
#   tune_brightness()
#   return an image with adjusting the input image with input value
#-------------------------------------------------------------------------------        
def tune_brightness(img, alpha):
    return cv2.convertScaleAbs(img, -1, alpha, 0)
    



#-------------------------------------------------------------------------------
#   add_noise()
#   return an imagewith adding noise (circles) with the given value.
#   'salt and pepper' is default mode. white/black only
#-------------------------------------------------------------------------------
def add_noise(img, noise_type, salt_color, pepper_color, value = 0.0):
    if noise_type == "gauss":
        row,col,ch= img.shape
        mean = 0
        var = 60
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = img + gauss
        return noisy
    elif noise_type == "s&p":
        np.random.seed(np.random.randint(50,200))
        row,col,ch = img.shape
        s_vs_p = 0.5
        amount = value/25000
        out = np.copy(img)
        # Salt mode
        num_noise = int(amount * img.size * s_vs_p)
        ##- random channel
        #coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
        #out[coords] = 255
        ## all channel
        
        coords_X = [int(x) for x in np.random.randint(img.shape[1], size = num_noise)]
        np.random.shuffle(coords_X)
        coords_Y = [int(x) for x in np.random.randint(img.shape[0], size = num_noise)]
        np.random.shuffle(coords_Y)
        coords = zip(coords_X, coords_Y)
        for i in range(num_noise):
                noise_color = np.random.randint(pepper_color, high = salt_color)
                rand_factor = np.random.randint(0, 255)
                #out[coords[i]] = [255,255,255]
                if (rand_factor & 0x10)==0:  ## one seventh probability to draw line
                    if (rand_factor & 8):
                        x2 = min(col,coords[i][0]+ ((rand_factor & 0x3C)>>2))
                    else:    
                        x2 = max(0, coords[i][0]- ((noise_color & 0x3C)>>2))

                    if (noise_color & 0x20):
                        y2 = min(row,coords[i][1]+ ((noise_color & 0x78)>>3))
                    else:    
                        y2 = max(0,coords[i][1]- ((rand_factor & 0x78)>>3))
                        
                    cv2.line(out, coords[i], (x2,y2), (noise_color,noise_color,noise_color), 1, cv2.LINE_AA)
                else:
                    if (rand_factor & 0x7)==0:
                        cv2.circle(out,coords[i], 4, (noise_color,noise_color,noise_color), -1)
                    else:    
                        cv2.circle(out,coords[i], 2, (noise_color,noise_color,noise_color), -1)
        
        #num_pepper = int(amount* img.size * (1. - s_vs_p))
        #np.random.shuffle(coords_Y)
        #coords_Y = [int(x) for x in np.random.randint(img.shape[1], size = num_pepper)]
        #np.random.shuffle(coords_X)
        #coords_X = [int(x) for x in np.random.randint(img.shape[0], size = num_pepper)]
        #coords = zip(coords_Y, coords_X)
        #for i in range(num_pepper):
        #        #out[coords[i]] = [0,0,0]
        #        cv2.circle(out,coords[i], 2, (0,0,0), -1)
        return out
  
    elif noise_type == "poisson":
        vals = len(np.unique(img))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(img * vals) / float(vals)
        return noisy
    elif noise_type =="speckle":
        row,col,ch = img.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = img + img * gauss
        return noisy





#-------------------------------------------------------------------------------
#   do_blur()
#   return an image with gaussian blur applied to the input image, 
#   input value as parameters of the kernel size
#-------------------------------------------------------------------------------
def do_blur(img, value):
    mode = \
    {
    0: 3,
    1: 5,
    2: 7
    }[value]
    img = cv2.blur(img,(5,5),0)
    return  cv2.blur(img,(mode,mode),0)






#-------------------------------------------------------------------------------
#   do_rotation()
#   return a rotated image with given degree after scaling down (down to 0.8)   
#-------------------------------------------------------------------------------
def do_rotation(img, value):

    cols,rows,_ = img.shape
    #print (0.8+0.2/5.0*(5.0-abs(value)))
    #print value
    #M = cv2.getRotationMatrix2D((rows/2,cols/2),value,0.8+0.2/5.0*(5.0-abs(value)))  ## calculate the scale number by referencing the value
    #print type(value)
    M = cv2.getRotationMatrix2D((rows/2,cols/2),value,0.8+0.05/5.0*(5.0-abs(value)))
    #print ('Rotation: ', str(rows), str(cols))
    
    return  cv2.warpAffine(img,M,(rows,cols))
    







#-------------------------------------------------------------------------------
#   turn_Left()
#   return a rotated image with turning 90 degrees to the left with given times
#-------------------------------------------------------------------------------
def turn_Left(img, value):

    cols,rows,_ = img.shape
    
    M = \
    {   
     1: cv2.getRotationMatrix2D((rows/2,cols/2),90,0.9),
     2: cv2.getRotationMatrix2D((rows/2,cols/2),180,1),
     3: cv2.getRotationMatrix2D((rows/2,cols/2),270,0.9)
     }[value]
    
    return  cv2.warpAffine(img,M,(rows,cols))




#-------------------------------------------------------------------------------
#   do_perspectiveTransform()
#   randomly pick the indent locations of the 4 corners bounded by the given value
#   return an image with indents    
#-------------------------------------------------------------------------------
def do_perspectiveTransform(img, value):

    cols,rows,ch = img.shape
    
    ## left-top X
    a1 = cols*np.random.random_sample()*value   
    ## left-top Y
    a2 = rows*np.random.random_sample()*value
    ## Right-top X
    b1 = cols*np.random.random_sample()*value
    ## Right-top Y
    b2 = rows*np.random.random_sample()*value
    ## left-bottom X
    c1 = cols*np.random.random_sample()*value
    ## left-bottom Y
    c2 = rows*np.random.random_sample()*value
    ## Right-bottom X
    d1 = cols*np.random.random_sample()*value
    ## Right-bottom Y
    d2 = rows*np.random.random_sample()*value

    #print a1, a2, b1, b2
    #print c1, c2, d1, d2
    #pts1 = np.float32([[0,0],[300,0],[0,300],[300,300]])
    pts1 = np.float32([[0,0],[cols,0],[0,rows],[cols,rows]])
    pts2 = np.float32([[a1,a2],[cols-b1,b2],[c1,rows-c2],[cols-d1,rows-d2]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
 
    return cv2.warpPerspective(img,M,(rows,cols)) 


    


#-------------------------------------------------------------------------------
#   add_effects()
#   according to the given collection, add different effects to the files in the specified source path      
#   all results will be output under dstPath        
#-------------------------------------------------------------------------------

def add_effects(collection):
    
    if (collection.noise[0] or collection.blur[0] or collection.brightness[0] or collection.perspective[0] or collection.rotate[0] or collection.turnLeft[0]):

        effects = [[x,y,z,m,n,o] for x in collection.noise for y in collection.blur for z in collection.brightness \
                                 for m in collection.perspective for n in collection.rotate for o in collection.turnLeft]
        
        for img_name in collection.srcFiles:
            
            basename, extname = os.path.splitext(img_name)
            
            for eff in effects:
                img = cv2.imread(join(collection.srcPath,img_name))
                fnStr = collection.dstPath + basename + '_' + collection.substr 

                n_str = blr_str = b_str = ppt_str = rot_str =tl_str = ''
                
                if (eff[0]!=None):
                    n_str = '_n('+str(eff[0])+')'
                    img = add_noise(img, "s&p", collection.salt, collection.pepper, float(eff[0]))
    
                if (eff[1]!=None):
                    blr_str = '_blr('+str(eff[1])+')'
                    img = do_blur(img, eff[1])
                
                if (eff[2]!=None):
                    b_str = '_b('+str(eff[2])+')'
                    if (eff[2]!=1):
                        img = tune_brightness(img, eff[2])
                    
                if (eff[3]!=None):
                    ppt_str = '_ppt('+str(eff[3])+')'
                    if (eff[4]!=1):
                        img = do_perspectiveTransform(img, float(eff[3]))

                if (eff[4]!=None):
                    rot_str = '_rot('+str(eff[4])+')'
                    if (eff[4]!=0):
                        img = do_rotation(img, float(eff[4]))
                if (eff[5]!=None):
                    tl_str = '_TL('+str(eff[5])+')'
                    img = turn_Left(img, eff[5])

                fnStr += n_str   +\
                         blr_str +\
                         b_str   +\
                         ppt_str +\
                         rot_str +\
                         tl_str  +\
                        '.jpg'
                cv2.imwrite(fnStr, img)   
    
        print ("%d files have been generated under %s\n" % (len(effects)*len(collection.srcFiles), collection.dstPath))
    else:
        print ("No function is enabled.\n")
    return 





 #-------------------------------------------------------------------------------
#   read_setting()
#   parsing the input setting file according to the rules
#   if unknown error happens, line number will be told    
#-------------------------------------------------------------------------------
def read_setting(fname):
    lineNo = 0   
    try:
        allCollection = []
        genImg = None
        with open(fname) as f:
            for line in f:
                lineNo +=1
                line = line.strip()     ## remove space before and after data, and '\n'
                if (len(line)== 0):
                    continue
                if (line[0] == '#'):    ## skip comment line
                    continue
                if (line[0] == '{'):
                    genImg = NewImgProcessingAug()     ## start a new bank
                    continue
                if (line[0] == '}'):
                    allCollection.append(genImg)     ## save this bank
                    continue
        
                line = re.sub("'","", line)
                #line = re.sub('/','\\\\', line)
                line = re.sub('\[\]\@\!\$\%','', line)      ## remove special character ([]@!$%)
                data = line.split('=')     ## remove space and split data into variable/value
                data = [x.strip() for x in data]
                
                if (data[0] == 'srcPath'):
                    genImg.srcPath = data[1]
                    genImg.srcFiles = [sf for sf in os.listdir(data[1]) if isfile(join(genImg.srcPath, sf)) and sf.endswith(".jpg")]
                    genImg.dstPath = genImg.srcPath + '/aug/'
                    if os.path.exists(genImg.dstPath) is False:
                        os.makedirs(genImg.dstPath)
                elif (data[0] == 'file_subStr'):
                    genImg.substr = data[1]
                elif (data[0] == 'brightness'): 
                    genImg.brightness = [max(0.6, min(1.2,float(x))) for x in splitStrWithComma(data[1],0)]
                elif (data[0] == 'noise'): 
                    genImg.noise = [max(1, min(9,int(x))) for x in splitStrWithComma(data[1],0)]
                elif (data[0] == 'noise_salt'): 
                    genImg.salt = max(0, min(255,int(data[1])))
                elif (data[0] == 'noise_pepper'): 
                    genImg.pepper = max(0, min(genImg.salt,int(data[1])))
                elif (data[0] == 'blur'): 
                    genImg.blur = [max(0,min(2,int(x))) for x in splitStrWithComma(data[1],0)]
                elif (data[0] == 'perspective'): 
                    genImg.perspective = [max(0,min(0.2,float(x))) for x in splitStrWithComma(data[1],0)]
                elif (data[0] == 'rotate'): 
                    genImg.rotate = [max(-5.0,min(5.0,float(x))) for x in splitStrWithComma(data[1],0)]
                elif (data[0] == 'turnLeft'): 
                    genImg.turnLeft = [max(0,min(3,int(x))) for x in splitStrWithComma(data[1],0)]

            f.close()    
            return allCollection
            
    except ValueError:
        print('data type error. Cound not convert data.')
    except IOError:
        print ('IO Error.')
    except:
        print ('error, line %d' % (lineNo))    





   
#-------------------------------------------------------------------------------
#   do_img_aug_process
#   read the setting file and add im effects to each collection    
#-------------------------------------------------------------------------------
def do_img_aug_process(infilename):
    collections = read_setting(infilename)
    
    if collections:
        for x in collections:
            add_effects(x)
    else:            
        print ('Setting Error')
    return
    
    
    
    
    

if __name__ == "__main__":    
    
    inputfilename = 'aug_setting.txt'
    
    infile = join(os.getcwd(),inputfilename)
    if (isfile(infile)):
        do_img_aug_process(infile)
    else:
        print ("\'%s\' is not found.\n" % inputfilename)
        
        