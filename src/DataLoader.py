import numpy as np
import os
import glob
import time
import cv2
import pickle
from PIL import Image
from io import BytesIO

class DataLoader:

    def __init__( self,
                  dataPath,
                  filenames,
                  lblFilename ,
                  batchSize = 20,
                  dim = 224,
                  timesteps = 16 ):
        self.dim       = dim
        self.timesteps = timesteps
        self.filenames = filenames
        self.length    = filenames.shape[ 0 ]
        
        self.setBatchSize( batchSize )
        self.shuffle()
        self.generateLabelsDict( lblFilename )
        with open( dataPath, 'rb' ) as f:
            self.data = pickle.load( f )
        
        self.curBatchLen = None
        self.batch  = None
        self.labels = None


    def generateLabelsDict( self, filename ):
        self.labelsDict = dict()
        f = open( filename , 'r' )
        for line in f.readlines():
            self.labelsDict[ line.split()[ 1 ] ] = line.split()[ 0 ]


    def shuffle( self ):
        self.ids = np.arange( self.length )
        np.random.shuffle( self.ids )
        self.index = 0


    def setBatchSize( self , batchSize ):
        self.batchSize = batchSize


    def incIndex( self ):
        self.index = self.index + self.batchSize
        if self.index >= self.length:
            self.shuffle()


    def allocateBatch( self, batchLen ):
        if self.curBatchLen != batchLen :
            self.curBatchLen = batchLen
            self.batch  = np.empty( ( self.curBatchLen, self.dim, self.dim, 2, self.timesteps),
                                      dtype = 'float32')
            self.labels = np.empty( (self.curBatchLen , 101) )


    def randomBatchPaths( self ):
        batchPaths = list()
        endIndex = self.index + self.batchSize
        if endIndex > self.length:
            endIndex = self.length
        for i in range( self.index , endIndex):
            batchPaths += [ self.filenames[ self.ids[ i ] ].split('.')[0] ]
        self.allocateBatch( len( batchPaths ) )
        self.incIndex()
        return batchPaths


    def randomCrop( self , inp ):
        dim = self.dim
        imgDimX = inp.shape[ 1 ]
        imgDimY = inp.shape[ 0 ]
        left    = np.random.randint( imgDimX - dim + 1 )
        top     = np.random.randint( imgDimY - dim + 1 )
        right   = left + dim
        bottom  = top  + dim
        crop    = inp[ top : bottom , left : right ]
        return crop


    def randomFlip( self , inp ):
        if np.random.random() > 0.5:
            inp = np.flip( inp , 1 )
        return inp


    def loadFlow( self, video, index ):
        u_img = Image.open( video ['u'] [index] )
        v_img = Image.open( video ['v'] [index] )
        u_range = video ['u_range'] [index]
        v_range = video ['v_range'] [index]

        u = np.array( u_img, dtype = 'float32' ).copy()
        v = np.array( v_img, dtype = 'float32' ).copy()
        cv2.normalize( u, u,  u_range[ 0 ], u_range[ 1 ], cv2.NORM_MINMAX )
        cv2.normalize( v, v,  v_range[ 0 ], v_range[ 1 ], cv2.NORM_MINMAX )
        return u, v


    def stackFlow( self, video, start, timesteps ):
        flowList = list()
        for i in range( start, start+timesteps ):
            u, v = self.loadFlow( video, i )
            flowList += [ np.array( [ u , v ] ) ]
        stack = np.array( flowList )
        stack = np.transpose( stack , [ 2 , 3 , 1 , 0 ] )
        stack = self.randomCrop( stack )
        stack = self.randomFlip( stack )
        return stack


    def randomBatchFlow( self ):
        batchPaths = self.randomBatchPaths()
        # labels = list()
        for i, batchPath in enumerate(batchPaths):
            # fullPath  = os.path.join( self.rootPath, batchPath )
            # video = pickle.load( open( fullPath + '.pickle' , 'rb' ) )
            video = self.data[ batchPath ]
            
            start = np.random.randint( len( video[ 'u' ] ) - self.timesteps )
            # batch  += [ self.stackFlow( video, start, self.timesteps ) ]

            stack = self.stackFlow( video, start, self.timesteps )
            self.batch[ i ] = self.stackFlow( video, start, self.timesteps )
            className = batchPath.split('/')[ 0 ]
            self.labels[ i ] = np.zeros(( 101 ) , dtype = 'float32')
            self.labels[ i ][ int( self.labelsDict[ className ] ) - 1 ] = 1.0
            # labels += [ label ]
            # self.labels[ i ] = label

        batch = np.reshape( self.batch , [ self.curBatchLen, 
                                           self.dim * self.dim * 2 * self.timesteps] )
        # labels = np.array( labels )
        return batch, self.labels


if __name__ == '__main__':
    # rootPath    = '/home/olorin/Documents/caetano/datasets/UCF-101_flow'
    # rootPath    = '/media/olorin/Documentos/caetano/datasets/UCF-101_flow'
    # rootPath    = '/home/caetano/Documents/datasets/UCF-101_flow'
<<<<<<< HEAD
    # rootPath    = '/lustre/cranieri/UCF-101_flow'
    dataPath    = '/lustre/cranieri/UCF-101_flow/data.pickle'
=======
    rootPath    = '/lustre/cranieri/UCF-101_flow'
>>>>>>> 1d1214a307abae264b381487f799cdc11999fe80
    filenames   = np.load( '../splits/trainlist011.npy' )
    lblFilename = '../classInd.txt'
    dataLoader = DataLoader( dataPath, filenames, lblFilename )
    # batch, labels =  dataLoader.randomBatchFlow()
    for i in range( 100 ):
        t = time.time()
        batch, labels =  dataLoader.randomBatchFlow()
        print( i , batch.shape , labels.shape )
        print( 'Total time:' , time.time() - t )







