#coding=utf-8
import numpy as np
import huffman
import bitstring
import struct
class iminfo(object):
    def __init__(self,imshape = None):
        self.imshape = imshape

class fmaps_info(object):
    def __init__(self,fmasp_shape = None):
        self.fmaps_shape = fmaps_shape

class bin_data(object):
    def __init__(self,data = None,codec = None,shape = None,lens = -1):
        self.data = data
        self.codec = codec
        self.iminfo = iminfo
        self.fmaps_info = fmaps_info
        self.lens = lens

def write_head(b_data,file_object):
    bytes = struct.pack('3i',b_data.iminfo.imshape[0],b_data.iminfo.imshape[1],b_data.iminfo.imshape[2])     # Image Shape
    offset = struct.calcsize('3i')
    file_object.write(bytes)

    bytes = struct.pack('4i',b_data.fmaps_info.fmaps_shape[0],b_data.fmaps_info.fmaps_shape[1],b_data.fmaps_info.fmaps_shape[2],b_data.fmaps_info.fmaps_shape[3])     # Feature maps Shape
    offset = struct.calcsize('4i')
    file_object.write(bytes)

    codec = b_data.codec
    bytes = struct.pack('i',len(codec))     # codec NUM
    offset = struct.calcsize('i') + offset
    file_object.write(bytes)

    for i in codec:
        cod = ''
        for j in codec[i]:
            cod = cod + str(j)
        #codec_str.append(cod)
        bytes = struct.pack('ii%ds'%len(codec[i]),i,len(codec[i]),cod.encode('utf-8'))
        offset = struct.calcsize('ii%ds'%len(codec[i])) + offset

        #bytes = struct.pack("ii",1,2)
        #print struct.unpack("ii", bytes)
        file_object.write(bytes)

    offset = struct.calcsize('i') + offset
    #print('bits:',offset)
    return offset*8

def read_head(file_object):

    #string = file_object.read(1)
    #print(string)
    b_data = bin_data()

    b_data.iminfo = struct.unpack('3i',file_object.read(12))
    b_data.fmaps_info = struct.unpack('4i',file_object.read(16))

    codec_num = struct.unpack("i",file_object.read(4))[0]
    dict_codec={}
    for i in range(codec_num):
        index, lens = struct.unpack("ii",file_object.read(8))
        #dict[index] = struct.unpack("%ds"%lens,file_object.read(lens))[0]
        str = struct.unpack("%ds"%lens,file_object.read(lens))[0]
        cod = []
        for j in str:
            cod.append(int(j))
        dict_codec[index] = cod
        #print index,dict[index]
    b_data.lens = struct.unpack("i",file_object.read(4))[0]
    #data= int(file_object.read().encode('hex'),16)
    #file_object.close()
    b_data.codec = dict_codec
    return b_data

def write(filename,im,arr,codec):
    #write
    b_data = bin_data(data=arr,codec=codec)
    file_object = open(filename, 'wb')

    b_data.fmaps_info.fmaps_shape = arr.shape
    b_data.iminfo.imshape = im.shape

    sizehead = write_head(b_data,file_object)

    sizearr = huffman.encode(arr,codec,file_object)
    sizearr = int(np.ceil(sizearr/8.0)*8)

    file_object.close()
    #print('head:',sizehead,'bytes')
    #print('arr:',sizearr,'bytes')
    #print('file sum:',sizehead + sizearr,'bytes / ',(sizehead + sizearr)/8,'bits')
    print 'head:',sizehead,'bytes'
    print 'arr:',sizearr,'bytes'
    print 'file sum:',sizehead + sizearr,'bytes /',(sizehead + sizearr)/8,'bits'
    print 'bpp: ', np.double(sizehead + sizearr) / H / W / 8.0 / 8.0
#read
def read(filename):
    file_object = open(filename, 'rb')

    b_data = read_head(file_object)
    #print(lens_arr)
    decoded = huffman.decode(b_data.lens,b_data.codec,file_object)

    file_object.close()

    #print("decoded: ",decoded)
    #print"decoded data: ",decoded
    return np.reshape(decoded,b_data.shape)
