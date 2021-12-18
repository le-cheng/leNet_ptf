import os
import numpy as np
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import data_prep_util
import indoor3d_util

# Constants
data_dir = os.path.join(ROOT_DIR, 'data')
indoor3d_data_dir = os.path.join(data_dir)
NUM_POINT = 4096
H5_BATCH_SIZE = 1000
data_dim = [NUM_POINT, 9]
label_dim = [NUM_POINT]
data_dtype = 'float32'
label_dtype = 'uint8'

# Set paths
filelist = os.path.join(BASE_DIR, 'meta/all_data_label.txt')
data_label_files = [os.path.join(indoor3d_data_dir, line.rstrip()) for line in open(filelist)]
output_dir = os.path.join(data_dir, 'indoor3d_sem_seg_hdf5_data')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_filename_prefix = os.path.join(output_dir, 'ply_data_all')
output_room_filelist = os.path.join(output_dir, 'room_filelist.txt')
fout_room = open(output_room_filelist, 'w')

# --------------------------------------
# ----- BATCH WRITE TO HDF5 -----
# --------------------------------------
batch_data_dim = [H5_BATCH_SIZE] + data_dim
batch_label_dim = [H5_BATCH_SIZE] + label_dim
h5_batch_data = np.zeros(batch_data_dim, dtype = np.float32)
h5_batch_label = np.zeros(batch_label_dim, dtype = np.uint8)
buffer_size = 0  # state: record how many samples are currently in buffer
h5_index = 0 # state: the next h5 file to save

def insert_batch(data, label, last_batch=False):
    global h5_batch_data, h5_batch_label
    global buffer_size, h5_index
    data_size = data.shape[0]
    # If there is enough space, just insert
    if buffer_size + data_size <= h5_batch_data.shape[0]:
        h5_batch_data[buffer_size:buffer_size+data_size, ...] = data
        h5_batch_label[buffer_size:buffer_size+data_size] = label
        buffer_size += data_size
    else: # not enough space
        capacity = h5_batch_data.shape[0] - buffer_size
        assert(capacity>=0)
        if capacity > 0:
           h5_batch_data[buffer_size:buffer_size+capacity, ...] = data[0:capacity, ...] 
           h5_batch_label[buffer_size:buffer_size+capacity, ...] = label[0:capacity, ...] 
        # Save batch data and label to h5 file, reset buffer_size
        h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
        data_prep_util.save_h5(h5_filename, h5_batch_data, h5_batch_label, data_dtype, label_dtype) 
        print('Stored {0} with size {1}'.format(h5_filename, h5_batch_data.shape[0]))
        h5_index += 1
        buffer_size = 0
        # recursive call
        insert_batch(data[capacity:, ...], label[capacity:, ...], last_batch)
    if last_batch and buffer_size > 0:
        h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
        data_prep_util.save_h5(h5_filename, h5_batch_data[0:buffer_size, ...], h5_batch_label[0:buffer_size, ...], data_dtype, label_dtype)
        print('Stored {0} with size {1}'.format(h5_filename, buffer_size))
        h5_index += 1
        buffer_size = 0
    return


sample_cnt = 0
for i, data_label_filename in enumerate(data_label_files):
    print(data_label_filename)
    data, label = indoor3d_util.room2blocks_wrapper_normalized(data_label_filename, NUM_POINT, block_size=1.0, stride=1.0,
                                                 random_sample=False, sample_num=None)
    print('{0}, {1}'.format(data.shape, label.shape))
    for _ in range(data.shape[0]):
        fout_room.write(os.path.basename(data_label_filename)[0:-4]+'\n')

    sample_cnt += data.shape[0]
    insert_batch(data, label, i == len(data_label_files)-1)

fout_room.close()
print("Total samples: {0}".format(sample_cnt))

# import h5py
# import numpy as np
# from open3d import *
# import open3d as o3d


# def read_label():
#     #filename = 'D:/document/dl_project/pointnet/sem_seg/data/ply_data_train.h5'
#     #f = h5py.File(filename, 'r')
#     #label = f['label'][:]
#     label = np.zeros(5)
#     return label  # (2048, 1)


# def read_data():
#     data = np.zeros((5, 1024, 3))
#     for i in range(1,6):
#         path = 'D:/document/dl_project/pointnet/sem_seg/pcd/rabbit' + str(i) + '.pcd'
#         pcd = o3d.io.read_point_cloud(path)
#         points = np.asarray(pcd.points)  # (1024,3)
#         data[i-1] = points
#     #print(data)
#     return data  # (2048,1024,3)


# def wh5():
#     data = read_data()
#     label = read_label()
#     f = h5py.File('D:/document/dl_project/pointnet/sem_seg/data/ply_data_train.h5', 'w')  # 创建一个h5文件，文件指针是f
#     f['data'] = data  # 将数据写入文件的主键data下面
#     f['label'] = label  # 将数据写入文件的主键labels下面
#     f.close()  # 关闭文件

# wh5()


import os
import sys
import numpy as np
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def loadDataFile(path):
    data = np.loadtxt(path)
    point_xyz = data[:,0:9]
    label = (data[:,9]).astype(int)   
    return point_xyz, label

def change_scale(data):
    #centre 
    xyz_min = np.min(data[:,0:3],axis=0)
    xyz_max = np.max(data[:,0:3],axis=0)
    xyz_move = xyz_min+(xyz_max-xyz_min)/2
    data[:,0:3] = data[:,0:3]-xyz_move
    #scale
    scale = np.max(data[:,0:3])
    return data[:,0:3]/scale
    
if __name__ == "__main__":
    DATA_FILES = getDataFiles(os.path.join(BASE_DIR, 'file_path.txt'))
    num_sample = 4096
    DATA_ALL = []
    for fn in range(len(DATA_FILES)):
        print(DATA_FILES[fn])
        current_data, label = loadDataFile(DATA_FILES[fn])
        #change_data = change_scale(current_data)
        #print(change_data)
        data_label = np.column_stack((current_data,label))
        DATA_ALL.append(data_label)
        
    output = np.vstack(DATA_ALL)
    print(output.shape)
    output = output.reshape(-1,num_sample,10)
         
    # 这里没将训练测试集单独分开        
    if not os.path.exists('ply_data_train1.h5'):
        with h5py.File('ply_data_train1.h5') as f:
            f['data'] = output[:,:,0:9]
            f['label'] = (output[:,:,9]).astype(int)   