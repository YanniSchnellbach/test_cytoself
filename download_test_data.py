# # download sample data
# from cytoself.datamanager.opencell import DataManagerOpenCell

# print("STEP 1: Download sample Data")
# data_ch = ['pro', 'nuc']
# datapath = 'sample_data'  # path to download sample data
# DataManagerOpenCell.download_sample_data(datapath)  # donwload data
# datamanager = DataManagerOpenCell(datapath, data_ch, fov_col=None)
# datamanager.const_dataloader(batch_size=32, label_name_position=1)

from cytoself.datamanager.opencell import DataManagerOpenCell

data_ch = ['pro', 'nuc']
datapath = 'Testdata' # path to download sample data
DataManagerOpenCell.download_sample_data(datapath) # donwload data
datamanager = DataManagerOpenCell(datapath, data_ch, fov_col=None)
