import h5py
import numpy as np


def read_hdf5_field(h5file, field_ref):
    """Read a single field from the HDF5 file using its reference."""
    if isinstance(field_ref, h5py.Reference):
        return h5file[field_ref][()]
    return field_ref


def parse_digit_struct(h5file, digit_struct_group):
    """Parse the digitStruct group and extract all information."""
    data = []
    for i in range(len(digit_struct_group['name'])):
        item = {}

        # Read name
        name_ref = digit_struct_group['name'][i].item()
        name_data = h5file[name_ref][()]
        if isinstance(name_data, bytes):
            item['name'] = name_data.decode('utf-8')
        elif isinstance(name_data, np.ndarray):
            item['name'] = ''.join(chr(c) for c in name_data)
        else:
            item['name'] = str(name_data)

        # Read bbox fields
        bbox_ref = digit_struct_group['bbox'][i].item()
        bbox_group = h5file[bbox_ref]
        bbox_data = {}
        for key in bbox_group.keys():
            values = bbox_group[key][:]
            if len(values.shape) == 0:
                value = read_hdf5_field(h5file, values.item())
            else:
                value = [read_hdf5_field(h5file, val.item()) for val in values]
            bbox_data[key] = value

        item['bbox'] = bbox_data
        data.append(item)

    return data


def load_h5(file_path):
    """Load .mat file using .h5py and return a list of dictionaries with the data."""
    with h5py.File(file_path, 'r') as f:
        digit_struct_group = f['digitStruct']
        parsed_data = parse_digit_struct(f, digit_struct_group)

    return parsed_data


if __name__ == "__main__":
    # Test the function
    svhn = "C:\\piao_programs\\py_programs\\DeepLearningProject\\Manifold_SmiLearn\\data\\processed\\svhn\\test\\digitStruct.mat"
    dct = load_h5(svhn)

    # 获取第一个图像的信息
    first_item = dct[0]
    print(f"First image name: {first_item['name']}")

    # 获取第一个图像的边界框信息
    first_bbox = first_item['bbox']

    # 打印边界框信息
    print("Bounding boxes for the first image:")
    for k in ['label', 'top', 'left', 'width', 'height']:
        if k in first_bbox:
            v = first_bbox[k]
            if isinstance(v, (list, np.ndarray)):
                print(f"{k}: {v}")
            else:
                print(f"{k}: [{v}]")