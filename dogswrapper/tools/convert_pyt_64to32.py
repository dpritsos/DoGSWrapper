import tables as tb
import numpy as np


hdf = tb.open_file(
    "/media/dimitrios/71e878e6-723d-4e8a-adea-b7ee4008d304/Openness_RFSE_W1G_Gensim_KI04_2017_11_13_PART_TWO.h5", 'r'
)

hdf_COPY = tb.open_file(
    "/media/dimitrios/71e878e6-723d-4e8a-adea-b7ee4008d304/Openness_RFSE_W1G_Gensim_KI04_2017_11_13_PART_TWO_COPY.h5", 'w'
)

for nd in hdf.walk_nodes(where='/', classname='Array'):

    print "Array Converting..."

    array_path_nodes_lst = nd._v_pathname.split('/')[0:-1]
    array_path = '/'.join(array_path_nodes_lst)

    print array_path

    array_name = nd.name
    f64_array = nd.read()

    f32_array = np.array(f64_array, dtype=np.float32)

    # hdf.remove_node(array_path, array_name)

    next_group = hdf_COPY.root
    for path_nd in array_path_nodes_lst[1::]:
        try:
            next_group = hdf_COPY.get_node(next_group, path_nd)
        except Exception as exp_signal:
            next_group = hdf_COPY.create_group(next_group, path_nd)

    print next_group
    f32_array_new = hdf_COPY.create_array(array_path, array_name, f32_array, "")

    hdf.flush()
    hdf_COPY.flush()

    print "Array Converted!"
    print "Array Copiend!"

hdf.close()
hdf_COPY.close()

print "File DONE!"


# 'predicted_Y'
# 'predicted_scores'
# 'max_sim_scores_per_iter'
# 'predicted_classes_per_iter'
