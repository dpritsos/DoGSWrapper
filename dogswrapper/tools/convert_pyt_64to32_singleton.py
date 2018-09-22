import tables as tb
import numpy as np
import os

src_fpath = "/mnt/turnstone/SANTINIS/TF_W3G_SANTINIS/"
dst_fpath = "/mnt/turnstone/SANTINIS/TF_W3G_SANTINIS/Old_style_index_files/"



for fname in os.listdir(src_fpath):

    if fname[-2::] == 'h5' and fname[0:6] == "Corpus":

        print "File: " + fname

        hdf = tb.open_file(src_fpath + fname, 'r')

        hdf_COPY = tb.open_file(dst_fpath + fname, 'w')

        nd = hdf.get_node('/', 'corpus_earray')  # classname='ARRAY')

        print "Array Converting..."

        array_name = nd.name

        f64_array = nd.read()

        f32_array = np.array(f64_array, dtype=np.float32)

        nd_copy_root = hdf_COPY.root

        f32_array_new = hdf_COPY.create_array(nd_copy_root, array_name, f32_array, "")

        hdf.flush()
        hdf_COPY.flush()

        print "Array Converted!"
        print "Array Copiend!"
        print

        hdf.close()
        hdf_COPY.close()

    else:

        print "File: " + fname + " Skipped"


print "File DONE!"
