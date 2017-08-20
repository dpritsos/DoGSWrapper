
import os
import numpy as np
import json


def LoadCrpsFnamesTags(raw_corpus_files_path=None, process_state_saving_path=None):

    # Replace the class instantiation defined variable self.corpus_files_path if any.
    if raw_corpus_files_path:
        self.corpus_files_path = raw_corpus_files_path

    if not os.path.exists(self.corpus_files_path):
        raise Exception("Corpus files path does not exist.")

    # Replace and create if required the path where the process-state files will be saved.
    if process_state_saving_path:
        self.state_save_path = process_state_saving_path

    if not os.path.exists(self.state_save_path):
        os.mkdir(self.state_save_path)
        print "New process-state saving path is: '" + self.state_save_path + "'"

    # Set the file names for the Filenames and Tags lists to be Loaded or Saved.
    corpus_files_lst_path = self.state_save_path + '/Corpus_filenames_sorted.lst'
    corpus_tags_lst_path = self.state_save_path + '/Corpus_tags_sorted.lst'

    if os.path.exists(corpus_files_lst_path) and os.path.exists(corpus_tags_lst_path):

        print "Loading HTML Filenames list and Classes Tags array..."

        # Load Filename and classes Tags.
        with open(corpus_files_lst_path, 'r') as f:
            html_file_l = json.load(f, encoding='utf-8')

        with open(corpus_tags_lst_path, 'r') as f:
            cls_tgs = json.load(f, encoding='utf-8')

    else:

        print "Loading HTML Filenames from given a file path..."

        html_file_l = list()
        cls_tgs = list()

        # Get the list of Genre argument as given to this Class and build html-file-list...
        # ...and class-genres-tags list
        for i, g in enumerate(self.classes_lst):

            # Get all files located to the genre's path 'g'
            gnrs_file_lst = file_list_frmpaths(self.corpus_files_path, [str(g + "/html/")])

            # Extends the list of html files with the set of files form genre 'g'.
            html_file_l.extend(gnrs_file_lst)

            # Extends the list of html files with the set of class tag form genre 'g',...
            # ...i.e. the index of the genre's list given as argument to this class...
            # ...(ParamGridCrossValBase()).
            cls_tgs.extend([i+1]*len(gnrs_file_lst))

        print "Saving Filenames list and Classes Tags Numpy array..."

        # Saving Filename and classes Tags lists
        with open(corpus_files_lst_path, 'w') as f:
            # HTML File List as founded in the Ext4 file system by python built-it os...
            # ...(python 2.7.x) lib
            json.dump(html_file_l, f, encoding='utf-8')

        with open(corpus_tags_lst_path, 'w') as f:
            # Assigned Genre Tags to files list Array
            json.dump(cls_tgs, f, encoding='utf-8')

    print "Returning the Corpus Filenames list and Classes Tags Numpy array."

    # Returning the filename list and the tags array.
    return np.array(html_file_l), np.array(cls_tgs)
