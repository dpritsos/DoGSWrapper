
import os
import sys
sys.path.append('../../synergeticprocessing/src')
sys.path.append('../../html2vectors/src')
import html2tf.tables.cngrams
import html2tf.tables.tbtools as tbtls

 
class Html2TF_Concurrent(object):
    
    def __init__(self, *args, **kwargs):
        if args:
            raise Exception("Only Keywords arguments are valid: 1.n=<ngrams size> if omitted 'word' terms assumed , lowecase=True/False is mandatory")
        case = kwargs.pop('lowercase', None)
        if not case: 
            raise Exception("keyword parameter lowecase=True/False is mandatory")
        nsize = kwargs.pop('n', 3)
        valid = kwargs.pop('valid_html', False)
        if nsize:
            self.h2tf = html2tf.tables.cngrams.Html2TF( n=nsize, lowercase=case, valid_html=valid)
        else:
            pass #self.h2tf = html2tf.tables.words.Html2TF( lowercase=case )
    
    def exec_for(self, basefrom_l, pathsfrom, h5file, tables_group, name_GenrePageListTable, load_encoding='utf-8', error_handling='strict'):
        #Find which base-path is present
        if isinstance(basefrom_l, list):
            for basefrom in basefrom_l:
                filepath = basefrom + pathsfrom    
                flist = [files for path, dirs, files in os.walk(filepath)]
                if flist:
                    break
        else:
            basefrom = basefrom_l
        #Building the filename-TF vectors Tables 
        tablesGroup, GenrePageListTable = self.h2tf.from_paths2tbls(h5file, tables_group, name_GenrePageListTable, basefrom, pathsfrom, load_encoding, error_handling)
        if tablesGroup and GenrePageListTable:
            report = [ "Tables with TF Vector read for Genre: %s" % tables_group]
        else:
            report = [ "Error Occurred for Genre's TF-tables creation - Genre:" % tables_group ]
        return report        






        