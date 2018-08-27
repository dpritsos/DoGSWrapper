


""" REQURIES UPDATE TO BE CONFORMED WITH THE NEW UDATE html2Vectors LIBRARY """


import os
import sys
sys.path.append('../../synergeticprocessing/src')
sys.path.append('../../html2vectors/src')
import filehandlers.tfdfileshandler
import filehandlers.basefilehandlers 
import html2tf.dictionaries.words
import html2tf.dictionaries.cngrams
import htmlattrib.regex
import re
 
class Html2TF_Concurrent(object):
    
    def __init__(self, *args, **kwargs):
        if args:
            raise Exception("Only Keywords arguments are valid: 1.n=<ngrams size> if omitted 'word' terms assumed , lowecase=True/False is mandatory")
        case = kwargs.pop('lowercase', None)
        if not case: 
            raise Exception("keyword parameter lowecase=True/False is mandatory")
        nsize = kwargs.pop('n', None)
        if nsize:
            self.h2tf = html2tf.dictionaries.cngrams.Html2TF( n=nsize, lowercase=case )
        else:
            self.h2tf = html2tf.dictionaries.words.Html2TF( lowercase=case )
        self.tfdfh = filehandlers.tfdfileshandler.TFdictFilesHandler()
        self.bfh = filehandlers.basefilehandlers.BaseFileHandler()
    
    def exec_for(self, basefrom_l, pathsfrom, tfv_file, tfd_file, err_file, load_encoding='utf-8', save_encoding='utf-8', error_handling='strict', low_mem='True'):
        #Find which base-path is present
        if isinstance(basefrom_l, list):
            for basefrom in basefrom_l:
                filepath = basefrom + pathsfrom    
                flist = [files for path, dirs, files in os.walk(filepath)]
                if flist:
                    break
        else:
            basefrom = basefrom_l
        #Build the list of filename-TF vectors 
        fn_tf_l = self.h2tf.from_paths(basefrom, pathsfrom, load_encoding, error_handling, low_mem)
        #Find HTML Files have returned None, i.e. are error pages or the have other problem under investigation
        rmv_wpgs = str()
        rmv_idxs_l = list()
        for i, (fn, tf) in enumerate(fn_tf_l):
            if not tf:
                rmv_wpgs = rmv_wpgs + str(fn) + ' => ' + str(tf) + '\n'
                rmv_idxs_l.append( i )
        #If error pages have occured 
        if len(rmv_idxs_l):
            #Remove this pages from the valid filename-TF vectors list 
            for i, rmv_i in enumerate(rmv_idxs_l):
                fn_tf_l.pop( rmv_i - i )
            #Save the error pages to an error file
            self.bfh.save_files(None, [ basefrom + err_file, len(rmv_idxs_l) ] )
        #Save the list of filename-TF vectors
        if self.tfdfh.save_tf_dct_lst(basefrom + tfv_file, fn_tf_l, save_encoding, error_handling):
            report = [ "TF Vector(s) File %s : Saved - Error pages encountered: %s" % (tfv_file, len(rmv_idxs_l)) ]
        else:
            report = [ "TF Vector(s) File %s : Error Occurred" % tfv_file ]
        return report        


class Text2TF_ngrams(htmlattrib.regex.BaseHtmlAttrib, html2tfd.charngrams.BaseString2TF, filehandlers.basefilehandlers.BaseFileHandler):
    
    def __init__(self, n=3, lowercase=False):
        html2tfd.charngrams.BaseString2TF.__init__(self, n)
        if lowercase:
            self._attrib = self.__attrib_lowercase
        
    def _attrib(self, xhtml_str):
        return self.nf_dict( xhtml_str )
    
    def __attrib_lowercase(self, xhtml_str):
        return self.nf_dict( xhtml_str.lower() )
    
    
class Text2TF_words(htmlattrib.regex.BaseHtmlAttrib, html2tfd.words.BaseString2TF, filehandlers.basefilehandlers.BaseFileHandler):
    
    def __init__(self, lowercase=False):
        html2tfd.words.BaseString2TF.__init__(self)
        if lowercase:
            self._attrib = self.__attrib_lowercase
        
    def _attrib(self, xhtml_str):
        return self.nf_dict( xhtml_str )
    
    def __attrib_lowercase(self, xhtml_str):
        return self.nf_dict( xhtml_str.lower() )


class Text2TF_Concurrent(object):
    
    def __init__(self, *args, **kwargs):
        if args:
            raise Exception("Only Keywords arguments are valid: 1.n=<ngrams size> if omitted 'word' terms assumed , lowecase=True/False is mandatory")
        case = kwargs.pop('lowercase', None)
        if not case: 
            raise Exception("keyword parameter lowecase=True/False is mandatory")
        nsize = kwargs.pop('n', None)
        if nsize:
            self.tx2tf = Text2TF_ngrams( n=nsize, lowercase=case )
        else:
            self.tx2tf = Text2TF_words( lowercase=case )
        self.tfdfh = filehandlers.tfdfileshandler.TFdictFilesHandler()
        self.bfh = filehandlers.basefilehandlers.BaseFileHandler()
        self.tab = re.compile('\t', re.UNICODE )
        self.nl = re.compile('\n', re.UNICODE ) 
        self.cr = re.compile('\r', re.UNICODE )                    
        self.f = re.compile('\f', re.UNICODE ) 
        self.vtab = re.compile('\v', re.UNICODE)
        self.strc = re.compile('\xc2\x85', re.UNICODE)
    
    def exec_for(self, basefrom_l, pathsfrom, tfv_file, tfd_file, err_file, load_encoding='utf-8', save_encoding='utf-8', error_handling='strict', low_mem='True'):
        #Find which base-path is present
        if isinstance(basefrom_l, list):
            for basefrom in basefrom_l:
                filepath = basefrom + pathsfrom    
                flist = [files for path, dirs, files in os.walk(filepath)]
                if flist:
                    break
        else:
            basefrom = basefrom_l
        #Build the list of filename-TF vectors 
        fn_tf_l = self.tx2tf.from_paths(basefrom, pathsfrom, load_encoding, error_handling, low_mem)
        #Find HTML Files have returned None, i.e. are error pages or the have other problem under investigation
        rmv_wpgs = str()
        rmv_idxs_l = list()
        for i, (fn, tf) in enumerate(fn_tf_l):
            if not tf:
                rmv_wpgs = rmv_wpgs + str(fn) + ' => ' + str(tf) + '\n'
                rmv_idxs_l.append( i )
        #If error pages have occured 
        if len(rmv_idxs_l):
            #Remove this pages from the valid filename-TF vectors list 
            for i, rmv_i in enumerate(rmv_idxs_l):
                fn_tf_l.pop( rmv_i - i )
            #Save the error pages to an error file
            self.bfh.save_files(None, [[ basefrom + err_file, str(rmv_idxs_l) ]] )
        ### TEMPORERARLY REPLACING WHITESPACE CHARS WITH CAPITAL LETTER RESPECTIVE TO THE LETTER OF WHITE SPACE USED IN REGEX and Pythom/C Strings
        for fn_tf in fn_tf_l:
            new_tfd = dict()
            for trm, frq in fn_tf[1].items():
                for c in range(len(trm)):
                    trm = trm.replace('\t', 'T') 
                    trm = trm.replace('\n', 'N')
                    trm = trm.replace('\r', 'R')
                    trm = trm.replace('\f', 'F')
                    trm = trm.replace('\v', 'V')
                    trm = trm.replace('\xc2\x85', 'T')
                    #trm = self.tab.sub('T', trm)
                    #trm = self.nl.sub('N', trm) 
                    #trm = self.cr.sub('R', trm)                    
                    #trm = self.f.sub('F', trm) 
                    #trm = self.vtab.sub('V', trm)
                    #trm = self.strc.sub('T', trm)
                new_tfd[trm] = frq
            fn_tf[1] = new_tfd 
        ### TEMP WHITESPACE REPLACEMENT  -  END
        #Save the list of filename-TF vectors
        if self.tfdfh.save_tf_dct_lst(basefrom + tfv_file, fn_tf_l, save_encoding, error_handling):
            report = [ "TF Vector(s) File %s : Saved - Error pages encountered: %s" % (tfv_file, len(rmv_idxs_l)) ]
        else:
            report = [ "TF Vector(s) File %s : Error Occurred" % tfv_file ]
        return report        




        