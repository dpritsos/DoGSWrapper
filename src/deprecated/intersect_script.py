"""

"""

import codecs
import re
import os

def load_sets_intersected(filepath, filename):
    try:
        f = codecs.open( filepath + str(filename), "r")
    except IOError, e:
        print("FILE %s ERROR: %s" % (filename,e))
        return None
    set_re = re.compile(r'negative_indices->\[([^::alpha::]+)\]\n')
    num_re = re.compile(r'(\d+),')
    set_l = list()
    try: 
        for lnum, fileline in enumerate(f):
            set_line = set_re.findall(fileline)
            if set_line:
                num_list = num_re.findall(set_line[0])
                set_l.append( set(num_list) )
        intersected_s = set_l[0]
        for s in set_l[1:0]:
            intersected_s.intersection_update(s)    
    except Exception as e:
        print(e)
        return None
    finally:
        f.close()
    #Return the TF Vector    
    return intersected_s

def load_negative_indeces(filepath, filename):
    try:
        f = codecs.open( filepath + str(filename), "r")
    except IOError, e:
        print("FILE %s ERROR: %s" % (filename,e))
        return None
    feature_len_re = re.compile(r'\^\^\^\^ Terms kept= ([^::alpha::]+) \^\^\^\^\n')
    nu_re = re.compile(r'\+\+\+\+ for nu= ([^::alpha::]+) \+\+\+\+\n')
    bin_format_re =  re.compile(r'\*\*\*\* Binary \*\*\*\*\n')
    norm_fromat_re = re.compile('\*\*\*\* Normalised by Max Term \*\*\*\*')
    set_re = re.compile(r'negative_indices->\[([^::alpha::]+)\]\n')
    set_num_re = re.compile(r'(\d+),')
    bin_form_list = list()
    norm_form_list = list()
    try:
        for fileline in f:
            if bin_format_re.findall(fileline):
                format = 1
            if norm_fromat_re.findall(fileline):
                format = 2
            feature_len = feature_len_re.findall(fileline)
            if feature_len:
                kept_featrs = feature_len[0]
            new_nu = nu_re.findall(fileline)
            if new_nu:
                nu = new_nu[0]
            negative_set = set_re.findall(fileline)
            if negative_set:
                set_num_list = set_num_re.findall(negative_set[0]) 
                if format == 1:
                    bin_form_list.append( (kept_featrs, nu, set(set_num_list)) )
                if format == 2:
                    norm_form_list.append( (kept_featrs, nu, set(set_num_list)) )
    finally:
        f.close()    
    return ( bin_form_list, norm_form_list )


#Wrappers for regular expression results - for load_tptn_matrix() function
def re_list(func):
    def list_re(*args, **kwargs):
        res = func(*args, **kwargs)
        if res:
            return list(res[0])
        else:
            return None
    return list_re

@re_list    
def re_m(regexp, str):
    regexp = re.compile(r'(\d+)\t(\d+)\t(\d+)\n')
    return regexp.findall(str)
#Wrappers for regular expression results - End

def load_tptn_matrix(filepath, filename):
    try:
        f = codecs.open( filepath + str(filename), "r")
    except IOError, e:
        print("FILE %s ERROR: %s" % (filename,e))
        return None
    feature_len_re = re.compile(r'\^\^\^\^ Terms kept= ([^::alpha::]+) \^\^\^\^\n')
    nu_re = re.compile(r'\+\+\+\+ for nu= ([^::alpha::]+) \+\+\+\+\n')
    bin_format_re =  re.compile(r'\*\*\*\* Binary \*\*\*\*\n')
    norm_fromat_re = re.compile('\*\*\*\* Normalised by Max Term \*\*\*\*')
    matrix_re = re.compile(r'(\d+)\t(\d+)\t(\d+)\n')
    bin_form_list = list()
    norm_form_list = list()
    matrixz_p = list()
    matrixz_n = list()
    try:
        for fileline in f:
            if bin_format_re.findall(fileline):
                format = 1
            if norm_fromat_re.findall(fileline):
                format = 2
            feature_len = feature_len_re.findall(fileline)
            if feature_len:
                kept_featrs = feature_len[0]
            new_nu = nu_re.findall(fileline)
            if new_nu:
                nu = new_nu[0]
            matrix_row = re_m(matrix_re,fileline)
            if matrix_row:
                matrixz_p.append(matrix_row[0])  
                matrixz_n.append(matrix_row[1])
                if len(matrixz_p) == 7 and format == 1: #7 was 5
                    bin_form_list.append( (kept_featrs, nu, [ matrixz_p ], [ matrixz_n ] ) )
                    matrixz_p = list()
                    matrixz_n = list()
                if len(matrixz_p) == 7 and format == 2:
                    norm_form_list.append( (kept_featrs, nu, [ matrixz_p ], [ matrixz_n ] ) )
                    matrixz_p = list()
                    matrixz_n = list()
    finally:
        f.close()    
    return ( bin_form_list, norm_form_list )

def intersect_indeces(form_l_1, form_l_2):
    """It changes the form_l_1 list and returns it / Needs some more explanatory comments"""
    for i in range(len(form_l_1)):
        ftrs, nu, neg_set = form_l_1[i]
        ftrs_2, nu_2, neg_set_2 = form_l_2[i]
        #print ftrs, ftrs_2, nu, nu_2
        if nu == nu_2: #### if ftrs == ftrs_2 and nu == nu_2: CHECK FOR ERRRRORRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR
            neg_set.intersection_update(neg_set_2)
        else:
            raise Exception("Unexpected: Not equal parameters")
    return form_l_1

def merge_matrixs(matrix_l_1, matrix_l_2):
    """It changes the matrix_l_1 list and returns it / Needs some more explanatory comments"""
    for i in range(len(matrix_l_1)):
        ftrs, nu, m_p, m_n = matrix_l_1[i]
        ftrs_2, nu_2, m_p_2, m_n_2 = matrix_l_2[i]
        if nu == nu_2: #### if ftrs == ftrs_2 and nu == nu_2: CHECK FOR ERRRRORRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR
            #print "LEN", len(m_p) 
            m_p.extend(m_p_2) 
            m_n.extend(m_n_2)
            #print "LEN", len(m_p)
        else:
            raise Exception("Unexpected: Not equal parameters")
    return matrix_l_1

def group_indeces(indices, groups, pgs_per_group):
    grps_l = [[i,list()] for i in range(40, (groups+1)*pgs_per_group, pgs_per_group)] #500
    for idx in indices:
        for up_lim, grp_list in grps_l:
            if float(idx) < float(up_lim):
                grp_list.append(idx)
                break 
    for tupl in grps_l:
        tupl[1] = [ int(s_num) for s_num in tupl[1] ] 
        tupl[1].sort()
    return grps_l
    
if __name__ == '__main__':
    base_filepath = "/home/dimitrios/Synergy-Crawler/Santini_corpus/OC-SVM_terms/T/" #saved_pages/OC-SVM_featurs10-710_Step50_TERMS/" #Santini_corpus/"
    flist = [files for path, dirs, files in os.walk(base_filepath)]
    flist = flist[0]
    plot_l = list()
    intrscted_s_l = list()
    for z, file in enumerate(flist):
        print file
        intersected_s = load_sets_intersected(base_filepath, file)
        print file," Intersection: ",intersected_s
        print "Intersected indices amount:",len(intersected_s)
        intrscted_s_l.append(intersected_s)
    total_inter_s = intrscted_s_l[0]
    for s in intrscted_s_l[1:]:
        total_inter_s.intersection_update(s)
    print "Total intersection: ",total_inter_s 
    print "Intersected indices amount: ",len(total_inter_s)
    
    bin_form_list_1, norm_form_list_1 = load_negative_indeces(base_filepath, flist[0])
    #print bin_form_list_1
    for z, file in enumerate(flist[1:]):
        bin_form_list_2, norm_form_list_2 = load_negative_indeces(base_filepath, file)
        bin_form_list_1 = intersect_indeces(bin_form_list_1, bin_form_list_2)
        norm_form_list_1 = intersect_indeces(norm_form_list_1, norm_form_list_2)
    
    #for line in bin_form_list_1: 
    #    print line[0], line[1]
    #    print len(line[2])
    #bin_form_list_1
    #print norm_form_list_1
    bin_form_list_matrix, norm_form_list_matrix = load_tptn_matrix(base_filepath, flist[0])
    for z, file in enumerate(flist[1:]):
        bin_matrix, norm_matrix = load_tptn_matrix(base_filepath, file)
        bin_form_list_matrix = merge_matrixs(bin_form_list_matrix, bin_matrix)
        norm_form_list_matrix = merge_matrixs(norm_form_list_matrix, norm_matrix)
    
    #print bin_form_list_matrix
    
    with open("/home/dimitrios/Synergy-Crawler/Santini_corpus/OC-SVM_terms/bin_confusion_matricies.txt" , "w") as f:
        #Total amount of indices Never Classified
        f.write("***\n\n") 
        f.write("TOTAL amount of Intersected Negative Indices (Bin & Normalized) = " + str(len(total_inter_s)) +"\n\n")
        group_list = group_indeces(total_inter_s, 7, 40) #5, 500)
        for y, genre in enumerate([ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage" ]): #"wiki_pages", "product_companies", "forum", "blogs", "news" ]):
            f.write( genre + "'s Intersected Negative Indices (" + str(len(group_list[y][1])) + ") = " + str(group_list[y][1]) +"\n\n")
        f.write("***\n\n\n")
        f.write("**** BINARY FORMAT EXPERIMENTS ****\n\n")
        #Confusion Matrix Per Experiments starts here 
        fname_l = list()
        for fname in flist:
            fname_l.append( fname[0:5] )
        for idx, (ftrs, nu, m_p, m_n) in enumerate(bin_form_list_matrix):
            f.write("EXPERIMENT No" + str(idx+1) + "(x5 Genres)\n")
            f.write("Features = " + ftrs  + "\tnu = " + nu + "\n")
            f.write("\t\t")
            for fname in fname_l:
                f.write( fname + "\t\t")
            f.write("\n")
            for i in range(len(m_p)):
                f.write(fname_l[i]+"\t\t")
                for z in range(len(m_p)):
                    f.write( m_p[z][i] + "\t\t" )
                f.write("\n")    
            #f.write("Intersected Negative Indices = " + str(bin_form_list_1[idx][2]) +"\n\n")
            f.write("Amount of Intersected Negative Indices = " + str(len(bin_form_list_1[idx][2])) +"\n\n")
            group_list = group_indeces(bin_form_list_1[idx][2], 7, 40) #7, 40
            for y, genre in enumerate([ "Forum", "Wiki_pages", "News", "Product_Companies", "Blogs"]):
                f.write( genre + "'s Intersected Negative Indices (" + str(len(group_list[y][1])) + ") = " + str(group_list[y][1]) +"\n\n")
            f.write("\n\n\n")
            
    
    
    
    
    
    
    
    