import sys
#*****************************************************
# Imputation conversion script
# ****************************************************
input_raw_file = sys.argv[1]
input_raw_converted_file = sys.argv[2]
raw_file = open(input_raw_file, 'r')
raw_converted_file = open(input_raw_converted_file, 'w') 
for line in raw_file:
    line = line.rstrip("\n\r")
    if line[:5] == 'wt_aa':
        raw_converted_file.write('wt_aa' +'\t' + 'pos' + '\t' + 'mut_aa' + '\t' + 'wt_codon' + '\t' + 'mut_codon' + '\t' + 'annotation' + '\t' + 'nonselect' + '\t' + 'select' + '\t' + 'controlNS' + '\t' + 'controlS' + '\t' + 'replicate_id' + '\n')
    else:
        lst_line = line.split('\t')
        raw_converted_file.write(lst_line[0] +'\t' + lst_line[1] + '\t' + lst_line[2] + '\t' + lst_line[3] + '\t' + lst_line[4]+ '\t' + lst_line[5] + '\t' + lst_line[6] + '\t' + lst_line[8] + '\t' + lst_line[10] + '\t' + lst_line[12] + '\t' + '1\n' )
        raw_converted_file.write(lst_line[0] +'\t' + lst_line[1] + '\t' + lst_line[2] + '\t' + lst_line[3] + '\t' + lst_line[4]+ '\t' + lst_line[5] + '\t' + lst_line[7] + '\t' + lst_line[9] + '\t' + lst_line[11] + '\t' + lst_line[13] + '\t' + '2\n' )

raw_file.close()
raw_converted_file.close()  
