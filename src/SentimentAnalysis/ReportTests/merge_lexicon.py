base_dir = "../../../data/Lexicons/"

def read_file(path):
    file = open(path, "r")
    dict = {}
    for line in file:
        lst = line.split("\t")
        dict[lst[0]] = float(lst[1])
    file.close()
    return dict

def merge(lex_1, lex_2, save_path):
    dict_1 = read_file(lex_1)
    dict_2 = read_file(lex_2)

    for key in dict_2:
        if key in dict_1:
            dict_1[key] = (dict_1[key] + dict_2[key])/2
        else:
            dict_1[key] = dict_2[key]

    output_string = ""
    for key in sorted(dict_1):
        output_string += (key + "\t" + str(dict_1[key]) + "\n")

    write_file = open(save_path, "w")
    write_file.write(output_string)
    write_file.close()

merge(base_dir + "msc_student.txt", base_dir + "NTUSD_lex.txt", base_dir + "NTUSD_plus_msc_student.txt")