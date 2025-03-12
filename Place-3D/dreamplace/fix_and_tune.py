import argparse


def fix_and_tune(input_def, output_def):
    new_def = ''
    with open(input_def, 'r', encoding='utf-8') as def_file:
        flag = False
        pre_line = ''
        for line in def_file:
            if 'COMPONENTS' in line:
                flag = True
                if 'END' in line:
                    flag = False
            if flag:
                if 'fakeram' in pre_line:
                    line = line.replace('PLACED', 'FIXED')
                    # only "N" is allowed
                    line = line.replace('FS', 'N')  
                    # adjust y coordinate for pin alignment
                    y = line.split()[4]
                    num_row = int((float(y) - 70) / 280)
                    y_ = str(280 * num_row + 70)
                    print(y, y_)
                    line = line.replace(y, y_)

            new_def = new_def + pre_line    
            pre_line = line
        new_def = new_def + line
    with open(output_def, 'w', encoding='utf-8') as def_file:
        def_file.write(new_def)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune the coordinates of macros and fix them. ")
    parser.add_argument('--in_def', type=str)
    parser.add_argument('--out_def', type=str)
    args = parser.parse_args()
    
    fix_and_tune(args.in_def, args.out_def)