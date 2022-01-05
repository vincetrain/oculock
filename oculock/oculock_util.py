import os

def mkdir(dir):
    '''
    Creates a directory within specified path.
    
    Uses .split() with provided file divider to generate a list (dir_list) containing each sub-directory of provided dir
    We then iterate between dir_list and create each subdirectory, skipping over the directories that already exist. 
    '''
    file_div = '/'
    if ('\\' in dir):
        file_div = '\\'
    dir_list = dir.split(file_div)
    
    temp = ''
    for i in (dir_list):
        temp+=i+file_div
        try:
            if not os.path.exists(temp):
                os.mkdir(temp)
        except:
            print('Error occured while making directory: \'', temp + '\'')
            break