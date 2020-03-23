from data_interface import WeaponDatasetGenerator

print('start generating data ...')

root = '/visinf/projects_students/Smiths_LKA_Weapons/ctix-lka-20190503/'
target = '/fastdata/Smiths_LKA_Weapons_Down/len_8/'

WeaponDatasetGenerator(root, target, side_len=8).generate_data()

print('... done generating data')
